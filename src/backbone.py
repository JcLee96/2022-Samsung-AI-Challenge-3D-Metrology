"""EfficientNet encoder adapted from lukemelas/EfficientNet-PyTorch (v0.7.0).

Changes vs. upstream: GroupNorm instead of BatchNorm, stride-1 stem, and
endpoints detected on width reduction, so 72x48 SEM images keep enough resolution.
"""
import collections
import math
import re

import torch
from torch import nn
from torch.nn import functional as F

GlobalParams = collections.namedtuple('GlobalParams', [
    'width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate',
    'batch_norm_epsilon', 'drop_connect_rate', 'depth_divisor', 'min_depth'])
BlockArgs = collections.namedtuple('BlockArgs', [
    'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
    'input_filters', 'output_filters', 'se_ratio', 'id_skip'])

# width, depth, nominal resolution (drives static padding), dropout
EFFICIENTNET_PARAMS = {
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
}

BLOCK_STRINGS = [
    'r1_k3_s11_e1_i32_o16_se0.25',
    'r2_k3_s22_e6_i16_o24_se0.25',
    'r2_k5_s22_e6_i24_o40_se0.25',
    'r3_k3_s22_e6_i40_o80_se0.25',
    'r3_k5_s11_e6_i80_o112_se0.25',
    'r4_k5_s22_e6_i112_o192_se0.25',
    'r1_k3_s11_e6_i192_o320_se0.25',
]


class Conv2dStaticSamePadding(nn.Conv2d):
    """TF-style 'SAME' conv whose padding is fixed from a nominal image size."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        pad_h = max((math.ceil(ih / sh) - 1) * sh + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((math.ceil(iw / sw) - 1) * sw + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        return F.conv2d(self.static_padding(x), self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class SwishImplementation(torch.autograd.Function):
    """Swish with a hand-written backward that does not keep the sigmoid output."""

    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


def round_filters(filters, gp):
    """Scale a channel count by the width multiplier, rounded to the depth divisor."""
    if not gp.width_coefficient:
        return filters
    divisor = gp.depth_divisor
    min_depth = gp.min_depth or divisor
    filters *= gp.width_coefficient
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, gp):
    """Scale a block repeat count by the depth multiplier."""
    return repeats if not gp.depth_coefficient else int(math.ceil(gp.depth_coefficient * repeats))


def drop_connect(inputs, p, training):
    """Stochastic depth: drop whole samples of a residual branch with probability p."""
    if not training:
        return inputs
    keep_prob = 1 - p
    random_tensor = keep_prob + torch.rand([inputs.shape[0], 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    return inputs / keep_prob * torch.floor(random_tensor)


def output_image_size(image_size, stride):
    """Nominal spatial size after a 'SAME' conv with the given stride."""
    h, w = (image_size, image_size) if isinstance(image_size, int) else image_size
    stride = stride if isinstance(stride, int) else stride[0]
    return [int(math.ceil(h / stride)), int(math.ceil(w / stride))]


def decode_block_string(block_string):
    """Parse a block string such as 'r1_k3_s11_e1_i32_o16_se0.25' into BlockArgs."""
    options = {}
    for op in block_string.split('_'):
        splits = re.split(r'(\d.*)', op)
        if len(splits) >= 2:
            options[splits[0]] = splits[1]
    return BlockArgs(
        num_repeat=int(options['r']), kernel_size=int(options['k']),
        stride=[int(options['s'][0])],  # list (not int) on purpose, as upstream v0.7.0
        expand_ratio=int(options['e']), input_filters=int(options['i']),
        output_filters=int(options['o']),
        se_ratio=float(options['se']) if 'se' in options else None,
        id_skip='noskip' not in block_string)


class MBConvBlock(nn.Module):
    """Mobile inverted bottleneck block with squeeze-excitation and GroupNorm."""

    def __init__(self, block_args, gp, image_size):
        super().__init__()
        self._block_args = block_args
        eps = gp.batch_norm_epsilon
        self.has_se = block_args.se_ratio is not None and 0 < block_args.se_ratio <= 1
        self.id_skip = block_args.id_skip

        inp = block_args.input_filters
        oup = inp * block_args.expand_ratio
        if block_args.expand_ratio != 1:
            self._expand_conv = Conv2dStaticSamePadding(inp, oup, 1, bias=False, image_size=image_size)
            self._gn0 = nn.GroupNorm(1, oup, eps=eps)

        self._depthwise_conv = Conv2dStaticSamePadding(oup, oup, block_args.kernel_size, stride=block_args.stride,
                                                       groups=oup, bias=False, image_size=image_size)
        self._gn1 = nn.GroupNorm(1, oup, eps=eps)
        image_size = output_image_size(image_size, block_args.stride)

        if self.has_se:
            squeezed = max(1, int(inp * block_args.se_ratio))
            self._se_reduce = Conv2dStaticSamePadding(oup, squeezed, 1, image_size=(1, 1))
            self._se_expand = Conv2dStaticSamePadding(squeezed, oup, 1, image_size=(1, 1))

        self._project_conv = Conv2dStaticSamePadding(oup, block_args.output_filters, 1, bias=False,
                                                     image_size=image_size)
        self._gn2 = nn.GroupNorm(1, block_args.output_filters, eps=eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._gn0(self._expand_conv(inputs)))
        x = self._swish(self._gn1(self._depthwise_conv(x)))
        if self.has_se:
            s = self._se_expand(self._swish(self._se_reduce(F.adaptive_avg_pool2d(x, 1))))
            x = torch.sigmoid(s) * x
        x = self._gn2(self._project_conv(x))
        args = self._block_args
        if self.id_skip and args.stride == 1 and args.input_filters == args.output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs
        return x


class EfficientNet(nn.Module):
    """EfficientNet encoder exposing multi-scale endpoints for a U-Net style decoder."""

    def __init__(self, model_name='efficientnet-b0', in_channels=1, out_classes=1):
        super().__init__()
        w, d, res, dropout = EFFICIENTNET_PARAMS[model_name]
        gp = GlobalParams(width_coefficient=w, depth_coefficient=d, image_size=res, dropout_rate=dropout,
                          batch_norm_epsilon=1e-3, drop_connect_rate=0.2, depth_divisor=8, min_depth=None)
        self._global_params = gp
        image_size = gp.image_size

        stem_channels = round_filters(32, gp)
        self._conv_stem = Conv2dStaticSamePadding(3, stem_channels, 3, stride=1, bias=False, image_size=image_size)
        self._gn0 = nn.GroupNorm(1, stem_channels, eps=gp.batch_norm_epsilon)
        image_size = output_image_size(image_size, 2)

        self._blocks = nn.ModuleList([])
        for block_args in map(decode_block_string, BLOCK_STRINGS):
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, gp),
                output_filters=round_filters(block_args.output_filters, gp),
                num_repeat=round_repeats(block_args.num_repeat, gp))
            self._blocks.append(MBConvBlock(block_args, gp, image_size))
            image_size = output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, gp, image_size))

        head_channels = round_filters(1280, gp)
        self._conv_head = Conv2dStaticSamePadding(block_args.output_filters, head_channels, 1, bias=False,
                                                  image_size=image_size)
        self._gn1 = nn.GroupNorm(1, head_channels, eps=gp.batch_norm_epsilon)
        self._fc = nn.Linear(head_channels, out_classes)  # unused by the depth model, kept for checkpoint parity
        self._swish = MemoryEfficientSwish()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        if in_channels != 3:
            self._conv_stem = Conv2dStaticSamePadding(in_channels, stem_channels, 3, stride=1, bias=False,
                                                      image_size=gp.image_size)

    def extract_endpoints(self, inputs):
        """Return {'reduction_1'..'reduction_5'}: features right before each width downsampling + head."""
        endpoints = {}
        x = self._swish(self._gn0(self._conv_stem(inputs)))
        prev_x = x
        for idx, block in enumerate(self._blocks):
            rate = self._global_params.drop_connect_rate * float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=rate)
            if prev_x.size(3) > x.size(3):
                endpoints[f'reduction_{len(endpoints) + 1}'] = prev_x
            prev_x = x
        x = self._swish(self._gn1(self._conv_head(x)))
        endpoints[f'reduction_{len(endpoints) + 1}'] = x
        return endpoints
