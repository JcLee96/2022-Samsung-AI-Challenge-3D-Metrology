"""Depth network (EfficientNet encoder + ASPP + residual decoder) and its losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.backbone import EfficientNet


class ResidualBlock(nn.Module):
    """Three GroupNorm-ReLU-Conv layers; the output is concatenated (not added) by the caller."""

    def __init__(self, n_channels):
        super().__init__()
        self.gn = nn.GroupNorm(1, n_channels)
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, groups=n_channels // 8, padding=1, bias=False)
        self.conv3 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.gn(x)))
        out = self.conv2(F.relu(self.gn(out)))
        return self.conv3(F.relu(self.gn(out)))


def _aspp_branch(in_ch, out_ch, dilation):
    kernel = 1 if dilation == 1 else 3
    padding = 0 if dilation == 1 else dilation
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel, padding=padding, dilation=dilation),
                         nn.GroupNorm(1, out_ch), nn.ReLU(inplace=True))


class SceneUnderstandingModule(nn.Module):
    """ASPP: image-level pooling + four dilated branches, fused back to the input width."""

    def __init__(self, n_channels, out_channels=64, dilations=(1, 3, 6, 9)):
        super().__init__()
        self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Conv2d(n_channels, out_channels, 1, bias=False),
                                        nn.GroupNorm(1, out_channels), nn.ReLU())
        self.aspp1, self.aspp2, self.aspp3, self.aspp4 = (_aspp_branch(n_channels, out_channels, d) for d in dilations)
        self.concat_process = nn.Sequential(nn.Conv2d(out_channels * 5, n_channels, 1),
                                            nn.GroupNorm(1, n_channels), nn.ReLU(inplace=True),
                                            nn.Dropout2d(p=0.5))

    def forward(self, x):
        pooled = self.image_pool(x).expand(-1, -1, x.shape[2], x.shape[3])
        branches = [pooled, self.aspp1(x), self.aspp2(x), self.aspp3(x), self.aspp4(x)]
        return self.concat_process(torch.cat(branches, dim=1))


def ordinal_probabilities(x):
    """Map 2K logits to K probabilities P(label > k) via a pairwise softmax over (even, odd) channels."""
    a = torch.clamp(x[:, ::2], min=1e-8, max=1e8)
    b = torch.clamp(x[:, 1::2], min=1e-8, max=1e8)
    return torch.softmax(torch.stack((a, b), dim=0), dim=0)[1]


class DepthNet(nn.Module):
    """Returns (depth in [0, 1] via sigmoid, ordinal probabilities) for a 1x72x48 SEM image."""

    def __init__(self, k_ordinal_class=17, model_name='efficientnet-b0', reduction=0.5, output_padding=(1, 0)):
        super().__init__()
        self.efficient_net = EfficientNet(model_name, in_channels=1, out_classes=1)
        self.scene_understanding_module = SceneUnderstandingModule(1280)
        self.transpose_conv1 = nn.ConvTranspose2d(1280, int(1280 * reduction), 2, stride=2,
                                                  output_padding=output_padding)
        self.resblock1 = ResidualBlock(752)
        self.transpose_conv2 = nn.ConvTranspose2d(1504, 120, 2, stride=2)
        self.resblock2 = ResidualBlock(160)
        self.transpose_conv3 = nn.ConvTranspose2d(320, 40, 2, stride=2)
        self.resblock3 = ResidualBlock(64)
        self.transpose_conv4 = nn.ConvTranspose2d(128, 8, 2, stride=2)
        self.resblock4 = ResidualBlock(24)
        self.ordinal_regression_conv1 = nn.Conv2d(48, k_ordinal_class * 2, kernel_size=1)
        self.last_conv = nn.Conv2d(48, 1, kernel_size=1)

    def forward(self, x):
        ep = self.efficient_net.extract_endpoints(x)
        out = self.scene_understanding_module(ep['reduction_5'])
        stages = [(self.transpose_conv1, 'reduction_4', self.resblock1),
                  (self.transpose_conv2, 'reduction_3', self.resblock2),
                  (self.transpose_conv3, 'reduction_2', self.resblock3),
                  (self.transpose_conv4, 'reduction_1', self.resblock4)]
        for up, skip, res in stages:
            out = torch.cat((ep[skip], up(out)), dim=1)
            out = torch.cat((out, res(out)), dim=1)
        depth = torch.sigmoid(self.last_conv(out))
        return depth, ordinal_probabilities(self.ordinal_regression_conv1(out))


def ordinal_regression_loss(probs, target):
    """Pixel-averaged ordinal NLL: log P(>k) for k <= label, log(1 - P(>k)) for k > label."""
    n, c, h, w = probs.shape
    k = torch.arange(c, device=probs.device).view(1, c, 1, 1)
    below = k <= target
    loss = torch.log(torch.clamp(probs[below], 1e-8, 1e8)).sum() \
        + torch.log(torch.clamp(1 - probs[~below], 1e-8, 1e8)).sum()
    return -loss / (n * h * w)


def depth_loss(pred_depth, pred_ord, depth, ordinal):
    """Total loss = L1 on normalized depth + ordinal regression loss; returns (total, l1, ord)."""
    l1 = F.l1_loss(pred_depth, depth)
    ord_loss = ordinal_regression_loss(pred_ord, ordinal)
    return l1 + ord_loss, l1, ord_loss
