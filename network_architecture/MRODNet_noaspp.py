import torch
import torch.nn as nn
import torch.nn.functional as F
from network_architecture.efficientnet_pytorch import EfficientNet


class Transition(nn.Module):
    def __init__(self, n_channels, n_output_channels):
        super(Transition, self).__init__()
        # self.bn1 = nn.BatchNorm3d(n_channels)
        # self.gn = nn.GroupNorm(int(n_channels), int(n_channels))
        self.gn = nn.GroupNorm(1, int(n_channels))
        #self.conv1 = nn.Conv3d(n_channels, n_output_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.conv1 = nn.Conv2d(n_channels, n_output_channels, kernel_size=3, padding=1, bias=False)
        self.avg_pool = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x):
        out = self.conv1(F.relu(self.gn(x)))
        out = self.avg_pool(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResidualBlock, self).__init__()
        # self.gn = nn.GroupNorm(int(n_channels), int(n_channels))
        self.gn = nn.GroupNorm(1, int(n_channels))
        # self.bn = nn.BatchNorm3d(n_channels)
        # self.conv1 = nn.Conv3d(n_channels, n_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        # self.conv2 = nn.Conv3d(n_channels, n_channels, kernel_size=(3, 3, 3), groups=int(n_channels / 8),
        #                        padding=(1, 1, 1), bias=False)
        # self.conv3 = nn.Conv3d(n_channels, n_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, groups=int(n_channels / 8), padding=1, bias=False)
        self.conv3 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.gn(x)))
        out = self.conv2(F.relu(self.gn(out)))
        out = self.conv3(F.relu(self.gn(out)))
        return out

class SceneUnderstandingModule(nn.Module):
    def __init__(self, n_channels, scale_factor, device=torch.device("cuda:0"), dilations=[1, 3, 6, 9]):
        super(SceneUnderstandingModule, self).__init__()
        out_n_channels = 64
        self.device = device
        self.n_channels = n_channels
        self.out_n_channels = out_n_channels
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.aspp1 = nn.Sequential(
            #nn.Conv3d(n_channels, out_n_channels, kernel_size=1, stride=1, padding=0, dilation=dilations[0]),
            nn.Conv2d(n_channels, out_n_channels, kernel_size=1, stride=1, padding=0, dilation=dilations[0]),
            nn.GroupNorm(1, out_n_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            #nn.Conv3d(n_channels, out_n_channels, kernel_size=3, stride=1, padding=dilations[1], dilation=dilations[1]),
            nn.Conv2d(n_channels, out_n_channels, kernel_size=3, stride=1, padding=dilations[1], dilation=dilations[1]),
            nn.GroupNorm(1, out_n_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp3 = nn.Sequential(
            #nn.Conv3d(n_channels, out_n_channels, kernel_size=3, stride=1, padding=dilations[2], dilation=dilations[2]),
            nn.Conv2d(n_channels, out_n_channels, kernel_size=3, stride=1, padding=dilations[2], dilation=dilations[2]),
            nn.GroupNorm(1, out_n_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp4 = nn.Sequential(
            #nn.Conv3d(n_channels, out_n_channels, kernel_size=3, stride=1, padding=dilations[3], dilation=dilations[3]),
            nn.Conv2d(n_channels, out_n_channels, kernel_size=3, stride=1, padding=dilations[3], dilation=dilations[3]),
            nn.GroupNorm(1, out_n_channels),
            nn.ReLU(inplace=True)
        )
        self.concat_process = nn.Sequential(
            #nn.Conv3d(out_n_channels * 5, n_channels, 1),
            nn.Conv2d(out_n_channels * 5, n_channels, 1),
            nn.GroupNorm(1, n_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

    def forward(self, x):
        d = x.shape[2]
        global_avg_pool = nn.Sequential(#nn.AdaptiveAvgPool3d((d, 1, 1)),
                                        #nn.Conv3d(self.n_channels, self.out_n_channels, 1, stride=1, bias=False),
                                        nn.AdaptiveAvgPool2d(1),
                                        nn.Conv2d(self.n_channels, self.out_n_channels, 1, stride=1, bias=False),
                                        nn.GroupNorm(1, self.out_n_channels),
                                        nn.ReLU()).to(self.device)
        x0 = global_avg_pool(x)
        x0 = self.upsample(x0)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x6 = torch.cat((x0, x1, x2, x3, x4), dim=1)
        # print('cat x6 size:', x6.size())
        out = self.concat_process(x6)
        return out

class OrdinalRegressionLayer(nn.Module):
    def __init__(self):
        super(OrdinalRegressionLayer, self).__init__()

    def forward(self, x):
        """
        :param x: N X H X W X C, N is batch_size, C is channels of features
        :return: ord_labels is ordinal outputs for each spatial locations , size is N x H X W X C (C = 2K, K is interval of SID)
                 decode_label is the ordinal labels for each position of Image I
        """
        xo = x
        # P is phase number. In this case, P = 1
        N, C, H, W = x.size()
        x = x.view(N, -1, 1, H, W)
        N, C, P, H, W = x.size()
        ord_num = C // 2

        """
        replace iter with matrix operation
        fast speed methods
        """
        # x = x.view(-1, 2, ord_num, P, D, H, W)
        # prob = nn.functional.softmax(x, dim=1)[:, 0, :, :, :, :, :]
        # prob = F.log_softmax(x, dim=1).view(-1, ord_num, P, D, H, W)
        # return prob
        A = x[:, ::2, :, :, :].clone()
        B = x[:, 1::2, :, :, :].clone()

        OA = A
        OB = B
        A = A.view(N, 1, ord_num * P * H * W)
        B = B.view(N, 1, ord_num * P * H * W)

        C = torch.cat((A, B), dim=1)
        C = torch.clamp(C, min=1e-8, max=1e8)  # prevent nans

        ord_c = nn.functional.softmax(C, dim=1)

        ord_c1 = ord_c[:, 1, :].clone()
        ord_c1 = ord_c1.view(-1, ord_num, H, W)
        return ord_c1

# Original DRNN is the neural network architecture which is the same to paper's NN architecture.
class EfficientnetBackboneDrnnSddLossNoAspp(nn.Module):
    # denblock_config = (6, 12, 24, 48)
    def __init__(self, n_channels, growth_rate, reduction, k_ordinal_class, dev_0, dev_1, dev_2, dev_3, model_name, scale_factor=(14, 14), output_padding=0):
        super(EfficientnetBackboneDrnnSddLossNoAspp, self).__init__()
        self.dev_0 = dev_0
        self.dev_1 = dev_1
        self.dev_2 = dev_2
        self.dev_3 = dev_3
        # dense 1
        self.efficient_net = EfficientNet.from_name(model_name, out_classes=1, in_channels=n_channels).to(dev_0)
        n_channels = 1280
        ##self.scene_understanding_module = SceneUnderstandingModule(1280, scale_factor, device=dev_0).to(dev_0)
        ##self.transpose_conv1 = nn.ConvTranspose3d(n_channels, int(n_channels * reduction), kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0)).to(dev_0)
        self.transpose_conv1 = nn.ConvTranspose2d(n_channels, int(n_channels * reduction), kernel_size=2, stride=2, output_padding=output_padding).to(dev_0)
        # # resblock
        self.resblock1 = ResidualBlock(752).to(dev_0)
        n_channels = 1504
        # n_channels = int(n_channels * 6) # n_channels = 512
        ##self.transpose_conv2 = nn.ConvTranspose3d(n_channels, 120, kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0)).to(dev_0)
        self.transpose_conv2 = nn.ConvTranspose2d(n_channels, 120, kernel_size=2, stride=2).to(dev_0)
        self.resblock2 = ResidualBlock(160).to(dev_0)
        n_channels = 320
        ##self.transpose_conv3 = nn.ConvTranspose3d(n_channels, 40, kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0)).to(dev_0)
        self.transpose_conv3 = nn.ConvTranspose2d(n_channels, 40, kernel_size=2, stride=2).to(dev_0)
        # n_channels = int(n_channels // 12)  # n_channels = 32
        self.resblock3 = ResidualBlock(64).to(dev_1)
        ##self.transpose_conv4 = nn.ConvTranspose3d(128, 8, kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0)).to(dev_1
        self.transpose_conv4 = nn.ConvTranspose2d(128, 8, kernel_size=2, stride=2).to(dev_1)
        # n_channels = int(n_channels // 12) # n_channels = 8
        self.resblock4 = ResidualBlock(24).to(dev_1)
        # self.last_gn = nn.GroupNorm(32, 32)
        # self.last_gn = nn.GroupNorm(1, 32).to(dev_1)
        ##self.ordinal_regression_conv1 = nn.Conv3d(48, k_ordinal_class * 10, kernel_size=(1, 1, 1)).to(dev_2)
        #self.ordinal_regression_conv1 = nn.Conv3d(48, k_ordinal_class * 10, kernel_size=(1, 1, 1)).to(dev_2)
        self.ordinal_regression_conv1 = nn.Conv2d(48, k_ordinal_class * 2, kernel_size=1).to(dev_2)
        self.orl1 = OrdinalRegressionLayer().to(dev_2)
        #self.last_conv = nn.Conv3d(48, 5, kernel_size=(1, 1, 1)).to(dev_0)
        self.last_conv = nn.Conv2d(48, 1, kernel_size=1).to(dev_0)

    def forward(self, x):
        # encoder
        x = x.to(self.dev_0, dtype=torch.float)
        endpoints = self.efficient_net.extract_endpoints(x)
        reduction_1, reduction_2, reduction_3, reduction_4, out = endpoints["reduction_1"], endpoints["reduction_2"], \
                                                                  endpoints["reduction_3"], endpoints["reduction_4"], \
                                                                  endpoints["reduction_5"]
        #out = self.scene_understanding_module(out)
        # decoder
        out = self.transpose_conv1(out)
        out = torch.cat((reduction_4, out), dim=1)
        del reduction_4
        res1 = self.resblock1(out)
        out = torch.cat((out, res1), dim=1)
        del res1
        #
        out = self.transpose_conv2(out)
        out = torch.cat((reduction_3, out), dim=1)
        del reduction_3
        res2 = self.resblock2(out)
        out = torch.cat((out, res2), dim=1)
        del res2
        #
        out = self.transpose_conv3(out)
        out = torch.cat((reduction_2, out), dim=1)
        # out = out.to(self.dev_1)
        del reduction_2
        res3 = self.resblock3(out)
        out = torch.cat((out, res3), dim=1)
        del res3
        #
        out = self.transpose_conv4(out)
        out = torch.cat((reduction_1, out), dim=1)
        del reduction_1
        res4 = self.resblock4(out)
        out = torch.cat((out, res4), dim=1)
        del res4

        reg_output = self.last_conv(out)
        out = out.to(self.dev_2)
        #
        ordinal_regression1 = self.ordinal_regression_conv1(out)
        #
        reg_output = torch.tanh(reg_output)
        return (reg_output.to(self.dev_2), self.orl1(ordinal_regression1))


if __name__ == '__main__':
    input = torch.rand(1, 1, 72, 48) #N, C, H, W
    net = EfficientnetBackboneDrnnSddLossNoAspp(n_channels=1,
                                                growth_rate=12,
                                                reduction=0.5,
                                                k_ordinal_class=5,
                                                dev_0=torch.device("cpu"),
                                                dev_1=torch.device("cpu"),
                                                dev_2=torch.device("cpu"),
                                                dev_3=torch.device("cpu"),
                                                model_name="efficientnet-b0",
                                                scale_factor=(4, 3),
                                                output_padding=(1, 0))
    # from utils.utils import count_parameters
    # print(count_parameters(net))
    outputs = net(input)

    for output in outputs:
        print(output.shape)
    # for y in ys:
    #     print(y.shape)


