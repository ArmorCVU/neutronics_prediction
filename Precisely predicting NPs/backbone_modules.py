import torch
import torch.nn as nn
import numpy as np
from torch.nn import Sequential as Seq
import torch.nn.functional as F
from model import _IncepModule


def _conv_dw(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


def _conv_st(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class _ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):  # groups=1普通卷积
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


class MobileV2Residual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        # expand_ratio 扩展因子
        super(MobileV2Residual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 point-wise conv
            layers.append(_ConvBNReLU(in_channel, hidden_channel, kernel_size=1))

        layers.extend([
            # 3x3 depth-wise conv
            _ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 point-wise conv (linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MixConv2d(nn.Module):
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        """
        :params c1: 输入feature map的通道数
        :params c2: 输出的feature map的通道数（这个函数的关键点就是对c2进行分组）
        :params k: 混合的卷积核大小 其实一般是[3, 5, 7...]用的比较多的
        :params s: 步长 stride
        :params equal_ch: 通道划分方式 有均等划分和指数划分两种方式  默认是均等划分
        """
        super(MixConv2d, self).__init__()
        groups = len(k)
        if equal_ch:  # 均等划分通道
            i = torch.linspace(0, groups - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
        else:  # 指数划分通道
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class _SEBlock(nn.Module):
    def __init__(self, ch_in, reduction=4):
        super(_SEBlock, self).__init__()
        # inter_ch = int(ch_in * se_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)  # 注意力作用每一个通道上


class MixBottleneckSE(nn.Module):
    def __init__(self, c_in, k=(3, 5, 7), expand_ratio=6, se_reduction=4):
        super().__init__()
        expand_ch = int(c_in * expand_ratio)
        self.conv1x1_1 = nn.Conv2d(c_in, expand_ch, 1, 1)
        self.mix_conv = MixConv2d(expand_ch, expand_ch, k)
        self.se = _SEBlock(expand_ch, reduction=se_reduction)
        self.conv1x1_2 = nn.Conv2d(expand_ch, c_in, 1, 1)

    def forward(self, x):
        out = self.conv1x1_1(x)
        out = self.mix_conv(out)
        out = self.se(out)
        out = self.conv1x1_2(out)
        return x + out


class MixBottleneck(nn.Module):
    def __init__(self, c_in, k=(3, 5, 7), expand_ratio=6):
        super().__init__()
        expand_ch = int(c_in * expand_ratio)
        self.conv1x1_1 = nn.Conv2d(c_in, expand_ch, 1, 1)
        self.mix_conv = MixConv2d(expand_ch, expand_ch, k)
        self.conv1x1_2 = nn.Conv2d(expand_ch, c_in, 1, 1)

    def forward(self, x):
        out = self.conv1x1_1(x)
        out = self.mix_conv(out)
        out = self.conv1x1_2(out)
        return x + out


class ResidualBlock(nn.Module):
    def __init__(self, c_in, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_in, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c_in, c_in, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_in)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResidualBlockSE(nn.Module):
    def __init__(self, c_in, stride=1, downsample=None, se_reduction=4):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_in, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c_in, c_in, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_in)
        self.downsample = downsample
        self.se = _SEBlock(c_in, reduction=se_reduction)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out


class IncepModule(nn.Module):
    def __init__(self, ch):
        """请忽略这个ch, 偷个懒.
        """
        super().__init__()
        self.module = _IncepModule()

    def forward(self, x):
        return self.module(x)


if __name__ == '__main__':
    m2d = MixBottleneckSE(3)
    rin = torch.randn(1, 3, 32, 32)
    out = m2d(rin)
    print(out.shape)
