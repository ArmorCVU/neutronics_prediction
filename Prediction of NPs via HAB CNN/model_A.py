import torch
import torch.nn as nn
from torch.nn import Sequential as Seq
import torch.nn.functional as F


class FeatureMapCut15(nn.Module):
    """
将特征图裁剪成 15x15.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:, :, :15, :15]


class _NetA(nn.Module):
    """
    input: (B, 32, 15, 15)
    output: (B, 128, 15, 15)
    """

    def __init__(self):
        super().__init__()
        self.group0 = Seq(
            nn.Conv2d(32, 32, 1, 1, 'same'),
            nn.BatchNorm2d(32),
        )

        self.group1 = Seq(
            nn.AvgPool2d(2, stride=1, padding=1),
            FeatureMapCut15(),
            nn.Conv2d(32, 32, 1, 1, 'same'),
            nn.BatchNorm2d(32)
        )

        self.group2 = Seq(
            nn.Conv2d(32, 64, 3, 1, 'same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1, 1, 'same'),
            nn.BatchNorm2d(32)
        )

    def forward(self, x):
        x0 = self.group0(x)
        x1 = self.group1(x)
        x2 = self.group2(x)

        x = torch.cat([x0, x1, x2], dim=1)

        # 使用1x1卷积来降低通道维度到32
        x = F.conv2d(x, weight=torch.randn(32, 96, 1, 1).to(x.device), stride=1, padding=0)

        return F.relu(x)


class _NetB(nn.Module):
    """
    input: (B, 128, 15, 15)
    output: (B, 128, 15, 15)
    """

    def __init__(self):
        super().__init__()
        self.group0 = Seq(
            nn.Conv2d(128, 64, 1, 1, 'same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1, 1, 'same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.group1 = Seq(
            nn.Conv2d(128, 32, 1, 1, 'same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 'same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.group2 = Seq(
            nn.Conv2d(64, 32, 1, 1, 'same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 'same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 128, 1, 1, 'same')
        )

        self.group3 = Seq(
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

    def forward(self, x):
        x0, x1 = self.group0(x), self.group1(x)
        out = torch.cat([x0, x1], dim=1)
        out = self.group2(out)
        out = out + x
        out = self.group3(out)
        return out


class _NetC(nn.Module):
    """
    input: (B, 128, 15, 15)
    output: (B, 32, 15, 15)
    """

    def __init__(self):
        super().__init__()
        self.group0 = Seq(
            nn.Conv2d(128, 8, 1, 1, 'same'),
            nn.BatchNorm2d(8),
        )

        self.group1 = Seq(
            nn.Conv2d(128, 64, 1, 1, 'same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 16, 3, 1, 'same'),
            nn.BatchNorm2d(16)
        )

        self.group2 = Seq(
            nn.AvgPool2d(2, stride=1, padding=1),
            FeatureMapCut15(),
            nn.Conv2d(128, 64, 1, 1, 'same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1, 1, 'same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 8, 3, 1, 'same'),
            nn.BatchNorm2d(8)
        )

    def forward(self, x):
        x0 = self.group0(x)
        x1 = self.group1(x)
        x2 = self.group2(x)
        x = torch.cat([x0, x1, x2], dim=1)
        return F.relu(x)


class _IncepResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 只包含一个_NetA实例
        self.netA = _NetA()

    def forward(self, x):
        # 只通过_NetA网络
        return self.netA(x)


class Model(nn.Module):
    """
    input: (B, 13, 15, 15)
    """

    # class Model(nn.Module):
    def __init__(self, input_channel=13, include_top=True):
        super().__init__()
        self.include_top = include_top

        # 输入层，将输入通道数从13转换为32
        self.input_conv = nn.Conv2d(input_channel, 32, 1, 1, 'same')
        self.bn_input = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

        # 只使用一个_IncepResNet实例，它只包含_NetA
        self.inception_resnet = _IncepResNet()

        # 根据原始模型的输出尺寸调整后续层
        # 假设原始模型最后一个全连接层的输入尺寸是32 * 15 * 15
        self.top = Seq(
            nn.Flatten(),  # 展平特征图 (B, 32 * 15 * 15)
            nn.Linear(32 * 15 * 15, 32),  # 根据实际情况调整
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 确保最终输出是单个值
        )

    def forward(self, x):
        # 通过输入卷积层、BN和ReLU
        x = self.relu(self.bn_input(self.input_conv(x)))

        # 通过_IncepResNet（只包含_NetA）
        x = self.inception_resnet(x)

        # 通过顶层全连接网络
        if self.include_top:
            out = self.top(x)
            out = out.view(-1)
            return out

        return x


if __name__ == '__main__':
    m = _IncepResNet()
    i = torch.randn(1, 32, 15, 15)
    o = m(i)
    print(o.shape)
