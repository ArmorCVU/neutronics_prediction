import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
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


class ResidualBlock_Dropout(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock_Dropout, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.drop1 = nn.Dropout(0.25)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(0.25)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.drop1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers: list, in_channels=1, num_classes=10, global_avg_size=4, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.conv = conv3x3(in_channels, 16)
        #
        self.in_channels = 16  # 输入的通道数为16
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        #
        self.layer1 = self.make_layer(block, out_channels=16, blocks=layers[0])
        self.layer2 = self.make_layer(block, out_channels=32, blocks=layers[1], stride=2)
        self.layer3 = self.make_layer(block, out_channels=64, blocks=layers[2], stride=2)

        if include_top:
            self.avg_pool = nn.AvgPool2d(global_avg_size)
            self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels: int, blocks: int, stride=1):
        """构造并返回一个层一个层
        """
        downsample = None

        # 判断当前层是否是下采样层，如果是，那么该层中第一个残查块进行下采样）
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

        # 第一个残差块
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        # 加入剩余的残差块（如有）
        layers += [block(out_channels, out_channels) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        if self.include_top:
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)

        return out
