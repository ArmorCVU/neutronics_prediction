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
            nn.Conv2d(32, 64, 1, 1, 'same'),
            nn.BatchNorm2d(64),
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


class _IncepModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = Seq(
            _NetA(), _NetB(), _NetC()
        )

    def forward(self, x):
        return self.seq(x)


class Model(nn.Module):
    """
    input: (B, 13, 15, 15)
    """

    def __init__(self, input_channel=13, num_inception_resnet=3, include_top=True):
        super().__init__()
        self.input_channel = input_channel
        self.num_inception_resnet = num_inception_resnet
        self.include_top = include_top
        incep_blocks = [_IncepModule() for _ in range(num_inception_resnet)]
        #
        self.seq = Seq(
            nn.Conv2d(input_channel, 32, 1, 1, 'same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=1, padding=1),
            FeatureMapCut15(),
            *incep_blocks,  # (B, 32, 15, 15)
        )
        #
        self.top = Seq(
            nn.Flatten(),  # (B, 32 * 15, 15)
            nn.Linear(32 * 15 * 15, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out = self.seq(x)
        if self.include_top:
            out = self.top(out)
            out = out.view(-1)
        return out


if __name__ == '__main__':
    m = _IncepModule()
    i = torch.randn(1, 32, 15, 15)
    o = m(i)
    print(o.shape)
