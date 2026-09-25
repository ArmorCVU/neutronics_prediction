import torch
import torch.nn as nn
from torch.nn import Sequential as Seq
import backbone_modules as bm
import torch.nn.functional as F

class FeatureMapCut15(nn.Module):
    """
    为了复现, 将特征图裁剪成 15x15.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:, :, :15, :15]


class Backbone(nn.Module):

    def __init__(self, module, module_inp_channel=128, input_channel=5, module_stack_num=5, include_top=True):
        super().__init__()
        self.input_channel = input_channel
        self.num_inception_resnet = module_stack_num
        self.include_top = include_top
        self.module_inp_channel = module_inp_channel

        self.stem = Seq(
            bm._ConvBNReLU(input_channel, module_inp_channel),
            # nn.AvgPool2d(2, stride=1, padding=1),
            # FeatureMapCut15()
        )

        self.blocks = Seq(
            *[module(module_inp_channel) for _ in range(module_stack_num)]
        )

        # 如果输出的通道不是 32, 那么经过一个 conv1x1 将它变为 32
        if module_inp_channel != 32:
            self.conv = nn.Conv2d(module_inp_channel, 32, 1, 1, padding='same')
        # self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.top = Seq(
            nn.Flatten(),  # (B, 32 * 15, 15)
            nn.Linear(32 * 15 * 15, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)

        x = self.conv(x)
        # x = self.global_pool(x)

        if self.include_top:
            x = self.top(x)

        return x


if __name__ == '__main__':
    net = Backbone(bm.MixBottleneckSE)
    rin = torch.randn(3, 5, 15, 15)
    out = net(rin)
    print(out.shape)