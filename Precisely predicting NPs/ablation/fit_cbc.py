import torch

from old_data.dataset import *
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import Tensor
import torch.nn.functional as F
import utils
from model import Model


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


class ResNet(nn.Module):
    def __init__(self, block, layers: list, in_channels=1, global_avg_size=4):
        super(ResNet, self).__init__()
        self.conv = conv3x3(in_channels, 16)
        #
        self.in_channels = 16  # 输入的通道数为16
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        #
        self.layer1 = self.make_layer(block, out_channels=16, blocks=layers[0])
        self.layer2 = self.make_layer(block, out_channels=32, blocks=layers[1], stride=2)
        self.layer3 = self.make_layer(block, out_channels=64, blocks=layers[2], stride=2)

        self.avg_pool = nn.AvgPool2d(global_avg_size)
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(),
                                nn.Linear(32, 1))

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
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out.view(-1)


DEVICE = 'cuda:3'
BATCH_SIZE = 32
TEST_BATCH_SIZE = 512
LR_INIT = 1e-5
EPOCH = 600
WEIGHT_DECAY = 0
# DEVICE_IDS = [0, 1, 2, 3]

# dataset
train, val = rand_train_val_split()
train_data = ReactorCoreDataset(train, normalize=True)
valid_data = ReactorCoreDataset(val, normalize=True)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE)

# model
# net = ResNet(ResidualBlock, [1, 1, 1], in_channels=13)
net = Model().to(DEVICE)

# 损失函数，优化器，学习率调节
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
lr_scheduler = CosineAnnealingLR(optimizer, EPOCH)

writer = utils.get_tensorboard_writer('fit_cbc5')


def cal_metric(model, loader):
    model.eval()
    with torch.no_grad():
        loss_sum = 0.
        for batch_idx, data in enumerate(loader):
            inp_tensor = utils.to_device(data['inp_tensor'], DEVICE)
            target = utils.to_device(data['cbc'], DEVICE)
            #
            out = net(inp_tensor)
            loss = criterion(out, target)
            loss_sum += loss.item()

    return loss_sum / len(loader)


for iter in range(EPOCH):
    # train a epoch
    loss_sum = 0.
    denorm_loss_mse_sum = 0.

    for batch_idx, data in enumerate(train_loader):
        inp_tensor = utils.to_device(data['inp_tensor'], DEVICE)
        target = utils.to_device(data['cbc'], DEVICE)
        #
        out = net(inp_tensor)
        loss = criterion(out, target)
        loss_sum += loss.item()
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 9:
            print(
                f'Epoch [{iter + 1}/{EPOCH}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss_sum / 10} '
                f'LR: {lr_scheduler.get_lr()} '
            )
            loss_sum = 0.
            denorm_loss_mse_sum = 0.

    lr_scheduler.step()
    # net.eval()
    train_metric = cal_metric(net, train_loader)
    valid_metric = cal_metric(net, valid_loader)
    print(f'Epoch [{iter + 1}/{EPOCH}] Train Metric: {train_metric} Valid Metric: {valid_metric}\n')
    utils.write_train_valid(writer, train_metric, valid_metric, iter)
    net.train()
