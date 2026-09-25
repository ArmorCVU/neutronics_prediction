import torch

from old_data.dataset import *
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import Tensor
import torch.nn.functional as F
import utils



DEVICE = 'cuda:0'
BATCH_SIZE = 32
TEST_BATCH_SIZE = 512
LR_INIT = 1e-5
EPOCH = 600
WEIGHT_DECAY = 0
# DEVICE_IDS = [0, 1, 2, 3]

# dataset
train, val = rand_train_val_split()
train_data = ReactorCoreDataset(train, normalize=False)
valid_data = ReactorCoreDataset(val, normalize=False)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE)

# model
# net = nn.Sequential(ResNet(ResidualBlock, [1, 1, 1], in_channels=13, include_top=False),
#                     nn.Flatten(), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 15 * 15)).to(DEVICE)


from model import Model
net = nn.Sequential(
    Model(include_top=False, num_inception_resnet=5),
    nn.Flatten(),
    nn.Linear(32 * 15 * 15, 4 * 15 * 15),
    nn.BatchNorm1d(4 * 15 * 15),
    nn.ReLU(),
    nn.Linear(4 * 15 * 15, 15 * 15)
).to(DEVICE)


# class MaskedMSELoss(nn.Module):
#     """只计算堆芯部分的 MSE 损失
#     """
#
#     def __init__(self, device):
#         super().__init__()
#         self.mask = torch.LongTensor(ReactorCoreDataset.VALID_POSITION_MASK).to(device)
#         self.mask.requires_grad = False
#
#     def forward(self, out: Tensor, target: Tensor):
#         # 沿着通道维度平均
#         out = out.view(-1, 15, 15)
#         out = out * self.mask
#         tar = target * self.mask
#         assert out.dtype == tar.dtype == torch.float32
#         return F.mse_loss(out, tar, reduction='sum') / self.mask.sum()


# class MaskedLoss(nn.Module):
#     """只计算堆芯部分的 MSE 损失
#     """
#
#     def __init__(self, device):
#         super().__init__()
#         self.valid_mask = torch.LongTensor(ReactorCoreDataset.VALID_POSITION_MASK).to(device)
#         self.valid_mask.requires_grad = False
#         self.invalid_mask = torch.from_numpy(ReactorCoreDataset.INVALID_POSITION_MASK).long().to(device)
#         self.invalid_mask.requires_grad = False
#
#     def forward(self, output: Tensor, target: Tensor):
#         # 沿着通道维度平均
#         outputs = output.view(-1, 15, 15)
#
#         # 把燃料周围的数值变为 0
#         outputs = outputs * self.valid_mask
#         # 把燃料周围的数值变为 1
#         target = self.invalid_mask + target * self.valid_mask
#
#         metric = outputs - target
#         metric = torch.abs(metric)
#         metric = metric / torch.abs(target)
#         return metric.mean()


# def get_denormalized_mse_mae(out: Tensor, target: Tensor):
#     mask = torch.LongTensor(ReactorCoreDataset.VALID_POSITION_MASK).to(DEVICE)
#     mask.requires_grad = False
#     out = out.detach()
#     out = out.view(-1, 15, 15)
#     target = target.detach()
#     out = train_data.denormalize_fq(out)
#     target = train_data.denormalize_fq(target)
#     return F.mse_loss(out, target).item(), torch.abs(out - target).mean().item()


writer = utils.get_tensorboard_writer('fdh3')


# 损失函数，优化器，学习率调节
# criterion = MaskedMSELoss(device=DEVICE)
# criterion = MaskedLoss(device=DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
lr_scheduler = CosineAnnealingLR(optimizer, EPOCH)


def cal_metric(model, loader):
    """有点复杂
    """
    model.eval()
    with torch.no_grad():
        loss_sum = 0.
        for batch_idx, data in enumerate(loader):
            inp_tensor = utils.to_device(data['inp_tensor'], DEVICE)
            target = utils.to_device(data['fdh'], DEVICE)
            #
            out = net(inp_tensor).view(-1, 15, 15)
            loss = criterion(out, target)
            loss_sum += loss.item()

    return loss_sum / len(loader)


for iter in range(EPOCH):
    # train a epoch
    loss_sum = 0.
    denorm_loss_mse_sum = 0.

    for batch_idx, data in enumerate(train_loader):
        inp_tensor = utils.to_device(data['inp_tensor'], DEVICE)
        target = utils.to_device(data['fdh'], DEVICE)
        #
        out = net(inp_tensor).view(-1, 15, 15)
        loss = criterion(out, target)
        loss_sum += loss.item()
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # denorm_loss_mse, _ = get_denormalized_mse_mae(out, target)
        # denorm_loss_mse_sum += denorm_loss_mse

        if batch_idx % 10 == 9:
            print(
                f'Epoch [{iter + 1}/{EPOCH}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss_sum / 10} '
                f'LR: {lr_scheduler.get_lr()} '
                # f'Denorm MSE: {denorm_loss_mse_sum / 10}'
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
