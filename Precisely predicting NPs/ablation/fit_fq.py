import torch

from old_data.dataset import *
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import ResNet, ResidualBlock
from torch import Tensor
import torch.nn.functional as F
import utils

DEVICE = 'cpu'
BATCH_SIZE = 32
TEST_BATCH_SIZE = 512
LR_INIT = 1e-2
EPOCH = 200
WEIGHT_DECAY = 0
# DEVICE_IDS = [0, 1, 2, 3]

# dataset
train, val = rand_train_val_split()
train_data = ReactorCoreDataset(train, normalize=True)
valid_data = ReactorCoreDataset(val, normalize=True)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE)

# model
net = nn.Sequential(ResNet(ResidualBlock, [2, 2], in_channels=13, include_top=False),
                    nn.Conv2d(32, 32, 3, 1, 1)).to(DEVICE)


class MaskedMSELoss(nn.Module):
    """只计算堆芯部分的 MSE 损失
    """
    def __init__(self, device):
        super().__init__()
        self.mask = torch.LongTensor(ReactorCoreDataset.VALID_POSITION_MASK).to(device)
        self.mask.requires_grad = False

    def forward(self, out: Tensor, target: Tensor):
        # 沿着通道维度平均
        out = out.mean(dim=1)
        out = out * self.mask
        tar = target * self.mask
        assert out.dtype == tar.dtype == torch.float32
        return F.mse_loss(out, tar, reduction='sum') / self.mask.sum()


class MaskedLoss(nn.Module):
    """只计算堆芯部分的 MSE 损失
    """
    def __init__(self, device):
        super().__init__()
        self.valid_mask = torch.LongTensor(ReactorCoreDataset.VALID_POSITION_MASK).to(device)
        self.valid_mask.requires_grad = False
        self.invalid_mask = torch.from_numpy(ReactorCoreDataset.INVALID_POSITION_MASK).long().to(device)
        self.invalid_mask.requires_grad = False

    def forward(self, output: Tensor, target: Tensor):
        # 沿着通道维度平均
        outputs = output.mean(dim=1)

        # 把燃料周围的数值变为 0
        outputs = outputs * self.valid_mask
        # 把燃料周围的数值变为 1
        target = self.invalid_mask + target * self.valid_mask

        metric = outputs - target
        metric = torch.abs(metric)
        metric = metric / torch.abs(target)
        return metric.mean()


def get_denormalized_mse_mae(out: Tensor, target: Tensor):
    mask = torch.LongTensor(ReactorCoreDataset.VALID_POSITION_MASK).to(DEVICE)
    mask.requires_grad = False
    out = out.detach()
    target = target.detach()
    out = out.mean(dim=1)
    out = train_data.denormalize_fq(out)
    target = train_data.denormalize_fq(target)
    return F.mse_loss(out, target).item(), torch.abs(out - target).mean().item()


# 损失函数，优化器，学习率调节
criterion = MaskedMSELoss(device=DEVICE)
# criterion = MaskedLoss(device=DEVICE)
optimizer = optim.Adam(net.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
lr_scheduler = CosineAnnealingLR(optimizer, EPOCH)


def cal_metric(model, loader):
    """有点复杂
    """
    valid_mask = torch.LongTensor(ReactorCoreDataset.VALID_POSITION_MASK).to(DEVICE)
    invalid_mask = torch.from_numpy(ReactorCoreDataset.INVALID_POSITION_MASK).long().to(DEVICE)
    valid_mask.requires_grad = False
    invalid_mask.requires_grad = False
    #
    model.eval()
    all_metric = []
    with torch.no_grad():
        for data in loader:
            inp_tensor = data['inp_tensor']
            target = data['fq']
            inp_tensor = utils.to_device(inp_tensor, DEVICE)
            target = utils.to_device(target, DEVICE)
            outputs = model(inp_tensor)
            outputs = outputs.mean(dim=1)

            # 把燃料周围的数值变为 0
            outputs = outputs * valid_mask
            # 把燃料周围的数值变为 1
            target = invalid_mask + target * valid_mask

            metirc = outputs - target
            metirc = torch.abs(metirc)
            metirc = metirc / torch.abs(target)
            metric = metirc.mean().item()
            all_metric.append(metric)
    return np.array(all_metric).mean()


for iter in range(EPOCH):
    # train a epoch
    loss_sum = 0.
    denorm_loss_mse_sum = 0.

    for batch_idx, data in enumerate(train_loader):
        inp_tensor = utils.to_device(data['inp_tensor'], DEVICE)
        target = utils.to_device(data['fq'], DEVICE)
        #
        out = net(inp_tensor)
        loss = criterion(out, target)
        loss_sum += loss.item()
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        denorm_loss_mse, _ = get_denormalized_mse_mae(out, target)
        denorm_loss_mse_sum += denorm_loss_mse

        if batch_idx % 10 == 9:
            print(
                f'Epoch [{iter + 1}/{EPOCH}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item() / 10} '
                f'LR: {lr_scheduler.get_lr()} Denorm MSE: {denorm_loss_mse_sum / 10}',
                flush=True)
            loss_sum = 0.
            denorm_loss_mse_sum = 0.

    lr_scheduler.step()
    # net.eval()
    # train_metric = cal_metric(net, train_loader)
    # valid_metric = cal_metric(net, valid_loader)
    # print(f'Epoch [{iter + 1}/{EPOCH}] Train Metric: {train_metric} Valid Metric: {valid_metric}\n')
    # net.train()
