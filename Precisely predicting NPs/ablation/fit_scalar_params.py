import torch

from dataset import *
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import Tensor
import torch.nn.functional as F
import utils
from model import Model

DEVICE = 'cuda:3'
BATCH_SIZE = 32
TEST_BATCH_SIZE = 512
LR_INIT = 5e-4
EPOCH = 600
WEIGHT_DECAY = 0
#
SCALAR = 'cbc'

# dataset
train, val = rand_train_val_split()
train_data = ReactorCoreDataset(train)
valid_data = ReactorCoreDataset(val)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE)

# model
net = Model(num_inception_resnet=5).to(DEVICE)

# 损失函数，优化器，学习率调节
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
lr_scheduler = CosineAnnealingLR(optimizer, EPOCH)

writer = utils.get_tensorboard_writer('cbc')


def cal_metric(model, loader):
    model.eval()
    with torch.no_grad():
        loss_sum = 0.
        for batch_idx, data in enumerate(loader):
            inp_tensor = utils.to_device(data['inp_tensor'], DEVICE)
            target = utils.to_device(data[SCALAR], DEVICE)
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
        target = utils.to_device(data[SCALAR], DEVICE)
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
