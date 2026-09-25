import math
from argparse import ArgumentParser

import torch
from model import Model
from dataset import *
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, StepLR
from torch import Tensor
import torch.nn.functional as F
import utils
from model import Model
from tqdm.auto import tqdm
from backbone import Backbone
import backbone_modules as bm

BATCH_SIZE = 512
TEST_BATCH_SIZE = 1024
LR_INIT = 1e-5
EPOCH = 300
WEIGHT_DECAY = 0
DEVICE_IDS = None

# dataset
train, val = rand_train_val_split()
train_data = ReactorCoreDataset(train)
valid_data = ReactorCoreDataset(val)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE)

#
parser = ArgumentParser()
valid_module_type = {'mix', 'mix_se', 'res', 'res_se', 'inception', 'mlp'}
parser.add_argument('--module', type=str, default='res', choices=valid_module_type)
parser.add_argument('--device', type=str, default='cuda:1')
args = parser.parse_args()
DEVICE = args.device

valid_module_dict = {
    'inception': (bm.IncepModule, 32),
    'mix': (bm.MixBottleneck, 128),
    'mix_se': (bm.MixBottleneckSE, 128),
    'res': (bm.ResidualBlock, 128),
    'res_se': (bm.ResidualBlockSE, 128)
}

module_type = args.module
print(f'Using module: {module_type} fitting CBC')

save_information = 'fit_fq_MixSE_9_LR_1e-5_WD_0.pkl'
tensorboard_information = f'fit_cbc_old_model_{module_type}_epoch300'

# # model
# if module_type != 'mlp':
#     net = Backbone(valid_module_dict[module_type][0], valid_module_dict[module_type][1],
#                  module_stack_num=5, input_channel=5).to(DEVICE)
# else:
#     net = nn.Sequential(
#         nn.Flatten(),
#         nn.Linear(5 * 15 * 15, 1024),
#         nn.BatchNorm1d(1024),
#         nn.ReLU(),
#         # =============================================
#         nn.Linear(1024, 256),
#         nn.BatchNorm1d(256),
#         nn.ReLU(),
#         # =============================================
#         nn.Linear(256, 64),
#         nn.BatchNorm1d(64),
#         nn.ReLU(),
#         # =============================================
#         nn.Linear(64, 1),
#     ).to(DEVICE)


# net = nn.Sequential(
#     nn.Conv2d(5, 7, 3, 1, 1),  # B, 32, 15, 15
#     nn.ReLU(),
#     nn.AvgPool2d(2, stride=2),  # B, 32, 7, 7
#     nn.ReLU(),
#     nn.Flatten(),
#     nn.Linear(7 * 7 * 7, 32),
#     nn.ReLU(),
#     nn.Linear(32, 1)
# ).to(DEVICE)


# net = nn.Sequential(
#     nn.Conv2d(5, 32, 3, 1, 1),  # B, 32, 15, 15
#     nn.ReLU(),
#     nn.Conv2d(32, 32, 3, 1, 1),  # B, 32, 15, 15
#     nn.ReLU(),
#     nn.Conv2d(32, 32, 3, 1, 1),  # B, 32, 15, 15
#     nn.ReLU(),
#     nn.MaxPool2d(2, stride=2),  # B, 32, 7, 7
#     nn.ReLU(),
#     #######################################################
#     nn.Conv2d(32, 64, 3, 1, 1),  # B, 64, 7, 7
#     nn.ReLU(),
#     nn.Conv2d(64, 64, 3, 1, 1),  # B, 64, 7, 7
#     nn.ReLU(),
#     nn.Conv2d(64, 64, 3, 1, 1),  # B, 64, 7, 7
#     nn.ReLU(),
#     nn.MaxPool2d(2, stride=2),  # B, 64, 3, 3
#     nn.ReLU(),
#     nn.Flatten(),
#     nn.Linear(64 * 3 * 3, 128),
#     nn.ReLU(),
#     nn.Linear(128, 32),
#     nn.ReLU(),
#     nn.Linear(32, 1)
# ).to(DEVICE)

if DEVICE_IDS is not None:
    net = nn.DataParallel(net, DEVICE_IDS)

# 损失函数，优化器，学习率调节
# criterion = utils.MAELoss()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)


class QuickStartCosineLR:
    def __init__(self, optimizer, total_epochs, warm_start_epochs=5, warm_start_lr=1e-5, cosine_init_lr=1e-5):
        """这个函数返回一个 lr_scheduler.LambdaLR.
        -如果当前 epoch <= warm_start_epochs, 学习率为 warm_start_lr.
        -如果当前 epoch > warm_start_epochs, 学习率从 cosine_init_lr 开始, 进行余弦退火学习率调节.
        """
        self._epoch = 1
        self._current_lr = warm_start_lr
        self.optimizer = optimizer

        for param_group in optimizer.param_groups:
            param_group['lr'] = warm_start_lr

        self.total_epochs = total_epochs
        self.warm_start_epochs = warm_start_epochs
        self.warm_start_lr = warm_start_lr
        self.cosine_init_lr = cosine_init_lr

    def step(self):
        self._epoch += 1
        if self._epoch <= self.warm_start_epochs:
            lr = self.warm_start_lr
        else:
            lr = (1 + math.cos((self._epoch - self.warm_start_epochs) /
                               (self.total_epochs - self.warm_start_epochs) * math.pi)) * 0.5 * self.cosine_init_lr

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        self._current_lr = lr

    def get_lr(self):
        return self._current_lr


# lr_scheduler = CosineAnnealingLR(optimizer, EPOCH)
lr_scheduler = QuickStartCosineLR(optimizer, total_epochs=EPOCH)
writer = utils.get_tensorboard_writer(tensorboard_information)


def cal_mean_absolute_error(model) -> float:
    """计算平均相对误差.
    """
    model.eval()
    with torch.no_grad():
        loss_sum = 0.
        count = 0
        for data in tqdm(valid_loader, desc='Calculating final criterion'):
            inp_tensor = utils.to_device(data['inp_tensor'], DEVICE)
            output = net(inp_tensor).view(-1)
            cbc = Tensor(data['cbc']).to(DEVICE)
            # assert cbc.min() > 0, cbc.min()
            # 分别获得 output (预测值) 和 cbc 最大位置的值, 并计算他们的相对误差.
            loss = torch.abs(output - cbc) / cbc
            loss = loss.sum()
            loss_sum += loss
            count += cbc.size(0)
    return loss_sum / count


def cal_metric(model, loader):
    """这里在数据集/验证集上根据训练的指标计算损失. 用于判断欠拟合/过拟合问题.
    """
    model.eval()
    with torch.no_grad():
        loss_sum = 0.
        for batch_idx, data in tqdm(enumerate(loader), desc='Calculating metric'):
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
    #
    _criterion = cal_mean_absolute_error(net)
    writer.add_scalars('Metric', {'metric': _criterion}, global_step=iter)
    print(f'Average Relative Error: {_criterion}\n')
    net.train()

torch.save(net.state_dict(), save_information)
