import math
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from dataset import *
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, StepLR
from torch import Tensor
import torch.nn.functional as F
import utils
from model_C import Model
from numba import jit
from tqdm.auto import tqdm

DEVICE = 'cuda:0'
BATCH_SIZE = 512
TEST_BATCH_SIZE = 1024
LR_INIT = 1e-5
EPOCH = 50 # 只运行50个epoch 为了加速出图
WEIGHT_DECAY = 0
DEVICE_IDS = [0]

# 数据集加载和分割
def load_and_split_data(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train, val = rand_train_val_split()
    train_data = ReactorCoreDataset(train)
    valid_data = ReactorCoreDataset(val)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE)
    return train_loader, valid_loader

# 模型定义
def create_model():
    net = Model(input_channel=5, include_top=True).to(DEVICE)
    net = nn.DataParallel(net, DEVICE_IDS)
    return net

# 自定义学习率调度器
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

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self._current_lr = lr

    def get_lr(self):
        return self._current_lr

# 损失函数，优化器，学习率调节
def setup_optimizer(net):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    lr_scheduler = QuickStartCosineLR(optimizer, total_epochs=EPOCH)
    return criterion, optimizer, lr_scheduler

# 计算平均相对误差
def cal_mean_absolute_error(model, loader):
    model.eval()
    with torch.no_grad():
        loss_sum = 0.
        count = 0
        for data in tqdm(loader, desc='Calculating final criterion'):
            inp_tensor = utils.to_device(data['inp_tensor'], DEVICE)
            output = model(inp_tensor).view(-1)
            cbc = Tensor(data['cbc']).to(DEVICE)
            # 将非正数替换为一个很小的正数
            cbc[cbc <= 0] = 1e-6
            loss = torch.abs(output - cbc) / cbc
            loss = loss.sum()
            loss_sum += loss
            count += cbc.size(0)
    return loss_sum / count

# 计算指标
def cal_metric(model, loader, criterion):
    model.eval()
    with torch.no_grad():
        loss_sum = 0.
        for batch_idx, data in enumerate(loader):
            inp_tensor = utils.to_device(data['inp_tensor'], DEVICE)
            target = utils.to_device(data['cbc'], DEVICE)
            out = model(inp_tensor)
            loss = criterion(out, target)
            loss_sum += loss.item()
    return loss_sum / len(loader)

# 训练模型
def train_model(seed, train_loader, valid_loader):
    net = create_model()
    criterion, optimizer, lr_scheduler = setup_optimizer(net)
    writer = utils.get_tensorboard_writer(f'fit_cbc_0121_C_seed{seed}')
    losses = []

    for iter in range(EPOCH):
        loss_sum = 0.
        for batch_idx, data in enumerate(train_loader):
            inp_tensor = utils.to_device(data['inp_tensor'], DEVICE)
            target = utils.to_device(data['cbc'], DEVICE)
            out = net(inp_tensor)
            loss = criterion(out, target)
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 9:
                print(f'Epoch [{iter + 1}/{EPOCH}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss_sum / 10} LR: {lr_scheduler.get_lr()}')
                loss_sum = 0.

        lr_scheduler.step()
        train_metric = cal_metric(net, train_loader, criterion)
        valid_metric = cal_metric(net, valid_loader, criterion)
        losses.append(valid_metric)
        print(f'Epoch [{iter + 1}/{EPOCH}] Train Metric: {train_metric} Valid Metric: {valid_metric}')
        utils.write_train_valid(writer, train_metric, valid_metric, iter)
        _criterion = cal_mean_absolute_error(net, valid_loader)
        writer.add_scalars('Metric', {'metric': _criterion}, global_step=iter)
        print(f'Average Relative Error: {_criterion}\n')
        net.train()

    torch.save(net.state_dict(), f'models/fit_cbc_0121_C_seed{seed}.pkl')
    np.save(f'results/val_losses_cbc_0121_C_seed{seed}.npy', losses)

# 主函数
def main():
    if not os.path.exists('results'):
        os.makedirs('results')
    for seed in range(1, 6):
        print(f'Running with seed: {seed}')
        train_loader, valid_loader = load_and_split_data(seed)
        train_model(seed, train_loader, valid_loader)

    for seed in range(1, 6):
        plt.plot(np.load(f'results/val_losses_cbc_0121_C_seed{seed}.npy'), label=f"Seed {seed} Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()