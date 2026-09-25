import torch
import matplotlib.pyplot as plt
from dataset import *
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import Tensor
import torch.nn.functional as F
import utils
from model_hab import Model
"""修改网络模型的引用.
"""
import numba
from tqdm.auto import tqdm

DEVICE = 'cuda:0'
BATCH_SIZE = 512
TEST_BATCH_SIZE = 1024
LR_INIT = 1e-5
EPOCH = 300
WEIGHT_DECAY = 0
DEVICE_IDS = [0]

# 数据集
train, val = rand_train_val_split()
train_data = ReactorCoreDataset(train)
valid_data = ReactorCoreDataset(val)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE)

# 模型: 去掉分类头, 然后连接多个线性层, 最后的输出是一个 15 * 15 的向量
net = nn.Sequential(
    Model(include_top=False, num_inception_resnet=3, input_channel=5),
    # num_inception_resnet修改个数，默认为3
    nn.Flatten(),
    nn.Linear(32 * 15 * 15, 4 * 15 * 15),
    nn.BatchNorm1d(4 * 15 * 15),
    nn.ReLU(),
    nn.Linear(4 * 15 * 15, 15 * 15)
).to(DEVICE)

net = nn.DataParallel(net, DEVICE_IDS)

# Tensorboard
writer = utils.get_tensorboard_writer('fit_fdh_1128_hab')

# 损失函数，优化器，学习率调节
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
lr_scheduler = CosineAnnealingLR(optimizer, EPOCH)


def cal_metric(model, loader) -> float:
    """这里在数据集/验证集上根据训练的指标计算损失. 用于判断欠拟合/过拟合问题.
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


def cal_mean_absolute_error(model) -> float:
    """计算平均相对误差.
    """
    model.eval()
    with torch.no_grad():
        loss_sum = 0.
        count = 0
        for data in tqdm(valid_loader, desc='Calculating final criterion'):
            inp_tensor = utils.to_device(data['inp_tensor'], DEVICE)
            output = net(inp_tensor).view(-1, 15, 15)
            fdh = Tensor(data['fdh']).to(DEVICE)
            valid_mask = Tensor(ReactorCoreDataset.VALID_POSITION_MASK).to(DEVICE)
            # invalid_mask = Tensor(ReactorCoreDataset.INVALID_POSITION_MASK).to(DEVICE)

            # 将堆芯外部区域的值设置为 0, 这样能确保最大值只出现在堆芯区域内
            # 实际上根本不需要这步操作, 因为其实在处理数据的时候已经进行过这步操作了, 这样之时更加保险
            fdh *= valid_mask

            # 获得 fdh 中最大值那个位置的下标
            output = output.view(-1, 225)
            fdh = fdh.view(-1, 225)
            max_idx = list(torch.argmax(fdh, dim=1).cpu().numpy())
            max_idx = [[i for i in range(len(max_idx))], max_idx]

            # 分别获得 output (预测值) 和 fdh 最大位置的值, 并计算他们的相对误差.
            loss = torch.abs(output[max_idx] - fdh[max_idx]) / fdh[max_idx]
            loss = loss.sum()
            loss_sum += loss
            count += fdh.size(0)
    return loss_sum / count


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

        if batch_idx % 10 == 9:
            print(
                f'Epoch [{iter + 1}/{EPOCH}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss_sum / 10} '
                f'LR: {lr_scheduler.get_lr()} '
            )
            loss_sum = 0.
            denorm_loss_mse_sum = 0.

    lr_scheduler.step()
    train_metric = cal_metric(net, train_loader)
    valid_metric = cal_metric(net, valid_loader)
    #
    print(f'Epoch [{iter + 1}/{EPOCH}] Train Metric: {train_metric} Valid Metric: {valid_metric}')
    utils.write_train_valid(writer, train_metric, valid_metric, iter)
    #
    _criterion = cal_mean_absolute_error(net)
    writer.add_scalars('Metric', {'metric': _criterion}, global_step=iter)
    print(f'Average Relative Error: {_criterion}\n')

    net.train()

torch.save(net.state_dict(), 'models/fit_fdh_1128_hab.pkl')