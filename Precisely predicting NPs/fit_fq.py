import torch

from dataset import *
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import Tensor
import torch.nn.functional as F
import utils
from model import Model
from backbone import Backbone
import backbone_modules as bm
from argparse import ArgumentParser

BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR_INIT = 1e-5
EPOCH = 200
WEIGHT_DECAY = 0
DEVICE_IDS = None

#
parser = ArgumentParser()
valid_module_type = {'mix', 'mix_se', 'res', 'res_se', 'inception', 'mlp'}
parser.add_argument('--module', type=str, default='mlp', choices=valid_module_type)
parser.add_argument('--device', type=str, default='cuda:1')
args = parser.parse_args()
DEVICE = args.device

#
valid_module_dict = {
    'inception': (bm.IncepModule, 32),
    'mix': (bm.MixBottleneck, 128),
    'mix_se': (bm.MixBottleneckSE, 128),
    'res': (bm.ResidualBlock, 128),
    'res_se': (bm.ResidualBlockSE, 128)
}

#
module_type = args.module
print(f'Using module: {module_type} fitting FQ')

save_information = f'fit_fq_{module_type}_2_layer_cnn.pkl'
tensorboard_information = f'fq_no_one_hot_epoch_300_conv'

# 数据集
train, val = rand_train_val_split(path='data/data_fq.pkl')
train_data = ReactorCoreDataset(train, encoding='onehot')
valid_data = ReactorCoreDataset(val, encoding='onehot')
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE)

net = nn.Sequential(
    Model(input_channel=5, num_inception_resnet=5, include_top=False),
    # nn.Flatten(),
    # nn.Linear(32 * 15 * 15, 4 * 15 * 15),
    # nn.BatchNorm1d(4 * 15 * 15),
    # nn.ReLU(),
    # nn.Linear(4 * 15 * 15, 15 * 15)
    nn.Conv2d(32, 8, 3, 1, 1),
    nn.ReLU(),
    nn.Conv2d(8, 1, 1, 1)
).to(DEVICE)

# if module_type != 'mlp':
#     # 模型: 去掉分类头, 然后连接多个线性层, 最后的输出是一个 15 * 15 的向量
#     net = nn.Sequential(
#         Backbone(valid_module_dict[module_type][0], valid_module_dict[module_type][1],
#                  module_stack_num=5, input_channel=5, include_top=False),
#         nn.Flatten(),
#         nn.Linear(32 * 15 * 15, 4 * 15 * 15),
#         nn.BatchNorm1d(4 * 15 * 15),
#         nn.ReLU(),
#         nn.Linear(4 * 15 * 15, 15 * 15)
#     ).to(DEVICE)
#
# else:
#     net = nn.Sequential(
#         nn.Flatten(),
#         nn.Linear(5 * 15 * 15, 1024),
#         nn.BatchNorm1d(1024),
#         nn.ReLU(),
#         # =============================================
#         nn.Linear(1024, 512),
#         nn.BatchNorm1d(512),
#         nn.ReLU(),
#         # =============================================
#         nn.Linear(512, 225)
#     ).to(DEVICE)
#

# net = nn.Sequential(
#     nn.Conv2d(5, 7, 3, 1, 1),  # B, 32, 15, 15
#     nn.ReLU(),
#     nn.MaxPool2d(2, stride=2),  # B, 32, 7, 7
#     nn.ReLU(),
#     nn.Flatten(),
#     nn.Linear(7 * 7 * 7, 15 * 15)
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
#     nn.Linear(64 * 3 * 3, 256),
#     nn.ReLU(),
#     nn.Linear(256, 225)
# ).to(DEVICE)

if DEVICE_IDS is not None:
    net = nn.DataParallel(net, DEVICE_IDS)

# Tensorboard
writer = utils.get_tensorboard_writer(tensorboard_information)

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
        for data in tqdm(loader, desc='Calculating metric'):
            inp_tensor = utils.to_device(data['inp_tensor'], DEVICE)
            target = utils.to_device(data['fq'], DEVICE)
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
            fq = Tensor(data['fq']).to(DEVICE)
            valid_mask = Tensor(ReactorCoreDataset.VALID_POSITION_MASK).to(DEVICE)
            # invalid_mask = Tensor(ReactorCoreDataset.INVALID_POSITION_MASK).to(DEVICE)

            # 将堆芯外部区域的值设置为 0, 这样能确保最大值只出现在堆芯区域内
            # 实际上根本不需要这步操作, 因为其实在处理数据的时候已经进行过这步操作了, 这样之时更加保险
            fq *= valid_mask

            # 获得 fq 中最大值那个位置的下标
            output = output.view(-1, 225)
            fq = fq.view(-1, 225)
            max_idx = list(torch.argmax(fq, dim=1).cpu().numpy())
            max_idx = [[i for i in range(len(max_idx))], max_idx]

            # 分别获得 output (预测值) 和 fq 最大位置的值, 并计算他们的相对误差.
            loss = torch.abs(output[max_idx] - fq[max_idx]) / fq[max_idx]
            loss = loss.sum()
            loss_sum += loss
            count += fq.size(0)
    return loss_sum / count


for iter in range(EPOCH):
    # train a epoch
    loss_sum = 0.
    denorm_loss_mse_sum = 0.

    for batch_idx, data in enumerate(train_loader):
        inp_tensor = utils.to_device(data['inp_tensor'], DEVICE)
        target = utils.to_device(data['fq'], DEVICE)
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

torch.save(net.state_dict(), save_information)
