import torch
import matplotlib.pyplot as plt
from dataset_157 import *
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import Tensor
import torch.nn.functional as F
import utils
from model_HAB import Model
import os
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
DEVICE_IDS = [0,1,2]

# 数据集
def load_and_split_data(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train, val = rand_train_val_split()
    train_data = ReactorCoreDataset(train)
    valid_data = ReactorCoreDataset(val)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE)
    return train_loader, valid_loader


# 模型: 输出端：去掉分类头, 然后连接多个线性层, 最后的输出是一个 15 * 15 的向量
# 此模型并非网络模型，网络模型在model.py中，此为网络模型到输出端的最后一层，即展平和还原为2维的操作
def create_model():
    net = nn.Sequential(
        Model(include_top=False, num_inception_resnet=3, input_channel=5),
        nn.Flatten(),
        nn.Linear(32 * 15 * 15, 4 * 15 * 15),
        nn.BatchNorm1d(4 * 15 * 15),
        nn.ReLU(),
        nn.Linear(4 * 15 * 15, 15 * 15)
    ).to(DEVICE)
    net = nn.DataParallel(net, DEVICE_IDS)
    return net

# 损失函数，优化器，学习率调节
def setup_optimizer(net):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    lr_scheduler = CosineAnnealingLR(optimizer, EPOCH)
    return criterion, optimizer, lr_scheduler

# 计算指标的函数
def cal_metric(model, loader, criterion):
    model.eval()
    with torch.no_grad():
        loss_sum = 0.
        for batch_idx, data in enumerate(loader):
            inp_tensor = utils.to_device(data['inp_tensor'], DEVICE)
            target = utils.to_device(data['fdh'], DEVICE)
            out = model(inp_tensor).view(-1, 15, 15)
            loss = criterion(out, target)
            loss_sum += loss.item()
    return loss_sum / len(loader)

def cal_mean_absolute_error(model, loader) -> float:
    """计算平均相对误差.
    """
    model.eval()
    with torch.no_grad():
        loss_sum = 0.
        count = 0
        for data in tqdm(loader, desc='Calculating final criterion'):
            inp_tensor = utils.to_device(data['inp_tensor'], DEVICE)
            output = model(inp_tensor).view(-1, 15, 15)
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

# 训练函数
def train_model(seed, train_loader, valid_loader):
    net = create_model()
    criterion, optimizer, lr_scheduler = setup_optimizer(net)
    writer = utils.get_tensorboard_writer(f'fit_fdh_0106_incp3hab_157_seed{seed}')
    losses = []

    for iter in range(EPOCH):
        loss_sum = 0.
        for batch_idx, data in enumerate(train_loader):
            inp_tensor = utils.to_device(data['inp_tensor'], DEVICE)
            target = utils.to_device(data['fdh'], DEVICE)
            out = net(inp_tensor).view(-1, 15, 15)
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

    torch.save(net.state_dict(), f'models/fit_fdh_incp3hab_157_seed{seed}.pkl')
    np.save(f'results/val_losses_incp3hab_157_seed{seed}.npy', losses)

# 主函数
def main():
    if not os.path.exists('results'):
        os.makedirs('results')
    for seed in range(1, 6):
        print(f'Running with seed: {seed}')
        train_loader, valid_loader = load_and_split_data(seed)
        train_model(seed, train_loader, valid_loader)

    for seed in range(1, 6):
        plt.plot(np.load(f'results/val_losses_incp3hab_157_seed{seed}.npy'), label=f"Seed {seed} Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()