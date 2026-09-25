import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
import dataset1
from dataset1 import *
import utils
from model_HAB import Model
import numba
import matplotlib.pyplot as plt
from cam_tools import *

# 这条脚本是为了可视化一条数据

DEVICE = 'cuda:0'
BATCH_SIZE = 512
TEST_BATCH_SIZE = 1024
DEVICE_IDS = [0]


# 准备数据集
val_data_path = 'data/data_cr.pkl'  # 验证数据集的路径
with open(val_data_path, 'rb') as f:
    val_data = pickle.load(f)

# 创建验证数据集对象
val_dataset = dataset1.ReactorCoreDataset(val_data, cal_min_max=False)  # val_dataset加载模式
valid_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)  # 创建数据加载器,确保按顺序读取

# 模型: 去掉分类头, 然后连接多个线性层, 最后的输出是一个 15 * 15 的向量
net = nn.Sequential(
    Model(include_top=False, num_inception_resnet=3, input_channel=5), # num的数量必须和训练时的数量对齐，否则会报错
    nn.Flatten(),
    nn.Linear(32 * 15 * 15, 4 * 15 * 15),
    nn.BatchNorm1d(4 * 15 * 15),
    nn.ReLU(),
    nn.Linear(4 * 15 * 15, 15 * 15)
).to(DEVICE)

net = nn.DataParallel(net, DEVICE_IDS)
last_conv_layer=get_last_conv(net)
last_conv_layer.register_forward_hook(save_activation('conv_out'))
last_conv_layer.register_backward_hook(save_gradient('conv_grad'))

criterion = nn.MSELoss()
net.load_state_dict(torch.load('models/fit_fdh_0103_hab_seed5.pkl'))
net.eval()

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

valid_metric = cal_metric(net, valid_loader)
print(valid_metric)


def show_input(input_tensor):
    import torch
    import matplotlib.pyplot as plt

# 将所有通道的特征图相加
    summed_tensor = input_tensor.sum(dim=0)  # 对所有通道求和，得到一个 (15, 15) 的张量

# 归一化
    summed_tensor = summed_tensor - summed_tensor.min()  # 将最小值设为0
    summed_tensor = summed_tensor / summed_tensor.max()  # 将最大值设为1

# 可视化叠加后的图像
    plt.imshow(summed_tensor.cpu().detach().numpy(), cmap='jet')
    plt.title("Summed Channels")
    plt.axis('off')
    plt.colorbar()
    # plt.savefig('input_tensor.png')
    plt.show()

input_tensor = None
for batch_idx, data in enumerate(valid_loader):
    batch_inp_tensor = utils.to_device(data['inp_tensor'], DEVICE)
    target = utils.to_device(data['fdh'], DEVICE)
    input_tensor = batch_inp_tensor[0]
    break
show_input(input_tensor)

heatmap = generate_grad_cam(input_tensor, net)
print(heatmap.shape)
# 可选：对 heatmap 进行归一化处理（0-1 范围内）
heatmap = np.array(heatmap)  # 假设 heatmap 是一个 15x15 的数组或 Tensor
heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # 归一化到 0-1 之间

# 创建一个热力图
plt.imshow(heatmap, cmap='jet', interpolation='nearest')
plt.colorbar()  # 显示颜色条
plt.title('Heatmap Visualization (15x15)')
# plt.savefig('heat.png')
plt.show()
