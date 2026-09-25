import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import dataset1
from dataset1 import ReactorCoreDataset
import utils
import pickle
from model_HAB import Model  # 确保导入 Model 类

# 设备配置
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 512
TEST_BATCH_SIZE = 1024
DEVICE_IDS = [0]


def analyze_model(model, input_shape=(5, 15, 15)):
    """
    分析模型的结构、参数量和神经元数量
    :param model: 要分析的PyTorch模型
    :param input_shape: 模型输入的形状 (channels, height, width)
    """
    print("=" * 80)
    print("模型分析报告")
    print("=" * 80)

    # 1. 打印模型架构
    print("\n[模型架构]")
    print(model)

    # 2. 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n[参数量统计]")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"不可训练参数量: {total_params - trainable_params:,}")

    # 3. 统计各层信息
    conv_layers = []
    linear_layers = []
    bn_layers = []
    activation_layers = []
    other_layers = []

    total_neurons = 0

    # 递归遍历所有模块
    def traverse_modules(module, name_prefix=""):
        nonlocal total_neurons
        for name, child in module.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name

            if isinstance(child, nn.Conv2d):
                conv_layers.append({
                    'name': full_name,
                    'in_channels': child.in_channels,
                    'out_channels': child.out_channels,
                    'kernel_size': child.kernel_size,
                    'stride': child.stride,
                    'padding': child.padding
                })
            elif isinstance(child, nn.Linear):
                linear_layers.append({
                    'name': full_name,
                    'in_features': child.in_features,
                    'out_features': child.out_features
                })
                total_neurons += child.out_features
            elif isinstance(child, nn.BatchNorm2d) or isinstance(child, nn.BatchNorm1d):
                bn_layers.append({
                    'name': full_name,
                    'num_features': child.num_features
                })
            elif isinstance(child, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)):
                activation_layers.append({
                    'name': full_name,
                    'type': type(child).__name__
                })
            else:
                other_layers.append({
                    'name': full_name,
                    'type': type(child).__name__
                })

            # 递归遍历子模块
            if len(list(child.children())) > 0:
                traverse_modules(child, full_name)

    traverse_modules(model)

    # 4. 打印各层详细信息
    if conv_layers:
        print("\n[卷积层信息]")
        for layer in conv_layers:
            print(f"层名: {layer['name']}")
            print(f"  输入通道: {layer['in_channels']}, 输出通道: {layer['out_channels']}")
            print(f"  卷积核: {layer['kernel_size']}, 步长: {layer['stride']}, 填充: {layer['padding']}")

    if linear_layers:
        print("\n[全连接层信息]")
        for layer in linear_layers:
            print(f"层名: {layer['name']}")
            print(f"  输入特征: {layer['in_features']}, 输出特征: {layer['out_features']}")

    if bn_layers:
        print("\n[批归一化层信息]")
        for layer in bn_layers:
            print(f"层名: {layer['name']}, 特征数: {layer['num_features']}")

    if activation_layers:
        print("\n[激活层信息]")
        for layer in activation_layers:
            print(f"层名: {layer['name']}, 类型: {layer['type']}")

    if other_layers:
        print("\n[其他层信息]")
        for layer in other_layers:
            print(f"层名: {layer['name']}, 类型: {layer['type']}")

    # 5. 神经元数量统计
    print("\n[神经元数量统计]")
    print(f"全连接层神经元总数: {total_neurons}")

    # 6. 计算模型大小（MB）
    param_size = total_params * 4 / (1024 ** 2)  # 假设float32（4字节）
    print(f"\n[模型大小估计]")
    print(f"模型参数大小: {param_size:.2f} MB (float32)")


# 主函数
def main():
    # 1. 准备数据集（可选，仅用于验证）
    val_data_path = 'data/data_test.pkl'
    with open(val_data_path, 'rb') as f:
        val_data = pickle.load(f)

    val_dataset = ReactorCoreDataset(val_data, cal_min_max=False)
    valid_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 2. 创建模型 - 修复了这里的语法错误
    model = nn.Sequential(
        Model(include_top=False, num_inception_resnet=3, input_channel=5),  # 使用 Model 类
        nn.Flatten(),
        nn.Linear(32 * 15 * 15, 4 * 15 * 15),
        nn.BatchNorm1d(4 * 15 * 15),
        nn.ReLU(),
        nn.Linear(4 * 15 * 15, 15 * 15)
    ).to(DEVICE)

    # 3. 加载预训练权重 - 添加了权重映射处理
    try:
        # 加载权重
        state_dict = torch.load('models/fit_fdh_incp3cbam_157_seed3.pkl', map_location=DEVICE)

        # 处理可能的 DataParallel 前缀
        if all(key.startswith('module.') for key in state_dict.keys()):
            # 移除 'module.' 前缀
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # 加载状态字典
        model.load_state_dict(state_dict)
        print("成功加载预训练权重")
    except Exception as e:
        print(f"警告: 无法加载预训练权重，将继续分析模型结构。错误信息: {str(e)}")

    # 4. 分析模型
    analyze_model(model, input_shape=(5, 15, 15))

    # 5. 验证模型性能（可选）
    criterion = nn.MSELoss()
    model.eval()

    with torch.no_grad():
        loss_sum = 0.
        for batch_idx, data in enumerate(valid_loader):
            inp_tensor = utils.to_device(data['inp_tensor'], DEVICE)
            target = utils.to_device(data['fdh'], DEVICE)
            out = model(inp_tensor).view(-1, 15, 15)
            loss = criterion(out, target)
            loss_sum += loss.item()
            if batch_idx >= 10:  # 只测试前10个样本以节省时间
                break

    valid_metric = loss_sum / min(len(valid_loader), 10)  # 使用实际测试的样本数
    print("\n[模型性能]")
    print(f"验证集MSE损失: {valid_metric:.6f}")


if __name__ == "__main__":
    main()