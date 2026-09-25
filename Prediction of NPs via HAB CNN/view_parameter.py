import torch
import torch.nn as nn

# 导入模型定义
from model import Model

def count_parameters(model):
    """
    计算模型的总参数量和可训练参数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

# 创建模型实例
model = Model(input_channel=13, num_inception_resnet=5, include_top=True)

# 计算参数量
total_params, trainable_params = count_parameters(model)

# 打印结果
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")