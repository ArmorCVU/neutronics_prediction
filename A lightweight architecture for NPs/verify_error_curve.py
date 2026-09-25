import torch
from torch.utils.data import DataLoader
import pickle
import dataset
from model_noatt import Model
verify.py

# 设备配置
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 准备数据集
val_data_path = 'data/data.pkl'  # 验证数据集的路径
with open(val_data_path, 'rb') as f:
    val_data = pickle.load(f)

# 创建验证数据集对象
val_dataset = dataset.ReactorCoreDataset(val_data, cal_min_max=False) #val_dataset加载模式
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)  # 创建数据加载器,确保按顺序读取

# 加载模型
model_path = 'models/fit_cbc_095.pkl'  # 替换为模型权重的路径
model = Model(input_channel=5, num_inception_resnet=3).to(device)  # 初始化模型
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.eval()  # 设置为评估模式

# 定义评估指标
criterion = torch.nn.MSELoss()  # 均方误差

# 进行预测和评估
val_loss = 0
with torch.no_grad():  # 禁用梯度计算
    for data in val_loader: # 使用val_loader加载验证集，如果使用val_dataset加载则需要相应更改。
        inp_tensor = data['inp_tensor'].to(device)  # 确保输入张量在正确的设备上
        target = data['cbc'].to(device)  # 确保目标张量在正确的设备上

        output = model(inp_tensor)  # 前向传播
        output = output.view(-1)  # 调整输出形状以匹配目标张量

        loss = criterion(output, target.float())  # 计算损失

        val_loss += loss.item() * inp_tensor.size(0)
        val_loss /= len(val_dataset)  # 计算平均损失

        # 计算平均损失
        print(f'Validation Loss: {val_loss}')

