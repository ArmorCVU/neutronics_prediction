import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn as F
from torch.utils.data import TensorDataset, DataLoader

device = "cpu"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("using {} device.".format(device))

input_data = np.load('data/input_matrix.npy')
output_data = np.load('data/output_file.npy')

"""
# preprocess
input_data_1 = input_data[:, 0, :, :]
# 指定填充的大小，例如在第二和第三维上分别填充 1 行和 1 列
padding = ((0, 0), (0, 1), (0, 1))

# 使用 np.pad 进行填充
padded_array_1 = np.pad(input_data_1, pad_width=padding, mode='constant', constant_values=0)

input_data_2 = input_data[:, 1, :, :]
# 指定填充的大小，例如在第二和第三维上分别填充 1 行和 1 列
padding = ((0, 0), (0, 1), (0, 1))

# 使用 np.pad 进行填充
padded_array_2 = np.pad(input_data_1, pad_width=padding, mode='constant', constant_values=-1)

input_data = np.stack((padded_array_1, padded_array_2), axis=1)

padding = ((0, 0), (0, 0), (0, 1), (0, 1))
output_data = np.pad(output_data, pad_width=padding, mode='constant', constant_values=-1)
"""

# 定义划分比例
split_ratio = 0.8  # 80%用于训练，20%用于测试

# 计算划分的索引
split_index = int(len(input_data) * split_ratio)

# 划分数据为训练集和测试集
train_input = input_data[:split_index]
train_output = output_data[:split_index]
test_input = input_data[split_index:]
test_output = output_data[split_index:]

# 将NumPy数组转换为PyTorch张量
train_input_tensor = torch.tensor(train_input, dtype=torch.float32)
train_output_tensor = torch.tensor(train_output, dtype=torch.float32)
test_input_tensor = torch.tensor(test_input, dtype=torch.float32)
test_output_tensor = torch.tensor(test_output, dtype=torch.float32)

# 创建训练和测试数据集
train_dataset = TensorDataset(train_input_tensor, train_output_tensor)
test_dataset = TensorDataset(test_input_tensor, test_output_tensor)

# 创建训练和测试数据加载器
batch_size = 64  # 指定批量大小
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义ResNet模型
class MultiTaskResNet(nn.Module):
    def __init__(self, block, layers, num_class):
        super(MultiTaskResNet, self).__init__()
        self.in_channels = 64
        neuron_num = 128  # 定义神经元数量
        num_class = 15 * 15
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)  # 使 用批量归一化
        self.relu = nn.ReLU(inplace=True)  # 使用ReLU作为激活函数
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 15 * 15)

        # 定义共享层
        self.shared_fc = nn.Linear(225, neuron_num)

        # 定义任务特定层
        self.fc1 = nn.Linear(neuron_num, num_class)  # task1
        self.fc2 = nn.Linear(neuron_num, num_class)  # task2

        # 定义残差模块

    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        shared_out = self.shared_fc(x)
        # shared_out = F.ReLU(shared_out)

        out1 = self.fc1(shared_out)
        out2 = self.fc2(shared_out)
        return out1, out2


# 定义Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


# 训练
model = MultiTaskResNet(ResidualBlock, [2, 2, 2, 2], 15 * 15)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000

for epoch in range(num_epochs):
    running_loss = 0.0
    num_samples = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        target1 = targets[:, 3, :, :]
        target2 = targets[:, 3, :, :]
        optimizer.zero_grad()
        output1, output2 = model(inputs)
        loss1 = criterion(output1, target1)
        loss2 = criterion(output2, target2)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_samples += 1

    if epoch % 50 == 0:
        print('output1:', output1)
        print('output2:', output2)
        print('\ntarget1:', target1)
        print('target2:', target2)

    print(f'Epoch {epoch + 1}, Loss: {running_loss / num_samples}')

print('Finished Training')

# 保存模型的权重和参数
torch.save(model.state_dict(), 'model_weights.pth')

# 加载模型的权重和参数
model = MultiTaskResNet(ResidualBlock, [2, 2, 2, 2])  # 重新创建一个相同结构的模型
model.load_state_dict(torch.load('model_weights.pth'))

# 测试
model.eval()

num_samples = 0
test_loss = 0.0
for inputs, targets in test_loader:
    inputs, targets = inputs.to(device), targets.to(device)
    target1 = targets[:, 3, :, :]
    target2 = targets[:, 3, :, :]
    output1, output2 = model(inputs)
    loss1 = criterion(output1, target1)
    loss2 = criterion(output2, target2)
    loss = loss1 + loss2

    num_samples += 1
    test_loss += loss.item()

print('average loss:', test_loss / num_samples)
