
from matplotlib import pyplot as plt
import numpy as np

plt.figure(figsize=(13, 6))

plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 12  # 可选：设置全局字体大小

# 定义不同形状的标记
markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', '+', 'x']
labels = ['6 Res-blocks', '5 Res-blocks', '4 Res-blocks','3 Res-blocks', '2 Res-blocks', '1 Res-block'] #修改标签的名称

# 加载数据
data = []
for idx, i in enumerate([1, 2, 3, 4, 5, 6]):
    seeds = []
    for j in range(5):
        seeds.append(np.load(f'results/val_losses_0106_incp{i}_seed{j+1}.npy')) #修改数据地址

    data_seed = np.array(seeds)
    data_mean = np.mean(data_seed, axis=0)
    data_std = np.std(data_seed, axis=0)


    upper_bound = data_mean + data_std * 2 #增大阴影面积
    lower_bound = data_mean - data_std * 2 #增大阴影面积

    print(f"Data for {labels[idx]}: Mean shape = {data_mean.shape}")

    # 绘制主图的折线，markerfacecolor='none', marker='o', markersize='5', linewidth=1.5
    # line, = plt.plot(data_mean, label=f'{i}', marker='o', markersize='3')
    # 只在每隔 5 个点绘制标记
    # marker_indices = range(0, len(data_mean), 10)
    # line, = plt.plot(data_mean, label=f'{i}')
    # plt.scatter(marker_indices, [data_mean[i] for i in marker_indices], marker='o', s=20, color=line.get_color())

    marker_indices = range(0, len(data_mean), 10)
    line, = plt.plot(data_mean, label=labels[idx], marker=markers[idx], markersize=5, linestyle='-', markevery=10)
    plt.scatter(marker_indices, [data_mean[i] for i in marker_indices], marker=markers[idx], s=10, color=line.get_color())


    # 绘制主图的阴影
    plt.fill_between(
        range(len(data_mean)),
        lower_bound,
        upper_bound,
        color=line.get_color(),
        alpha=0.15
    )


# 设置主图的标签和标题
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Loss of Variations (MSE)', fontsize=20)
# plt.title('The Generalization of Block Quantities on $F_Q$ Over Epochs', fontsize=16)
plt.xlim(0, 49) # 主图的epoch范围
plt.ylim(0,0.02) # 主图的loss范围
plt.tick_params(axis='both', which='major', labelsize=16) # 设置坐标轴数字的字体大小
plt.legend(fontsize=18)
plt.grid(True)
plt.savefig('画图/depth.pdf')
# plt.savefig('画图/size20250118.png')
# plt.show()

