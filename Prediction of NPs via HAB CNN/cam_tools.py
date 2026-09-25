import numpy as np
import torch
import cv2
import torch.nn.functional as F
import torch.nn as nn

# Hook to get activations and gradients
activations = []
gradients = []

def save_activation(name):
    def hook(model, input, output):
        activations.append(output)
    return hook

def save_gradient(name):
    def hook(model, grad_in, grad_out):
        gradients.append(grad_out[0])
    return hook

'''
# 用于存储激活和梯度的列表
activations = [] 
gradients = [] 

# 用于捕获激活和梯度的钩子
def  forward_hook ( module , input, output ): 
    activations.append(output) 

def  behind_hook ( module , grad_input, grad_output ): 
    gradations.append(grad_output[ 0 ]) 

target_layer.register_forward_hook(forward_hook) 
target_layer.register_full_backward_hook(backward_hook)


import numpy as np 

# 计算权重 
weights = torch.mean(gradients[ 0 ], dim=[ 2 , 3 ]) 

# 计算 Grad-CAM 热图
heatmap = torch. sum (weights *activations[ 0 ], dim= 1 ).squeeze() 
heatmap = np.maximum(heatmap.cpu().detach().numpy(), 0 ) 
heatmap /= np.max ( heatmap)
'''

def generate_grad_cam(input_image, net):

    # Example usage:
    net.eval()
    output = net(input_image.reshape(1, 5, 15, 15))  # 增加 batch 维度
    # 清除梯度
    net.zero_grad()
    # 假设我们关注输出矩阵中的最大值位置（可以修改为其他回归输出）
    output = output.view(-1, 15, 15)  # 调整输出的形状为 [batch_size, 15, 15]
    target_value = torch.max(output)  # 获取输出中的最大值

    # 反向传播：计算该最大值位置的梯度
    target_value.backward()

    # Get the activations and gradients
    grad = gradients[-1]
    act = activations[-1]

    # Calculate the weights by averaging the gradients across the width and height of the feature maps
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # global average pooling for each channel

    # Compute the weighted sum of the activations
    weighted_activation_map = torch.sum(weights * act, dim=1, keepdim=True)

    # ReLU on the weighted activation map (to retain positive values)
    cam = F.relu(weighted_activation_map)

    # Normalize the heatmap between 0 and 1
    cam = cam.squeeze().cpu().data.numpy()
    cam -= np.min(cam)
    cam /= np.max(cam)

    return cam

def overlay_grad_cam(heatmap, image, alpha=0.5):
    """
    Overlay the Grad-CAM heatmap on the input image.
    heatmap: The generated Grad-CAM heatmap.
    image: The input image (numpy array or torch tensor).
    alpha: The transparency factor for overlaying the heatmap.
    """
    # Convert the image to numpy if it's a torch tensor
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)  # Convert from CxHxW to HxWxC

    # Resize the heatmap to match the image dimensions
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Normalize the heatmap to 0-255
    heatmap_resized = np.uint8(255 * heatmap_resized)

    # Apply the colormap to the heatmap
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # Overlay the heatmap on the original image
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

    return overlay

def get_last_conv(net):
    model_ = net.module
    # 遍历模型的所有子模块，找出卷积层
    conv_layers = []
    for name, layer in model_.named_modules():
        if isinstance(layer, nn.Conv2d):
            conv_layers.append((name, layer))

    # 打印所有卷积层
    # print("All Convolutional Layers:")
    for name, conv_layer in conv_layers:
        # print(f"{name}: {conv_layer}")
        continue

    # 获取最后一个卷积层
    last_conv_layer = conv_layers[-1][1]
    print("Last Convolutional Layer:", last_conv_layer)
    return last_conv_layer