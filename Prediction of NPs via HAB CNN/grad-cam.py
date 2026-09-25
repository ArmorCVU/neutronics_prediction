import torch
from torch.autograd import Variable

# 不再使用本脚本

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM.

        Parameters:
        - model: PyTorch model
        - target_layer: Layer from which gradients are computed (last convolutional layer in this case)
        """
        self.model = model.eval()
        self.target_layer = target_layer
        self.feature = None
        self.gradient = None
        self.handlers = []

        # Register forward hooks
        self.handlers.append(target_layer.register_forward_hook(self.forward_hook_function))
        # Register backward hooks
        self.handlers.append(target_layer.register_backward_hook(self.backward_hook_function))

    def forward_hook_function(self, module, input, output):
        self.feature = output.data.cpu()

    def backward_hook_function(self, module, grad_in, grad_out):
        self.gradient = grad_out[0].data.cpu()

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def generate_cam(self, input_tensor, class_idx=None, retain_graph=False):
        """
        Generate Grad-CAM for given input tensor.

        Parameters:
        - input_tensor: Input tensor for which Grad-CAM is generated
        - class_idx: Target class index (if not provided, the highest scoring class is used)
        - retain_graph: Whether to retain the computational graph
        """
        # Forward pass
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output.cpu().data).item()

        # Zero out gradients
        self.model.zero_grad()

        # One-hot encoding of the target class
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        one_hot = Variable(one_hot.to(input_tensor.device), requires_grad=True)

        # Compute gradients
        output.backward(gradient=one_hot, retain_graph=retain_graph)

        # Compute weights by averaging gradients over spatial dimensions
        weights = self.gradient.mean((2, 3), keepdim=True)

        # Compute weighted feature maps
        weighted_feature = weights * self.feature
        cam = weighted_feature.sum(dim=1)

        # Resize to match input image size
        cam = F.interpolate(cam.unsqueeze(0), size=input_tensor.shape[2:], mode='bilinear', align_corners=False)[0]

        # Normalize CAM
        cam = cam - cam.min()
        cam = cam / cam.max()

        # Remove registered hooks
        self.remove_handlers()

        return cam