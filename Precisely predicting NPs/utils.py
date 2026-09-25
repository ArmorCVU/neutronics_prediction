import math
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn


def to_device(obj, device):
    """将 dict 中的内容转移到 device 上并返回.
    """
    if torch.is_tensor(obj):  # Tensor
        return obj.to(device)
    if isinstance(obj, tuple):  # Tuple
        return tuple(to_device(t, device) for t in obj)
    if isinstance(obj, list):  # List
        return [to_device(t, device) for t in obj]
    if isinstance(obj, dict):  # Dict
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (int, float, str)):  # int / float / str
        return obj
    raise ValueError(f'{obj} has unsupported type {type(obj)}')


def get_tensorboard_writer(file_name):
    writer = SummaryWriter(f'logs/{file_name}', flush_secs=10)
    return writer


def write_train_valid(writer: SummaryWriter, train, valid, step):
    writer.add_scalars(
        'Log', {'Train MSE': train, 'Valid MSE': valid}, global_step=step
    )


# def get_warm_start_cosine_lr(optimizer, total_epochs, warm_start_epochs=5, warm_start_lr=1e-3, cosine_init_lr=1e-5):
#     """这个函数返回一个 lr_scheduler.LambdaLR.
#     -如果当前 epoch <= warm_start_epochs, 学习率为 warm_start_lr.
#     -如果当前 epoch > warm_start_epochs, 学习率从 cosine_init_lr 开始, 进行余弦退火学习率调节.
#     """
#     def lr_lambda(epoch):
#         if epoch <= warm_start_epochs:
#             return warm_start_lr
#         else:
#             lr = (1 + math.cos((epoch - warm_start_epochs) /
#                                (total_epochs - warm_start_epochs) * math.pi)) * 0.5 * cosine_init_lr
#             return lr
#
#     return LambdaLR(optimizer, lr_lambda=lr_lambda)


class MAELoss(nn.Module):
    def __init__(self, reduction='mean'):
        assert reduction in ['mean', 'sum', None]
        self.reduction = reduction
        super().__init__()

    def forward(self, out, target):
        ret = torch.abs(out - target)
        if self.reduction == 'mean':
            return ret.mean()
        if self.reduction == 'sum':
            return ret.sum()
        return ret
