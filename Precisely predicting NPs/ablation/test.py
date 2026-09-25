import torch.nn as nn
import torch
from torch import Tensor, LongTensor

emb = nn.Embedding(3, embedding_dim=5)
inp = LongTensor([[[0, 1, 0],
                  [1, 2, 1],
                  [0, 1, 0]]])
out = emb(inp)
print(out)
print(out.shape)

# original: B, H, W, EMB
# new     : B, EMB, H, W
out = out.permute(0, 3, 1, 2)
print(out)
print(out[0, :, 0, 0])