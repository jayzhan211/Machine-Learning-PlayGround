import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.randn(2, 10, 3)
norm_x = x.norm(p=2, dim=1, keepdim=True)
z = x / norm_x
_z = F.normalize(x, p=2, dim=1)
assert z.allclose(_z), "error"
