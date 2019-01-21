from torch import nn
import torch
from torch.nn import functional as F


x = torch.randn(1,3,1,1)
z = [x] + [x]
print (x.shape)
print (z)
print (torch.cat(z, 1))