import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


x = torch.randn(10,3,3,3)
y = torch.randn(10,3,5,5)

z = x.size()[-2:]
print(z)
k = F.interpolate(y, x.size()[-2:], mode='bilinear', align_corners=True)
print(k.shape)
