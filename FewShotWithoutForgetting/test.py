import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

num_feat = 10
x = torch.randn(2, 3)
y = torch.randn(3)
z = y.expand_as(x)
_z = x * z
print(x)
print(z)
print(_z)