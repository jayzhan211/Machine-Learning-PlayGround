import torch
import torch.nn as nn



x = torch.randn(3, 10)
z = x[1::]
print(x)
print(z)

