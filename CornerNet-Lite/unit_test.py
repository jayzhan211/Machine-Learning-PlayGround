import torch
import torch.nn as nn

b, c, h, w = 2, 3, 5, 5
K = 10

feat = torch.randn(b, h*w, c)
ind = torch.randn(1, 1)

z = ind.unsqueeze(1).expand_as(feat)
print(z.size())