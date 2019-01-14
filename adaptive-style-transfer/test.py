from torch import nn
import torch
import torch.nn.functional as F
x = torch.randint(5,(1,2,3,3)).type(torch.float32)
m = nn.AvgPool2d(kernel_size=2, stride=1)
y = F.avg_pool2d(x, kernel_size=2, stride=1)
print(x)
z = m(x)
print(y)
print(z)
print(y.shape)
print(z.shape)