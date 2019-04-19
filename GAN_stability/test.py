import torch
from torch import nn
from torch.nn import functional as F
import math
from torch.autograd import Variable

# x = torch.log2(torch.Tensor([3.0])).int()[]
batch_size = 256
num_classes = 1000

x = torch.randn(batch_size, num_classes)
y = torch.randint(low=1, high=num_classes, size=(batch_size,))

idx = Variable(torch.LongTensor(range(batch_size)))
print(y.size())
out = x[idx, y]
_idx = range(batch_size)
_out = x[_idx, y]
# print(out)
# print(_idx)
print(out[:10])
print(_out[:10])
assert out.allclose(_out), "WTF"
