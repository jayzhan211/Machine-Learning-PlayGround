import math
import torch
from torch import nn
import time
import torch.nn.functional as F

from torch.autograd import Variable, grad
import torch

x = torch.tensor([3.0,6.0,2.0])
x = x.norm(2)
print(x)