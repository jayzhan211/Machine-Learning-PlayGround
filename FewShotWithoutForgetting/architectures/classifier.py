import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


# Matrix ElementWise Multiplication
class LinearDiag(nn.Module):
    def __init__(self, num_feat, bias=False):
        super(LinearDiag, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_feat))
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_feat))
        else:
            self.register_parameter('bias', None)
    def forward(self, x):
        assert (x.dim() == 2 and x.size(1) == self.weight.size(0))
        out = x * self.weight.expand_as(x)
        if self.bias is not None:
            out = out + self.bias.expand_as(out)
        return out

