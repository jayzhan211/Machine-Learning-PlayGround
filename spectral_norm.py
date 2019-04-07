import torch
from torch.optim.optimizer import Optimizer, required

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter
import copy
from torch.nn.utils import spectral_norm
import numpy as np
from torch.nn.functional import normalize

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        for _ in range(self.power_iterations):
            v = normalize(w.t().mv(u), dim=0)
            u = normalize(w.mv(v), dim=0)

        sigma = u.dot(w.mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        # print(self.module)
        weight = getattr(self.module, self.name)
        print(type(weight.data))
        h, w = weight.size()
        # height = w.data.shape[0]
        # width = w.view(height, -1).data.shape[1]
        # print(type(w))
        u = Parameter(weight.data.new(h).normal_(0, 1), requires_grad=False)
        v = Parameter(weight.data.new(w).normal_(0, 1), requires_grad=False)
        # u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        # v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        # print(w.data)
        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


def check():

    n, m = 4, 3
    x = torch.randn(n, m)
    h = nn.Linear(3, 4)
    m = SpectralNorm(copy.deepcopy(h), power_iterations=99)
    a = copy.deepcopy(h).weight
    b = copy.deepcopy(h).weight
    print(a)
    print(b)
    print(m.module.weight_bar)
    z = m(x)
    print(m.module.weight)
    _A = m.module.weight
    _, s, _ = np.linalg.svd(b.detach())
    print(s)
    _B = b / s[0]
    print(a)
    print(b)
    if not torch.allclose(_A, _B):
        raise ValueError("Error 404")


check()

