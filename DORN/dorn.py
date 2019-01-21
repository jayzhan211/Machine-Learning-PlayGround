import torch
from torch import nn
class FullImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_pooling = nn.AvgPool2d(8)