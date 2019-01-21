import torch
from torch import nn
import math

class RRDB_Net(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=2):
        super().__init__()
        n_upscale = int(math.log2(upscale))
        feature_conv = B.conv_block(in_nc, nf, kernel_size=3)