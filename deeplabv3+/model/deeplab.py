import torch
import torch.nn as nn
import torch.nn.functional as F
from model.sync_bn.batch_norm import SynchronizedBatchNorm2d
from model.aspp import ASPP

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21, sync_bn=True, freeze_bn=False):
        super().__init__()
        if backbone == 'drn':
            output_stride = 8

