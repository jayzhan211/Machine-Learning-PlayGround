import collections
import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast

from .comm import SyncMaster

__all__ = ['SynchronizedBatchNorm1d', 'SynchronizedBatchNorm2d', 'SynchronizedBatchNorm3d']

class _SynchronizedBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=.1, affine=True):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=affine)
        self._sync_master = SyncMaster(self._data_parallel_master)
