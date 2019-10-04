from abc import ABC, abstractmethod
import torch.utils.data as data

class BaseDataset(data.Dataset, ABC):
    def __init__(self, opt):
        """
        abstract base class for datasets
        :param opt:
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass
