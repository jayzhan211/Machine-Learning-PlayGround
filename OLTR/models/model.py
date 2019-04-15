import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

        # print(self.__str__())
        # print(self.__repr__())
    def forward(self, x):

        return x
class MnistMode1l(nn.Module):
    def __init__(self, num_classes=10):
        super(MnistMode1l, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

        # print(self.__str__())
        # print(self.__repr__())
    def forward(self, x):

        return x

    def __repr__(self):
        return super().__repr__() + '123'


import torch
x= torch.randn(1,1,32,32)

m = MnistModel()
n = MnistMode1l()
print(m)
print(n)
# print(m(x))
# import numpy as np
# a = sum(np.prod(i) for i in [(3,2), (50,10,3), 2])
# print(a)