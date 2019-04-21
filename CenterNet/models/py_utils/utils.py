import torch
import torch.nn as nn

class conv_bn_act(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch, stride=1, use_bn=True):
        super(conv_bn_act, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class fc_bn_act(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=True):
        super(fc_bn_act, self).__init__()

        self.fc = nn.Linear(in_ch, out_ch);
        self.bn = nn.BatchNorm1d(out_ch) if use_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.fc(x)))


