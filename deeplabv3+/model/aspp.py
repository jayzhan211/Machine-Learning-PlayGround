import math
import torch
from torch import nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super().__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else: # resnet 50-152  => 512 * 4 (=2048)
            inplanes = 2048

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        # padding = (kernel_size - 1) * dilations  // 2
        # (3 -1) * dilations // 2 = dilations
        self.aspp1 = ASPPBlock(inplanes, 256, 1, padding=0, dilations=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = ASPPBlock(inplanes, 256, 3, padding=dilations[1], dilations=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = ASPPBlock(inplanes, 256, 3, padding=dilations[2], dilations=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = ASPPBlock(inplanes, 256, 3, padding=dilations[3], dilations=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # bn will remove channel mean, which is bias of conv_layer , so set bias as False
            nn.Conv2d(inplanes, 256, kernel_size=1, stride=1, bias=False),
            BatchNorm(256),
            nn.ReLU()
        )
        # 1280 = 256 * 5
        self.conv1 = nn.Conv2d(1280, 256, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.relu(self.bn(self.conv1(x)))
        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d) or isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)



class ASPPBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super().__init__()

        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()
        self._init_weight()
    def forward(self, x):
        out = self.relu(self.bn(self.atrous_conv(x)))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d) or isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)



# https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/aspp.py