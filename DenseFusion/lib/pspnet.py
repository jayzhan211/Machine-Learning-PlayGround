import torch
from torch import nn
import torch.nn.functional as F

import lib.extractors as extractors


class PSPModule(nn.Module):

    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stages(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    @staticmethod
    def _make_stage(features, size):
        prior = nn.AdaptiveAvgPool2d(size)
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, x):
        h, w = x.size()[2:]
        priors = [F.interpolate(stage(x), size=(h, w), mode='bilinear') for stage in self.stages] + [x]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSPUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)


class PSPNet(nn.Module):
    def __init__(self, sizes=(1, 2, 3, 6), psp_size=2048, backend='resnet18'):
        super(PSPNet, self).__init__()
        self.feats = getattr(extractors, backend)()
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(0.3)
        self.drop_2 = nn.Dropout2d(0.15)
        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.fianl = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        f, _ = self.feats(x)
        p = self.drop_1(self.psp(f))
        p = self.drop_2(self.up_1(p))
        p = self.drop_2(self.up_2(p))
        p = self.up_3(p)
        return self.fianl(p)
