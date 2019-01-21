import torch
from torch import nn
import torch.nn.functional as F
import math

def swish(x):
    return x * F.sigmoid(x)

class Pixel_Suffle_Block(nn.Module):
    def __init__(self, in_channels, out_channels, up_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_factor)
    def forward(self, x):
        return swish(self.pixel_shuffle(self.conv(x)))

class Residual_Block(nn.Module):
    def __init__(self, inplanes, planes, kernelsize=3, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernelsize, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernelsize, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = swish(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual



class SRGAN(nn.Module):
    def __init__(self):
        super().__init__()
        class Generator(nn.Module):
            def __init__(self, n_residual_blocks=16, upscale_factor=4):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
                self.n_residual_blocks = n_residual_blocks
                self.residual_block = self.res_layer(n_blocks=n_residual_blocks)
                self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
                self.bn2 = nn.BatchNorm2d(64)
                self.upsample = self.up_layer(upscale_factor)
                self.conv3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

            def res_layer(self, n_blocks):
                layers = []
                for _ in range(n_blocks):
                    layers.append(Residual_Block(inplanes=64, planes=64))
                return nn.Sequential(*layers)
            def up_layer(self, up_scale):
                layers = []
                n_up = int(math.log(up_scale, 2))
                for _ in range(n_up):
                    layers.append(Pixel_Suffle_Block(in_channels=64, out_channels=256))
                return nn.Sequential(*layers)

            def forward(self, x):
                out = swish(self.conv1(x))
                out_residual = out
                out = self.residual_block(out)
                out = out_residual + self.bn2(self.conv2(out))
                out = self.upsample(out)
                out = self.conv3(out)
                return out
        # end
        class Discriminator(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

                self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
                self.bn2 = nn.BatchNorm2d(64)
                self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
                self.bn3 = nn.BatchNorm2d(128)
                self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
                self.bn4 = nn.BatchNorm2d(128)
                self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
                self.bn5 = nn.BatchNorm2d(256)
                self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
                self.bn6 = nn.BatchNorm2d(256)
                self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
                self.bn7 = nn.BatchNorm2d(512)
                self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
                self.bn8 = nn.BatchNorm2d(512)

                self.conv9 = nn.Conv2d(512, 1, kernel_size=1)
            def forward(self, x):
                x = swish(self.conv1(x))
                x = swish(self.bn2(self.conv2(x)))
                x = swish(self.bn3(self.conv3(x)))
                x = swish(self.bn4(self.conv4(x)))
                x = swish(self.bn5(self.conv5(x)))
                x = swish(self.bn6(self.conv6(x)))
                x = swish(self.bn7(self.conv7(x)))
                x = swish(self.bn8(self.conv8(x)))
                x = self.conv9(x)
                return torch.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)
        # end

    def forward(self):


# Reference
# https://github.com/aitorzip/PyTorch-SRGAN/blob/master/models.py
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# https://github.com/xinntao/BasicSR/tree/master/codes/models

