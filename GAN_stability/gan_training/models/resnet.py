# Resnet Model for ImageNet
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import normalize
import torch


class ResnetBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(ResnetBlock, self).__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.conv_0 = nn.Conv2d(self.inplanes, self.planes, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(self.planes, self.planes, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        if inplanes != planes:
            self.downsample = nn.Conv2d(self.inplanes, self.planes, kernel_size=1)

    def forward(self, x):
        if self.inplanes != self.planes:
            residual = self.downsample(x)
        else:
            residual = x
        out = self.conv_0(self.activation(x))
        out = self.conv_1(self.activation(x))
        out = residual + out


class Generator(nn.Module):
    def __init__(self, num_classes, n_z=256, n_embed=256):
        super(Generator, self).__init__()
        self.n_z = n_z
        self.embedding = nn.Embedding(num_classes, n_embed)
        self.fc = nn.Linear(n_z + n_embed, self.n_f * self.n_z * self.n_z)
        self.layer1 = ResnetBlock(1024, 1024)
        self.layer2 = ResnetBlock(1024, 1024)
        self.layer3 = ResnetBlock(1024, 512)
        self.layer4 = ResnetBlock(512, 256)
        self.layer5 = ResnetBlock(256, 128)
        self.layer6 = ResnetBlock(128, 64)

        self.conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, z, y):
        assert (z.size(0) == y.size(0))
        batch_size = z.size(0)
        if y.dtype is torch.int64:
            y_emb = self.embedding(y)
        else:
            y_emb = y
        y_emb = normalize(y_emb)
        input = torch.cat([z, y_emb], dim=1)
        out = self.fc(input)
        out = F.interpolate(self.layer1(out), scale_factor=2)
        out = F.interpolate(self.layer2(out), scale_factor=2)
        out = F.interpolate(self.layer3(out), scale_factor=2)
        out = F.interpolate(self.layer4(out), scale_factor=2)
        out = F.interpolate(self.layer5(out), scale_factor=2)
        out = self.conv(F.leaky_relu(out, negative_slope=0.2))
        out = F.tanh(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, n_z, num_classes):
        super(Discriminator, self).__init__()

        self.num_classes = num_classes
        self.n_z = n_z
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.layer1 = ResnetBlock(64, 128)
        self.layer2 = ResnetBlock(128, 256)
        self.layer3 = ResnetBlock(256, 512)
        self.layer4 = ResnetBlock(512, 1024)
        self.layer5 = ResnetBlock(1024, 1024)
        self.layer6 = ResnetBlock(1024, 1024)
        self.d_resnet = nn.Sequential(
            ResnetBlock(64, 128),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            ResnetBlock(128, 256),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            ResnetBlock(256, 512),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            ResnetBlock(512, 1024),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            ResnetBlock(1024, 1024),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            ResnetBlock(1024, 1024)
        )
        self.fc = nn.Linear(1024 * 4 * 4, self.num_classes)

    def forward(self, x, y):
        assert (x.size(0) == y.size(0))
        batch_size = x.size(0)
        out = self.conv(x)
        out = self.d_resnet(out)
        out = out.view(batch_size, -1)
        out = self.fc(F.leaky_relu(out, negative_slope=0.2))

        idx = range(batch_size)
        out = out[idx, y]
        return out
