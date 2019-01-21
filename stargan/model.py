
from torch import nn
import torch
import numpy as np
class ResidualBlock(nn.Module):
    def __init__(self, in_nc, out_nc):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_nc, out_nc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_nc, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_nc, affine=True)
        )
    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    def __init__(self, gf=64, c_dim=5, res_blocks=6):
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, gf, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(gf, affine=True))
        layers.append(nn.ReLU(inplace=True))


        cur_dim = gf
        for _ in range(2):
            layers.append(nn.Conv2d(cur_dim, cur_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(cur_dim * 2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            cur_dim = cur_dim * 2

        for _ in range(res_blocks):
            layers.append(ResidualBlock(cur_dim, cur_dim))

        for _ in range(2):
            layers.append(nn.ConvTranspose2d(cur_dim, cur_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(cur_dim // 2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            cur_dim = cur_dim // 2

        layers.append(nn.Conv2d(cur_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1).repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, image_size=128, df=64, c_dim=5, num_convs=6):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(3, df, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        cur_dim = df
        for _ in range(1, num_convs):
            layers.append(nn.Conv2d(cur_dim, cur_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            cur_dim = cur_dim * 2

        kernel_size = int(image_size / np.power(2, num_convs))
        self.main = nn.Sequential(*layers)
        self.conv_src = nn.Conv2d(cur_dim, 1, kernel_size=3, padding=1, bias=False)
        self.conv_cls = nn.Conv2d(cur_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        out = self.main(x)
        src = self.conv_src(out)
        cls = self.conv_cls(out)
        return src, cls.view(cls.size(0), cls.size(1))





