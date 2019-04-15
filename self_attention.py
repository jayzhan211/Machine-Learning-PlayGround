import torch
from torch import nn
import torch.nn.functional as F



# A non-local block as used in SA-GAN
# code from https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py
class Self_Attn_Non_Local(nn.Module):
    def __init__(self, channel, conv=SNconv2d):
        super(Self_Attn_Non_Local, self).__init__()
        self.channel = channel
        self.conv = conv
        self.theta = self.conv(self.channel, self.channel // 8, kernel_size=1, padding=0, bias=False)
        self.phi = self.conv(self.channel, self.channel // 8, kernel_size=1, padding=0, bias=False)
        self.g = self.conv(self.channel, self.channel // 2, kernel_size=1, padding=0, bias=False)
        self.o = self.conv(self.channel // 2, self.channel, kernel_size=1, padding=0, bias=False)
        self.gamma = nn.Parameter(torch.zeros(), requires_grad=True)

    def forward(self, x):
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2, 2]) # b, c, h/2, w/2
        g = F.max_pool2d(self.g(x), [2, 2]) # b, c, h/2, w/2
        theta = theta.view(-1, self.ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.ch // 2, x.shape[2] * x.shape[3] // 4)
        # -1, c/8, h*w transpose -> -1, h*w/ c/8
        # -1, c/8, h*w/4
        # -1 ,h*w, h*w/4
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # -1, c/2, h*w/4 * -1, h*w/4, h*w  --> -1, c/2, h*w
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x
