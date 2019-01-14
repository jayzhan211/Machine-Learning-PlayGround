import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.in0 = nn.InstanceNorm2d(3, affine=True)

        self.conv1 = Reflection_Pad_Conv2d(3, 32, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = Reflection_Pad_Conv2d(32, 32, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(32, affine=True)
        self.conv3 = Reflection_Pad_Conv2d(32, 64, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(64, affine=True)
        self.conv4 = Reflection_Pad_Conv2d(64, 128, kernel_size=3, stride=2)
        self.in4 = nn.InstanceNorm2d(128, affine=True)
        self.conv5 = Reflection_Pad_Conv2d(128, 256, kernel_size=3, stride=2)
        self.in5 = nn.InstanceNorm2d(256, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.in0(x)
        out = self.relu(self.in1(self.conv1(out)))
        out = self.relu(self.in2(self.conv2(out)))
        out = self.relu(self.in3(self.conv3(out)))
        out = self.relu(self.in4(self.conv5(out)))
        out = self.relu(self.in5(self.conv5(out)))
        return out


class Reflection_Pad_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(256)
        self.res3 = ResidualBlock(256)
        self.res4 = ResidualBlock(256)
        self.res5 = ResidualBlock(256)
        self.res6 = ResidualBlock(256)
        self.res7 = ResidualBlock(256)
        self.res8 = ResidualBlock(256)
        self.res9 = ResidualBlock(256)

        self.deconv1 = UpsampleBlock(256, 256, kernel_size=3, stride=2)
        self.in1 = nn.InstanceNorm2d(256)
        self.deconv2 = UpsampleBlock(256, 128, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(128)
        self.deconv3 = UpsampleBlock(128, 64, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(64)
        self.deconv4 = UpsampleBlock(64, 32, kernel_size=3, stride=2)
        self.in4 = nn.InstanceNorm2d(32)
        self.deconv5 = Reflection_Pad_Conv2d(32, 3, kernel_size=7, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.res1(x)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        out = self.res6(out)
        out = self.res7(out)
        out = self.res8(out)
        out = self.res9(out)

        out = self.relu(self.in1(self.deconv1(out)))
        out = self.relu(self.in2(self.deconv2(out)))
        out = self.relu(self.in3(self.deconv3(out)))
        out = self.relu(self.in4(self.deconv4(out)))
        out = self.sigmoid(self.deconv5(out))

        out = out * 2.0 - 1.0

        return  out




class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()


        self.conv1 = Reflection_Pad_Conv2d(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels)
        self.conv2 = Reflection_Pad_Conv2d(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(x))
        out =  out + x
        return  out

class UpsampleBlock(nn.Module):
    def __init__(self, in_channesl, out_channels, kernel_size, stride, scale_factor=None):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv2d = Reflection_Pad_Conv2d(in_channesl, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x_in = x
        if self.scale_factor:
            x_in = F.interpolate(x_in, mode="nearest", scale_factor=self.scale_factor)
        out = self.conv2d(x_in)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = Reflection_Pad_Conv2d(3, 128, kernel_size=5, stride=2)
        self.in1 = nn.InstanceNorm2d(128)
        self.conv2 = Reflection_Pad_Conv2d(128, 128, kernel_size=5, stride=2)
        self.in2 = nn.InstanceNorm2d(128)
        self.conv3 = Reflection_Pad_Conv2d(128, 256, kernel_size=5, stride=2)
        self.in3 = nn.InstanceNorm2d(256)
        self.conv4 = Reflection_Pad_Conv2d(256, 512, kernel_size=5, stride=2)
        self.in4 = nn.InstanceNorm2d(512)
        self.conv5 = Reflection_Pad_Conv2d(512, 512, kernel_size=5, stride=2)
        self.in5 = nn.InstanceNorm2d(512)
        self.conv6 = Reflection_Pad_Conv2d(512, 1024, kernel_size=5, stride=2)
        self.in6 = nn.InstanceNorm2d(1024)
        self.conv7 = Reflection_Pad_Conv2d(1024, 1024, kernel_size=5, stride=2)
        self.in7 = nn.InstanceNorm2d(1024)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        self.pred = Reflection_Pad_Conv2d(1024, 1, kernel_size=3, stride=1)
        self.in_pred =  nn.InstanceNorm2d(1)

        self.auxiliary1 = nn.Conv2d(128, 1, kernel_size=5, stride=2)
        self.auxiliary2 = nn.Conv2d(128, 1, kernel_size=5, stride=2)
        self.auxiliary4 = nn.Conv2d(512, 1, kernel_size=5, stride=2)
        self.auxiliary6 = nn.Conv2d(1024, 1, kernel_size=5, stride=2)


    def forward(self, x):
        out = self.lrelu(self.in1(self.conv1(x)))
        out1 = self.auxiliary1(out)
        out = self.lrelu(self.in2(self.conv2(out)))
        out2 = self.auxiliary1(out)
        out = self.lrelu(self.in3(self.conv3(out)))
        out = self.lrelu(self.in4(self.conv4(out)))
        out4 = self.auxiliary1(out)
        out = self.lrelu(self.in5(self.conv5(out)))
        out = self.lrelu(self.in6(self.conv6(out)))
        out6 = self.auxiliary1(out)
        out = self.lrelu(self.in7(self.conv7(out)))

        out = self.in_pred(self.pred(out))


        return {
            "scale_1": out1,
            "scale_2": out2,
            "scale_4": out4,
            "scale_6": out6,
            "scale_7": out
        }



class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.blockT = nn.AvgPool2d(kernel_size=10, stride=1)
    def forward(self, x):
        out = self.blockT(x)
        return out

