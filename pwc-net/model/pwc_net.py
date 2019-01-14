import torch
from torch import nn
import torch.nn.functional as F
from correlation import correlation


Grid = {}
Partial = {}
def Warp(image, flow):
    """
        :param image:  [B,3,H,W]
        :param flow: [B,2,H,W]
        :return: image with warped
        """
    B, C, H, W = image.size()

    # identity_grid 0~H-1 normal to [-1,1]
    # grid_sample(input, identity_grid) = input

    if str(flow.size()) not in Grid:
        Horizontal = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
        Vertical = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
        Grid[str(flow.size())] = torch.cat([Horizontal, Vertical], dim=1).cuda()

    # Partial is to check whether smaple is in [-1, 1]
    # if grid_sample(1, identity_grid) == 1 :  it is in [1,1] otherwise will < 1.0
    if str(flow.size()) not in Partial:
        Partial[str(flow.size())] = flow.new_ones(B, 1, H, W)

    flow = torch.cat([flow[:,0,:,:] / (W - 1.0) / 2.0, flow[:,1,:,:] / (H - 1.0) / 2.0], dim=1)
    image = torch.cat([image, Partial(str(flow.size()))], dim=1)

    output = F.grid_sample(input=image, grid=(Grid[str(flow.size())] + flow).permute(0, 2, 3, 1), mode=' bilinear', padding_mode='zeros')
    mask = output[:,-1,:,:]
    mask[mask < 0.9999] = 0.0
    mask[mask > 0.0] = 1.0

    return output[:,:-1,:,:] * mask

# end



######################################################

class PWC_Net(nn.Module):
    def __init__(self):
        super().__init__()

        class Extractor(nn.Module):
            def __init__(self):
                super().__init__()

                self.level_1 = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(negative_slope=0.1),
                    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=0.1),
                    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=0.1)
                )
                self.level_2 = nn.Sequential(
                    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(negative_slope=0.1),
                    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=0.1),
                    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=0.1)
                )
                self.level_3 = nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(negative_slope=0.1),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=0.1),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=0.1)
                )
                self.level_4 = nn.Sequential(
                    nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(negative_slope=0.1),
                    nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=0.1),
                    nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=0.1)
                )
                self.level_5 = nn.Sequential(
                    nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(negative_slope=0.1),
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=0.1),
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=0.1)
                )
                self.level_6 = nn.Sequential(
                    nn.Conv2d(128, 196, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(negative_slope=0.1),
                    nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=0.1),
                    nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=0.1)
                )

            # end

            def forward(self, x):
                out_1 = self.level_1(x)
                out_2 = self.level_2(out_1)
                out_3 = self.level_3(out_2)
                out_4 = self.level_4(out_3)
                out_5 = self.level_5(out_4)
                out_6 = self.level_6(out_5)

                return [out_1, out_2, out_3, out_4, out_5, out_6]

            # end
        # Extractor end

        class Decoder(nn.Module):
            def __init__(self, intLevel):
                super().__init__()

                '''
                                md = 4 , maximum_displacementt
                                nd = (md * 2 + 1) ** 2 (=81)
                                
                                '''
                intPrevious = \
                [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][
                    intLevel + 1]
                intCurrent = \
                [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][
                    intLevel + 0]

                if intLevel < 6:
                    # Upsampled flow
                    self.Upflow = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)
                    self.Upfeature = nn.ConvTranspose2d(intPrevious + 128 + 128 + 96 + 64 + 32, 2, kernel_size=4, stride=2, padding=1)
                    self.scales = [None, None, None, 5.0, 2.5, 1.25, 0.625, None][intLevel + 1]

                self.conv1 = nn.Sequential(
                    nn.Conv2d(intCurrent, 128, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=0.1)
                )
                self.conv2 = nn.Sequential(
                    nn.Conv2d(intCurrent + 128, 128, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=0.1)
                )
                self.conv3 = nn.Sequential(
                    nn.Conv2d(intCurrent + 128 + 128, 96, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=0.1)
                )
                self.conv4 = nn.Sequential(
                    nn.Conv2d(intCurrent + 128 + 128 + 96, 64, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=0.1)
                )
                self.conv5 = nn.Sequential(
                    nn.Conv2d(intCurrent + 128 + 128 + 96 + 64, 32, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=0.1)
                )
                self.conv6 = nn.Sequential(
                    nn.Conv2d(intCurrent + 128 + 128 + 96 + 64 + 32, 2, kernel_size=3, stride=1, padding=1),
                )
            # end

            def forward(self, img1, img2, objectPrevious):
                Flow = None
                Features = None

                if objectPrevious is None:
                    Cost_Volume = F.leaky_relu(
                        input=correlation.FunctionCorrelation(
                            tensorFirst=img1,
                            tensorSecond=img2
                        ),
                        negative_slope=.1
                    )
                    Features = Cost_Volume

                else:
                    Flow = self.Upflow(objectPrevious['flow'])
                    Features = self.Upfeature(objectPrevious['feature'])
                    Cost_Volume = F.leaky_relu(
                        input=correlation.FunctionCorrelation(
                            tensorFirst=img1,
                            tensorSecond=Warp(img2, Flow * self.scales)
                        ),
                        negative_slope=.1
                    )
                    Features = torch.cat([Cost_Volume, img1, Flow, Features], dim=1)

                # end

                Features = torch.cat([self.conv1(Features), Features], dim=1)
                Features = torch.cat([self.conv2(Features), Features], dim=1)
                Features = torch.cat([self.conv3(Features), Features], dim=1)
                Features = torch.cat([self.conv4(Features), Features], dim=1)
                Features = torch.cat([self.conv5(Features), Features], dim=1)

                Flow = self.conv6(Features)
                return {
                    'flow':Flow,
                    'feature':Features
                }
            # end
        # end

        class Context_Net(nn.Module):
            def __init__(self):
                super().__init__()

                self.convs = nn.Sequential(
                    nn.Conv2d(81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, 128, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=.1),
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),
                    nn.LeakyReLU(negative_slope=.1),
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=4, dilation=4),
                    nn.LeakyReLU(negative_slope=.1),
                    nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=8, dilation=8),
                    nn.LeakyReLU(negative_slope=.1),
                    nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=16, dilation=16),
                    nn.LeakyReLU(negative_slope=.1),
                    nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=.1),
                    nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)

                )
            def forward(self, x):
                out = self.convs(x)
                return out

        # end

        self.extractor = Extractor()
        self.flow2 = Decoder(2)
        self.flow3 = Decoder(3)
        self.flow4 = Decoder(4)
        self.flow5 = Decoder(6)
        self.flow6 = Decoder(6)
        self.context = Context_Net()

    def forward(self, img1, img2):
        img1 = self.extractor(img1)
        img2 = self.extractor(img2)

        flow_estimate = self.flow6(img1[-1], img2[-1], None)
        flow_estimate = self.flow5(img1[-2], img2[-2], flow_estimate)
        flow_estimate = self.flow4(img1[-3], img2[-3], flow_estimate)
        flow_estimate = self.flow3(img1[-4], img2[-4], flow_estimate)
        flow_estimate = self.flow2(img1[-5], img2[-5], flow_estimate)

        return  flow_estimate['flow'] + self.context(flow_estimate['feature'])

    # end
# end

model = PWC_Net()















# Reference
# https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/models/PWCNet.py
# https://github.com/sniklaus/pytorch-pwc/blob/master/run.py
