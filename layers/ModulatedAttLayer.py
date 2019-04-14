import torch
from torch import nn
from torch.nn import functional as F


class ModulatedAttLayer(nn.Module):
    def __init__(self, in_channels, reduction=2, mode='embedded_gaussian'):
        super(ModulatedAttLayer, self).__init__()
        self.mode = mode
        self.reduction = reduction
        self.in_channels = in_channels
        self.inter_channels = in_channels // reduction
        assert self.mode in ['embedded_gaussian']

        self.g = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)
        self.theta = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)
        self.W = nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1, bias=False)

        self.fc_spatial = nn.Linear(7 * 7 * self.in_channels, 7 * 7)
        self.init_weight()
    def init_weight(self):
        for m in [self.g, self.theta, self.phi]:
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.W.weight, 0)
    def embedded_gaussian(self, x):
        batch_size, _, height, width = x.size()
        ### Non-local block
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        # ( b, hw, inter_c ) x ( b, inter_c, hw) = (b, hw, hw)
        map_t_p = torch.matmul(theta_x, phi_x)
        mask_t_p = F.softmax(map_t_p, dim=-1)
        # (b, hw, hw) x (b, hw, inter_c) = (b, hw, inter_c)
        map_ = torch.matmul(mask_t_p, g_x)
        map_ = map_.permute(0, 2, 1).contiguous()
        map_ = map_.view(batch_size, self.inter_channels, height, width)
        mask = self.conv_mask(map_)
        ###  Non-local block
        ###  Spatial Attention
        x_flatten = x.view(-1, 7 * 7 * self.in_channels)
        spatial_att = self.fc_spatial(x_flatten)
        spatial_att.softmax(dim=1)
        spatial_att = spatial_att.view(-1, 7, 7).unsqueeze(1)
        spatial_att = spatial_att.expand(-1, self.in_channels, -1, -1)
        ###  Spatial Attention
        f_att = x + spatial_att * mask
        return f_att, [x, spatial_att, mask]

    def forward(self, x):
        if self.mode == 'embedded_gaussian':
            output, feature_maps = self.embedded_gaussian(x)
        else:
            raise NotImplemented("The code has not been implemented.")
        return output, feature_maps



