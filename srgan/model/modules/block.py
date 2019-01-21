import torch
from torch import nn
from collections import OrderedDict


# To select which activation to use
def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace=inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(negative_slope=neg_slope, inplace=inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{}] is not found'.format(act_type))
    return layer

# Select Normaliztion
def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batchnorm':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instancenorm':
        layer = nn.InstanceNorm2d(nc, affine=True)
    else:
        raise NotImplementedError('normalization layer [{}] is not found'.format(norm_type))
    return layer

def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if pad_type == 'None': # for Zero Padding , do it in conv layer
        return None
    if pad_type == 'reflectionpad':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicationpad':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{}] is not found'.format(pad_type))
    return layer

def get_same_padding(kernel_size, dilation):
    # N + 2p - (k-1) * d = N
    return (kernel_size - 1) * dilation // 2

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('seqential doesnt support OrderedDict input')
        return args[0]
    modules = []
    for model in args:
        if isinstance(model, nn.Sequential):
            for submodel in model:
                modules.append(submodel)
        elif isinstance(model, nn.Module):
            modules.append(model)
    return nn.Sequential(*modules)





def conv_block(in_nc, out_nc, kernel_size, stride=1, pad_type=None, dilation=1, groups=1, bias=True,
               norm_type=None, act_type=None, mode='CNA'):

    '''
        mode:
         CNA  conv->norm->act
         NAC norm->act->conv
        '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'error conv mode [{}] '.format(mode)
    padding = get_same_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type else None
    if p is not None: padding = 0
    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode: # CNA or CNAC
        n = norm(norm_type, nc=out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        n = norm(norm_type, nc=in_nc) if norm_type else None
        # if get to relu before input it will modify input
        if n is None and act_type is not None:
            a = act(act_type, inplace=False)
        return sequential(n, a, p, c)

class ResNet_Block(nn.Module):
    '''
        resnet 3-3 style in EDSR
        '''
    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, stride=1, dilation=1, groups=1, bias=True,
                 pad_type=None, norm_type=None, act_type='relu', mode='CNA', res_scale=1):
        super().__init__()
        conv0 = conv_block(in_nc, mid_nc, kernel_size=kernel_size, stride=stride, dilation=dilation,
                           groups=groups, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        if mode == 'CNA': # CNACN  SRResNet
            act_type = None
        elif mode == 'CNAC': # CAC  EDSR
            act_type = None
            norm_type = None
        conv1 = conv_block(mid_nc, out_nc, kernel_size=kernel_size, stride=stride, dilation=dilation,
                           groups=groups, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale
    def forward(self, x):
        residual = x
        return  residual + self.res(x) * self.res_scale

class ResidualDenseBlock(nn.Module):
    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type=None,
                 norm_type=None, act_type='leakyrelu', mode='CNA'):
        super().__init__()
        self.conv1 = conv_block(nc, gc, kernel_size=kernel_size, stride=stride, bias=bias, pad_type=pad_type,
                                norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = conv_block(nc + gc, gc, kernel_size=kernel_size, stride=stride, bias=bias, pad_type=pad_type,
                                norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = conv_block(nc + gc * 2, gc, kernel_size=kernel_size, stride=stride, bias=bias, pad_type=pad_type,
                                norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv4 = conv_block(nc + gc * 3, gc, kernel_size=kernel_size, stride=stride, bias=bias, pad_type=pad_type,
                                norm_type=norm_type, act_type=act_type, mode=mode)

        if mode == 'CNA':
            act_type = None

        self.conv5 = conv_block(nc + gc * 4, nc, kernel_size=3, stride=stride, bias=bias, pad_type=pad_type,
                                norm_type=norm_type, act_type=act_type, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat([x, x1], dim=1))
        x3 = self.conv3(torch.cat([x, x1, x2], dim=1))
        x4 = self.conv4(torch.cat([x, x1, x2, x3], dim=1))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    # (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type=None,
                 norm_type=None, act_type='leakyrelu', mode='CNA'):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nc, kernel_size=kernel_size, gc=gc, stride=stride, bias=bias,
                                       pad_type=pad_type,norm_type=norm_type, act_type=act_type, mode=mode)
        self.rdb2 = ResidualDenseBlock(nc, kernel_size=kernel_size, gc=gc, stride=stride, bias=bias,
                                       pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.rdb3 = ResidualDenseBlock(nc, kernel_size=kernel_size, gc=gc, stride=stride, bias=bias,
                                       pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
    def forward(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return x + out * 0.2

def pixelshuffle_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True,
                       pad_type=None, norm_type=None, act_type='relu'):
    conv = conv_block(in_nc, out_nc * (upscale_factor ** 2), kernel_size, stride, pad_type=pad_type,
                      norm_type=None, act_type=None, bias=bias)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)

def upconv_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True,
                       pad_type=None, norm_type=None, act_type='relu', mode='nearest'):
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias,
                      pad_type=pad_type, norm_type=norm_type, act_type=act_type)

    return sequential(upsample, conv)
