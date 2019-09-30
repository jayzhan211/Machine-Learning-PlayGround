import functools
import torch.nn as nn
###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """
    :param norm_type: instance | batchnorm | none
    :return:

    for batchnorm, we use learnable affine parameters
    for instancenorm, we dont use learnalbe affine paramters
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [{}] is not found'.format(norm_type))
    return norm_layer


class ResnetBlock(object):
    pass


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc,
                 output_nc,
                 ngf=64,
                 norm_layer=nn.BatchNorm2d,
                 use_dropout=False,
                 n_blocks=6,
                 padding_type='reflect'):
        """

        :param input_nc:
        :param output_nc:
        :param ngf:
        :param norm_layer:
        :param use_dropout:
        :param n_blocks: number of ResNet block
        :param padding_type: padding in conv: reflect | replicate | zero
        """
        super(ResnetGenerator, self).__init__()
        """
        Question: Why add bias if norm=instance ?
        """
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            # mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, ngf * mult / 2,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(ngf * mult / 2),
                      nn.ReLU(True)]
            mult >>= 1  # mult / 2
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

def deinf_G(input_nc,
            output_nc,
            ngf,
            net_G,
            norm='batch',
            use_dropout=False,
            init_type='normal',
            init_gain=0.02,
            gpu_ids=[]):
    """

    :param input_nc: number of channels in input_images
    :param output_nc: ouput_images
    :param ngf: number of filters in last conv_layer
    :param net_G: resnet_6 | resnet_9
    :param norm: batch_norm | instance_norm | none
    :param use_dropout:
    :param init_type: initialize method
    :param init_gain: scaling factor for normal, xavier, orthogonal
    :param gpu_ids: e.g., 0,1,2
    :return:

    use ReLU for non-linearity
    """
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    else:
        raise NotImplementedError('Generator model name [{}] is not recognized'.format(netG))
    return init_net(net, init_type, init_gain, gpu_ids)
    
