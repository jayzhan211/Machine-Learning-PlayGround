import torch
from torch import nn
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label=1.0, fake_label=0.0):
        super().__init__()
        self.gan_type = gan_type.lower()
        self.real_label = real_label
        self.fake_label = fake_label
        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':
            def wgan_loss(input, target):
                return -1 * input.mean() if target else input.mean()
            self.loss = wgan_loss
        else:
            raise NotImplementedError('gan_type:[{}] not found'.format(gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label)
        else:
            return torch.empty_like(input).fill_(self.fake_label)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(target_is_real)
        loss = self.loss(input, target_label)
        return loss

class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super().__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interpolates, interpolates_D):
        grad_outputs = self.get_grad_outputs(interpolates_D)
        grad_interp = torch.autograd.grad(outputs=interpolates_D, inputs=interpolates,
                                          grad_outputs=grad_outputs, create_graph=True, retain_graph=True,
                                          only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)
        loss = ((grad_interp_norm - 1.0) ** 2).mean()
        return loss


# https://github.com/xinntao/BasicSR/tree/master/codes/models/modules



