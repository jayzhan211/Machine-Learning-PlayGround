import torch
import torch.nn as nn

from .utils import residual, upsample, merge, _decode


def _make_layer(input_dim, output_dim, modules):
    layers = [residual(input_dim, output_dim)]
    layers += [residual(output_dim, output_dim) for _ in range(1, modules)]
    return nn.Sequential(*layers)
def _make_layer_reverse(input_dim, output_dim, modules):
    layers = [residual(input_dim, input_dim) for _ in range(modules - 1)]
    layers += [residual(input_dim, output_dim)]
    return nn.Sequential(*layers)

def _make_pool_layer():
    return nn.MaxPool2d(kernel_size=2, stride=2)
def _make_unpool_layer():
    return upsample(scale_factor=2)
def _make_merge_layer():
    return merge()



class saccade_net(nn.Module):
    def __init__(
        self, hg, tl_modules, br_modules, tl_heats, br_heats,
        tl_tags, br_tags, tl_offs, br_offs, att_modules, up_start=0
    ):
        super(saccade_net, self).__init__()
        self._decode = _decode
        self.hg = hg
        self.tl_modules = tl_modules
        self.br_modules = br_modules
        self.tl_heats = tl_heats
        self.br_heats = br_heats
        self.tl_tags = tl_tags
        self.br_tags = br_tags
        self.tl_offs = tl_offs
        self.br_offs = br_offs
        self.att_modules = att_modules
        self.up_start = up_start

    def _train(self, *xs):
        image = xs[0]
        cnvs, ups = self.hg(image)
        ups = [up[self.up_start:] for up in ups]
        tl_modules = [tl_mod_(cnv) for tl_mod_, cnv in zip(self.tl_modules, cnvs)]
        br_modules = [br_mod_(cnv) for br_mod_, cnv in zip(self.br_modules, cnvs)]
        tl_heats = [tl_heat_(tl_mod) for tl_heat_, tl_mod in zip(self.tl_heats, tl_modules)]
        br_heats = [br_heat_(br_mod) for br_heat_, br_mod in zip(self.br_heats, br_modules)]
        tl_tags = [tl_tag_(tl_mod) for tl_tag_, tl_mod in zip(self.tl_tags, tl_modules)]
        br_tags = [br_tag_(br_mod) for br_tag_, br_mod in zip(self.br_tags, br_modules)]
        tl_offs = [tl_off_(tl_mod) for tl_off_, tl_mod in zip(self.tl_offs, tl_modules)]
        br_offs = [br_off_(br_mod) for br_off_, br_mod in zip(self.br_offs, br_modules)]
        atts = [[att_mod_(u) for att_mod_, u in zip(att_mods, up)] for att_mods, up in zip(self.att_modules, ups)]
        return [tl_heats, br_heats, tl_tags, br_tags, tl_offs, br_offs, atts]
