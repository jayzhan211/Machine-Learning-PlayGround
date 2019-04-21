import torch
import torch.nn as nn
import torch.nn.functional as F


def _gather_feat(feat, index, mask=None):
    dim = feat.size(2)


class conv_bn_act(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch, stride=1, use_bn=True):
        super(conv_bn_act, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class fc_bn_act(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=True):
        super(fc_bn_act, self).__init__()

        self.fc = nn.Linear(in_ch, out_ch);
        self.bn = nn.BatchNorm1d(out_ch) if use_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.fc(x)))


def _nms(heat, kernel=1):
    pad = (kernel -1) // 2
    heat_maxpool =  F.max_pool2d(heat, kernel, stride=1, padding=pad)
    keep = (heat_maxpool == heat).float()
    return heat * keep

def _topk(scores, K=20):
    batch_size, _, height, width = scores.size()
    topk_scores, topk_indexs = torch.topk(scores.view(batch_size, -1), K)
    topk_classes = (topk_indexs / (height * width)).int()
    topk_indexs = topk_indexs % (height * width)
    topk_ys = (topk_indexs / width).int().float()
    topk_xs = (topk_indexs % width).int().float()
    return topk_scores, topk_indexs, topk_classes, topk_ys, topk_xs


def  _gather_feat(feat, idx, mask=None):
    # feat: size(1, h*w, c), idx: size(1, K)
    dim = feat.size(2)
    idx = idx.unsqueeze(2).expand(idx.size(0), idx.size(1), dim)
    feat = feat.gather(1, idx)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _decode(tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr,
            K=100, kernel=1, ae_threshold=1, num_dets=1000, no_border=False):

    # tl: top-left
    # br: bottom-right

    batch_size, _, height, width = tl_heat.size()

    tl_heat = torch.sigmoid(tl_heat)
    br_heat = torch.sigmoid(br_heat)
    tl_heat = _nms(tl_heat, kernel=kernel)
    br_heat = _nms(br_heat, kernel=kernel)

    # inds: indexs
    # clses: classes
    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)

    tl_ys = tl_ys.view(batch_size, K, 1).expand(batch_size, K, K)
    tl_xs = tl_xs.view(batch_size, K, 1).expand(batch_size, K, K)
    br_ys = br_ys.view(batch_size, 1, K).expand(batch_size, K, K)
    br_xs = br_xs.view(batch_size, 1, K).expand(batch_size, K, K)


    if no_border:
        tl_ys_binds = (tl_ys == 0)
        tl_xs_binds = (tl_xs == 0)
        br_ys_binds = (br_ys == height - 1)
        br_xs_binds = (br_xs == width - 1)

