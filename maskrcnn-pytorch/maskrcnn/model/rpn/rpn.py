import torch
import torch.nn.functional as F
from torch import nn
from .anchor_generator import make_anchor_generator

class RPNHead(nn.Module):
    '''rpn head with classifier and regression'''

    def __init__(self, cfg, in_channels, num_anchors):
        '''
              Arguments::
                    cfg              : config
                    in_channels (int): number of channels of the input feature
                    num_anchors (int): number of anchors to be predicted
              '''
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

        def forward(self, x):
            logits=[]
            bbox_reg=[]
            for feature in x:
                t = F.relu(self.conv(feature))
                logits.append(self.cls_logits(t))
                bbox_reg.append(self.bbox_pred(t))
            return logits, bbox_reg

class RPNModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        anchor_generator = make_anchor_generator(cfg)