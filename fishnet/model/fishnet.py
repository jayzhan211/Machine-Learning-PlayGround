import torch
from torch import nn




class Fish(nn.Module):
    def __init__(self, block, num_cls=1000, num_down_sample=5, num_up_sample=3, trans_map=(2, 1, 0, 6, 5, 4),
                 network_planes=None, num_res_blks=None, num_trans_blks=None):
        self.block = block
        self.trans_map = trans_map
        self.upsample = nn.Upsample(scale_factor=2)
        self.down_sample = nn.MaxPool2d(2, stride=2)
        self.num_cls = num_cls
        self.num_down = num_down_sample
        self.num_up = num_up_sample
        self.network_planes = network_planes[1:]
        self.depth = len(self.network_planes)
        self.num_trans_blks = num_trans_blks
        self.num_res_blks = num_res_blks
        self.fish = self._make_fish(network_planes[0])

    def _make_score(self, in_ch, out_ch=1000, has_pool=False):
        conv = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True)
        )
        if has_pool:
            fc = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_ch // 2, out_ch, kernel_size=1)
            )
            fc = nn.Conv2d(in_ch // 2, out_ch, kernel_size=1)

        return [conv, fc]

    def _make_residual_block(self, inplanes, outplanes, n_blocks, is_up=False, k=1, dilation=1):
        layers = []
        if is_up:
            layers.append(self.block(inplanes, outplanes, mode='UP', dilation=dilation, k=k))
        else:
            layers.append(self.block(inplanes, outplanes, stride=1))
        for i in range(1, n_blocks):
            layers.append(self.block(outplanes, outplanes, stride=1, dilation=dilation))
        return nn.Sequential(*layers)

    def _make_stage(self, is_down_sample, inplanes, outplanes, n_blk, has_trans=True,
                    has_score=False, trans_planes=0, no_sampling=False, num_trans=2, **kwargs):
        sample_block = []
        if has_score: # i == num.down
            sample_block.extend(self._make_score(outplanes, outplanes * 2, has_pool=False))

        if no_sampling or is_down_sample:
            res_block = self._make_residual_block(inplanes, outplanes, n_blk, **kwargs)
        else:
            res_block = self._make_residual_block(inplanes, outplanes, n_blk, is_up=True, **kwargs)

        sample_block.append(res_block)
        if has_trans:
            trans_in_planes = self.in_planes if trans_planes == 0 else trans_planes
            sample_block.append(self._make_residual_block(trans_in_planes, trans_in_planes, num_trans))

        if not no_sampling:
            if is_down_sample:
                sample_block.append(self.down_sample)
            else:
                sample_block.append(self.upsample)

        return nn.ModuleList(sample_block)


    def _make_fish(self, in_planes):
        def get_trans_planes(index):
            map_id = self.trans_map[index - self.num_down - 1] - 1
            p = in_planes if map_id == -1 else cated_planes[map_id]
            return p
        def get_trans_blk(index):
            return self.num_trans_blks[index - self.num_down - 1]
        def get_cur_planes(index):
            return self.network_planes[index]
        def get_blk_num(index):
            return self.num_res_blks[index]
        cated_planes, fish = [in_planes] * self.depth, []
        for i in range(self.depth):
            is_down, has_trans, no_sampling = i < self.num_down, i > self.num_down, i == self.num_down

            cur_planes, trans_planes, cur_blocks, num_trans = \
                get_cur_planes(i), get_trans_planes(i) , get_blk_num(i), get_trans_blk(i)

            stg_args = [is_down, cated_planes[i - 1], cur_planes, cur_blocks]

            if is_down or no_sampling:
                k, dilation = 1, 1
            else:
                k, dilation = cated_planes[i - 1] // cur_planes

            sample_block = self._make_stage(*stg_args, has_trans=has_trans, trans_planes=trans_planes,
                                            has_score=(i == self.num_down), num_trans=num_trans, k=k,
                                            dilation=dilation, no_sampling=no_sampling)

            if i + 1 == self.depth:
                sample_block.




    def _fish_forward(self, all_feat):
        stg_id = 0
        # tail
        while stg_id < self.depth:
            stg_blk = stage_factory(*self.fish[stg_id])

    def forward(self, x):
        all_feat = [None] * (self.depth + 1)
        all_feat[0] = x
        return self._fish_forward(all_feat)


class FishNet(nn.Module):
    def __init__(self, block, **kwargs):
        super(FishNet, self).__init__()

        inplanes = kwargs['network_planes'][0]
        # 224 x 224
        self.conv1 = self.conv_bn_relu(3, inplanes // 2)
        self.conv2 = self.conv_bn_relu(inplanes // 2, inplanes // 2)
        self.conv3 = self.conv_bn_relu(inplanes // 2, inplanes)

        self.pool1 = nn.MaxPool2d(3, padding=1, stride=2)
        self.fish = Fish(block, **kwargs)

    def conv_bn_relu(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)




