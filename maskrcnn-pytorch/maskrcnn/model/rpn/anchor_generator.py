import math
import numpy as np
import torch
from torch import nn

class BufferList(nn.Module):
    def __init__(self, buffers=None):
        super().__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return  self

    def __len__(self):
        return len(self._buffers)
    def __iter__(self):
        return iter(self._buffers.values())

class AnchorGenerator(nn.Module):
    def __init__(self,
                 sizes=(128,256,512),
                 aspect_ratios=(0.5,1.0,2.0),
                 anchor_strides=(8,16,32),
                 straddle_thresh=0):
        super().__init__()

        if len(anchor_strides) == 1:
            anchor_stride = anchor_strides[0]
            cell_anchor = [
                generate_anchors(anchor_stride, sizes, aspect_ratios).float()
            ]
        else:
            if len(anchor_strides) != len(sizes):
                raise RuntimeError("FPN should have #anchor_strides == #sizes")
            cell_anchors = [
                generate_anchors(anchor_stride, (size,), aspect_ratios).float()
                for anchor_stride, size in zip(anchor_strides, sizes)
            ]

        self.strides = anchor_strides
        self.cell_anchors = BufferList(cell_anchors)
        self.straddle_thresh = straddle_thresh

    def num_anchors_per_location(self):
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self.strides, self.cell_anchors):
            grid_height, grid_width = size
            device = base_anchors.device
            shifts_x = torch.arange(
                0, grid_width * stride, step=stride, dtype=torch.float32, device=device
            )
            shifts_y = torch.arange(
                0, grid_height * stride, step=stride, dtype=torch.float32, device=device
            )
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors

    def add_visibility_to(self, boxlist):

    def forward(self, image_list, feature_maps):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
        anchors = []
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                boxlis = BoxList(
                    anchors_per_feature_map, (image_width, image_height), mode="xyxy"
                )


def generate_anchors(anchor_stride=16,
                     sizes=(32, 64, 128, 256, 512),
                     aspect_ratios=(0.5, 1.0, 2.0)):
    return _generate_anchors(anchor_stride,
                            np.array(sizes, dtype=np.float) / anchor_stride,
                            np.array(aspect_ratios, dtype=np.float))
def _generate_anchors(base_size, scales, aspect_ratios):
    anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
    anchors = _ratio_enum(anchor, aspect_ratios)
    anchors = np.vstack(
        [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
    )
    return torch.from_numpy(anchors)



def _whctrs(anchor):
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + (w - 1) * 0.5
    y_ctr = anchor[1] + (h - 1) * 0.5

    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1)
        )
    )
    return anchors

def _ratio_enum(anchor, ratios):
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    # anchor area = w * h and w / h = ratios

    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchor = _mkanchors(ws, hs, x_ctr, y_ctr)
    return  anchor

def _scale_enum(anchor, scales):
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors