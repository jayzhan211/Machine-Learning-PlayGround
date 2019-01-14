import torch

FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

class BoxList(object):
    def __init__(self, bbox, image_size, mode='xyxy'):
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if bbox.ndimension() != 2:
            raise ValueError(
                "bbox should have 2 dimensions, got {}".format(bbox.ndimension())
            )
        if bbox.size(-1) != 4:
            raise ValueError(
                "last dimenion of bbox should have a "
                "size of 4, got {}".format(bbox.size(-1))
            )
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")

        self.bbox = bbox
        self.size = image_size  # (image_width, image_height)
        self.mode = mode
        self.extra_fields = {}
    def add_field(self, field, field_data):
        self.extra_field[field]= field_data

    def get_field(self, field):
        return self.extra_field[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v