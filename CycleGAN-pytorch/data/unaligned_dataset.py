from data.image_folder import make_dataset
from .base_dataset import BaseDataset
import os


class Unaligned_dataset(BaseDataset):
    def __init__(self, opt):
        super(Unaligned_dataset, self).__init__()
        self.dir_A = os.path.join(opt.data_root, opt.phase + 'A')
        self.dir_B = os.path.join(opt.data_root, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A, opt.max_dataset_size)
        self.B_paths = make_dataset(self.dir_B, opt.max_dataset_size)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        b2a = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if b2a else self.opt.input_nc
        output_nc = self.opt.input_nc if b2a else self.opt.output_nc
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))


    def __len__(self):
        pass
