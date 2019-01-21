import os
import os.path
import numpy as np
import torch.utils.data as data
import h5py

IMG_EXTENSIONS = ['.h5', ]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)










class MyDataloader(data.Dataset):
    modality_names = ['rgb', 'rgbd', 'd']
    color_jitter = trans