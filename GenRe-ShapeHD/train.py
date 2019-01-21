import argparse
import random
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=int, default=0,
                        help='resume training by loading checkpoint.pt or best.pt. Use 0 for training from scratch, -1 for last and -2 for previous best. Use positive number for a specific epoch. \
                            Most options will be overwritten to resume training with exactly same environment')

parser.add_argument('--n_epoch', type=int, default=500, help='number of epochs to train')
# Dataset
parser.add_argument('--dataset', type=str, default=None, help='dataset to use')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--classes', type=str, default='chair', help='class to use')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')

opt = parser.parse_args()

def main():
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

