import torch
import os
from torch import nn
import argparse

parser = argparse.ArgumentParser(description='NYUDepth')
parser.add_argument('-b', '--batch_size', default=6, type=int, help='mini-batch size (default: 4)')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run (default: 200)')
parser.add_argument('-lr', default=0.0001, type=float, metavar='learning_rate', help='initial learning rate (default 0.0001)')


args = parser.parse_args()
print(args)




def NYUDepth_loader(data_path, batch_size=32, isTrain=True):
    if isTrain:
        train_dir = os.path.join(data_path, 'train')
        print(train_dir)
        if not os.path.exists(train_dir):
            train_dataset = 



def main():
    global args, best_result, output_directory

    n_gpu = torch.cuda.device_count()

    if n_gpu > 1:
        args.batch_size = args.batch_size * n_gpu
        train_loader = NYUDepth_loader(args.data_path, batch_size=args.batch_size, isTrain=True)



