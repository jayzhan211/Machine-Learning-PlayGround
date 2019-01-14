import torch.nn.functional as F
import time
import numpy as np
import torch

dtype = torch.FloatTensor
dtype_long = torch.LongTensor
def bilinear_interpolate_torch(im, x, y):
    x0 = torch.floor(x).type(dtype_long)
    x1 = x0 + 1

    y0 = torch.floor(y).type(dtype_long)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0][0]
    Ib = im[y1, x0][0]
    Ic = im[y0, x1][0]
    Id = im[y1, x1][0]

    print(x1.type(dtype) - x, x - x0.type(dtype))
    print(y1.type(dtype) - y, y - y0.type(dtype))

    wa = (x1.type(dtype) - x) * (y1.type(dtype) - y)
    wb = (x1.type(dtype) - x) * (y - y0.type(dtype))
    wc = (x - x0.type(dtype)) * (y1.type(dtype) - y)
    wd = (x - x0.type(dtype)) * (y - y0.type(dtype))

    return torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(
        torch.t(Id) * wd)

def bilinear_interpolate_torch_gridsample(image, samples_x, samples_y):


    # input image is: W x H x C
    image = image.permute(2, 0, 1)  # change to:      C x W x H
    image = image.unsqueeze(0)  # change to:  1 x C x W x H
    samples_x = samples_x.unsqueeze(2)
    samples_x = samples_x.unsqueeze(3)
    samples_y = samples_y.unsqueeze(2)
    samples_y = samples_y.unsqueeze(3)

    samples_x = torch.randn(1, 1, 5, 5)
    samples_y = torch.randn(1, 1, 5, 5)

    samples = torch.cat([samples_x, samples_y], 1).view(1,5,5,2)
    #samples[:, :, :, 0] = (samples[:, :, :, 0] / (W - 1))  # normalize to between  0 and 1
    #samples[:, :, :, 1] = (samples[:, :, :, 1] / (H - 1))  # normalize to between  0 and 1
    #samples = samples * 2 - 1  # normalize to between -1 and 1
    flow = torch.randn(1,5,5,2)
    samples += flow
    samples += 5
    #samples = torch.FloatTensor([0.6, 0.7]).view(1, 1, 1, 2)
    print(samples)
    image = torch.ones(image.size())

    return torch.nn.functional.grid_sample(image, samples)

B, W, H, C = 1, 5, 5, 1
test_image = torch.ones(W,H,C).type(dtype)
test_image[3,3,:] = 4
test_image[3,4,:] = 3

test_samples_x = torch.FloatTensor([[3.2]]).type(dtype)
test_samples_y = torch.FloatTensor([[3.4]]).type(dtype)


print(bilinear_interpolate_torch(test_image, test_samples_x, test_samples_y))
print(bilinear_interpolate_torch_gridsample(test_image, test_samples_x, test_samples_y))


def warp(image, samples_x, samples_y):
    image = image.permute(2, 0, 1)  # change to:      C x W x H
    image = image.unsqueeze(0)  # change to:  1 x C x W x H
    samples_x = torch.arange(0, W).view(1,-1).repeat(H,1)

    samples_y = torch.arange(0, H).view(-1, 1).repeat(1, W)
    samples_x = samples_x.view(1, 1, H, W).repeat(B, 1, 1, 1)
    samples_y = samples_y.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((samples_x, samples_y), 1).float()
    vgrid = grid
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    #print(vgrid)
    return F.grid_sample(image, vgrid)
print(warp(test_image, test_samples_x, test_samples_y))