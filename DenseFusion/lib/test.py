import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import lib.extractors as extractors

# camera data
cam_f = 28.6 # mm, camera focus length
cam_imgw = 640 # pixels
cam_imgh = 480 # pixels
cam_w = 32.  # mm,  sensor width
cam_h = cam_w * cam_imgh / cam_imgw
cam_cx = cam_w/2 - 0.009 # mm, camera center shift horizontal (+ or - ?)
cam_cy = cam_h/2 + 0.003 # mm, camera center shift vertical

print(cam_cx, cam_cy)

