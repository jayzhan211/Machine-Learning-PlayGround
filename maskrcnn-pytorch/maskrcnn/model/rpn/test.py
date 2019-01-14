import numpy as np
import torch

a = np.random.randn(10,2)
b= [k[-2:] for k in a]
print(b)