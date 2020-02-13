import torch
from torch import nn

def realHingeLoss(x):
    if x == 1:
        return 0
    else:
        return x-1

def fakeHingeLoss(x):
    if x==0:
        return -1
    else:
        return -(x+1)

def pixelLoss(x, y):
    mse = nn.MSELoss()
    return mse(x,y)

disc_loss = nn.ReLU()(1.0 - discriminator(data)).mean() + nn.ReLU()(1.0 + discriminator(generator(z))).mean()