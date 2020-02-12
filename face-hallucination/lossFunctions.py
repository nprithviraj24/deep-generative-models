import torch


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

