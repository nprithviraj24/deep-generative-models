import math
import torch
from torch import nn
from specNorm.spectral_normalization import SpectralNorm

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.block0 = DResBlock(64)
        self.block1 = DResBlock(64)
        self.block2 = DResBlock(64)
        self.block3 = DResBlock(64)

        self.block4 = DResBlock(64)
        self.block5 = DResBlock(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = SpectralNorm(nn.Linear(16, 10))

    def forward(self, x):
        
        b0 = self.block0(x)
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        
        b3 = self.block3(b2)
        b3 = self.pool(b3)

        b4 = self.block4(b3)
        b4 = self.pool(b4)

        out = self.block5(b4)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return torch.sigmoid(out)


class DResBlock(nn.Module):
    def __init__(self, channels):
        super(DResBlock, self).__init__()
        self.conv1 = SpectralNorm(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        self.relu = nn.ReLU()
        self.conv2 = SpectralNorm(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        
    def forward(self, x):
        residual = self.relu(x)
        residual = self.conv1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        return x + residual
