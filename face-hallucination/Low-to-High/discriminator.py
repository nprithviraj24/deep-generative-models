import math
import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, channels=64):
        super(Discriminator, self).__init__()

        self.conv = nn.Conv2d(3, channels, kernel_size=3, padding=1  )
        self.conv2 = nn.Conv2d(channels, 1, kernel_size=3, padding=1  )

        self.block0 = DResBlock(channels)
        self.block1 = DResBlock(channels)
        self.block2 = DResBlock(channels)
        self.block3 = DResBlock(channels)

        self.block4 = DResBlock(channels)
        self.block5 = DResBlock(channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv(x)
        b0 = self.block0(x)
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        b2 = self.pool(b2)

        b3 = self.block3(b2)
        b3 = self.pool(b3)

        b4 = self.block4(b3)
        b4 = self.pool(b4)

        out = self.block5(b4)
        out = self.pool(out)
        out = self.conv2(out)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fc(out)
        
        return torch.sigmoid(out)


class DResBlock(nn.Module):
    def __init__(self, channels):
        super(DResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        residual = self.relu(x)
        residual = self.conv1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        return x + residual
# cuda = torch.device("cuda:0")
# gen = Discriminator().to(cuda)
# # print(gen.zeroPhase.weight)
# v = torch.ones([8,3,64,64], dtype=torch.float, device=cuda)
# b = gen(v)
# print(b.size())