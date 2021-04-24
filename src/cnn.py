import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import orthogonal_init

class MyUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool0 = nn.MaxPool2d(3, 2, padding=1)  # 256 -> 128
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(3, 2, padding=1)  # 128 -> 64
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(3, 2, padding=1)  # 64 -> 32
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(3, 2, padding=1)  # 32 -> 16

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(scale_factor=(2,2))  # 16 -> 32
        self.dec_conv0 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=(2,2))  # 32 -> 64
        self.dec_conv1 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=(2,2))  # 64 -> 128
        self.dec_conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=(2,2))  # 128 -> 256
        self.dec_conv3 = nn.Conv2d(128, 3, 3, padding=1)
        
        self.apply(orthogonal_init)
        
    def forward(self, x):
        # encoder
        e0 = self.pool0(F.relu(self.enc_conv0(x)))
        e1 = self.pool1(F.relu(self.enc_conv1(e0)))
        e2 = self.pool2(F.relu(self.enc_conv2(e1)))
        e3 = self.pool3(F.relu(self.enc_conv3(e2)))

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))
        

        # decoder
        skip0 = torch.cat([self.upsample0(b), F.relu(self.enc_conv3(e2))], 1)
        d0 = F.relu(self.dec_conv0(skip0))
        skip1 = torch.cat([self.upsample1(d0), F.relu(self.enc_conv2(e1))], 1)
        d1 = F.relu(self.dec_conv1(skip1))
        skip2 = torch.cat([self.upsample2(d1), F.relu(self.enc_conv1(e0))], 1)
        d2 = F.relu(self.dec_conv2(skip2))
        skip3 = torch.cat([self.upsample3(d2), F.relu(self.enc_conv0(x))], 1)
        d3 = self.dec_conv3(skip3)  # no activation
        return d3