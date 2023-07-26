import numpy as np
import torch
import torch.nn as nn


class EncDec(nn.Module):

    def __init__(self):
        super(EncDec, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=3,
                      stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_features=16),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_features=32),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU()
        )

        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=64,
                      out_channels=32,
                      kernel_size=3,
                      stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_features=32),
        )
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2,  mode='bilinear'),
            nn.Conv2d(in_channels=32,
                      out_channels=16,
                      kernel_size=3,
                      stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_features=16),
        )
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2,  mode='bilinear'),
            nn.Conv2d(in_channels=16,
                      out_channels=1,
                      kernel_size=3,
                      stride=1, padding=1),
            nn.LeakyReLU())
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        dec1_out = self.dec1(enc3_out)
        dec2_out = self.dec2(dec1_out)
        return self.dec3(dec2_out)
