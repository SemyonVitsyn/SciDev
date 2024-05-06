import torch.nn as nn
import torch


class ResNetBottleNeckBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.layer_3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = out + residual
        return out


class ResUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.enc_conv1 = ResNetBottleNeckBlock(3, 32)
        self.enc_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_conv2 = ResNetBottleNeckBlock(32, 64)
        self.enc_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_conv3 = ResNetBottleNeckBlock(64, 128)
        self.enc_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_conv4 = ResNetBottleNeckBlock(128, 256)
        self.enc_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bridge = ResNetBottleNeckBlock(256, 512)
        
        self.dec_upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, kernel_size=1)
        )
        self.dec_conv1 = ResNetBottleNeckBlock(512, 256)

        self.dec_upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=1)
        )
        self.dec_conv2 = ResNetBottleNeckBlock(256, 128)

        self.dec_upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=1)
        )
        self.dec_conv3 = ResNetBottleNeckBlock(128, 64)

        self.dec_upsample4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=1)
        )
        self.dec_conv4 = ResNetBottleNeckBlock(64, 32)
        
        self.out = nn.Conv2d(32, 3, kernel_size=1)
        
    def forward(self, x):
        enc1 = self.enc_conv1(x)
        enc2 = self.enc_conv2(self.enc_pool1(enc1))
        enc3 = self.enc_conv3(self.enc_pool2(enc2))
        enc4 = self.enc_conv4(self.enc_pool3(enc3))
        
        bridge = self.bridge(self.enc_pool4(enc4))
        
        out = self.dec_upsample1(bridge)
        out = torch.cat([out, enc4], dim=1)
        out = self.dec_conv1(out)
        
        out = self.dec_upsample2(out)
        out = torch.cat([out, enc3], dim=1)
        out = self.dec_conv2(out)

        out = self.dec_upsample3(out)
        out = torch.cat([out, enc2], dim=1)
        out = self.dec_conv3(out)
        
        out = self.dec_upsample4(out)
        out = torch.cat([out, enc1], dim=1)
        out = self.dec_conv4(out)
        
        out = self.out(enc1)
        return out