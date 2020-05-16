""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels // 2, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
        
class UNet(nn.Module):
    def __init__(self, n_channels = 3, out_channels_id = 9, out_channels_uv = 256, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.out_channels_id = out_channels_id
        self.out_channels_uv = out_channels_uv
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024//factor)


        #ID MASK
        self.up1_id = Up(1024, 512, bilinear)
        self.up2_id = Up(512, 256, bilinear)
        self.up3_id = Up(256, 128, bilinear)
        self.up4_id = Up(128, 64 * factor, bilinear)
        self.outc_id = OutConv(64, out_channels_id)

        #UV Mask
        self.up1_uv = Up(1024, 512, bilinear)
        self.up2_uv = Up(512,512,bilinear)
        self.outc_uv1 = OutConv(256, out_channels_uv)
        self.outc_uv2 = OutConv(256, out_channels_uv)
        self.outc_uv3 = OutConv(256, out_channels_uv)
        self.outc_uv4 = OutConv(256, out_channels_uv)
        self.up3_uv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4_uv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # ID mask
        x_id = self.up1_id(x5, x4)
        x_id = self.up2_id(x_id, x3)
        x_id = self.up3_id(x_id, x2)
        x_id = self.up4_id(x_id, x1)
        logits_id = self.outc_id(x_id)

        # U mask
        x_u = self.up1_uv(x5, x4)
        x_u = self.up2_uv(x_u,x3)
        x_u = self.outc_uv1(x_u)
        x_u = self.outc_uv2(x_u)
        x_u = self.outc_uv3(x_u)
        x_u = self.up3_uv(x_u)
        x_u = self.up4_uv(x_u)
        logits_u = self.outc_uv4(x_u)

        # V mask
        x_v = self.up1_uv(x5, x4)
        x_v = self.up2_uv(x_v,x3)
        x_v = self.outc_uv1(x_v)
        x_v = self.outc_uv2(x_v)
        x_v = self.outc_uv3(x_v)
        x_v = self.up3_uv(x_v)
        x_v = self.up4_uv(x_v)
        logits_v = self.outc_uv4(x_v)
        
        return logits_id,logits_u, logits_v