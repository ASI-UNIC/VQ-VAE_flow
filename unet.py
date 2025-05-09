""" Parts of the U-Net model """

import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
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
    """Upscaling then double conv (without residual connections)"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2=None):  # x2 is no longer used
        x1 = self.up(x1)
        return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, n_channels, num_embeddings, codebook_size, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, num_embeddings))
        self.down1 = (Down(num_embeddings, num_embeddings * 2))
        self.down2 = (Down(num_embeddings * 2, num_embeddings * 4))
        self.down3 = (Down(num_embeddings * 4, num_embeddings * 8))
        factor = 2 if bilinear else 1
        self.down4 = (Down(num_embeddings * 8, num_embeddings * 16 // factor))
        self.up1 = (Up(num_embeddings * 16, num_embeddings * 8 // factor, bilinear))
        self.up2 = (Up(num_embeddings * 8, num_embeddings * 4 // factor, bilinear))
        self.up3 = (Up(num_embeddings * 4, num_embeddings * 2 // factor, bilinear))
        self.up4 = (Up(num_embeddings * 2, num_embeddings, bilinear))
        self.outc = (OutConv(num_embeddings, n_classes))

        self.embedding_dim = num_embeddings * 16
        self.vq = VectorQuantize(dim=self.embedding_dim, codebook_size=codebook_size, decay=0.8, commitment_weight=1.)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        orig_shape = x5.shape
        x5 = x5.view(orig_shape[0], self.embedding_dim, -1)
        x5 = x5.permute(0, 2, 1)
        quantized, indices, commit_loss = self.vq(x5)
        quantized = quantized.permute(0, 2, 1)
        quantized = quantized.view(*orig_shape)

        # Remove skip connections (x4, x3, x2, x1 are no longer used in the upsampling path)
        x = self.up1(quantized)  # No x4 passed
        x = self.up2(x)          # No x3 passed
        x = self.up3(x)          # No x2 passed
        x = self.up4(x)          # No x1 passed
        logits = self.outc(x)
        return logits, commit_loss