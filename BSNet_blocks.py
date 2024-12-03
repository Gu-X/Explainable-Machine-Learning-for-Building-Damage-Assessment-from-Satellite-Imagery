import torch
import torch.nn as nn
import torch.nn.functional as F


# "Multi-Scale Adaptive Feature Fusion Module" (MSAFF Module)
class MSAFF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSAFF, self).__init__()
        #Fine-Scale Feature Extractor (FSFE)
        self.FSFE = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # Coarse-Scale Feature Extractor (CSFE)
        self.CSFE = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv_fusion = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]  # Get the spatial dimensions (H, W)

        # Apply the 1x1, 3x3 (with different dilations), and global average pooling
        x1 = self.FSFE(x)
        x2 = self.CSFE(x)
        x3 = self.global_avg_pool(x)
        x3 = F.interpolate(x3, size=size, mode='bilinear', align_corners=False)  # Upsample to match the input size

        # Concatenate along the channel dimension
        x = torch.cat([x1, x2, x3], dim=1)

        # Fuse the concatenated outputs
        return self.conv_fusion(x)


# "Multi-Scale Contextual Aggregation Downsampling Block" (MCAD Block)
class MCAD(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(MCAD, self).__init__()
        self.MSAFF = MSAFF(in_channels, out_channels)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        # ASPP + Batch Norm + Dropout
        skip_out = self.MSAFF(x)
        skip_out = self.batch_norm(skip_out)
        skip_out = self.dropout(skip_out)

        # Downsample with 1x1 convolution
        down_out = self.down_sample(skip_out)

        return down_out, skip_out
