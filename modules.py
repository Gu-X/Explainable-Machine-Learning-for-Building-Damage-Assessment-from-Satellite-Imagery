import torch
import torch.nn as nn
from BSNet_blocks import MCAD
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample_mode):
        super(UpBlock, self).__init__()
        if up_sample_mode == 'conv_transpose':
            self.up_sample = nn.ConvTranspose2d(in_channels - out_channels, in_channels - out_channels, kernel_size=2,
                                                stride=2)
        elif up_sample_mode == 'bilinear':
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)")
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)

# Learnable Weighted Aggregation (LWA)
class LWA(nn.Module):
    def __init__(self, num_features):
        super(LWA, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_features) / num_features)  # Initialize to equal weights

        # Create a resize layer to transform 64x256x256 to 512x16x16
        self.resize_layer_64  = nn.Conv2d(64, 512, kernel_size=4, stride=16)  # Adjust kernel and stride
        self.resize_layer_128 = nn.Conv2d(128, 512, kernel_size=4, stride=8)  # Resize 128x128x128 to 512x16x16
        self.resize_layer_256 = nn.Conv2d(256, 512, kernel_size=4, stride=4)  # Resize 128x128x128 to 512x16x16
        self.resize_layer_512 = nn.Conv2d(512, 512, kernel_size=2, stride=2)  # Resize 128x128x128 to 512x16x16

        self.double_conv = DoubleConv(512, 1024)

    def forward(self, features, x):
        # Apply softmax to the weights to ensure they sum to 1
        weights = torch.softmax(self.weights, dim=0)

        # Resize the first feature map
        resized_features = []
        resized_features.append(self.resize_layer_64(features[0]))  # Resize according to the feature index
        resized_features.append(self.resize_layer_128(features[1]))  # Resize according to the feature index
        resized_features.append(self.resize_layer_256(features[2]))  # Resize according to the feature index
        resized_features.append(self.resize_layer_512(features[3]))  # Resize according to the feature index


        # Weighted sum of features
        weighted_features = sum(w * f for w, f in zip(weights, resized_features))

        x = self.double_conv(weighted_features*x)

        features[0] = features[0] * self.weights[0]
        features[1] = features[1] * self.weights[1]
        features[2] = features[2] * self.weights[2]
        features[3] = features[3] * self.weights[3]
        return x, features



class EntropyBasedAdaptiveLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EntropyBasedAdaptiveLayer, self).__init__()

        self.low_entropy_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.medium_entropy_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.high_entropy_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=9, padding=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=9, padding=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Weight parameters for each entropy level
        self.low_weight = nn.Parameter(torch.ones(1))
        self.medium_weight = nn.Parameter(torch.ones(1))
        self.high_weight = nn.Parameter(torch.ones(1))


    def calculate_entropy(self, x):
        """Estimate entropy over local patches."""
        # Calculate pixel-level probabilities by normalizing
        prob_map = F.softmax(x, dim=1)
        entropy_map = -prob_map * torch.log(prob_map + 1e-6)
        entropy_map = entropy_map.sum(dim=1, keepdim=True)
        return entropy_map

    def forward(self, x):
        entropy_map = self.calculate_entropy(x)
        # Define thresholds for low, medium, and high entropy
        # Calculate mean and standard deviation of entropy
        mean_entropy = entropy_map.mean()
        std_entropy = entropy_map.std()

        # Define thresholds dynamically
        low_threshold = mean_entropy - std_entropy*2
        high_threshold = mean_entropy + std_entropy*2

        # Masks based on entropy
        low_entropy_mask = (entropy_map < low_threshold).float()
        medium_entropy_mask = ((entropy_map >= low_threshold) & (entropy_map < high_threshold)).float()
        high_entropy_mask = (entropy_map >= high_threshold).float()

        # Apply different convolutions based on entropy levels
        low_features = self.low_entropy_conv(x)*low_entropy_mask
        medium_features = self.medium_entropy_conv(x)*medium_entropy_mask
        high_features = self.high_entropy_conv(x)*high_entropy_mask

        # Combine features with learnable weights
        out = (self.low_weight * low_features +  self.medium_weight * medium_features + self.high_weight * high_features)

        return out



class BSNet(nn.Module):
    def __init__(self, out_classes=1, up_sample_mode='conv_transpose'):
        super(BSNet, self).__init__()
        self.up_sample_mode = up_sample_mode

        # Down-sampling Path
        self.MCAD_1 = MCAD(3, 64)
        self.MCAD_2 = MCAD(64, 128)
        self.MCAD_3 = MCAD(128, 256)
        self.MCAD_4 = MCAD(256, 512)

        # EAL (Entropy Adaptive Layer)
        self.EAL1 = EntropyBasedAdaptiveLayer(64, 64)
        self.EAL2 = EntropyBasedAdaptiveLayer(128,128)
        self.EAL3 = EntropyBasedAdaptiveLayer(256,256)
        self.EAL4 = EntropyBasedAdaptiveLayer(512, 512)

        # Up-sampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode)
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode)
        self.up_conv2 = UpBlock(128 + 256, 128, self.up_sample_mode)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode)

        # Final Convolution
        self.conv_last = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1),
                                       nn.Conv2d(128, out_classes, kernel_size=1))

        # Initialize the Learnable Weighted Aggregation (LWA)
        self.LWA = LWA(num_features=4)  # 4 features from skip connections


    def forward(self, x):
        x, skip1_out = self.MCAD_1(x)
        x = self.EAL1(x)
        x, skip2_out = self.MCAD_2(x)
        x = self.EAL2(x)
        x, skip3_out = self.MCAD_3(x)
        x = self.EAL3(x)
        x, skip4_out = self.MCAD_4(x)
        x = self.EAL4(x)

        # Apply the LWA
        x, weighted_skip = self.LWA([skip1_out,skip2_out,skip3_out,skip4_out], x)


        # Apply learnable weights for each upsampling step
        x = self.up_conv4(x, weighted_skip[3])
        x = self.up_conv3(x, weighted_skip[2])
        x = self.up_conv2(x, weighted_skip[1])
        x = self.up_conv1(x, weighted_skip[0])

        x = self.conv_last(x)

        return x
