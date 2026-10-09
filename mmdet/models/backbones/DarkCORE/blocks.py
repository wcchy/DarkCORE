import torch
import torch.nn as nn
import math
import pdb
from torch.nn import init
import torch.nn.functional as F

def pair_downsampler(img):
    # img has shape B C H W
    c = img.shape[1]
    filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c, 1, 1, 1)
    filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)
    output1 = torch.nn.functional.conv2d(img, filter1, stride=2, groups=c)
    output2 = torch.nn.functional.conv2d(img, filter2, stride=2, groups=c)
    return output1,output2

class LocalMean(nn.Module):
    def __init__(self, patch_size=5):
        super(LocalMean, self).__init__()
        self.patch_size = patch_size
        self.padding = self.patch_size // 2

    def forward(self, image):
        image = F.pad(image, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        patches = image.unfold(2, self.patch_size, 1).unfold(3, self.patch_size, 1)
        return patches.mean(dim=(4, 5))

class TextureDifference(nn.Module):
    def __init__(self, patch_size=5, constant_C=1e-5,threshold=0.99):
        super(TextureDifference, self).__init__()
        self.patch_size = patch_size
        self.constant_C = constant_C
        self.threshold = threshold

    def forward(self, image1, image2):
        # Convert RGB images to grayscale
        image1 = self.rgb_to_gray(image1)
        image2 = self.rgb_to_gray(image2)

        stddev1 = self.local_stddev(image1)
        stddev2 = self.local_stddev(image2)
        numerator = 2 * stddev1 * stddev2
        denominator = stddev1 ** 2 + stddev2 ** 2 + self.constant_C
        diff = numerator / denominator

        # Apply threshold to diff tensor
        binary_diff = torch.where(diff > self.threshold, torch.tensor(1.0, device=diff.device),
                                  torch.tensor(0.0, device=diff.device))

        return binary_diff

    def local_stddev(self, image):
        padding = self.patch_size // 2
        image = F.pad(image, (padding, padding, padding, padding), mode='reflect')
        patches = image.unfold(2, self.patch_size, 1).unfold(3, self.patch_size, 1)
        mean = patches.mean(dim=(4, 5), keepdim=True)
        squared_diff = (patches - mean) ** 2
        local_variance = squared_diff.mean(dim=(4, 5))
        local_stddev = torch.sqrt(local_variance+1e-9)
        return local_stddev

    def rgb_to_gray(self, image):
        # Convert RGB image to grayscale using the luminance formula
        gray_image =  0.144 * image[:, 0, :, :] + 0.5870 * image[:, 1, :, :] + 0.299 * image[:, 2, :, :]
        return gray_image.unsqueeze(1)  # Add a channel dimension for compatibility

class Mlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ChannelAttention(nn.Module):
    """
    CBAM混合注意力机制的通道注意力
    """

    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            # 全连接层
            # nn.Linear(in_planes, in_planes // ratio, bias=False),
            # nn.ReLU(),
            # nn.Linear(in_planes // ratio, in_planes, bias=False)

            # 利用1x1卷积代替全连接，避免输入必须尺度固定的问题，并减小计算量
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
       avg_out = self.fc(self.avg_pool(x))
       max_out = self.fc(self.max_pool(x))
       out = avg_out + max_out
       out = self.sigmoid(out)
       return out * x

class SpatialAttention(nn.Module):
    """
    CBAM混合注意力机制的空间注意力
    """

    def __init__(self, kernel_size=7):
        super().__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out * x

class CBAM(nn.Module):
    """
    CBAM混合注意力机制
    """

    def __init__(self, in_channels, ratio=2, kernel_size=3):
        super().__init__()
        self.channelattention = ChannelAttention(in_channels, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channelattention(x)
        x = self.spatialattention(x)
        return x
    
class ECA(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.gap(x)  # bs, c, 1, 1
        y = y.squeeze(-1).permute(0, 2, 1)  # bs, 1, c
        y = self.conv(y)  # bs, 1, c
        y = self.sigmoid(y)  # bs, 1, c
        y = y.permute(0, 2, 1).unsqueeze(-1)  # bs, c, 1, 1
        return x * y.expand_as(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        self.atten = ECA()

    def forward(self, x):
        return self.maxpool_conv(x)
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class IlluminationEnhanceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_cov = DoubleConv(in_channels=1,out_channels=8)
        self.blocks1 = CustomBlock(in_channels=8, out_channels=8, dw_kernel_size=3, downsample=True)
        self.blocks2 = CustomBlock(in_channels=16, out_channels=8, dw_kernel_size=3, downsample=True)
        self.maxpool1 = nn.Sequential(nn.MaxPool2d(2),DoubleConv(in_channels=8,out_channels=8))
        self.maxpool2 = nn.Sequential(nn.MaxPool2d(2),DoubleConv(in_channels=16,out_channels=8))
        self.eca = ECA()
        self.outc = nn.Conv2d(in_channels=16,out_channels=1,kernel_size=3,padding=1,stride=1)

    def forward(self,x):
        x = self.in_cov(x)
        x1 = self.blocks1(x)
        x1 = self.eca(x1) + x1
        x2 = self.maxpool1(x)
        x2 = self.eca(x2) + x2

        x = torch.cat((x1,x2),dim=1)

        x3 = self.blocks2(x)
        x3 = self.eca(x3) + x3
        x4 = self.maxpool2(x)
        x4 = self.eca(x4) + x4

        x = torch.cat((x3,x4),dim=1)

        return self.outc(x)
    
class CustomBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dw_kernel_size, downsample=False, use_pixel_unshuffle=False):
        super().__init__()
        # self.inconv = DoubleConv(in_channels, out_channels)
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=dw_kernel_size, padding=dw_kernel_size//2, groups=in_channels)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # self.cbam = CBAM(out_channels) if use_cbam else None
        self.eca = ECA()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2) if downsample else None
        self.pixel_unshuffle = nn.PixelUnshuffle(2) if use_pixel_unshuffle else None
        self.max_pool = nn.MaxPool2d(2) if not downsample and not use_pixel_unshuffle else None

    def forward(self, x):
        # x = self.inconv(x)
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        # if self.cbam:
        #     x = self.cbam(x)
        # if self.eca:
        x = self.eca(x)
        x = self.leaky_relu(x)
        if self.pixel_unshuffle:
            x = self.pixel_unshuffle(x)
            # print(f"After PixelUnshuffle: {x.shape}")
        elif self.downsample:
            x = self.downsample(x)
        elif self.max_pool:
            x = self.max_pool(x)
        return x
    
class EnhanceReflectance(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnhanceReflectance, self).__init__()
        self.block1 = CustomBlock(in_channels, in_channels, dw_kernel_size=5, use_pixel_unshuffle=True)
        self.block2 = CustomBlock(in_channels*4, in_channels, dw_kernel_size=3, downsample=True)
        self.block3 = CustomBlock(in_channels, in_channels, dw_kernel_size=3, downsample=False)
        self.block4 = CustomBlock(in_channels, out_channels, dw_kernel_size=1, downsample=False)
        

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # pdb.set_trace()
        return x
