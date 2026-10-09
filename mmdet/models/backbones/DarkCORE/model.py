import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchvision.transforms as transforms
import torchvision
import pdb
from ...builder import BACKBONES
from .blocks import EnhanceReflectance, pair_downsampler, Mlp, IlluminationEnhanceNet
from timm.models.layers import DropPath, trunc_normal_
import einops


from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init)
from mmcv.runner import BaseModule
import warnings
    
class Denoise(nn.Module):
    def __init__(self, chan_embed=32):
        super(Denoise, self).__init__()

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(3, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, 3, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class ReCom(nn.Module):
    def __init__(self, hidden_dim=18):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,hidden_dim,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(num_features=hidden_dim),
            nn.LeakyReLU(0.2)
        )
        self.depthConv1 = DepthwiseSeparableConv(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.depthConv2 = DepthwiseSeparableConv(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_dim,hidden_dim,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(num_features=hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.conv1x1 = nn.Conv2d(hidden_dim*3, out_channels=4,kernel_size=1,padding=0,stride=1)
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
 
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.depthConv1(x1) + x1
        x3 = self.depthConv2(x1) + x1
        x4 = self.layer2(x1) + x1
        x = torch.cat((x2, x3, x4), dim=1)
        x = self.conv1x1(x)
        R, L = x[:, :3, :, :], x[:, 3:, :, :]
        R = self.tanh1(R)
        L = self.tanh2(L)
        return R, L


class GammaEnhance(nn.Module):
    def __init__(self, out_dim=1):
        super(GammaEnhance, self).__init__()
        self.ien = IlluminationEnhanceNet()
        self.adAvp = nn.AdaptiveMaxPool2d(2)
        self.mlp_1 = Mlp(in_features=out_dim * 2 * 2, hidden_features=out_dim, out_features=2)
        self.flat_1 = nn.Flatten(start_dim=1)
        self.ccw_base = nn.Parameter(torch.eye((1)), requires_grad=True)
        self.gamma_base = nn.Parameter(torch.ones((1)), requires_grad=True)
    
    def apply_color(self, image, ccw):
        shape = image.shape
        image = image.view(-1, 1)
        image = torch.tensordot(image, ccw, dims=[[1], [0]])
        image = image.view(shape)
        return torch.clamp(image, 1e-8, 1.0)
    
    def forward(self, L):
        out = self.ien(L)
        out = self.adAvp(out)
        out = self.flat_1(out)
        out = self.mlp_1(out)
        b = L.shape[0]
        gamma = out[:, 0:1] + self.gamma_base
        ccw = out[:,1:] + self.ccw_base
        img = L.permute(0, 2, 3, 1)  # (B,C,H,W) -- (B,H,W,C)
        img = torch.stack([self.apply_color(img[i, :, :, :], ccw[i, :]) ** gamma[i, :] for i in range(b)], dim=0)
        img = img.permute(0, 3, 1, 2)  # (B,H,W,C) -- (B,C,H,W)

        return img

class ReflectanceCCM(nn.Module):
    def __init__(self, in_dim=3, out_dim=12):
        super(ReflectanceCCM, self).__init__()
        self.denoise = Denoise()
        self.denoise.apply(self.denoise_weights_init)
        self.fea = EnhanceReflectance(in_channels=in_dim, out_channels=out_dim)
        self.adAvp = nn.AdaptiveMaxPool2d(2)
        self.mlp_1 = Mlp(in_features=out_dim*4, hidden_features=out_dim, out_features=9)
        self.flat_1 = nn.Flatten(start_dim=1)
        self.ccm_base = nn.Parameter(torch.ones(3,3), requires_grad=True)

    
    def denoise_weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias != None:
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)


    
    def apply_color(self, image, ccm):
        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, ccm, dims=[[1], [0]])
        image = image.view(shape)
        return torch.clamp(image, 1e-8, 1.0)
    
    def forward(self, R):
        down_1, down_2 = pair_downsampler(R)
        denoi_pred1 = down_1 - self.denoise(down_1)
        denoi_pred2 = down_2 - self.denoise(down_2)
        deno_r = R - self.denoise(R)
        out = self.fea(deno_r)
        out = self.adAvp(out)
        out = self.flat_1(out)
        out = self.mlp_1(out)
        b = deno_r.shape[0]
        ccm = torch.reshape(out, shape=(b, 3, 3)) + self.ccm_base
        img = deno_r.permute(0, 2, 3, 1)  # (B,C,H,W) -- (B,H,W,C)
        img = torch.stack([self.apply_color(img[i, :, :, :], ccm[i, :, :]) for i in range(b)], dim=0)
        img = img.permute(0, 3, 1, 2)  # (B,H,W,C) -- (B,C,H,W)
        return img, down_1, down_2, denoi_pred1, denoi_pred2, deno_r


@BACKBONES.register_module()
class DarkCORE(nn.Module):
    def __init__(self, pretrained=None,
                 init_cfg=None):
        super(DarkCORE, self).__init__()
        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')
        
        self.decom = ReCom()
        self.enrle = ReflectanceCCM()
        self.enilm = GammaEnhance()

        print('total parameters:', sum(param.numel() for param in self.parameters()))

    def forward(self,x):
        # decom
        r, l = self.decom(x)
        e_r, down_1, down_2, denoi_pred1, denoi_pred2, deno_r = self.enrle(r)
        e_l = self.enilm(l)
        
        img_high =  e_r * e_l

        return r, l, e_r, e_l, down_1, down_2, denoi_pred1, denoi_pred2, deno_r, img_high