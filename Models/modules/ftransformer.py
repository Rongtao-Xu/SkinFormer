
import torch
import torch.nn as nn
from functools import partial

from .multihead_isa_pool_attention import InterlacedPoolAttention2 as InterlacedPoolAttention
from .ffn_block import MlpDWBN,MlpLight,Mlp

BN_MOMENTUM = 0.1


def drop_path(x, drop_prob: float = 0.0, training: bool = False):

    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return "drop_prob={}".format(self.drop_prob)

class DilatedConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels
           kSize: kernel size
           stride: optional stride rate for down-sampling
           d: dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False,
                              dilation=d)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output

class GeneralTransformerBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes,
            planes,
            num_heads,
            window_size=7,  #8
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,#0.1, #0.0
            drop_path=0.0,#0.2, #0.0
            act_layer=nn.GELU,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super(GeneralTransformerBlock, self).__init__()
        self.dim = inplanes
        self.out_dim = planes
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.attn = InterlacedPoolAttention(
            self.dim,
            num_heads=num_heads,
            window_size=window_size,
            rpe=True,
            dropout=attn_drop,
        )

        self.norm1 = norm_layer(self.dim)
        self.norm2 = norm_layer(self.out_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(self.dim * mlp_ratio)


        self.mlp = Mlp(
            in_features=self.dim,
            hidden_features=mlp_hidden_dim,
            out_features=self.out_dim,
        )
        self.weight_level_1 = DilatedConv(256, 64, 3, 3)
        self.weight_level_2 = DilatedConv(256, 64, 3, 6)
        self.weight_level_3 = DilatedConv(256, 64, 3, 9)
        self.weight_levels = nn.Conv2d(192, 256, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.aspp = ASPP()
    def forward(self, x,y, mask=None):
        B, C, H, W = x.size()
        # reshape
        x = x.view(B, C, -1).permute(0, 2, 1)
        y = y.view(B, C, -1).permute(0, 2, 1)
        # Attention
        x = x + self.drop_path(self.attn(self.norm1(x),self.norm1(y), H, W))
        #print(x.shape)torch.Size([16, 1024, 256])
        B, N, C = x.shape
        if N == (H * W + 1):
            x_ = x[:, 1:, :].permute(0, 2, 1).reshape(B, C, H, W)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        #print(x_.shape)16, 256, 32, 32]

        # level_0 = self.weight_level_1(x_)
        # level_1 = self.weight_level_2(x_)
        # level_2 = self.weight_level_3(x_)
        # print(level_0.shape)
        # print(level_1.shape)
        # print(level_2.shape)
        levels_weight = self.aspp(x_)
        levels_weight = levels_weight.view(B, C, -1).permute(0, 2, 1)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(levels_weight), H, W))
        #x = x + self.drop_path(self.mlp(self.norm2(x)))
        # reshape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return "num_heads={}, window_size={}, mlp_ratio={}".format(
            self.num_heads, self.window_size, self.mlp_ratio
        )

import torch.nn.functional as F
class ASPP(nn.Module):
    def __init__(self, in_channel=256, depth=256):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=3, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=12)
        self.conv_1x1_output = nn.Conv2d(depth * 3, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block1 = F.upsample(atrous_block1, size=size, mode='bilinear')
        atrous_block6 = F.upsample(atrous_block6, size=size, mode='bilinear')
        atrous_block12 = F.upsample(atrous_block12, size=size, mode='bilinear')
        #print(atrous_block1.shape)torch.Size([16, 128, 32, 32])

        net = self.conv_1x1_output(torch.cat([atrous_block1, atrous_block6,
                                              atrous_block12], dim=1))
        return net
