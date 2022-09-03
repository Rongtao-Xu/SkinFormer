
import torch.nn as nn
from .multihead_attention import MultiheadAttention as MHA_
from .CA import ComprehensiveAttentionalBlock
from .multihead_isa_attention import PadBlock, LocalPermuteModule

class InterlacedPoolAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, window_size=7,
                 rpe=True, **kwargs):
        super(InterlacedPoolAttention, self).__init__()
        
        self.dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.with_rpe = rpe
        #print('print(embed_dim)', embed_dim)
        self.fullyattn = ComprehensiveAttentionalBlock(128)
        self.attn = MHA_(embed_dim, num_heads, **kwargs)
        self.pad_helper = PadBlock(window_size)
        self.permute_helper = LocalPermuteModule(window_size)

    def forward(self, x, H, W, **kwargs):
        B, N, C = x.shape

        x = x.view(B, H, W, C)
        # attention
        #print('print(x.shape)',x.shape)print(x_permute.shape) torch.Size([16, 16, 16, 128])

        x_ = self.fullyattn(x.transpose(1, 3).transpose(2, 3))
        #print(x_.shape)torch.Size([16, 128, 16, 16])
        x = x_.transpose(1, 3)

        # pad
        x_pad = self.pad_helper.pad_if_needed(x, x.size())
        # permute
        x_permute = self.permute_helper.permute(x_pad, x_pad.size())
        # attention
        # orch.Size([49, 144, 128])
        out = self.attn(x_permute, x_permute, x_permute, **kwargs)
        # reverse permutation
        out = self.permute_helper.rev_permute(out, x_pad.size())
        out = self.pad_helper.depad_if_needed(out, x.size())
        return out.reshape(B, N, C)


class InterlacedPoolAttention2(nn.Module):
    r""" interlaced sparse multi-head self attention (ISA) module with relative position bias.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): Window size.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, embed_dim, num_heads, window_size=7,
                 rpe=True, **kwargs):
        super(InterlacedPoolAttention2, self).__init__()

        self.dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.with_rpe = rpe
        self.fullyattn = ComprehensiveAttentionalBlock(embed_dim)
        self.attn = MHA_(embed_dim, num_heads, **kwargs)
        self.pad_helper = PadBlock(window_size)
        self.permute_helper = LocalPermuteModule(window_size)

    def forward(self, x,y, H, W, **kwargs):
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        # attention
        #print(x.shape)
        x_ = self.fullyattn(x.transpose(1, 3).transpose(2, 3))
        #print(x_.shape)torch.Size([16, 128, 16, 16])
        x = x_.transpose(1, 3)
        # pad
        x_pad = self.pad_helper.pad_if_needed(x, x.size())
        #print(x_pad.shape)
        # permute
        x_permute = self.permute_helper.permute(x_pad, x_pad.size())
        # print(x_permute.shape)
        # torch.Size([8, 128, 128, 32])
        # torch.Size([8, 133, 133, 32])
        # torch.Size([49, 2888, 32])
        # print('------------')
        y = x.view(B, H, W, C)
        # attention
        # pad
        y_pad = self.pad_helper.pad_if_needed(y, y.size())
        # permute
        y_permute = self.permute_helper.permute(y_pad, y_pad.size())
        # attention
        out = self.attn(x_permute, y_permute, y_permute, **kwargs)
        # reverse permutation
        out = self.permute_helper.rev_permute(out, x_pad.size())
        out = self.pad_helper.depad_if_needed(out, x.size())
        return out.reshape(B, N, C)
