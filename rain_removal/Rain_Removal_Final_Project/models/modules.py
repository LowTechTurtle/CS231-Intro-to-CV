import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab


def get_heatmap(mask):
    lum_img = np.maximum(
        np.maximum(
            mask[:, :, 0],
            mask[:, :, 1],
        ),
        mask[:, :, 2],
    )
    imgplot = plt.imshow(lum_img)
    imgplot.set_cmap("jet")
    plt.colorbar()
    plt.axis("off")
    pylab.show()
    return


class DSConv(nn.Module):
    """
    Depthwise Separable Convolution.
    Note that depthwise separable convolution is made up of depthwise and pointwise convolutions.
    The depthwise convolution is applied to each channel separately (setting groups=in_channels),
    and the pointwise convolution is applied to the entire feature map with kernel size equal to 1.

    Params
    ------
    in_channels: int
        Number of input channels
    out_channels: int
        Number of output channels
    kernel_size: int
        Kernel size of the depthwise convolution
    stride: int
        Stride of the depthwise convolution
    padding: int
        Padding of the depthwise convolution
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1
    ):
        super(DSConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, local_conv=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        self.local_conv = local_conv
        if local_conv:
            self.local_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        if self.local_conv:
            # Optionally apply a convolution over the sequence dimension.
            # Rearrange from (b, n, dim) -> (b, dim, n)
            x_conv = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
            x = x + x_conv  # fuse local context

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
