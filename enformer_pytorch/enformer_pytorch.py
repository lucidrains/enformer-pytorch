import math
import torch
from torch import nn, einsum
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

import torch.nn.functional as F

# constants

SEQUENCE_LENGTH = 196_608
TARGET_LENGTH = 896

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def map_values(fn, d):
    return {key: fn(values) for key, values in d.items()}

def exponential_linspace_int(start, end, num, divisible_by = 1):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]

# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x

class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size = 2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p = 2)
        self.to_attn_logits = nn.Parameter(torch.eye(dim))

    def forward(self, x):
        remainder = x.shape[-1] % self.pool_size
        if remainder > 0:
            x = F.pad(x, (0, remainder), value = 0)

        attn_logits = einsum('b d n, d e -> b e n', x, self.to_attn_logits)
        x = self.pool_fn(x)
        attn = self.pool_fn(attn_logits).softmax(dim = -1)
        return (x * attn).sum(dim = -1)

class TargetLengthCrop(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        seq_len, target_len = x.shape[-2], self.target_length
        if seq_len < target_len:
            raise ValueError(f'sequence length {seq_len} is less than target length {target_len}')

        trim = (target_len - seq_len) // 2
        return x[:, -trim:trim]

def ConvBlock(dim, dim_out = None, kernel_size = 1):
    return nn.Sequential(
        nn.BatchNorm1d(dim),
        GELU(),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding = kernel_size // 2)
    )

# attention classes

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        heads = 8,
        dim_key = 64,
        dim_value = 64,
        dropout = 0.
    ):
        super().__init__()
        self.scale = dim_key ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, dim_key * heads, bias = False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias = False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias = False)
        self.attn_dropout = nn.Dropout(dropout)

        self.to_out = nn.Linear(dim_value * heads, dim)

    def forward(self, x):
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# main class

class Enformer(nn.Module):
    def __init__(
        self,
        *,
        dim = 1536,
        depth = 11,
        heads = 8,
        output_heads = dict(human = 5313, mouse= 1643),
        target_length = TARGET_LENGTH,
        dropout_rate = 0.4,
        num_alphabet = 5,
        attn_dim_key = 64
    ):
        super().__init__()
        half_dim = dim // 2
        twice_dim = dim * 2

        # create stem

        self.stem = nn.Sequential(
            Rearrange('b n d -> b d n'),
            nn.Conv1d(num_alphabet, half_dim, 15, padding = 7),
            Residual(ConvBlock(half_dim)),
            AttentionPool(half_dim, pool_size = 2)
        )

        # create conv tower

        filter_list = exponential_linspace_int(half_dim, dim, num = 6, divisible_by = 128)
        filter_list = [half_dim, *filter_list]

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(nn.Sequential(
                ConvBlock(dim_in, dim_out, kernel_size = 5),
                Residual(ConvBlock(dim_out, dim_out, 1)),
                AttentionPool(dim_out, pool_size = 2)
            ))

        self.conv_tower = nn.Sequential(*conv_layers)

        # transformer

        transformer = []
        for _ in range(depth):
            transformer.append(nn.Sequential(
                Residual(nn.Sequential(
                    nn.LayerNorm(dim),
                    Attention(
                        dim,
                        heads = heads,
                        dim_key = attn_dim_key,
                        dim_value = dim // heads
                    ),
                    nn.Dropout(dropout_rate)
                )),
                Residual(nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim * 2),
                    nn.Dropout(dropout_rate),
                    nn.ReLU(),
                    nn.Linear(dim * 2, dim),
                    nn.Dropout(dropout_rate)
                ))
            ))

        self.transformer = nn.Sequential(
            Rearrange('b d n -> b n d'),
            *transformer
        )

        # target cropping

        self.target_length = target_length
        self.crop_final = TargetLengthCrop(target_length)

        # final pointwise

        self.final_pointwise = nn.Sequential(
            nn.Linear(filter_list[-1], twice_dim, 1),
            nn.Dropout(dropout_rate / 8),
            GELU()
        )

        # create trunk sequential module

        self._trunk = nn.Sequential(
            self.stem,
            self.conv_tower,
            self.transformer,
            self.crop_final,
            self.final_pointwise
        )

        # create final heads for human and mouse

        self._heads = map_values(lambda features: nn.Sequential(
            nn.Linear(twice_dim, features, 1),
            nn.Softplus()
        ), output_heads)

    @property
    def trunk(self):
        return self._trunk

    @property
    def heads(self):
        return self._heads
    
    def forward(self, x):
        x = self._trunk(x.float())
        return map_values(lambda fn: fn(x), self._heads)
