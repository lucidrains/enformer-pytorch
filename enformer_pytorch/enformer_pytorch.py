import math
import torch
from torch import nn, einsum
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

import torch.nn.functional as F

# constants

SEQUENCE_LENGTH = 196_608
BIN_SIZE = 128
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
        seq_len, target_len = x.shape[-1], self.target_length
        if seq_len < target_len:
            raise ValueError(f'sequence length {seq_len} is less than target length {target_len}')

        trim = (target_len - seq_len) // 2
        return x[..., -trim:trim]

def ConvBlock(dim, dim_out = None, kernel_size = 1):
    return nn.Sequential(
        nn.BatchNorm1d(dim),
        GELU(),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding = kernel_size // 2)
    )

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
        num_alphabet = 5
    ):
        super().__init__()
        half_dim = dim // 2
        twice_dim = dim * 2

        self.stem = nn.Sequential(
            nn.Conv1d(num_alphabet, half_dim, 15, padding = 7),
            Residual(ConvBlock(half_dim)),
            AttentionPool(half_dim, pool_size = 2)
        )

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
        self.target_length = target_length
        self.crop_final = TargetLengthCrop(target_length)

        self.final_pointwise = nn.Sequential(
            nn.Conv1d(filter_list[-1], twice_dim, 1),
            nn.Dropout(dropout_rate / 8),
            GELU()
        )

        self._heads = map_values(lambda features: nn.Sequential(
            nn.Conv1d(twice_dim, features, 1),
            nn.Softplus()
        ), output_heads)

    @property
    def heads(self):
        return self._heads
    
    def forward(self, x):
        x = rearrange(x.float(), 'b n d -> b d n')
        x = self.stem(x)
        x = self.conv_tower(x)
        x = self.crop_final(x)
        x = self.final_pointwise(x)
        return map_values(lambda fn: fn(x), self._heads)
