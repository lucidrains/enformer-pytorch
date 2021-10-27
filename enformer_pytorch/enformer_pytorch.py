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
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p = 2)
        self.to_attn_logits = nn.Parameter(torch.eye(dim))

    def forward(self, x):
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

def ConvBlock(dim, kernel_size = 1, dim_out = None):
    return nn.Sequential(
        nn.BatchNorm1d(dim),
        GELU(),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size)
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
        dropout_rate = 0.4
    ):
        super().__init__()
        init_dim = dim // 2
        twice_dim = dim * 2

        self.stem = nn.Sequential(
            nn.Conv1d(4, init_dim, 15),
            Residual(ConvBlock(init_dim)),
            AttentionPool(init_dim, pool_size = 2)
        )

        self.target_length = target_length
        self.crop_final = TargetLengthCrop(target_length)

        self.final_pointwise = nn.Sequential(
            nn.Conv1d(init_dim, twice_dim, 1),
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

        x = self.crop_final(x)
        x = self.final_pointwise(x)
        return map_values(lambda fn: fn(x), self._heads)
