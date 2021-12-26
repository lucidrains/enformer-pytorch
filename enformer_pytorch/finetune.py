import torch
from contextlib import contextmanager
from torch import nn, einsum
from einops import rearrange
from enformer_pytorch.enformer_pytorch import Enformer, poisson_loss

def exists(val):
    return val is not None

@contextmanager
def null_context():
    yield

class ContextAdapterWrapper(nn.Module):
    def __init__(
        self,
        *,
        enformer,
        enformer_dim,
        context_dim
    ):
        super().__init__()
        assert isinstance(enformer, Enformer)
        self.enformer = enformer

        self.to_context_weights = nn.Parameter(torch.randn(context_dim, enformer_dim * 2))
        self.to_context_bias = nn.Parameter(torch.randn(context_dim))

    def forward(
        self,
        seq,
        *,
        context,
        target = None,
        freeze_enformer = False
    ):
        enformer_context = null_context if freeze_enformer else torch.no_grad

        with enformer_context():
            _, embeddings = self.enformer(seq, return_embeddings = True)

            if freeze_enformer:
                embeddings.detach_()

        weights = einsum('t d, d e -> t e', context, self.to_context_weights)
        bias = einsum('t d, d -> t', context, self.to_context_bias)

        pred = einsum('b n d, t d -> b n t', embeddings, weights) + bias

        if not exists(target):
            return pred

        return poisson_loss(pred, target)
