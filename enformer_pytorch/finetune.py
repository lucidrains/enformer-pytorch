import torch
from contextlib import contextmanager
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange
from enformer_pytorch.enformer_pytorch import Enformer, poisson_loss

def exists(val):
    return val is not None

@contextmanager
def null_context():
    yield

def freeze_batchnorms(model):
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]

    for bn in bns:
        bn.eval()
        bn.requires_grad = False
        bn.track_running_stats = False

def get_enformer_embeddings(model, seq, freeze = False):
    freeze_batchnorms(model)
    enformer_context = null_context() if not freeze else torch.no_grad

    with enformer_context:
        embeddings = model(seq, return_only_embeddings = True)

        if freeze:
            embeddings.detach_()

    return embeddings

# fine-tune wrapper classes

class HeadAdapterWrapper(nn.Module):
    def __init__(
        self,
        *,
        enformer,
        num_tracks
    ):
        super().__init__()
        assert isinstance(enformer, Enformer)
        self.enformer = enformer

        self.to_tracks = nn.Sequential(
            nn.Linear(enformer.dim * 2, num_tracks),
            nn.Softplus()
        )

    def forward(
        self,
        seq,
        *,
        target = None,
        freeze_enformer = False
    ):
        embeddings = get_enformer_embeddings(self.enformer, seq, freeze = freeze_enformer)
        preds = self.to_tracks(embeddings)

        if not exists(target):
            return preds

        return poisson_loss(preds, target)

class ContextAdapterWrapper(nn.Module):
    def __init__(
        self,
        *,
        enformer,
        context_dim
    ):
        super().__init__()
        assert isinstance(enformer, Enformer)
        self.enformer = enformer

        self.to_context_weights = nn.Parameter(torch.randn(context_dim, enformer.dim * 2))
        self.to_context_bias = nn.Parameter(torch.randn(context_dim))

    def forward(
        self,
        seq,
        *,
        context,
        target = None,
        freeze_enformer = False
    ):
        embeddings = get_enformer_embeddings(self.enformer, seq, freeze = freeze_enformer)

        weights = einsum('t d, d e -> t e', context, self.to_context_weights)
        bias = einsum('t d, d -> t', context, self.to_context_bias)

        pred = einsum('b n d, t d -> b n t', embeddings, weights) + bias

        pred = F.softplus(pred)

        if not exists(target):
            return pred

        return poisson_loss(pred, target)
