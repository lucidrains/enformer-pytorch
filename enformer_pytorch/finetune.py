import torch
from contextlib import contextmanager
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
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

# extra head projection, akin to how human and mouse tracks were trained

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

# wrapper that allows one to supply each track with a context dimension
# the context embedding will be projected into the weights and biases of the head linear projection (hypernetwork)

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

# wrapper that does attention aggregation of the context, which can be a list of tokens (batch x seq x dim)

class ContextAttentionAdapterWrapper(nn.Module):
    def __init__(
        self,
        *,
        enformer,
        context_dim,
        heads = 8,
        dim_head = 64
    ):
        super().__init__()
        assert isinstance(enformer, Enformer)
        self.enformer = enformer
        enformer_hidden_dim = enformer.dim * 2

        self.query_norm = nn.LayerNorm(enformer_hidden_dim)
        self.key_values_norm = nn.LayerNorm(context_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = heads * dim_head
        self.to_queries = nn.Linear(enformer_hidden_dim, inner_dim)

        self.null_key = nn.Parameter(torch.randn(inner_dim))
        self.null_value = nn.Parameter(torch.randn(inner_dim))

        self.to_key_values = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, enformer_hidden_dim)

        self.to_pred  = nn.Sequential(
            nn.Linear(enformer_hidden_dim, 1),
            Rearrange('b c ... 1 -> b ... c'),
            nn.Softplus()
        )

    def forward(
        self,
        seq,
        *,
        context,
        target = None,
        freeze_enformer = False
    ):
        h = self.heads
        embeddings = get_enformer_embeddings(self.enformer, seq, freeze = freeze_enformer)

        # perform cross attention from genetic -> context

        if context.ndim == 2:
            context = rearrange(context, 'b d -> b 1 d')

        q = self.to_queries(self.query_norm(embeddings))
        k, v = self.to_key_values(self.key_values_norm(context)).chunk(2, dim = -1)

        null_k, null_v = map(lambda t: repeat(t, 'd -> b 1 d', b = context.shape[0]), (self.null_key, self.null_value))

        k = torch.cat((null_k, k), dim = 1)
        v = torch.cat((null_v, v), dim = 1)

        # split out head

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        sim = einsum('b h i d, c h j d -> b c h i j', q, k) * self.scale

        # attention

        attn = sim.softmax(dim = -1)

        # aggregate

        out = einsum('b c h i j, c h j d -> b c h i d', attn, v)

        out = rearrange(out, 'b c h n d -> b c n (h d)', h = h)

        # combine heads

        branch_out = self.to_out(out)

        # residual

        embeddings = embeddings + branch_out

        # to prediction

        pred = self.to_pred(embeddings)

        if not exists(target):
            return pred

        return poisson_loss(pred, target)
