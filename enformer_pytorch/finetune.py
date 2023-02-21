import torch
from typing import Optional

from copy import deepcopy
from contextlib import contextmanager
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from enformer_pytorch.modeling_enformer import Enformer, poisson_loss

from discrete_key_value_bottleneck_pytorch import DiscreteKeyValueBottleneck

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

@contextmanager
def null_context():
    yield

# better sequential

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

# controlling freezing of layers

def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)

def unfreeze_all_layers_(module):
    set_module_requires_grad_(module, True)

def freeze_batchnorms_(model):
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]

    for bn in bns:
        bn.eval()
        bn.track_running_stats = False
        set_module_requires_grad_(bn, False)

def freeze_all_but_layernorms_(model):
    for m in model.modules():
        set_module_requires_grad_(m, isinstance(m, nn.LayerNorm))

def freeze_all_but_last_n_layers_(enformer, n):
    assert isinstance(enformer, Enformer)
    freeze_all_layers_(enformer)

    transformer_blocks = enformer.transformer

    for module in transformer_blocks[-n:]:
        set_module_requires_grad_(module, True)

# get enformer embeddings

def get_enformer_embeddings(
    model,
    seq,
    freeze = False,
    train_layernorms_only = False,
    train_last_n_layers_only = None,
    enformer_kwargs: dict = {}
):
    freeze_batchnorms_(model)

    if train_layernorms_only:
        assert not freeze, 'you set the intent to train the layernorms of the enformer, yet also indicated you wanted to freeze the entire model'
        freeze_all_but_layernorms_(model)

    if exists(train_last_n_layers_only):
        assert not freeze, 'you set the intent to train last N layers of enformer, but also indicated you wanted to freeze the entire network'
        freeze_all_but_last_n_layers_(model, train_last_n_layers_only)

    enformer_context = null_context() if not freeze else torch.no_grad()

    with enformer_context:
        embeddings = model(seq, return_only_embeddings = True, **enformer_kwargs)

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
        num_tracks,
        post_transformer_embed = False, # whether to take the embeddings from right after the transformer, instead of after the final pointwise convolutional - this would add another layernorm
        discrete_key_value_bottleneck = False,
        bottleneck_num_memories = 256,
        bottleneck_num_codebooks = 4,
        bottleneck_decay = 0.9,
        transformer_embed_fn: nn.Module = nn.Identity(),
        output_activation: Optional[nn.Module] = nn.Softplus(),
        auto_set_target_length = True
    ):
        super().__init__()
        assert isinstance(enformer, Enformer)
        enformer_hidden_dim = enformer.dim * (2 if not post_transformer_embed else 1)

        self.discrete_key_value_bottleneck = discrete_key_value_bottleneck

        if discrete_key_value_bottleneck:
            enformer = DiscreteKeyValueBottleneck(
                encoder = enformer,
                dim = enformer_hidden_dim,
                num_memory_codebooks = bottleneck_num_codebooks,
                num_memories = bottleneck_num_memories,
                dim_memory = enformer_hidden_dim // bottleneck_num_codebooks,
                decay = bottleneck_decay,
            )

        self.post_transformer_embed = post_transformer_embed

        self.enformer = enformer

        self.auto_set_target_length = auto_set_target_length

        if post_transformer_embed:
            self.enformer = deepcopy(enformer)
            self.enformer._trunk[-1] = nn.Identity()
            self.enformer.final_pointwise = nn.Identity()

        self.post_embed_transform = Sequential(
            transformer_embed_fn,
            nn.LayerNorm(enformer_hidden_dim) if post_transformer_embed else None
        )

        self.to_tracks = Sequential(
            nn.Linear(enformer_hidden_dim, num_tracks),
            output_activation
        )

    def forward(
        self,
        seq,
        *,
        target = None,
        freeze_enformer = False,
        finetune_enformer_ln_only = False,
        finetune_last_n_layers_only = None
    ):
        enformer_kwargs = dict()

        if exists(target) and self.auto_set_target_length:
            enformer_kwargs = dict(target_length = target.shape[-2])

        if self.discrete_key_value_bottleneck:
            embeddings = self.enformer(seq, return_only_embeddings = True, **enformer_kwargs)
        else:
            embeddings = get_enformer_embeddings(self.enformer, seq, freeze = freeze_enformer, train_layernorms_only = finetune_enformer_ln_only, train_last_n_layers_only = finetune_last_n_layers_only, enformer_kwargs = enformer_kwargs)

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
        context_dim,
        discrete_key_value_bottleneck = False,
        bottleneck_num_memories = 256,
        bottleneck_num_codebooks = 4,
        bottleneck_decay = 0.9,
        auto_set_target_length = True,
        output_activation: Optional[nn.Module] = nn.Softplus()
    ):
        super().__init__()
        assert isinstance(enformer, Enformer)
        enformer_hidden_dim = enformer.dim * 2

        self.discrete_key_value_bottleneck = discrete_key_value_bottleneck

        if discrete_key_value_bottleneck:
            enformer = DiscreteKeyValueBottleneck(
                encoder = enformer,
                dim = enformer_hidden_dim,
                num_memory_codebooks = bottleneck_num_codebooks,
                num_memories = bottleneck_num_memories,
                dim_memory = enformer_hidden_dim // bottleneck_num_codebooks,
                decay = bottleneck_decay,
            )

        self.enformer = enformer

        self.auto_set_target_length = auto_set_target_length

        self.to_context_weights = nn.Parameter(torch.randn(context_dim, enformer_hidden_dim))
        self.to_context_bias = nn.Parameter(torch.randn(context_dim))

        self.activation = default(output_activation, nn.Identity())

    def forward(
        self,
        seq,
        *,
        context,
        target = None,
        freeze_enformer = False,
        finetune_enformer_ln_only = False,
        finetune_last_n_layers_only = None
    ):
        enformer_kwargs = dict()

        if exists(target) and self.auto_set_target_length:
            enformer_kwargs = dict(target_length = target.shape[-2])

        if self.discrete_key_value_bottleneck:
            embeddings = self.enformer(seq, return_only_embeddings = True, **enformer_kwargs)
        else:
            embeddings = get_enformer_embeddings(self.enformer, seq, freeze = freeze_enformer, train_layernorms_only = finetune_enformer_ln_only, train_last_n_layers_only = finetune_last_n_layers_only, enformer_kwargs = enformer_kwargs)

        weights = einsum('t d, d e -> t e', context, self.to_context_weights)
        bias = einsum('t d, d -> t', context, self.to_context_bias)

        pred = einsum('b n d, t d -> b n t', embeddings, weights) + bias

        pred = self.activation(pred)

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
        dim_head = 64,
        discrete_key_value_bottleneck = False,
        bottleneck_num_memories = 256,
        bottleneck_num_codebooks = 4,
        bottleneck_decay = 0.9,
        auto_set_target_length = True,
        output_activation: Optional[nn.Module] = nn.Softplus()
    ):
        super().__init__()
        assert isinstance(enformer, Enformer)
        enformer_hidden_dim = enformer.dim * 2

        self.discrete_key_value_bottleneck = discrete_key_value_bottleneck

        if discrete_key_value_bottleneck:
            enformer = DiscreteKeyValueBottleneck(
                encoder = enformer,
                dim = enformer_hidden_dim,
                num_memory_codebooks = bottleneck_num_codebooks,
                num_memories = bottleneck_num_memories,
                dim_memory = enformer_hidden_dim // bottleneck_num_codebooks,
                decay = bottleneck_decay,
            )

        self.enformer = enformer

        self.auto_set_target_length = auto_set_target_length

        self.query_norm = nn.LayerNorm(enformer_hidden_dim)
        self.key_values_norm = nn.LayerNorm(context_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = heads * dim_head
        self.to_queries = nn.Linear(enformer_hidden_dim, inner_dim, bias = False)

        self.null_key = nn.Parameter(torch.randn(inner_dim))
        self.null_value = nn.Parameter(torch.randn(inner_dim))

        self.to_key_values = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, enformer_hidden_dim)

        self.to_pred  = Sequential(
            nn.Linear(enformer_hidden_dim, 1),
            Rearrange('b c ... 1 -> b ... c'),
            output_activation
        )

    def forward(
        self,
        seq,
        *,
        context,
        context_mask = None,
        target = None,
        freeze_enformer = False,
        finetune_enformer_ln_only = False,
        finetune_last_n_layers_only = None
    ):
        """
        b - batch
        n - sequence length
        c - number of contexts (tracks)
        d - dimension
        i - sequence length (query embeddings)
        j - sequence length (keys / values contexts)
        h - attention heads
        """

        h = self.heads

        enformer_kwargs = dict()

        if exists(target) and self.auto_set_target_length:
            enformer_kwargs = dict(target_length = target.shape[-2])

        if self.discrete_key_value_bottleneck:
            embeddings = self.enformer(seq, return_only_embeddings = True, **enformer_kwargs)
        else:
            embeddings = get_enformer_embeddings(self.enformer, seq, freeze = freeze_enformer, train_layernorms_only = finetune_enformer_ln_only, train_last_n_layers_only = finetune_last_n_layers_only, enformer_kwargs = enformer_kwargs)

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

        # masking

        if exists(context_mask):
            context_mask = F.pad(context_mask, (1, 0), value = True)
            context_mask =rearrange(context_mask, 'b j -> b 1 1 1 j')
            sim = sim.masked_fill(~context_mask, -torch.finfo(sim.dtype).max)

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
