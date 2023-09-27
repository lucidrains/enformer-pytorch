from transformers import PretrainedConfig

class EnformerConfig(PretrainedConfig):
    model_type = "enformer"

    def __init__(
        self,
        dim = 1536,
        depth = 11,
        heads = 8,
        output_heads = dict(human = 5313, mouse= 1643),
        target_length = 896,
        attn_dim_key = 64,
        dropout_rate = 0.4,
        attn_dropout = 0.05,
        pos_dropout = 0.01,
        use_checkpointing = False,
        use_convnext = False,
        num_downsamples = 7,    # genetic sequence is downsampled 2 ** 7 == 128x in default Enformer - can be changed for higher resolution
        dim_divisible_by = 128,
        use_tf_gamma = False,
        **kwargs,
    ):
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.output_heads = output_heads
        self.target_length = target_length
        self.attn_dim_key = attn_dim_key
        self.dropout_rate = dropout_rate
        self.attn_dropout = attn_dropout
        self.pos_dropout = pos_dropout
        self.use_checkpointing = use_checkpointing
        self.num_downsamples = num_downsamples
        self.dim_divisible_by = dim_divisible_by
        self.use_tf_gamma = use_tf_gamma

        super().__init__(**kwargs)