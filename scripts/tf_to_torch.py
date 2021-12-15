def copy_bn(mod, vars, path):
  bn_running_mean = vars[f'{path}moving_mean/average:0']
  bn_running_var = vars[f'{path}moving_variance/average:0']
  bn_offset = vars[f'{path}offset:0']
  bn_scale = vars[f'{path}scale:0']

  mod.running_var.data.copy_(rearrange(bn_running_var, '() () d -> d'))
  mod.running_mean.data.copy_(rearrange(bn_running_mean, '() () d -> d'))
  mod.weight.data.copy_(bn_scale)
  mod.bias.data.copy_(bn_offset)

def copy_conv(mod, vars, path):
  bias = vars[f'{path}b:0']
  weight = vars[f'{path}w:0']
  mod.weight.data.copy_(rearrange(weight, 'k i o -> o i k'))
  mod.bias.data.copy_(bias)

def copy_attn_pool(mod, vars, path):
  attn_pool_proj = vars[path]
  mod.to_attn_logits.data.copy_(attn_pool_proj)

def copy_linear(mod, vars, path):
  weight = vars[f'{path}w:0']
  bias = vars[f'{path}b:0']
  mod.weight.data.copy_(rearrange(weight, 'i o -> o i'))
  mod.bias.data.copy_(bias)

def copy_ln(mod, vars, path):
  weight = vars[f'{path}scale:0']
  bias = vars[f'{path}offset:0']
  mod.weight.data.copy_(weight)
  mod.bias.data.copy_(bias)

def copy_attn(mod, var_path):
  pass

def get_tf_vars(tf_model):
  return {v.name: (torch.from_numpy(v.numpy()) if isinstance(v.numpy(), np.ndarray) else None) for v in tf_model.variables}

def copy_tf_to_pytorch(tf_model, pytorch_model):
  tf_vars = get_tf_vars(tf_model)
  stem_conv = pytorch_model.stem[1]
  stem_point_bn = pytorch_model.stem[2].fn[0]
  stem_point_conv = pytorch_model.stem[2].fn[2]
  stem_attn_pool = pytorch_model.stem[3]

  copy_bn(stem_point_bn, tf_vars, 'enformer/trunk/stem/stem/pointwise_conv_block/pointwise_conv_block/batch_norm/')
  copy_conv(stem_conv, tf_vars, 'enformer/trunk/stem/stem/conv1_d/')
  copy_conv(stem_point_conv, tf_vars, 'enformer/trunk/stem/stem/pointwise_conv_block/pointwise_conv_block/conv1_d/')
  copy_attn_pool(stem_attn_pool, tf_vars, 'enformer/trunk/stem/stem/softmax_pooling/linear/w:0')

  for ind, tower_block in enumerate(pytorch_model.conv_tower):
    tower_bn = tower_block[0][0]
    tower_conv = tower_block[0][2]
    tower_point_bn = tower_block[1].fn[0]
    tower_point_conv = tower_block[1].fn[2]
    tower_attn_pool = tower_block[2]

    conv_path = f'enformer/trunk/conv_tower/conv_tower/conv_tower_block_{ind}/conv_tower_block_{ind}/conv_block/conv_block/conv1_d/'
    bn_path = f'enformer/trunk/conv_tower/conv_tower/conv_tower_block_{ind}/conv_tower_block_{ind}/conv_block/conv_block/batch_norm/'
    point_conv_path = f'enformer/trunk/conv_tower/conv_tower/conv_tower_block_{ind}/conv_tower_block_{ind}/pointwise_conv_block/pointwise_conv_block/conv1_d/'
    point_bn_path = f'enformer/trunk/conv_tower/conv_tower/conv_tower_block_{ind}/conv_tower_block_{ind}/pointwise_conv_block/pointwise_conv_block/batch_norm/'
    attn_pool_path = f'enformer/trunk/conv_tower/conv_tower/conv_tower_block_{ind}/conv_tower_block_{ind}/softmax_pooling/linear/w:0'

    copy_bn(tower_bn, tf_vars, bn_path)
    copy_conv(tower_conv, tf_vars, conv_path)
    copy_bn(tower_point_bn, tf_vars, point_bn_path)
    copy_conv(tower_point_conv, tf_vars, point_conv_path)
    copy_attn_pool(tower_attn_pool, tf_vars, attn_pool_path)

  for ind, transformer_block in enumerate(pytorch_model.transformer):
    attn_ln_path = f'enformer/trunk/transformer/transformer/transformer_block_{ind}/transformer_block_{ind}/mha/mha/layer_norm/'
    ff_ln_path = f'enformer/trunk/transformer/transformer/transformer_block_{ind}/transformer_block_{ind}/mlp/mlp/layer_norm/'

    # https://github.com/deepmind/deepmind-research/blob/master/enformer/enformer.py#L119
    # needs to be edited to snt.Linear(channels * 2, name = 'project_in') and snt.Linear(channels, name = 'project_out') or variables are not accessible
    ff_linear1_path = f'enformer/trunk/transformer/transformer/transformer_block_{ind}/transformer_block_{ind}/mlp/mlp/project_in/'
    ff_linear2_path = f'enformer/trunk/transformer/transformer/transformer_block_{ind}/transformer_block_{ind}/mlp/mlp/project_out/'

    attn = transformer_block[0]
    attn_ln = attn.fn[0]

    ff = transformer_block[-1]
    ff_ln = ff.fn[0]
    ff_linear1 = ff.fn[1]
    ff_linear2 = ff.fn[4]

    copy_ln(attn_ln, tf_vars, attn_ln_path)

    copy_ln(ff_ln, tf_vars, ff_ln_path)
    copy_linear(ff_linear1, tf_vars, ff_linear1_path)
    copy_linear(ff_linear2, tf_vars, ff_linear2_path)

  final_bn = pytorch_model.final_pointwise[1][0]
  final_conv = pytorch_model.final_pointwise[1][2]

  copy_bn(final_bn, tf_vars, 'enformer/trunk/final_pointwise/final_pointwise/conv_block/conv_block/batch_norm/')
  copy_conv(final_conv, tf_vars, 'enformer/trunk/final_pointwise/final_pointwise/conv_block/conv_block/conv1_d/')

  human_linear = pytorch_model._heads['human'][0]
  mouse_linear = pytorch_model._heads['mouse'][0]

  copy_linear(human_linear, tf_vars, 'enformer/heads/head_human/head_human/linear/')
  copy_linear(mouse_linear, tf_vars, 'enformer/heads/head_mouse/head_mouse/linear/')

  print('success')

# copy_tf_to_pytorch(tf_model, pytorch_model)
