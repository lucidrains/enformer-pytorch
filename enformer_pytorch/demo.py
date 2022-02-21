from enformer_pytorch import EnformerConfig, Enformer
# from enformer_pytorch import load_pretrained_model

# load a model as follows

model = Enformer.from_pretrained('EleutherAI/enformer-preview', target_length = 128, dropout_rate = 0.1)

# for name, param in model.named_parameters():
#     print(name, param.shape)

# the code below was used to upload model weights to the hub

# model_name = '191k_corr_coef_obj'

# config = EnformerConfig()

# model = Enformer(config)

# # load pretrained weights
# model = load_pretrained_model(model_name)

# # push to hub
# model.push_to_hub(f"enformer-{model_name}", organization="nielsr", commit_message="Add model", use_temp_dir=True)