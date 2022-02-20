from enformer_pytorch import EnformerConfig, Enformer, load_pretrained_model

# load a model as follows

model = Enformer.from_pretrained("nielsr/enformer-preview")

# the code below was used to upload model weights to the hub

# model_name = 'corr_coef_obj'

# config = EnformerConfig()

# model = Enformer(config)

# # load pretrained weights
# model = load_pretrained_model(model_name)

# # push to hub
# model.push_to_hub(f"enformer-{model_name}", organization="nielsr", commit_message="Add model", use_temp_dir=True)