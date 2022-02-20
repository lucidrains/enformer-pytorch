from enformer_pytorch import EnformerConfig, Enformer, load_pretrained_model

model = Enformer.from_pretrained("nielsr/enformer-preview")

# config = EnformerConfig()

# model = Enformer(config)

# # load pretrained weights
# model = load_pretrained_model('preview')

# # push to hub
# model.push_to_hub("enformer-preview", organization="nielsr", commit_message="Add model", use_temp_dir=True)