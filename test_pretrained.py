import torch
from enformer_pytorch import from_pretrained

enformer = from_pretrained('EleutherAI/enformer-official-rough', use_tf_gamma = False).cuda()
enformer.eval()

data = torch.load('./data/test-sample.pt')
seq, target = data['sequence'].cuda(), data['target'].cuda()

with torch.no_grad():
    corr_coef = enformer(
        seq,
        target = target,
        return_corr_coef = True,
        head = 'human'
    )

print(corr_coef)
assert corr_coef > 0.1
