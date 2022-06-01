import torch
from enformer_pytorch import Enformer

# download pytorch weights from official set of deepmind enformer weights
# from https://drive.google.com/u/0/uc?id=1sg41meLWKPMaM6hMx4aBWSwlVOfXbe0R

model_path = './official-enformer-rough.pt'

enformer = Enformer.from_hparams().cuda()
enformer.load_state_dict(torch.load(model_path))
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