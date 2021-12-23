import os
import torch
from pathlib import Path
from enformer_pytorch.enformer_pytorch import Enformer

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# make gdown optional dep

try:
    import gdown
except ImportError:
    print('unable to import gdown - please `pip install gdown` before using enformer pretraining models')

# constants

CACHE_PATH = default(os.getenv('CACHE_PATH'), os.path.expanduser('~/.cache.enformer'))

CONFIG = dict(
    preview = dict(
        id = '1-5nXoOCcRQxFULV4gac9GikY9j3iO3p3',
        params = dict(
            dim = 1536,
            depth = 11,
            heads = 8,
            output_heads = dict(human = 5313, mouse= 1643),
            target_length = 896
        )
    )
)

# functions

def load_pretrained_model(slug, force = False):
    if slug not in CONFIG:
        print(f'model {slug} not found among available choices: [{", ".join(CONFIG.keys())}]')
        exit()

    config = CONFIG[slug]

    # download model from gdrive

    base_path = Path(CACHE_PATH)
    base_path.mkdir(parents = True, exist_ok = True)

    url = f'https://drive.google.com/uc?id={config["id"]}'
    save_path = base_path / f'{slug}.pt'

    if force or not save_path.exists():
        gdown.download(url, str(save_path), quiet = False)

    # load

    model = Enformer(**config['params'])
    model.load_state_dict(torch.load(str(save_path)))

    print(f'loaded {slug} successfully')
    return model
