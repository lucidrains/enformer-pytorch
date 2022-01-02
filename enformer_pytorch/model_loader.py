import os
import torch
import yaml
from pathlib import Path
from enformer_pytorch.enformer_pytorch import Enformer

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def remove_nones(d):
    return dict((k, v) for k, v in d.items() if exists(v))

# make gdown optional dep

try:
    import gdown
except ImportError:
    print('unable to import gdown - please `pip install gdown` before using enformer pretraining models')

# constants

CACHE_PATH = default(os.getenv('CACHE_PATH'), os.path.expanduser('~/.cache.enformer'))
CONFIG_PATH = Path(__file__).parents[0] / 'config.yml'

with open(str(CONFIG_PATH)) as stream:    
    CONFIG = yaml.safe_load(stream)

# functions

def load_pretrained_model(
    slug,
    force = False,
    model = None,
    verbose = True,
    **kwargs
):
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

    override_params = remove_nones(kwargs)
    params = {**config['params'], **override_params}

    if not exists(model):
        model = Enformer(**config['params'])
    else:
        assert len(kwargs) == 0, 'you are trying to override enformer parameters, but you are already passing a reference to an instantiated enformer model'

    model.load_state_dict(torch.load(str(save_path)))

    if verbose and 'description' in config:
        print(f"\n{config['description']}\n")

    print(f'Enformer model "{slug}" loaded successfully')
    return model
