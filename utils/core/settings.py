import os
from pathlib import Path
import yaml
from dataclasses import dataclass
from typing import Optional

PROJ_DIR = Path(os.getenv('PROJ', '.'))

def _load_config(file):
    """Load the setup from the config file specified by EINV_CONFIG environment variable."""
    config_file = os.getenv('EINV_CONFIG', file)
    config_path = PROJ_DIR / "configs" / config_file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def parse_layers(layers: str):
    parsed_layers = []
    for layer in layers.split(','):
        if "-" in layer:
            start, end = layer.split('-')
            parsed_layers.extend(range(int(start), int(end) + 1))
        else:
            parsed_layers.append(int(layer))
    parsed_layers.sort()
    return parsed_layers

config = _load_config('setup.yaml')


def _model_label_from_id(model_id: Optional[str]) -> str:
    label = {
        "meta-llama/Llama-3.1-8B": "llama8B",
        "meta-llama/Llama-3.2-3B": "llama3B",
        "mistralai/Mistral-7B-v0.1": "mistral7B",
        "google/gemma-7b": "gemma7B",
        "Qwen/Qwen2.5-32B": "qwen32B",
    }
    if model_id in label:
        return label[model_id]
    raise ValueError(f"Unknown model_id: {model_id}. Please add a label in _model_label_from_id.")

@dataclass
class ModelParams:
    dec_id: str
    emb_id: str
    model_name: str
    embedding_layer: Optional[int]

@dataclass
class ProjSettings:
    hf_token: Optional[str]
    openai_key: Optional[str]

@dataclass
class DatasetParams:
    data_name: str
    max_len: int
    overlap_len: int
    train_size: int
    embedding_batch_size: int
    save_every: int
    pretrain_ratio: float
    max_seq_len: Optional[int]
    gen_max_len: Optional[int]
    shuffle_pairing: bool = False
    shuffle_pairing_seed: int = 42

@dataclass
class ProjConfig:
    n_vecs: int
    ei_proj_dropout: float
    ei_proj_expansion: float
    pad_side: str
    use_patch_token: bool

@dataclass
class ProjPath:
    model_name: str
    setting_name: str
    tmp_dir: Path
    data_dir: Path
    prompt_dir: Path
    model_dir: Path
    result_dir: Path
    embedding_dir: Path
    fig_dir: Path
    weight_dir: Path
    checkpoint_dir: Path
    adapter_dir: Path
    config_dir: Path
    token_dir: Path

@dataclass
class ProjArgs:
    path: ProjPath
    settings: ProjSettings
    model: ModelParams
    data: DatasetParams
    proj: ProjConfig

# --- Build model params ---
models_cfg = config['models']

dec_id = models_cfg.get('dec_id')
emb_id = models_cfg.get('emb_id')
if dec_id is None or emb_id is None:
    raise ValueError("dec_id and emb_id must be specified in setup.yaml.")

model_name = models_cfg.get('name') or _model_label_from_id(emb_id)
embedding_layer = models_cfg.get('emb_layer')
if embedding_layer is None:
    embedding_layer = config.get(model_name, {}).get('embedding_layer')

model = ModelParams(
    dec_id=dec_id,
    emb_id=emb_id,
    model_name=model_name,
    embedding_layer=embedding_layer,
)

# --- Build dataset params ---
params_cfg = config['params']

data = DatasetParams(
    data_name=params_cfg['train_data'],
    max_len=params_cfg['max_len'],
    overlap_len=params_cfg['overlap_len'],
    train_size=params_cfg['train_size'],
    embedding_batch_size=params_cfg['embedding_batch_size'],
    save_every=params_cfg['save_every'],
    pretrain_ratio=params_cfg['pretrain_ratio'],
    max_seq_len=params_cfg.get('max_seq_len'),
    gen_max_len=params_cfg.get('gen_max_len'),
    shuffle_pairing=bool(params_cfg.get('shuffle_pairing', False)),
    shuffle_pairing_seed=int(params_cfg.get('shuffle_pairing_seed', 42)),
)

# --- Build projector params ---
proj_cfg = config['projector']

proj = ProjConfig(
    n_vecs=int(proj_cfg['n_vecs']),
    ei_proj_dropout=float(proj_cfg['ei_proj_dropout']),
    ei_proj_expansion=float(proj_cfg['ei_proj_expansion']),
    pad_side=proj_cfg['pad_side'],
    use_patch_token=proj_cfg['use_patch_token'],
)

# --- Build paths ---
emb_name = models_cfg.get('emb_name') or _model_label_from_id(model.emb_id)
layer_label = f"l{model.embedding_layer}"
token_label = f"t{data.max_len}"
projector_label = f"p{proj.n_vecs}"
label = f"{layer_label}{token_label}{projector_label}"
setting_name = os.path.join(model.model_name, label)

paths_cfg = config['dir']

tmp_dir = Path(PROJ_DIR / paths_cfg['tmp_dir'])
data_dir = Path(PROJ_DIR / paths_cfg['data_dir'] / params_cfg['train_data'])
prompt_dir = Path(PROJ_DIR / paths_cfg['data_dir'] / paths_cfg['prompt_dir'])
model_dir = Path(models_cfg.get('path', config.get(model.model_name, {}).get('path', '')))

result_dir = Path(PROJ_DIR / paths_cfg['result_dir'] / setting_name)
embedding_dir = Path(PROJ_DIR / paths_cfg['embedding_dir'] / emb_name / f"{layer_label}{token_label}")
fig_dir = Path(PROJ_DIR / paths_cfg['fig_dir'] / setting_name)
weight_dir = Path(PROJ_DIR / paths_cfg['weight_dir'])
checkpoint_dir = Path(PROJ_DIR / paths_cfg['checkpoint_dir'] / setting_name)
adapter_dir = Path(PROJ_DIR / "weights" / setting_name)
config_dir = Path(PROJ_DIR / "configs")

# Isolate tokens by embedding model to avoid cross-tokenizer contamination.
token_dir = Path(PROJ_DIR / paths_cfg['token_dir'] / emb_name)

path = ProjPath(
    model_name=model.model_name,
    setting_name=setting_name,
    tmp_dir=tmp_dir,
    data_dir=data_dir,
    prompt_dir=prompt_dir,
    model_dir=model_dir,
    result_dir=result_dir,
    embedding_dir=embedding_dir,
    fig_dir=fig_dir,
    weight_dir=weight_dir,
    checkpoint_dir=checkpoint_dir,
    adapter_dir=adapter_dir,
    config_dir=config_dir,
    token_dir=token_dir,
)

settings = ProjSettings(
    hf_token=os.getenv('HF_API_KEY'),
    openai_key=os.getenv('OPENAI_API_KEY'),
    # openai_key=os.getenv('OPENAI_API_KEY_MY'),
)

p_args = ProjArgs(
    path=path,
    settings=settings,
    model=model,
    data=data,
    proj=proj,
)
