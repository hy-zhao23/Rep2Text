from pathlib import Path
from .settings import p_args
import os

_file_label = f"l{p_args.data.max_len}_o{p_args.data.overlap_len}"

pt_data_idx_path = Path(p_args.path.data_dir) / f"pretrain_index_{p_args.data.train_size}_{_file_label}.pkl"
ft_data_idx_path = Path(p_args.path.data_dir) / f"finetune_index_{p_args.data.train_size}_{_file_label}.pkl"

# Paths for pretrain train/val/test splits
pt_train_idx_path = Path(p_args.path.data_dir) / f"pretrain_train_index_{p_args.data.train_size}_{_file_label}.pkl"
pt_val_idx_path = Path(p_args.path.data_dir) / f"pretrain_val_index_{p_args.data.train_size}_{_file_label}.pkl"
pt_test_idx_path = Path(p_args.path.data_dir) / f"pretrain_test_index_{p_args.data.train_size}_{_file_label}.pkl"

# Paths for finetune train/val/test splits
ft_train_idx_path = Path(p_args.path.data_dir) / f"finetune_train_index_{p_args.data.train_size}_{_file_label}.pkl"
ft_val_idx_path = Path(p_args.path.data_dir) / f"finetune_val_index_{p_args.data.train_size}_{_file_label}.pkl"
ft_test_idx_path = Path(p_args.path.data_dir) / f"finetune_test_index_{p_args.data.train_size}_{_file_label}.pkl"

def get_train_checkpoint_dir():
    return os.path.join(p_args.path.checkpoint_dir, f"{os.environ.get('SLURM_JOB_ID', 'nojobid')}")

def get_gpt_scores_path(split: str, model_type: str, epoch: int = None) -> str:
    _dir = os.path.join(p_args.path.result_dir, os.getenv('TRAIN_TAG', 'NA'))
    if epoch is not None:
        return os.path.join(_dir, f"{split}_{model_type}_epoch{epoch}_gpt_scores.json")
    return os.path.join(_dir, f"{split}_{model_type}_gpt_scores.json")

def get_decoded_path(split: str, model_type: str, epoch: int = None) -> str:
    _dir = os.path.join(p_args.path.result_dir, os.getenv('TRAIN_TAG', 'NA'))
    if epoch is not None:
        return os.path.join(_dir, f"{split}_{model_type}_epoch{epoch}_decoded.json")
    return os.path.join(_dir, f"{split}_{model_type}_decoded.json")

def get_metrics_path(split: str, model_type: str, epoch: int = None) -> str:
    _dir = os.path.join(p_args.path.result_dir, os.getenv('TRAIN_TAG', 'NA'))
    if epoch is not None:
        return os.path.join(_dir, f"{split}_{model_type}_epoch{epoch}_metrics.json")
    return os.path.join(_dir, f"{split}_{model_type}_metrics.json")

def get_metrics_sum_path(split: str, model_type: str, epoch: int = None) -> str:
    _dir = os.path.join(p_args.path.result_dir, os.getenv('TRAIN_TAG', 'NA'))
    if epoch is not None:
        return os.path.join(_dir, f"{split}_{model_type}_epoch{epoch}_metrics_summary.json")
    return os.path.join(_dir, f"{split}_{model_type}_metrics_summary.json")


def _get_adapter_dir():
    _label = os.getenv('TRAIN_TAG', 'NA')
    if _label != 'NA':
        return os.path.join(p_args.path.adapter_dir, _label)
    else:
        return os.path.join(p_args.path.adapter_dir)