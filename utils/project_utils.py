import torch
from utils.core.save_states import _get_projector
from utils import logger, p_args, gh
import os

import socket
import time
import torch.distributed as dist

def _dbg_bcast(t: torch.Tensor, name: str, src: int = 0):
    r = dist.get_rank()
    host = socket.gethostname()

    # 关键断言：早崩比挂死好
    assert t.is_cuda, f"{name} is not CUDA on rank {r}: {t.device}"
    assert t.device.index == torch.cuda.current_device(), \
        f"{name} on wrong device rank={r}: {t.device}, current={torch.cuda.current_device()}"

    print(f"[{host}][rank={r}] bcast BEGIN {name} shape={tuple(t.shape)} dtype={t.dtype} dev={t.device}", flush=True)
    dist.barrier(device_ids=[torch.cuda.current_device()])

    torch.cuda.synchronize()
    st = time.time()
    dist.broadcast(t, src=src)   # ✅ 注意：这里强制用 WORLD group
    torch.cuda.synchronize()
    dt = time.time() - st

    dist.barrier(device_ids=[torch.cuda.current_device()])
    print(f"[{host}][rank={r}] bcast END   {name} dt={dt:.3f}s sum={float(t.float().sum().item())}", flush=True)



def pre_cache_model(model_id=None):
    """Pre-download and cache the model(s) to avoid download issues during processing."""
    if model_id is None:
        model_ids = [p_args.model.dec_id]
    elif isinstance(model_id, (list, tuple, set)):
        model_ids = list(model_id)
    else:
        model_ids = [model_id]

    hf_home = os.getenv("HF_HOME")
    cache_dir = os.path.join(hf_home, "hub") if hf_home else None
    if cache_dir is None:
        cache_dir = str(p_args.path.weight_dir)

    from huggingface_hub import snapshot_download
    for mid in model_ids:
        logger.info("📥 Pre-caching model %s...", mid)
        snapshot_download(
            repo_id=mid,
            cache_dir=cache_dir,
            local_files_only=False,
        )

    logger.info("✅ Model fully cached.")

def sync_projector_paras(model):
    projector = _get_projector(model)
    if projector is None or not gh.distributed:
        return 
    projector.reset_parameters()

    projector.to(gh.device)
    # Broadcast from rank 0 to double-check all ranks have identical values
    for name, p in projector.named_parameters():
        # gh.broadcast(p.data, src=0)
        _dbg_bcast(p.data, name, src=0)
        if not torch.isfinite(p.data).all():
            raise RuntimeError(f"[RANK:{gh.rank}] Broadcast failed for parameter {name} with non-finite values")
    if gh.is_main:
        logger.info(f"Projector parameters synchronized from rank {0} to all ranks")


def sync_lora_paras(model, src: int = 0):
    """Synchronize LoRA parameters across all ranks"""
    if not gh.distributed:
        return
    
    # Find all LoRA parameters
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_' in name.lower():
            lora_params.append((name, param))
    
    if not lora_params:
        return
    
    # Reset LoRA parameters on ALL ranks with the same seed to ensure consistency
    torch.manual_seed(42)  # Use fixed seed for reproducible initialization
    
    for name, param in lora_params:
        # LoRA-specific initialization based on parameter type
        if 'lora_A' in name.lower():
            # LoRA A matrix: typically initialized with Kaiming uniform
            torch.nn.init.kaiming_uniform_(param, a=0, mode='fan_in', nonlinearity='relu')
        elif 'lora_B' in name.lower():
            # LoRA B matrix: typically initialized to zero
            torch.nn.init.zeros_(param)
        else:
            # Fallback for other LoRA parameters
            if param.dim() >= 2:
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.uniform_(param, -0.1, 0.1)

def set_global_seed(seed: int = 42):
    import random
    import torch
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
