import torch.nn.functional as F
import torch
import os
import time
import json
import shutil
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

from utils import logger
from utils.models.constants import IGNORE_INDEX
from utils import gh
from utils.core.path import _get_adapter_dir


def setup_train_paras(model) -> None:
    """Setup trainable parameters for the model, optimizing for single-pass iteration."""
    _total_params = 0
    _projector_params = 0
    _lora_params = 0


    actual_model = model.module if hasattr(model, 'module') else model

    # Single pass through parameters for both counting and gradient setup
    if hasattr(actual_model, 'peft_config'):
        for name, param in actual_model.named_parameters():
            param_count = param.numel()
            _total_params += param_count
            
            if 'lora' in name or 'projector' in name.lower():
                param.requires_grad = True
                if 'lora' in name:
                    _lora_params += param_count
                elif 'projector' in name.lower():
                    _projector_params += param_count
            else:
                param.requires_grad = False
    elif hasattr(actual_model, 'model') and hasattr(actual_model.model, 'embedding_projector'):
        projector = actual_model.model.embedding_projector
        
        # First pass: count all parameters and disable gradients
        for param in list(actual_model.parameters()):  # Convert to list
            _total_params += param.numel()
            param.requires_grad = False
        
        # Second pass: enable projector parameters
        for param in projector.parameters():
            param_count = param.numel()
            _projector_params += param_count
            param.requires_grad = True
    else:
        raise ValueError("❌ No embedding projector found in model")
    if gh.is_main:
        if _total_params > 0:
            logger.info(f"📊 Projector parameters: {_projector_params/_total_params*100:.2f}%")
            logger.info(f"📊 LoRA parameters: {_lora_params/_total_params*100:.2f}%")

def custom_loss(model, batch: Dict[str, Any]):
    """Compute custom loss with optional similarity penalty. Optimized version."""
    debug_sync = os.getenv("DEBUG_SYNC", "0") == "1"
    debug_max_calls = int(os.getenv("DEBUG_SYNC_MAX_CALLS", "2"))
    if not hasattr(custom_loss, "_dbg_calls"):
        custom_loss._dbg_calls = {}
    rank = gh.rank
    dbg_calls = custom_loss._dbg_calls.get(rank, 0)
    do_dbg = debug_sync and dbg_calls < debug_max_calls
    ts = time.strftime("%H:%M:%S")
    if do_dbg:
        logger.info(f"[DBG][{ts}][rank={rank}] custom_loss: before model")
    # Forward pass
    outputs = model(
        input_ids=batch['input_ids'],
        labels=batch['labels'],  # Don't compute loss in model
        hidden_embeds=batch['embedding'],
        use_cache=False,  # Disable KV cache for training
        output_hidden_states=False  # Save 10-15GB memory
    )
    if do_dbg:
        logger.info(f"[DBG][{ts}][rank={rank}] custom_loss: after model")
    
    # Compute loss - this creates a tensor with requires_grad=True
    valid_mask = (batch['labels'] != IGNORE_INDEX)
    if not valid_mask.any():
        return torch.tensor(0.0, device=outputs.logits.device, requires_grad=True)
    
    logits_flat = outputs.logits[valid_mask]
    labels_flat = batch['labels'][valid_mask]
    
    # This loss tensor has requires_grad=True and can be backpropagated
    loss = F.cross_entropy(logits_flat, labels_flat, reduction="mean")
    
    outputs.loss = loss
    if do_dbg:
        logger.info(f"[DBG][{ts}][rank={rank}] custom_loss: after loss")
        custom_loss._dbg_calls[rank] = dbg_calls + 1
    return outputs    


# ------------------------------
# 🧠 Checkpoint helpers (fine-tune)
# ------------------------------
def resume_from_latest_if_enabled(
    enabled: bool = True,
    fine_tune: bool = False,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
) -> Tuple[int, int]:
    """Load latest checkpoint if enabled; return (start_epoch, global_step)."""
    if not enabled:
        return 0, 0

    save_dir = os.path.join(_get_adapter_dir(), f"checkpoints")
    if fine_tune:
        save_dir = os.path.join(_get_adapter_dir(), f"lora_checkpoints")

    if not os.path.exists(save_dir):
        if gh.is_main:
            logger.info("No existing checkpoint found; starting from epoch 0")
        return 0, 0

    if gh.is_main:
        logger.info(f"🔄 Resuming from checkpoint: {save_dir}")

    if gh.accelerator is not None:
        gh.accelerator.load_state(save_dir)
    else:
        if model is None:
            if gh.is_main:
                logger.warning("⚠️ model is None; cannot load checkpoint without accelerator")
            return 0, 0

        rank_ckpt = os.path.join(save_dir, f"rank_{gh.rank}.pt")
        if not os.path.exists(rank_ckpt):
            if gh.is_main:
                logger.warning(f"⚠️ Rank checkpoint not found: {rank_ckpt}; starting from scratch")
            return 0, 0

        ckpt = torch.load(rank_ckpt, map_location="cpu")
        # strict=False: trainable-only checkpoints omit the frozen backbone,
        # which is already loaded from the base model
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if gh.is_main:
            logger.info(f"🔄 Loaded {len(ckpt['model'])} tensors "
                        f"({len(missing)} frozen keys taken from base model)")
        if unexpected and gh.is_main:
            logger.warning(f"⚠️ Unexpected keys in checkpoint: {unexpected[:5]}...")

        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])

        if scheduler is not None and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])

    # Try to get epoch from meta.json first, fallback to directory name parsing
    meta_path = os.path.join(save_dir, "my_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                return int(meta.get("epoch", 0)), int(meta.get("global_step", 0))
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fall through to directory name parsing
            return 0, 0
    return 0, 0

def save_step_checkpoint(
    epoch: int,
    global_step: int,
    fine_tune: bool = False,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
) -> None:
    """Save accelerate state and meta.json for a specific epoch. Returns checkpoint dir."""
    save_dir = os.path.join(_get_adapter_dir(), f"checkpoints")
    if fine_tune:
        save_dir = os.path.join(_get_adapter_dir(), f"lora_checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    
    if gh.accelerator is not None:
        gh.accelerator.save_state(save_dir)
    else:
        if model is None:
            if gh.is_main:
                logger.warning("⚠️ model is None; skip checkpoint save without accelerator")
            return

        rank_ckpt = os.path.join(save_dir, f"rank_{gh.rank}.pt")
        # Only persist trainable params (+ buffers, which are tiny); the frozen
        # backbone is reloaded from the base model on resume
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        buffers = {n for n, _ in model.named_buffers()}
        state = {
            "model": {k: v for k, v in model.state_dict().items()
                      if k in trainable or k in buffers},
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        }
        torch.save(state, rank_ckpt)

        if gh.distributed:
            gh.barrier()
    
    meta = {
        "epoch": epoch,
        "global_step": global_step
    }
    if gh.is_main:
        with open(os.path.join(save_dir, "my_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        logger.info(f"💾 Saved checkpoint at {global_step} to {save_dir}")