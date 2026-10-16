import torch
import torch.nn as nn
import re
from utils import logger, gh 
import math
from torch.optim.lr_scheduler import LambdaLR

def _build_param_groups_filtered(model, include_fn):
    """
    Same as _build_param_groups but only includes params where include_fn(name, param) is True.
    """
    decay, no_decay = [], []

    norm_module_types = (
        nn.LayerNorm,
        getattr(nn, "RMSNorm", tuple()),
        nn.GroupNorm,
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    )
    embedding_types = (nn.Embedding,)

    norm_param_ids = set()
    emb_param_ids  = set()
    for module in model.modules():
        if isinstance(module, norm_module_types):
            for p in module.parameters(recurse=False):
                norm_param_ids.add(id(p))
        if isinstance(module, embedding_types):
            for p in module.parameters(recurse=False):
                emb_param_ids.add(id(p))

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if not include_fn(name, param):
            continue

        is_bias = name.endswith(".bias")
        is_norm = id(param) in norm_param_ids or ("norm" in name.lower() or "layernorm" in name.lower() or "ln" in name.lower())
        is_emb  = id(param) in emb_param_ids or ("embedding" in name.lower() or "embeddings" in name.lower())

        if is_bias or is_norm or is_emb:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": 0.0},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def _is_proj_param(name: str) -> bool:
    # EInv projector lives under 'embedding_projector'
    return "embedding_projector" in name

def _is_lora_param(name: str) -> bool:
    # PEFT LoRA params usually contain 'lora_' (e.g., lora_A, lora_B)
    return "lora_" in name

def _extract_layer_number(param_name: str):
    """
    Extract layer number from parameter name.
    Examples:
      - "model.layers.5.self_attn.q_proj.lora_A" -> 5
      - "base_model.model.model.layers.23.mlp.gate_proj.lora_B" -> 23
    Returns None if no layer number found.
    """
    match = re.search(r'\.layers\.(\d+)\.', param_name)
    if match:
        return int(match.group(1))
    return None

def _build_layer_stratified_lora_groups(model, layer_lr_config, base_lr, base_wd):
    """
    Build parameter groups with different LRs for different layer ranges.
    
    Args:
        model: The model
        layer_lr_config: Dict with layer groups, e.g.:
            {
                "early": {"layers": [0, 15], "multiplier": 2.5},
                "mid": {"layers": [11, 21], "multiplier": 1.0},
                "late": {"layers": [22, 31], "multiplier": 0.5}
            }
        base_lr: Base learning rate
        base_wd: Base weight decay
    
    Returns:
        List of parameter groups
    """
    norm_module_types = (
        nn.LayerNorm,
        getattr(nn, "RMSNorm", tuple()),
        nn.GroupNorm,
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    )
    embedding_types = (nn.Embedding,)

    norm_param_ids = set()
    emb_param_ids = set()
    for module in model.modules():
        if isinstance(module, norm_module_types):
            for p in module.parameters(recurse=False):
                norm_param_ids.add(id(p))
        if isinstance(module, embedding_types):
            for p in module.parameters(recurse=False):
                emb_param_ids.add(id(p))

    # Create storage for each layer group
    layer_groups = {}
    for group_name, group_cfg in layer_lr_config.items():
        layer_groups[group_name] = {
            "decay": [],
            "no_decay": [],
            "config": group_cfg
        }
    
    # Collect ungrouped params (LoRA params not in any layer)
    ungrouped_decay = []
    ungrouped_no_decay = []

    # Categorize all LoRA parameters
    for name, param in model.named_parameters():
        if not param.requires_grad or not _is_lora_param(name):
            continue

        layer_num = _extract_layer_number(name)
        
        # Determine if this param should decay
        is_bias = name.endswith(".bias")
        is_norm = id(param) in norm_param_ids or ("norm" in name.lower() or "layernorm" in name.lower() or "ln" in name.lower())
        is_emb = id(param) in emb_param_ids or ("embedding" in name.lower() or "embeddings" in name.lower())
        should_decay = not (is_bias or is_norm or is_emb)

        # Find which layer group this belongs to
        assigned = False
        if layer_num is not None:
            for group_name, group_data in layer_groups.items():
                layer_range = group_data["config"]["layers"]
                if layer_range[0] <= layer_num <= layer_range[1]:
                    if should_decay:
                        group_data["decay"].append(param)
                    else:
                        group_data["no_decay"].append(param)
                    assigned = True
                    break
        
        # If not assigned to any group, add to ungrouped
        if not assigned:
            if should_decay:
                ungrouped_decay.append(param)
            else:
                ungrouped_no_decay.append(param)

    # Build final parameter groups
    param_groups = []
    
    for group_name, group_data in layer_groups.items():
        group_cfg = group_data["config"]
        lr_multiplier = group_cfg.get("multiplier", 1.0)
        group_lr = base_lr * lr_multiplier
        group_wd = group_cfg.get("weight_decay", base_wd)
        
        if group_data["decay"]:
            param_groups.append({
                "params": group_data["decay"],
                "lr": group_lr,
                "weight_decay": group_wd,
                "name": f"lora_{group_name}_decay"
            })
        if group_data["no_decay"]:
            param_groups.append({
                "params": group_data["no_decay"],
                "lr": group_lr,
                "weight_decay": 0.0,
                "name": f"lora_{group_name}_no_decay"
            })
    
    # Add ungrouped params with base LR
    if ungrouped_decay:
        param_groups.append({
            "params": ungrouped_decay,
            "lr": base_lr,
            "weight_decay": base_wd,
            "name": "lora_ungrouped_decay"
        })
    if ungrouped_no_decay:
        param_groups.append({
            "params": ungrouped_no_decay,
            "lr": base_lr,
            "weight_decay": 0.0,
            "name": "lora_ungrouped_no_decay"
        })

    return param_groups

def get_linear_schedule_with_warmup_and_min_lr(
    optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.0, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = (current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_cosine_schedule_with_warmup_and_min_lr(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    last_epoch: int = -1,
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warm-up from 0 → 1
            return float(current_step) / float(max(1, num_warmup_steps))

        # progress from 0 → 1 for decay phase
        progress = (current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )

        # cosine from 1 → 0, then scale to [min_lr_ratio, 1.0]
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def proj_optimizer(model, config):
    """
    Create AdamW for projector (and only projector) parameters.
    Uses training.learning_rate, training.weight_decay, training.betas, training.eps.
    """
    train_cfg = config.get("training", {})
    lr    = float(train_cfg.get("learning_rate"))
    wd    = float(train_cfg.get("weight_decay", 0.01))
    betas_cfg = train_cfg.get("betas", (0.9, 0.95))
    betas = (float(betas_cfg[0]), float(betas_cfg[1])) if isinstance(betas_cfg, (list, tuple)) else (0.9, 0.95)
    eps   = float(train_cfg.get("eps", 1e-8))

    param_groups = _build_param_groups_filtered(model, lambda n, p: _is_proj_param(n))
    # apply hyperparams
    param_groups[0]["weight_decay"] = wd
    for g in param_groups:
        g["lr"] = lr

    try:
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps, fused=True)  # type: ignore[arg-type]
        fused_used = True
    except TypeError:
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)
        fused_used = False

    if logger and gh.is_main:
        n_decay = sum(p.numel() for p in param_groups[0]["params"]) if len(param_groups) > 0 else 0
        n_nodec = sum(p.numel() for p in param_groups[1]["params"]) if len(param_groups) > 1 else 0
        logger.info("📊 Created AdamW (projector)")
        logger.info(f"   LR={lr}  WD(decay group)={wd}  betas={betas}  eps={eps}  fused={fused_used}")
        logger.info(f"   Params: decay={n_decay:,}  no_decay={n_nodec:,}")

    return optimizer

def lora_optimizer(model, config):
    """
    Create AdamW for LoRA parameters only. Returns None if no LoRA params are found.
    Supports layer-stratified learning rates via training.lora.layer_lr_strategy.
    """
    train_cfg = config.get("training", {})
    lora_cfg = train_cfg.get("lora", {})
    if lora_cfg is None:
        return None

    base_lr = float(lora_cfg.get("lr", train_cfg.get("learning_rate")))
    base_wd = float(lora_cfg.get("weight_decay", 0.0))
    betas_cfg = train_cfg.get("betas", (0.9, 0.95))
    betas = (float(betas_cfg[0]), float(betas_cfg[1])) if isinstance(betas_cfg, (list, tuple)) else (0.9, 0.95)
    eps = float(train_cfg.get("eps", 1e-8))

    # Check if layer-stratified LR is configured
    layer_lr_config = lora_cfg.get("layer_lr_strategy", None)
    
    if layer_lr_config is None:
        # Original behavior: single LR for all LoRA params
        param_groups = _build_param_groups_filtered(model, lambda n, p: _is_lora_param(n))
        
        if all(len(g["params"]) == 0 for g in param_groups):
            return None
        
        param_groups[0]["weight_decay"] = base_wd
        for g in param_groups:
            g["lr"] = base_lr
    else:
        # Layer-stratified LRs
        param_groups = _build_layer_stratified_lora_groups(
            model, 
            layer_lr_config, 
            base_lr, 
            base_wd
        )
        
        if len(param_groups) == 0:
            return None

    try:
        optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps, fused=True)  # type: ignore[arg-type]
        fused_used = True
    except TypeError:
        optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
        fused_used = False

    if logger and gh.is_main:
        if layer_lr_config is None:
            # Original logging
            n_decay = sum(p.numel() for p in param_groups[0]["params"]) if len(param_groups) > 0 else 0
            n_nodec = sum(p.numel() for p in param_groups[1]["params"]) if len(param_groups) > 1 else 0
            logger.info("📊 Created AdamW (LoRA)")
            logger.info(f"   LR={base_lr}  WD(decay group)={base_wd}  betas={betas}  eps={eps}  fused={fused_used}")
            logger.info(f"   Params: decay={n_decay:,}  no_decay={n_nodec:,}")
        else:
            # Layer-stratified logging
            logger.info("📊 Created AdamW (LoRA) with layer-stratified LRs")
            logger.info(f"   Base LR={base_lr}  Base WD={base_wd}  betas={betas}  eps={eps}  fused={fused_used}")
            for g in param_groups:
                n_params = sum(p.numel() for p in g["params"])
                group_name = g.get("name", "unknown")
                logger.info(f"   {group_name}: LR={g['lr']:.2e}  WD={g.get('weight_decay', 0.0)}  Params={n_params:,}")

    return optimizer