import torch
import os
import time
import torch.nn as nn
import wandb
from typing import Any, Optional
from utils import logger, gh, p_args
from collections import deque
from .optimizers import proj_optimizer, lora_optimizer, get_cosine_schedule_with_warmup_and_min_lr
from .train_utils import custom_loss, save_step_checkpoint
from utils.eval.test_utils import compute_and_log_token_topk

def total_batch_size(config):
    total_batch_size = int(config['training']['batch_size']) * int(config['training']['grad_accum_steps']) * gh.world_size

    if gh.is_main:
        logger.info(f"📊 Total effective batch size: {total_batch_size}")
    return total_batch_size

def prepare_model_and_optimizer(model, config, total_steps):
    """
    Prepare the model, optimizer, and scheduler for training with Accelerate.
    Returns: model, optimizer, scheduler
    """
    # Create optimizers
    proj_opt = proj_optimizer(model, config)
    lora_opt = lora_optimizer(model, config)

    w_steps = int(total_steps * config['training'].get('warmup_ratio', 0.05))
    lora_w_steps = int(total_steps * config['training'].get('lora', {}).get('warmup_ratio', 0.05))

    def _make_scheduler(opt, steps, min_lr_ratio):
        return get_cosine_schedule_with_warmup_and_min_lr(opt, num_warmup_steps=steps, num_training_steps=total_steps, min_lr_ratio=min_lr_ratio)

    if lora_opt is not None:
        min_lr_ratio = config['training'].get('min_lr_ratio', 0.15)
        optimizer = [proj_opt, lora_opt]
        scheduler = [_make_scheduler(proj_opt, w_steps, min_lr_ratio), _make_scheduler(lora_opt, lora_w_steps, min_lr_ratio)]
    else:
        min_lr_ratio = config['training'].get('min_lr_ratio', 0.15)
        optimizer = proj_opt
        scheduler = _make_scheduler(proj_opt, w_steps, min_lr_ratio)

    if gh.is_main:
        logger.info(f"📊 Total training steps: {total_steps}")
        logger.info(f"📊 Warmup steps: {w_steps}")
        if lora_opt is not None:
            logger.info(f"📊 LoRA warmup steps: {lora_w_steps}")
        logger.info(f"📊 Scheduler type: warmup -> cosine (min_lr_ratio)")
    
    return model, optimizer, scheduler

def _trainable_params(module: nn.Module):
    for p in module.parameters():
        if p.requires_grad and p.grad is not None:
            yield p

@torch.no_grad()
def _run_val_eval(model, val_loader, config) -> float:
    """Run forward-only loss eval over up to val_max_samples; returns gathered mean val_loss."""
    if val_loader is None:
        return float("nan")
    val_max_samples = int(config['training'].get('val_max_samples', 5000))
    per_rank_batch = int(config['training']['batch_size'])
    max_batches = max(1, val_max_samples // (per_rank_batch * gh.world_size))

    was_training = model.training
    model.eval()
    losses = []
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        if not isinstance(batch, dict) or any(k not in batch for k in ('input_ids', 'labels', 'embedding')):
            continue
        if batch['input_ids'].shape[0] == 0:
            continue
        with gh.accelerator.autocast():
            out = custom_loss(model, batch)
        loss = out.loss.detach()
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        gathered = gh.accelerator.gather(loss).mean()
        losses.append(gathered)
    if was_training:
        model.train()
    if not losses:
        return float("nan")
    return float(torch.stack(losses).mean().item())


def train_epoch(
    model,
    train_loader,
    optimizer: Any,
    scheduler: Any,
    epoch,
    global_step,
    config,
    fine_tune: bool = False,
    val_loader=None,
    es_state: Optional[dict] = None,
    best_tracker=None,
) -> None:
    model.train()

    total_loss = 0.0
    batch_means = []
    steps_per_epoch = len(train_loader) // config['training']['grad_accum_steps']
    _epochs = (global_step + 1) // steps_per_epoch
    # when resuming at the end step of an epoch
    if _epochs > epoch:
        return global_step

    _optimizer_step = max((global_step+1) % steps_per_epoch-1, 0)

    log_every = int(config['logging'].get('log_every', 50))
    max_grad_norm = float(config['training']['max_grad_norm'])
    
    recent_losses = deque(maxlen=int(config["logging"].get("recent_window", 20)))
    debug_sync = os.getenv("DEBUG_SYNC", "0") == "1"

    def _dbg(msg: str) -> None:
        if not debug_sync:
            return
        ts = time.strftime("%H:%M:%S")
        logger.info(f"[DBG][{ts}][rank={gh.rank}] {msg}")

    # tqdm only on main process
    progress_bar: Optional[Any] = None
    if gh.is_main:
        from tqdm import tqdm
        progress_bar = tqdm(
            total=len(train_loader) // config['training']['grad_accum_steps'],
            initial=_optimizer_step,  # start at resume point (e.g. 199 if resumed mid-epoch)
            desc=f"Epoch {epoch}",
            miniters=int(config['logging'].get('progress_update_freq', 50)),
            mininterval=1e9,  # disable time-based refresh; only step-count triggers
        )

    # An OOM that repeats forever is indistinguishable from healthy training at
    # the SLURM level: the job stays RUNNING, burns its walltime and produces
    # nothing. Bail out instead of swallowing the exception indefinitely.
    consecutive_oom = 0
    MAX_CONSECUTIVE_OOM = 30

    for _step, batch in enumerate(train_loader):
        try:
            if debug_sync and global_step == 0 and _step < config['training']['grad_accum_steps']:
                _dbg(f"enter step _step={_step} accum_end={((_step + 1) % config['training']['grad_accum_steps'] == 0)}")
            if (_step + 1) // config['training']['grad_accum_steps'] < _optimizer_step:
                continue
            # Basic batch checks
            if not isinstance(batch, dict) or any(k not in batch for k in ('input_ids', 'labels', 'embedding')):
                if gh.is_main:
                    logger.error(f"❌ Invalid batch format at global step {_step}")
                continue
            if batch['input_ids'].shape[0] == 0:
                if gh.is_main:
                    logger.warning(f"⚠️ Skipping empty batch at global step {_step}")
                continue

            # ----- Forward + backward with Accelerate -----
            with gh.accelerator.accumulate(model):
                if debug_sync and global_step == 0 and _step < config['training']['grad_accum_steps']:
                    _dbg("before forward")
                with gh.accelerator.autocast():
                    outputs: Any = custom_loss(model, batch)
                    raw_loss = outputs.loss.detach()
                if debug_sync and global_step == 0 and _step < config['training']['grad_accum_steps']:
                    _dbg("after forward")

                # Skip bad losses
                if torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
                    if gh.is_main:
                        logger.warning(f"⚠️ Skipping global step {_step} due to NaN/Inf loss")
                    continue

                if debug_sync and global_step == 0 and _step < config['training']['grad_accum_steps']:
                    _dbg("before backward")
                gh.accelerator.backward(outputs.loss)
                if debug_sync and global_step == 0 and _step < config['training']['grad_accum_steps']:
                    _dbg("after backward")
                
                # Step only when gradients are synced (end of accumulation)
                if gh.accelerator.sync_gradients:
                    gh.accelerator.clip_grad_norm_(_trainable_params(model), max_grad_norm)

                    if debug_sync and global_step == 0:
                        _dbg("before gather")
                    gathered_loss = gh.accelerator.gather(raw_loss).mean()
                    if debug_sync and global_step == 0:
                        _dbg("after gather")

                    # Step optimizers and schedulers
                    if isinstance(optimizer, list):
                        for opt, sch in zip(optimizer, scheduler if isinstance(scheduler, list) else [scheduler]):
                            opt.step()
                            sch.step()
                        for opt in optimizer:
                            opt.zero_grad()
                    else:
                        optimizer.step()
                        scheduler.step()  # ← Single call per accumulation cycle
                        optimizer.zero_grad()

                    recent_losses.append(float(gathered_loss.item()))
                    total_loss += float(gathered_loss.item())

                    # Clear cache to reduce memory pressure
                    if hasattr(gh.accelerator, 'empty_cache'):
                        gh.accelerator.empty_cache()
                    elif hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()

                    # Expensive metrics on cadence
                    acc = None
                    if (global_step + 1) % log_every == 0 and gh.is_main:
                        try:
                            acc, _, _ = compute_and_log_token_topk(outputs, batch, k=10)
                        except Exception as e:
                            logger.error(f"❌ Error in train_eval: {e}")
                            pass

                    # save checkpoints
                    if (global_step + 1) % p_args.data.save_every == 0 and gh.is_main:
                        save_step_checkpoint(epoch, global_step, fine_tune=fine_tune)
                    if debug_sync and global_step == 0:
                        _dbg("before barrier")
                    gh.barrier()
                    if debug_sync and global_step == 0:
                        _dbg("after barrier")

                    # ----- periodic val eval + early stop -----
                    eval_every = int(config['training'].get('eval_every', 0) or 0)
                    if val_loader is not None and eval_every > 0 and (global_step + 1) % eval_every == 0:
                        val_loss = _run_val_eval(model, val_loader, config)
                        if gh.is_main:
                            logger.info(f"[Val] Ep {epoch} Step {_optimizer_step+1}: val_loss={val_loss:.4f}")
                            if not (val_loss != val_loss):  # not NaN
                                wandb.log({'val/loss': val_loss, 'val/epoch': epoch, 'val/step': _optimizer_step+1})
                        if es_state is not None:
                            es_cfg = config['training'].get('early_stop') or {}
                            if es_cfg.get('enabled'):
                                min_delta = float(es_cfg.get('min_delta', 1e-3))
                                patience = int(es_cfg.get('patience', 3))
                                if val_loss < es_state['best_val_loss'] - min_delta:
                                    es_state['best_val_loss'] = val_loss
                                    es_state['no_improve'] = 0
                                    # Snapshot best weights whenever val improves
                                    if best_tracker is not None:
                                        unwrapped = gh.accelerator.unwrap_model(model)
                                        best_tracker.update(unwrapped, val_loss, save_lora=fine_tune)
                                else:
                                    es_state['no_improve'] += 1
                                if gh.is_main:
                                    logger.info(f"[ES] best={es_state['best_val_loss']:.4f}, no_improve={es_state['no_improve']}/{patience}")
                                if es_state['no_improve'] >= patience:
                                    es_state['stop'] = True
                        gh.barrier()
                        if es_state is not None and es_state.get('stop'):
                            break

                    recent_avg = (sum(recent_losses) / len(recent_losses)) if recent_losses else float(raw_loss.item())
                    # Clamp to avoid inf in exp
                    perplexity = float(torch.exp(torch.clamp(torch.tensor(recent_avg), max=20.0)).item())
                    lr_proj, lr_lora = None, None
                    if isinstance(optimizer, list):
                        lr_proj, lr_lora = float(optimizer[0].param_groups[0]['lr']), float(optimizer[1].param_groups[0]['lr'])
                    else:
                        lr_proj = float(optimizer.param_groups[0]['lr'])


                    _optimizer_step += 1
                    global_step += 1
                    lr_str = f'{lr_proj:.2e}, {lr_lora:.2e}' if isinstance(optimizer, list) else f'{lr_proj:.2e}'
                    if gh.is_main and progress_bar is not None:
                        # set_postfix first with refresh=False so update(1) is the only refresh,
                        # respecting miniters=50 / mininterval=30 — avoids double-print per step.
                        progress_bar.set_postfix({
                            'loss': f'{recent_avg:.4f}',
                            'acc': f'{(acc or 0.0):.4f}',
                            'lr': lr_str
                        }, refresh=False)
                        progress_bar.update(1)

                        wandb.log({
                            'train/lossstep': float(raw_loss.item()),
                            'train/loss_recent_avg': recent_avg,
                            'train/loss_epoch_avg': (total_loss + float(raw_loss.item())) / (_step+1),
                            'train/perplexity_recent': perplexity,
                            'train/epoch': epoch,
                            'train/step': _optimizer_step, # actually we log using optimizer steps
                            'train/token_top10_accuracy': acc,
                            'train/lr_proj': lr_proj,
                            'train/lr_lora': lr_lora,
                        })
                        
                        logger.info(f"[Train] Ep {epoch} Step {_optimizer_step}: "
                                    f"loss={float(raw_loss.item()):.4f}, recent={recent_avg:.4f}, "
                                    f"ppl={perplexity:.4f}, acc={(acc or 0.0):.4f}, lr={lr_str}")
                                    # f"ppl={perplexity:.4f}, lr={lr_str}")

            consecutive_oom = 0

        except torch.cuda.OutOfMemoryError:
            consecutive_oom += 1
            if consecutive_oom >= MAX_CONSECUTIVE_OOM:
                raise RuntimeError(
                    f"{consecutive_oom} consecutive OOMs at optimizer step {_optimizer_step} "
                    f"(batch_size={config['training']['batch_size']}/GPU): the step never fits, "
                    "so retrying cannot help. Lower batch_size or grad_accum_steps."
                )
            if gh.is_main:
                logger.warning(
                    f"⚠️ OOM during training step ({consecutive_oom}/{MAX_CONSECUTIVE_OOM}); "
                    "clearing cache and continuing"
                )
            torch.cuda.empty_cache()
            if isinstance(optimizer, list):
                for opt in optimizer:
                    opt.zero_grad(set_to_none=True)
            else:
                optimizer.zero_grad(set_to_none=True)
            continue
        except Exception as e:
            if gh.is_main:
                logger.error(f"❌ Error in training step {_optimizer_step}: {e}")
            continue

    # End-of-epoch diagnostics
    if gh.is_main:
        if batch_means:
            import numpy as np
            batch_means_np = np.stack(batch_means)
            across_batch_var = batch_means_np.var(axis=0).mean()
            logger.info(f"Projector output variance (across-batch means): {across_batch_var:.6f}")
        processed = int(progress_bar.n) if progress_bar is not None else 0
        logger.info(f"Epoch {epoch}: processed {processed} batches, {_optimizer_step} optimizer steps")
        if progress_bar is not None:
            progress_bar.close()

    return global_step