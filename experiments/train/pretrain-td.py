import os
import argparse
import yaml
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import wandb
from transformers import AutoConfig

from utils import p_args, gh, logger
from utils.core import print_config, save_final
from utils.project_utils import set_global_seed, sync_projector_paras
from utils.train import setup_train_paras, resume_from_latest_if_enabled, save_step_checkpoint
from utils.train.train_utils import custom_loss
from utils.eval.test_utils import compute_and_log_token_topk
from utils.models import load_pretrained_model
from utils.data import create_dataloader

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)

set_global_seed()


def _get_ranks(tp: int, dp: int):
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])

    if world != tp * dp:
        raise RuntimeError(f"world_size={world} must equal tp*dp={tp*dp} (tp={tp}, dp={dp})")

    # DP rank = node id (so both TP ranks on same node share data)
    dp_rank = int(os.environ.get("SLURM_NODEID", rank // tp))
    tp_rank = local
    return rank, dp_rank, tp_rank


def _build_mesh(tp: int, dp: int):
    mesh = init_device_mesh("cuda", (dp, tp), mesh_dim_names=("dp", "tp"))
    return mesh, mesh["dp"], mesh["tp"]


def _get_mesh_group(mesh_dim):
    # robust access across torch versions
    if hasattr(mesh_dim, "get_group"):
        return mesh_dim.get_group()
    if hasattr(mesh_dim, "group"):
        return mesh_dim.group
    raise RuntimeError(f"Cannot get process group from mesh dim. attrs={dir(mesh_dim)}")


def _apply_tp_plan_llama_like(model, tp_mesh):
    backbone = getattr(model, "model", None) or getattr(model, "transformer", None)
    if backbone is None:
        raise RuntimeError("Cannot find transformer backbone on model (.model or .transformer).")

    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise RuntimeError("Cannot find transformer layers (expected backbone.layers).")

    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)
        if attn is None or mlp is None:
            continue

        plan = {}
        if hasattr(attn, "q_proj"): plan["self_attn.q_proj"] = ColwiseParallel()
        if hasattr(attn, "k_proj"): plan["self_attn.k_proj"] = ColwiseParallel()
        if hasattr(attn, "v_proj"): plan["self_attn.v_proj"] = ColwiseParallel()
        if hasattr(attn, "o_proj"): plan["self_attn.o_proj"] = RowwiseParallel()

        if hasattr(mlp, "gate_proj"): plan["mlp.gate_proj"] = ColwiseParallel()
        if hasattr(mlp, "up_proj"):   plan["mlp.up_proj"]   = ColwiseParallel()
        if hasattr(mlp, "down_proj"): plan["mlp.down_proj"] = RowwiseParallel()

        if plan:
            parallelize_module(layer, tp_mesh, plan)

    return model


@torch.no_grad()
def _run_val_eval(model, val_loader, config, dp_group, max_batches: int) -> float:
    """Forward-only val loss, summed over the DP group so every rank agrees.

    Only DP ranks hold different data — TP ranks within a node see the same
    batch — so the reduction is over dp_group, not the world.
    """
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        if not isinstance(batch, dict) or any(k not in batch for k in ("input_ids", "labels", "embedding")):
            continue
        if batch["input_ids"].shape[0] == 0:
            continue
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(gh.device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = custom_loss(model, batch)
        loss = out.loss.detach()
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        total += loss.item()
        n += 1
    if was_training:
        model.train()

    acc = torch.tensor([total, float(n)], device=gh.device, dtype=torch.float64)
    dist.all_reduce(acc, op=dist.ReduceOp.SUM, group=dp_group)
    return (acc[0] / acc[1]).item() if acc[1].item() > 0 else float("nan")


def _snapshot_trainable(model):
    # The backbone is frozen, so "best weights" is just the projector — a few
    # hundred MB, cheap enough to keep a CPU copy of.
    return {k: v.detach().to("cpu", copy=True) for k, v in model.named_parameters() if v.requires_grad}


def _restore_trainable(model, state):
    with torch.no_grad():
        for k, v in model.named_parameters():
            if k in state:
                v.copy_(state[k].to(v.device, dtype=v.dtype))


def _dp_allreduce_trainable_grads(model, dp_group, dp_size: int):
    # average grads across DP only
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=dp_group)
            p.grad.div_(dp_size)


def train_one_epoch(model, loader, optimizer, scheduler, epoch, global_step, config, dp_group, dp_size: int):
    model.train()
    grad_accum = int(config["training"].get("grad_accum_steps", 1))
    log_every = int(config["logging"].get("log_every", 20))
    max_grad_norm = float(config["training"].get("max_grad_norm", 0.0))

    steps_per_epoch = max(1, len(loader) // grad_accum)
    completed_full_epochs = global_step // steps_per_epoch

    if epoch < completed_full_epochs:
        if gh.is_main:
            logger.info(
                f"⏩ Skip epoch {epoch}: already completed (global_step={global_step}, steps/epoch={steps_per_epoch})"
            )
        return global_step

    resume_opt_step = global_step % steps_per_epoch if epoch == completed_full_epochs else 0
    if gh.is_main and resume_opt_step > 0:
        logger.info(
            f"🔁 Resuming epoch {epoch} from optimizer step {resume_opt_step}/{steps_per_epoch}"
        )
    
    total_loss = 0.0
    batch_means = []

    for step, batch in enumerate(loader):
        # Skip already-completed accumulation windows when resuming mid-epoch.
        if (step // grad_accum) < resume_opt_step:
            continue

        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(gh.device, non_blocking=True)

        do_step = ((step + 1) % grad_accum == 0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Use custom_loss which handles input_ids, labels, embedding (as hidden_embeds)
            out = custom_loss(model, batch)
            loss = out.loss

        (loss / grad_accum).backward()

        if do_step:
            _dp_allreduce_trainable_grads(model, dp_group, dp_size)

            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_grad_norm,
                )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

            global_step += 1

            if global_step % p_args.data.save_every == 0:
                save_step_checkpoint(
                    epoch=epoch,
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
            
            # Track loss for averaging
            raw_loss = loss.item()
            total_loss += raw_loss
            batch_means.append(raw_loss)
            if len(batch_means) > log_every:
                batch_means.pop(0)
            
            recent_avg = sum(batch_means) / len(batch_means)
            epoch_avg = total_loss / (step // grad_accum + 1)
            perplexity = torch.exp(torch.tensor(recent_avg)).item()
            
            # Console output and wandb logging
            if gh.is_main:
                lr_proj = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
                
                # Compute accuracy on log cadence
                acc = None
                if global_step % log_every == 0:
                    try:
                        acc, _, _ = compute_and_log_token_topk(out, batch, k=10)
                    except Exception as e:
                        logger.warning(f"⚠️ Error computing accuracy: {e}")
                        acc = None

                wandb.log({
                    'train/lossstep': raw_loss,
                    'train/loss_recent_avg': recent_avg,
                    'train/loss_epoch_avg': epoch_avg,
                    'train/perplexity_recent': perplexity,
                    'train/step': global_step,
                    'train/token_top10_accuracy': acc,
                    'train/lr_proj': lr_proj,
                })

                logger.info(f"[Train] Ep {epoch} Step {global_step}: "
                            f"loss={raw_loss:.4f}, recent={recent_avg:.4f}, "
                            f"ppl={perplexity:.4f}, acc={(acc or 0.0):.4f}, lr={lr_proj:.2e}")

    return global_step


def main():
    ap = argparse.ArgumentParser("Adapter pretraining with TP(node-local)+DP(inter-node), no ZeRO")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--dp", type=int, required=True)
    ap.add_argument("--load_4bit", action="store_true", help="Load base model in 4-bit (bitsandbytes)")
    ap.add_argument("--load_8bit", action="store_true", help="Load base model in 8-bit (bitsandbytes)")
    ap.add_argument("--use_flash_attn", action="store_true", help="Enable flash_attention_2 when supported")
    ap.add_argument("--lazy_load", action="store_true",
                    help="Shard weights while reading them off disk instead of loading the full "
                         "checkpoint into host RAM first (needed for 70B+, cuts peak CPU mem ~30x)")
    args = ap.parse_args()

    # IMPORTANT: set device BEFORE init_process_group (nccl)
    torch.cuda.set_device(gh.local_rank)
    gh.init_distributed()

    try:
        if dist.get_backend() != "nccl":
            raise RuntimeError(f"Expected default PG backend=nccl, got {dist.get_backend()}")

        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        rank, dp_rank, tp_rank = _get_ranks(tp=args.tp, dp=args.dp)

        if gh.is_main:
            print_config(args.config)
            wandb.init(
                project=config["logging"]["wandb_project"],
                config=config,
                group=config["logging"].get("group"),
                name=str(os.getenv("TRAIN_TAG", "tpdp")),
            )
        else:
            os.environ["WANDB_DISABLED"] = "true"

        # mesh + dp_group
        _, dp_mesh, tp_mesh = _build_mesh(tp=args.tp, dp=args.dp)
        dp_group = _get_mesh_group(dp_mesh)

        # model
        # Eager path loads the whole checkpoint on CPU per rank (4 ranks x 145 GB
        # for a 72B), then shards. Lazy path hands the mesh to from_pretrained so
        # each rank reads only its own slice straight onto its GPU. The built-in
        # "auto" plan matches _apply_tp_plan_llama_like exactly.
        lazy_kwargs = {"tp_plan": "auto", "device_mesh": tp_mesh} if args.lazy_load else {}

        tokenizer, model, context_len = load_pretrained_model(
            model_type="einv",
            resume_from=False,
            device_map=None,
            load_4bit=args.load_4bit,
            load_8bit=args.load_8bit,
            use_flash_attn=args.use_flash_attn,
            **lazy_kwargs,
        )

        # TP (apply before moving to device to avoid OOM)
        if not args.lazy_load:
            model = _apply_tp_plan_llama_like(model, tp_mesh)

        # Enable gradient checkpointing to reduce memory
        if config.get('training', {}).get('use_gradient_checkpointing', False):
            if hasattr(model.model, 'gradient_checkpointing_enable'):
                model.model.gradient_checkpointing_enable()
                if gh.is_main:
                    print("✅ Gradient checkpointing enabled")
            elif hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                if gh.is_main:
                    print("✅ Gradient checkpointing enabled")
        
        # Now move to device after TP sharding (lazy path is already on-device)
        if not args.lazy_load:
            model = model.to(gh.device)

        setup_train_paras(model)  # mark adapters/projector trainable

        gh.barrier()
        sync_projector_paras(model)  # if you want DP-only sync later, we can add group=dp_group
        gh.barrier()

        # dataloader (DP-sharded, TP-shared within node)
        emb_cfg = AutoConfig.from_pretrained(p_args.model.emb_id)
        emb_hidden = getattr(emb_cfg, "hidden_size", None)
        if emb_hidden is None:
            raise ValueError("emb_id hidden_size not found")

        base_loader = create_dataloader(config, tokenizer, emb_hidden, context_len, "train")
        if not isinstance(base_loader, DataLoader):
            raise RuntimeError("create_dataloader must return a DataLoader")

        dataset = base_loader.dataset
        sampler = DistributedSampler(
            dataset,
            num_replicas=args.dp,   # DP only (nodes)
            rank=dp_rank,
            shuffle=True,
            drop_last=True,
        )

        train_loader = DataLoader(
            dataset,
            batch_size=getattr(base_loader, "batch_size", 1),
            sampler=sampler,
            num_workers=getattr(base_loader, "num_workers", 0),
            pin_memory=getattr(base_loader, "pin_memory", True),
            collate_fn=base_loader.collate_fn,
            persistent_workers=(getattr(base_loader, "num_workers", 0) > 0),
        )

        grad_accum = int(config["training"].get("grad_accum_steps", 1))
        total_steps = len(train_loader) * int(config["training"]["num_epochs"]) // grad_accum

        from utils.train import prepare_model_and_optimizer
        model, optimizer, scheduler = prepare_model_and_optimizer(model=model, config=config, total_steps=total_steps)

        gh.barrier()
        start_epoch, global_step = resume_from_latest_if_enabled(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        # Early stopping. training.py has this for the accelerate path; the
        # TP+DP path had no validation at all, so num_epochs was a hard commit.
        es_cfg = config["training"].get("early_stop") or {}
        es_on = bool(es_cfg.get("enabled"))
        patience = int(es_cfg.get("patience", 5))
        min_delta = float(es_cfg.get("min_delta", 1e-3))
        val_batches = int(es_cfg.get("val_max_batches", 40))
        best_val, no_improve, best_state = float("inf"), 0, None

        val_loader = None
        if es_on:
            val_loader = create_dataloader(config, tokenizer, emb_hidden, context_len, "val")
            if gh.is_main:
                logger.info(f"⏱️ Early stopping on: patience={patience}, min_delta={min_delta}, "
                            f"val_batches={val_batches}")

        num_epochs = int(config["training"]["num_epochs"])
        for epoch in range(start_epoch, num_epochs):
            train_loader.sampler.set_epoch(epoch)
            global_step = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                epoch, global_step, config,
                dp_group=dp_group, dp_size=args.dp,
            )
            gh.barrier()

            if es_on:
                val_loss = _run_val_eval(model, val_loader, config, dp_group, val_batches)
                improved = val_loss == val_loss and val_loss < best_val - min_delta
                if improved:
                    best_val, no_improve = val_loss, 0
                    best_state = _snapshot_trainable(model)
                else:
                    no_improve += 1
                if gh.is_main:
                    logger.info(f"[Val] Ep {epoch}: val_loss={val_loss:.4f}  "
                                f"[ES] best={best_val:.4f} no_improve={no_improve}/{patience}")
                    wandb.log({'val/loss': val_loss, 'val/epoch': epoch, 'val/best': best_val})
                if no_improve >= patience:
                    if gh.is_main:
                        logger.info(f"⏹️ Early stop at epoch {epoch} (best val_loss={best_val:.4f})")
                    break
                gh.barrier()

        if es_on and best_state is not None:
            # Ship the best checkpoint, not whatever the last epoch happened to be.
            _restore_trainable(model, best_state)
            if gh.is_main:
                logger.info(f"↩️ Restored best weights (val_loss={best_val:.4f}) before final save")

        if gh.is_main:
            save_final(model, save_lora=False)

        gh.barrier()

    finally:
        # always attempt clean shutdown
        try:
            gh.barrier()
        except Exception:
            pass
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()