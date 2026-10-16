import wandb
import argparse
import os
import yaml
from transformers import AutoConfig
from utils.models import load_pretrained_model
from utils import p_args, gh, logger
from accelerate import Accelerator
from utils.train import train_epoch
from utils.data import create_dataloader
from utils.train import prepare_model_and_optimizer
from utils.project_utils import set_global_seed, sync_projector_paras
from utils.train import setup_train_paras, resume_from_latest_if_enabled
from utils.core import print_config, save_final, BestTracker

set_global_seed()

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Adapter Pretraining with Accelerate")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank")
    args = parser.parse_args()

    gh.init_distributed()  # ✅ 显式调用，确保 default PG = NCCL
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if gh.is_main:
        print_config(args.config)
        logger.info("🧠 Base model (dec_id): %s", p_args.model.dec_id)
        wandb.init(
            project=config['logging']['wandb_project'],
            config=config,
            group=config['logging']['group'],
            name=f"{os.getenv('TRAIN_TAG', 'default')}"
        )

    # ------------------------------
    # 🧠 Load model
    # ------------------------------
    tokenizer, model, context_len = load_pretrained_model(
        model_type="einv",
        resume_from=False, # Set False for training
        device_map=None
    )

    # ------------------------------
    # 🧠 Create dataloaders
    # ------------------------------
    # Create dataloaders first to get actual dataset size
    emb_cfg = AutoConfig.from_pretrained(p_args.model.emb_id)
    emb_hidden = getattr(emb_cfg, "hidden_size", None)
    if emb_hidden is None:
        raise ValueError("emb_id hidden_size not found; cannot build dataloader")
    train_loader = create_dataloader(config, tokenizer, emb_hidden, context_len, "train")

    val_loader = None
    if config['training'].get('eval_every'):
        val_loader = create_dataloader(config, tokenizer, emb_hidden, context_len, "val")

    # need to calculate before accelerator.prepare
    grad_accum_steps = config['training'].get('grad_accum_steps', 1)
    total_steps = len(train_loader) * int(config['training']['num_epochs']) // grad_accum_steps
    
    # ------------------------------
    # 🧠 Synchronize projector parameters AFTER loading onto GPUs
    # ------------------------------ 
    sync_projector_paras(model)
    setup_train_paras(model)

    gh.accelerator = Accelerator(gradient_accumulation_steps=config['training']['grad_accum_steps'])
    gh.enable_accelerator(gh.accelerator)  # ✅ Enable accelerator for global access
    # ------------------------------
    # 🧠 Initialize training components, optimizer, model, dataloader
    # ------------------------------
    model, optimizer, scheduler = prepare_model_and_optimizer(model=model, config=config, total_steps=total_steps)
    model, optimizer, scheduler, train_loader = gh.accelerator.prepare(model, optimizer, scheduler, train_loader)
    if val_loader is not None:
        val_loader = gh.accelerator.prepare(val_loader)

    gh.barrier()

    start_epoch, global_step = resume_from_latest_if_enabled()

    # ------------------------------
    # 🧠 Training process
    # ------------------------------
    num_epochs = int(config['training']['num_epochs'])
    es_cfg = config['training'].get('early_stop') or {}
    es_state = {"best_val_loss": float("inf"), "no_improve": 0, "stop": False}
    best_tracker = BestTracker() if es_cfg.get('enabled') else None
    for epoch in range(start_epoch, num_epochs):
        global_step = train_epoch(model, train_loader, optimizer, scheduler, epoch, global_step, config,
                                   val_loader=val_loader, es_state=es_state, best_tracker=best_tracker)
        gh.barrier()
        if es_state.get("stop"):
            if gh.is_main:
                logger.info(f"⏹️ Early stop triggered at epoch {epoch}, global_step {global_step}")
            break

    # ------------------------------
    # 🧠 Save final model projector first
    # ------------------------------
    save_final(gh.accelerator.unwrap_model(model), save_lora=False, best_tracker=best_tracker)

    gh.barrier()

if __name__ == "__main__":
    main()
