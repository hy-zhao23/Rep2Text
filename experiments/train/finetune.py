from utils import p_args, logger, gh
from utils.train import setup_train_paras
import yaml
import argparse
from transformers import AutoConfig
import wandb
import os
from utils.core import save_final, BestTracker, print_config
from utils.models import load_pretrained_model
from utils.data import create_dataloader
from utils.project_utils import sync_lora_paras, set_global_seed
from accelerate import Accelerator
from utils.train import resume_from_latest_if_enabled 

set_global_seed()

def train(args):
    # ------------------------------
    # 🧠 Setup correct device for this process
    # 🧠 Load training config 
    # ------------------------------
    gh.init_distributed()  # ✅ 显式调用，确保 default PG = NCCL

    with open(args.config, 'r', encoding='utf-8') as f:
        train_config = yaml.safe_load(f)

    gh.accelerator = Accelerator(gradient_accumulation_steps=train_config['training'].get('grad_accum_steps', 1))
    gh.enable_accelerator(gh.accelerator)


    # ------------------------------
    # 🧠 Initialize wandb
    # ------------------------------
    if gh.is_main:
        print_config(args.config)  
        logger.info("🧠 Base model (dec_id): %s", p_args.model.dec_id)
        wandb.init(
            project=train_config['logging']['wandb_project'],
            config=train_config,
            group=train_config['logging']['group'],
            name=f"lora_{os.getenv('TRAIN_TAG', 'default')}"
        )

    # ------------------------------
    # 🧠 Load adapter and base model
    # ------------------------------
    tokenizer, model, context_len = load_pretrained_model(
        model_type="einv_lora",
        resume_from=False, # Set False for training
        device_map=None,
        train_config=train_config
    )
    
    
    # ------------------------------
    # 🧠 Synchronize parameters across ranks
    # ------------------------------
    # Sync LoRA parameters, projector parameters are loaded the same
    setup_train_paras(model)
    sync_lora_paras(model, src=0)

    # ------------------------------
    # 🧠 Setup data loaders
    # -----------------------------    from utils.data.builder import create_dataloader
    
    emb_cfg = AutoConfig.from_pretrained(p_args.model.emb_id)
    emb_hidden = getattr(emb_cfg, "hidden_size", None)
    if emb_hidden is None:
        raise ValueError("emb_id hidden_size not found; cannot build dataloader")

    train_loader = create_dataloader(
        train_config,
        tokenizer,
        emb_hidden,
        context_len=context_len,
        split_type="train",
        data_type="finetune"
    )

    val_loader = None
    if train_config['training'].get('eval_every'):
        val_loader = create_dataloader(
            train_config,
            tokenizer,
            emb_hidden,
            context_len=context_len,
            split_type="val",
            data_type="finetune"
        )

    # no test loader used during training

    # For quick testing: use only 1 step (batch) for train, val, and test loaders
    # from torch.utils.data import DataLoader, Subset

    # test_batch_count = 1
    # batch_size = train_config['training']['batch_size']

    # def limit_loader(loader, batch_count):
    #     # loader is a Dataset, not a DataLoader yet
    #     return Subset(loader.dataset, range(batch_count * batch_size))

    # train_loader = DataLoader(
    #     limit_loader(train_loader, test_batch_count),
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=train_config['data']['num_workers'],
    #     collate_fn=train_loader.collate_fn
    # )
    # val_loader = DataLoader(
    #     limit_loader(val_loader, test_batch_count),
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=train_config['data']['num_workers'],
    #     collate_fn=val_loader.collate_fn
    # )
    # test_loader = DataLoader(
    #     limit_loader(test_loader, test_batch_count),
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=train_config['data']['num_workers'],
    #     collate_fn=test_loader.collate_fn
    # )


    # ------------------------------
    # 🧠 Setup training components
    # ------------------------------
    grad_accum_steps = train_config['training'].get('grad_accum_steps', 1)
    total_steps = len(train_loader) * train_config['training']['num_epochs'] // grad_accum_steps
    
    from utils.train.training import prepare_model_and_optimizer, train_epoch
    model, optimizer, scheduler = prepare_model_and_optimizer(model, train_config, total_steps)

    # Prepare with Accelerate (include scheduler so checkpoint state can be restored correctly)
    model, train_loader, optimizer, scheduler = gh.accelerator.prepare(
        model, train_loader, optimizer, scheduler
    )
    if val_loader is not None:
        val_loader = gh.accelerator.prepare(val_loader)

    # ------------------------------
    # 🧠 Resume from latest checkpoint (if enabled and exists)
    # ------------------------------
    start_epoch, global_step = resume_from_latest_if_enabled(
        fine_tune=True,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # ------------------------------
    # 🧠 Training loop
    # ------------------------------
    if gh.is_main:
        logger.info("🚀 Starting fine-tuning with LoRA + Adapter...")

    es_cfg = train_config['training'].get('early_stop') or {}
    es_state = {"best_val_loss": float("inf"), "no_improve": 0, "stop": False}
    best_tracker = BestTracker() if es_cfg.get('enabled') else None

    for epoch in range(start_epoch, train_config['training']['num_epochs']):
        global_step = train_epoch(
            model, train_loader, optimizer, scheduler, epoch, global_step, train_config,
            fine_tune=True, val_loader=val_loader, es_state=es_state, best_tracker=best_tracker,
        )
        gh.barrier()
        if es_state.get("stop"):
            if gh.is_main:
                logger.info(f"⏹️ Early stop triggered at epoch {epoch}, global_step {global_step}")
            break

    # ------------------------------
    # 🧠 Final save
    # ------------------------------
    save_final(gh.accelerator.unwrap_model(model), save_lora=True, best_tracker=best_tracker)

    gh.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Fine-tuning LLM with LoRA + Adapter")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    train(args)

