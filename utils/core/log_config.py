from .logger import logger
from .settings import p_args
import yaml


def print_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info("-------------------------------------------------------------------")
    logger.info("--------------------Training Parameter Settings--------------------")
    logger.info("-------------------------------------------------------------------")
    logger.info(f"Embedding Layer: {p_args.model.embedding_layer}")
    logger.info(f"Batch Size: {config['training']['batch_size']}")
    logger.info(f"Learning Rate: {config['training']['learning_rate']}")
    logger.info(f"Num Epochs: {config['training']['num_epochs']}")
    logger.info(f"Grad Accum Steps: {config['training']['grad_accum_steps']}")
    logger.info(f"Weight Decay: {config['training']['weight_decay']}")
    logger.info(f"Betas: {config['training']['betas']}")
    logger.info(f"Eps: {config['training']['eps']}")
    logger.info(f"Max Grad Norm: {config['training']['max_grad_norm']}")
    logger.info(f"Warmup Ratio: {config['training']['warmup_ratio']}")
    logger.info(f"Use Scheduler: {config['training']['use_scheduler']}")
    logger.info(f"Label Smoothing: {config['training']['label_smoothing']}")
    logger.info(f"Tau: {config['training']['tau']}")
    logger.info(f"Sim Penalty: {config['training']['sim_penalty']}")
    logger.info(f"Sim Weight: {config['training']['sim_weight']}")
    logger.info(f"Dropout: {config['training']['dropout']}")


    logger.info("-------------------------------------------------------------------")
    logger.info("---------------------Dataset Parameter Settings--------------------")
    logger.info("-------------------------------------------------------------------")
    logger.info(f"Data Name: {p_args.data.data_name}")
    logger.info(f"Max Len: {p_args.data.max_len}")
    logger.info(f"Overlap Len: {p_args.data.overlap_len}")
    logger.info(f"Train Size: {p_args.data.train_size}")
    logger.info(f"Embedding Batch Size: {p_args.data.embedding_batch_size}")
    logger.info(f"Checkpoint Steps: {p_args.data.save_every}")
    logger.info(f"Pretrain Ratio: {p_args.data.pretrain_ratio}")

    if config.get('lora', {}).get('train', False):
        logger.info(f"LoRA Attention R: {config.get('lora', {}).get('att_r')}")
        logger.info(f"LoRA MLP R: {config.get('lora', {}).get('mlp_r')}")
        logger.info(f"LoRA Attention Alpha: {config.get('lora', {}).get('att_alpha')}")
        logger.info(f"LoRA MLP Alpha: {config.get('lora', {}).get('mlp_alpha')}")
        logger.info(f"LoRA Dropout: {config.get('lora', {}).get('dropout')}")
        logger.info(f"LoRA Target Att Modules: {config.get('lora', {}).get('target_atts')}")
        logger.info(f"LoRA Target MLP Modules: {config.get('lora', {}).get('target_mlps')}")
        logger.info(f"LoRA Layers: {config.get('lora', {}).get('layers')}")
        logger.info(f"LoRA Bias: {config.get('lora', {}).get('bias')}")
        logger.info(f"LoRA Task Type: {config.get('lora', {}).get('task_type')}")

