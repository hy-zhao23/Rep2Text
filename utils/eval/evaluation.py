import torch
from utils import logger, gh
from tqdm import tqdm
from utils.train.train_utils import custom_loss
from .test_utils import greedy_decode_batch
from .metrics import EvaluationMetrics
from utils.core.path import get_metrics_path, get_decoded_path
from utils.core.files import write_json

def run_eval(model, config, dataloader, tokenizer, save_results=True, epoch=None, model_type="einv", split="test", decode_all=False):
    """
    Run comprehensive evaluation on a given dataloader for both val and test splits.

    decode_all: generate text for EVERY batch instead of every log_every-th one.
        Set this when the dataloader has already been subsampled to a fixed eval
        size (see eval_gen.py --max_eval_samples); otherwise the log_every gate
        would thin that subsample a second time.
    """
    if gh.is_main:
        logger.info(f"🔄 Running {split} evaluation" + (f" for epoch {epoch}" if epoch is not None else "") + f" with {model_type} model")

    # Use provided metrics calculator or create default one
    metrics_cal = EvaluationMetrics()  # Default: all metrics enabled

    # Run evaluation and collect metrics
    _eval_with_metrics(model, config, dataloader, tokenizer, metrics_cal, split=split, decode_all=decode_all)
    
    # Aggregate metrics across processes
    if gh.is_main and save_results:
        # Save metrics summary
        file = get_metrics_path(split=split, model_type=model_type, epoch=epoch)
        metrics_cal.save_metrics(file)

        # Save decoded pairs for downstream GPT scoring
        if metrics_cal.batch_decoded:
            decoded_file = get_decoded_path(split=split, model_type=model_type, epoch=epoch)
            write_json(metrics_cal.batch_decoded, decoded_file)

def _eval_with_metrics(model, config, dataloader, tokenizer, metrics_cal, split, decode_all=False):
    """
    Evaluate model on a given dataloader and collect comprehensive metrics.
    
    Args:
        model: The model to evaluate
        config: Configuration object
        dataloader: DataLoader for evaluation
        tokenizer: Tokenizer for text processing
        metrics_calculator: EvaluationMetrics instance to collect metrics
        split: Split name for logging
    """
    model.eval()
    
    if gh.is_main:
        logger.info(f"🔍 Starting {split} evaluation with {len(dataloader)} batches")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating on {split}")):
            try:
                # Ensure embeddings have the same dsplit as the model
                batch = _prepare_batch(batch)
                
                # Forward pass
                base_model = model.unwrap_module() if hasattr(model, 'unwrap_module') else model
                outputs = custom_loss(base_model, batch)
                batch_loss = outputs.loss.detach().item()

                if (decode_all or batch_idx % config['logging']['log_every'] == 0) and gh.is_main:
                    batch_decoded = greedy_decode_batch(model, batch, tokenizer, log=(batch_idx % config['logging']['log_every'] == 0))
                    metrics_cal.add_batch_results(
                        batch_loss=batch_loss,
                        batch_decoded=batch_decoded
                    )
                else:
                    # Add batch results to metrics calculator
                    metrics_cal.add_batch_results(
                        batch_loss=batch_loss
                    )
                
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"OOM during {split} evaluation, skipping batch")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                logger.error(f"Error during {split} evaluation: {str(e)}")
                continue
        
    model.train()

def _prepare_batch(batch):
    """Move batch to device."""
    batch = {k: v.to(gh.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    return batch