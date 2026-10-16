from transformer_lens import HookedTransformer
from ..core import p_args, logger, gh
import torch as th
from typing import Any, List, Optional, Union
import numpy as np
from tqdm import tqdm
from huggingface_hub import login
import os
import glob

login(token=p_args.settings.hf_token)
_file_label = f"l{p_args.data.max_len}_o{p_args.data.overlap_len}"

@th.no_grad()
def _get_embeddings(
    tokens: th.Tensor,
    model: HookedTransformer,
    attention_mask: Optional[th.Tensor] = None,
) -> th.Tensor:
    # TODO: pay attention to the layer here
    layer_name = f"blocks.{p_args.model.embedding_layer}.hook_resid_post"

    device = next(model.parameters()).device
    model_kwargs = {}
    if attention_mask is not None:
        model_kwargs["attention_mask"] = attention_mask.to(device)

    _, cache = model.run_with_cache(
        tokens.to(device), 
        stop_at_layer=p_args.model.embedding_layer + 1,
        names_filter=lambda name: name == layer_name,
        **model_kwargs,
    )

    # Extract embeddings from cache. .float() is required: the model runs in
    # bf16 and numpy has no bfloat16 dtype, so .numpy() would raise. Stored as
    # fp32 exactly as before -- the dataloader casts back to bf16 on read.
    hidden_states = cache[layer_name]
    if attention_mask is None:
        embeddings = hidden_states[:, -1, :]
    else:
        attention_mask = attention_mask.to(hidden_states.device)
        last_indices = (
            attention_mask.shape[1]
            - 1
            - attention_mask.long().flip(dims=(1,)).argmax(dim=1)
        )
        embeddings = hidden_states[
            th.arange(hidden_states.shape[0], device=hidden_states.device),
            last_indices,
        ]
    embeddings = embeddings.float().cpu().numpy()
    
    return embeddings

def _load_existing_checkpoint(batch_idx: int) -> tuple:
    """Load existing checkpoint if available and return resume information."""
    checkpoint_file = f"{p_args.path.embedding_dir}/embeddings_r{gh.rank}_b{batch_idx}_{_file_label}_checkpoint.npz"
    
    if not os.path.exists(checkpoint_file):
        return None, 0, [], []
    
    # Load checkpoint embeddings
    checkpoint_data = np.load(checkpoint_file)
    batch_embeddings = checkpoint_data['embeddings']
    
    # Load corresponding tokens
    tok_checkpoint = checkpoint_file.replace('embeddings_', 'tokens_')
    tok_data = np.load(tok_checkpoint)
    tokens_processed = list(tok_data['tokens'])
    
    # Calculate starting index using the minimum length to ensure consistency
    embeddings_count = len(batch_embeddings) if hasattr(batch_embeddings, '__len__') else 0
    tokens_count = len(tokens_processed)
    start_idx = min(embeddings_count, tokens_count)
    
    logger.info(f"🔄 Rank {gh.rank} resuming batch {batch_idx}, processed {start_idx} tokens")
    return checkpoint_file, start_idx, batch_embeddings, tokens_processed

def _initialize_new_batch(batch_idx: int, tokens: list) -> tuple:
    """Initialize a new batch with empty state."""
    batch_embeddings = []
    tokens_processed = []
    start_idx = 0
    logger.info(f"🚀 Rank {gh.rank} starting new batch {batch_idx} with {len(tokens)} tokens")
    
    return batch_embeddings, tokens_processed, start_idx

def _process_remaining_tokens(tokens: list, model: HookedTransformer, 
                            batch_idx: int, start_idx: int,
                            batch_embeddings: list, tokens_processed: list) -> tuple:
    """Process the remaining tokens and save checkpoints periodically."""
    for i in tqdm(range(start_idx, len(tokens), p_args.data.embedding_batch_size), 
                 desc=f"Rank {gh.rank} Batch {batch_idx}"):
        right_idx = min(i + p_args.data.embedding_batch_size, len(tokens))
        th.cuda.empty_cache()

        embeddings = _get_embeddings(th.tensor(tokens[i:right_idx]), model)
        batch_embeddings.append(embeddings)
        tokens_processed.extend(tokens[i:right_idx])
        
        # Save checkpoint every save_every
        current_step = len(batch_embeddings)
        if current_step % p_args.data.save_every == 0:
            _save_checkpoint(
                p_args.path.embedding_dir, 
                batch_idx, 
                current_step,
                batch_embeddings, 
                tokens_processed
            )
    
    return batch_embeddings, tokens_processed

def _save_final_batch_files(batch_embeddings: list, tokens: list, batch_idx: int) -> None:
    """Save the final batch files and clean up checkpoints."""
    if batch_embeddings:
        batch_embeddings = np.vstack(batch_embeddings)
    else:
        raise ValueError("No embeddings found")

    # Atomic write for final batch files
    os.makedirs(p_args.path.embedding_dir, exist_ok=True)
    emb_temp = os.path.join(p_args.path.embedding_dir, f"tmp_embeddings_r{gh.rank}_b{batch_idx}_{_file_label}.npz")
    tok_temp = os.path.join(p_args.path.embedding_dir, f"tmp_tokens_r{gh.rank}_b{batch_idx}_{_file_label}.npz")
    emb_final = os.path.join(p_args.path.embedding_dir, f"embeddings_r{gh.rank}_b{batch_idx}_{_file_label}.npz")
    tok_final = os.path.join(p_args.path.embedding_dir, f"tokens_r{gh.rank}_b{batch_idx}_{_file_label}.npz")
    
    try:
        np.savez(emb_temp, embeddings=batch_embeddings)
        np.savez(tok_temp, tokens=tokens)
        
        # Atomic rename
        os.rename(emb_temp, emb_final)
        os.rename(tok_temp, tok_final)
        
        logger.info(f"✅ Rank {gh.rank} Saved final embeddings and tokens for batch {batch_idx}")
        
        # Clean up checkpoint files after successful completion
        _cleanup_checkpoint_files(batch_idx)
        
    except Exception as e:
        # Only clean up temp files on error (keep them for debugging)
        logger.error(f"Rank {gh.rank} Failed to save final batch {batch_idx}: {e}")
        raise

def _cleanup_checkpoint_files(batch_idx: int) -> None:
    """Clean up checkpoint files after successful batch completion."""
    checkpoint_file = f"{p_args.path.embedding_dir}/embeddings_r{gh.rank}_b{batch_idx}_{_file_label}_checkpoint.npz"
    tok_checkpoint = checkpoint_file.replace('embeddings_', 'tokens_')
    
    try:
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            logger.info(f"🧹 Rank {gh.rank} Cleaned up checkpoint: {checkpoint_file}")
        if os.path.exists(tok_checkpoint):
            os.remove(tok_checkpoint)
            logger.info(f"🧹 Rank {gh.rank} Cleaned up checkpoint: {tok_checkpoint}")
    except Exception as e:
        logger.warning(f"Rank {gh.rank} Failed to clean up checkpoint files: {e}")
        

def _process_batch(tokens: Union[List[Any], np.ndarray], model: HookedTransformer, batch_idx: int) -> None:
    """Process a batch of tokens, with checkpointing support for resuming interrupted processing."""
    
    # Try to load existing checkpoint
    checkpoint_info = _load_existing_checkpoint(batch_idx)

    batch_embeddings, tokens_processed, start_idx = _initialize_new_batch(batch_idx, tokens)
    
    if checkpoint_info[0] is not None:
        # Resume from existing checkpoint
        _, start_idx, batch_embeddings, tokens_processed = checkpoint_info
    
    # Process remaining tokens
    batch_embeddings, tokens_processed = _process_remaining_tokens(
        tokens, model, batch_idx, start_idx,
        batch_embeddings, tokens_processed
    )

    # Save final checkpoint if we haven't saved one recently
    if batch_embeddings and len(batch_embeddings) % p_args.data.save_every != 0:
        _save_checkpoint(
            p_args.path.embedding_dir, 
            batch_idx, 
            len(batch_embeddings),
            batch_embeddings, 
            tokens_processed
        )

    # Save final batch files
    _save_final_batch_files(batch_embeddings, tokens, batch_idx)

def _save_checkpoint(embedding_dir, batch_idx, step, batch_embeddings, tokens_so_far):
    # Create temporary file names (using different naming to avoid confusion)
    os.makedirs(embedding_dir, exist_ok=True)
    emb_temp = os.path.join(embedding_dir, f"tmp_embeddings_r{gh.rank}_b{batch_idx}_{_file_label}_checkpoint.npz")
    tok_temp = os.path.join(embedding_dir, f"tmp_tokens_r{gh.rank}_b{batch_idx}_{_file_label}_checkpoint.npz")
    emb_final = os.path.join(embedding_dir, f"embeddings_r{gh.rank}_b{batch_idx}_{_file_label}_checkpoint.npz")
    tok_final = os.path.join(embedding_dir, f"tokens_r{gh.rank}_b{batch_idx}_{_file_label}_checkpoint.npz")
    
    try:
        # Convert list of embeddings to numpy array
        if isinstance(batch_embeddings, list):
            embeddings_array = np.vstack(batch_embeddings)
        else:
            embeddings_array = batch_embeddings
            
        # Write to temporary files first
        np.savez(emb_temp, embeddings=embeddings_array, step=step)
        np.savez(tok_temp, tokens=np.array(tokens_so_far))
        
        # Atomic rename to final names
        os.rename(emb_temp, emb_final)
        os.rename(tok_temp, tok_final)
        
        logger.info(f"💾 Rank {gh.rank} Saved checkpoint for batch {batch_idx} at step {step}")
        
    except Exception as e:
        # Only log error, keep temp files for debugging
        logger.error(f"❌ Rank {gh.rank} Failed to save checkpoint: {e}")
        raise

def tokens_to_embeddings(tokens: Union[List[Any], np.ndarray]):
    logger.info(
        "📦 Embedding cache model: %s | dir: %s",
        p_args.model.emb_id,
        p_args.path.embedding_dir,
    )
    # bf16 to match the rest of the pipeline: the dataloader casts every cached
    # embedding to bfloat16 on read anyway (see dataloader.py __getitem__), and
    # the projector/decoder are bf16, so an fp32 cache stores precision that is
    # discarded immediately. Measured on a 40G MIG slice (Llama-3.2-3B, bsz=512,
    # 128-token seqs): 21.3 s/it at fp32 vs 2.47 s/it at bf16 (8.6x), and peak
    # VRAM drops 26.4 -> 13.6 GB. Kept on from_pretrained (not
    # from_pretrained_no_processing, which transformer_lens suggests for reduced
    # precision) so the weight processing matches every earlier cache.
    model = HookedTransformer.from_pretrained(
        p_args.model.emb_id,
        device="cuda",
        dtype=th.bfloat16,
        n_devices=int(os.getenv('SLURM_GPUS_ON_NODE', 1)),
        cache_dir=p_args.path.weight_dir
    )

    # Check for existing completed batches
    completed_batches = set()
    existing_files = glob.glob(f"{p_args.path.embedding_dir}/embeddings_r{gh.rank}_b*_{_file_label}.npz")
    for file in existing_files:
        if '_checkpoint' not in file:  # Only count final batch files, not checkpoints
            filename = os.path.basename(file)
            try:
                # Extract batch number from filename pattern: embeddings_r{rank}_b{batch_num}_{file_label}.npz
                # Find the part after '_b' and before the next '_'
                batch_part = filename.split('_b')[-1].split('_')[0]
                batch_num = int(batch_part)
                completed_batches.add(batch_num)
            except (ValueError, IndexError) as e:
                logger.warning(f"Skipping file with unexpected naming pattern: {filename}")
                continue
    
    if completed_batches:
        logger.info(f"📋 Rank {gh.rank} Found {len(completed_batches)} completed batches: {sorted(completed_batches)}")

    # Process batches (100000 sequences per batch)
    batch_size = 100000
    for batch_idx in range(0, len(tokens), batch_size):
        batch_num = batch_idx // batch_size
        
        # Skip if batch is already completed
        if batch_num in completed_batches:
            logger.info(f"⏭️ Rank {gh.rank} Skipping completed batch {batch_num}")
            continue
            
        right_idx = min(batch_idx + batch_size, len(tokens))
        th.cuda.empty_cache()
        _process_batch(tokens[batch_idx:right_idx], model, batch_num)
    
    logger.info(f"🎉 Rank {gh.rank} All batches completed")
