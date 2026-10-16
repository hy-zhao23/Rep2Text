import torch
from utils.models.constants import IGNORE_INDEX
from utils import logger, p_args

def compute_and_log_token_topk(
    outputs,
    batch,
    k: int = 10,
    ignore_index: int = IGNORE_INDEX,
):
    """
    Computes next-token top-k accuracy from model outputs and logs a readable top-k table.
    Returns: (accuracy: float, num_correct: int, num_total: int)
    """
    # Guardrails
    if not hasattr(outputs, "logits"):
        return 0.0, 0, 0
    if not isinstance(batch, dict) or "labels" not in batch:
        return 0.0, 0, 0

    # Align: logits at t predict labels at t+1
    logits_full = outputs.logits.detach()             # (B, T, V)
    labels_full = batch["labels"]                     # (B, T)
    mask_full   = (labels_full != ignore_index)       # (B, T)

    if logits_full.size(1) < 2 or labels_full.size(1) < 2:
        return 0.0, 0, 0  # need at least 2 tokens to compute next-token accuracy

    logits = logits_full[:, :-1, :]                   # (B, T-1, V)
    labels = labels_full[:, 1:]                       # (B, T-1)
    mask   = mask_full[:, 1:]                         # (B, T-1)

    # Top-k correctness
    topk_idx = logits.topk(k, dim=-1).indices         # (B, T-1, k)
    labels_exp = labels.unsqueeze(-1).expand_as(topk_idx)
    correct_any = ((topk_idx == labels_exp) & mask.unsqueeze(-1)).any(dim=-1)  # (B, T-1)

    num_correct = int(correct_any.sum().item())
    num_total   = int(mask.sum().item())
    accuracy = (num_correct / num_total) if num_total > 0 else 0.0

    return accuracy, num_correct, num_total

def greedy_decode_batch(
    model,
    batch,
    tokenizer,
    log: bool = True,
):
    """
    Greedy decode for *every* sample in the batch.
      - Uses batch['input_ids'] and batch['embedding'] as inputs
      - Compares generated continuation vs ground-truth continuation from labels
    Returns a list of dicts, one per sample.
    """
    results = []
    # Store original training state to restore later
    was_training = model.training
    model.eval()
    
    bsz = batch["input_ids"].size(0)
    for i in range(bsz):
        input_ids = batch["input_ids"][i].unsqueeze(0)      # (1, T)
        embeddings = batch["embedding"][i].unsqueeze(0)     # (1, T, D)
        labels = batch["labels"][i].unsqueeze(0)          # (1, T)
        
        # ---- 1) Find answer start from labels mask
        non_ign = (labels != IGNORE_INDEX).nonzero(as_tuple=False)
        if non_ign.numel() == 0:
            raise ValueError("No answer tokens found (all labels are IGNORE_INDEX).")
        ans_start = non_ign[0][1].item() + 1  # first supervised token
        if ans_start <= 0:
            raise ValueError("Answer starts at position 0; need at least one context token.")

        # ---- 2) Truncate inputs to context window
        input_ids = input_ids[:, :ans_start]          # (1, ans_start)
        attn_mask = input_ids.ne(tokenizer.pad_token_id).to(input_ids.device)

        max_new_tokens = p_args.data.max_len
        gen_max_len = getattr(p_args.data, "gen_max_len", None)
        if gen_max_len is not None:
            max_new_tokens = int(gen_max_len)

        # Greedy generate
        generated = None
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                hidden_embeds=embeddings,
                attention_mask=attn_mask,
                do_sample=False,
                num_beams=1,
                max_new_tokens=p_args.data.max_len,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        # Ground-truth continuation: labels already align to next tokens
        greedy_ids = generated[0, ans_start:].tolist()
        greedy_text = tokenizer.decode(greedy_ids, skip_special_tokens=True)

        # Get ground truth IDs and truncate at EOS token
        gt_ids = batch['input_ids'][i, ans_start:]
        eos_token_id = tokenizer.eos_token_id
        
        # Find EOS token position and truncate
        eos_pos = (gt_ids == eos_token_id).nonzero(as_tuple=True)[0]
        if eos_pos.numel() > 0:
            # Truncate at first EOS token
            gt_ids = gt_ids[:eos_pos[0].item()]
        
        gt_ids = gt_ids[:max_new_tokens]
        gt_text = tokenizer.decode(gt_ids.tolist(), skip_special_tokens=True)

        result = {
            "gt_ids": gt_ids.tolist(),
            "greedy_ids": greedy_ids,
            "gt_text": gt_text,
            "greedy_text": greedy_text,
        }
        results.append(result)

        if log:
            logger.info("========== Greedy decode vs GT ==========")
            logger.info(f"[Sample {i}] GT next: {gt_text}")
            logger.info(f"[Sample {i}] GEN greedy: {greedy_text}")

    # Restore original training state
    if was_training:
        model.train()
    
    return results