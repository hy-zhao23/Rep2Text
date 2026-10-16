"""Run our trained Qwen14B inverter on the 500 full-sentence embeddings.

Reads NPZ from tmp/embeddings/full_sentence/mistral7b/, runs batched greedy
generation with KV cache enabled, and saves results to
results/mistral7B/l10t32p32/full_sentence/ours/val_einv_decoded.json
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# EINV_CONFIG must be set before importing utils
os.environ.setdefault("EINV_CONFIG", "transfer/q14b-32token.yaml")
os.environ.setdefault("TRAIN_TAG", "transfer-q14b-32token-lr15")

from utils import p_args, gh, logger
from utils.models import load_pretrained_model
from utils.models.constants import IGNORE_INDEX, DEFAULT_EM_TOKEN
from transformers import AutoTokenizer

PROJ = Path(__file__).resolve().parents[2]
NPZ_DIR = PROJ / "tmp/embeddings/full_sentence/mistral7b"
SENT_PATH = PROJ / "results/mistral7B/l10t32p32/full_sentence/sentences.json"
OUT_DIR = PROJ / "results/mistral7B/l10t32p32/full_sentence/ours"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 4   # keep small for multi-GPU sharded 14B; bump if memory permits


def build_prefix(tokenizer, sys_prompt, user_prompt):
    em_tokens = DEFAULT_EM_TOKEN * p_args.proj.n_vecs
    bos = tokenizer.bos_token if tokenizer.bos_token is not None else ""
    return f"{bos}{em_tokens}\n{sys_prompt}\n{user_prompt}\nAnswer:"


def main():
    print(f"[ours] Loading model with device_map='auto' ...", flush=True)
    tokenizer, model, _ = load_pretrained_model(
        model_type="einv", resume_from=True,
        device="cuda", device_map="auto",
    )
    base_model = model.unwrap_module() if hasattr(model, "unwrap_module") else model
    input_device = next(base_model.parameters()).device
    print(f"[ours] Model loaded. Input device = {input_device}", flush=True)

    # Source tokenizer (mistral) for decoding stored token ids → text
    src_tok = AutoTokenizer.from_pretrained(p_args.model.emb_id, use_fast=True)
    if src_tok.pad_token is None:
        src_tok.pad_token = src_tok.eos_token

    # Load NPZ
    embs = np.load(NPZ_DIR / "embeddings_r0_b0_l32_o0.npz")["embeddings"]
    src_tokens = np.load(NPZ_DIR / "tokens_r0_b0_l32_o0.npz")["tokens"]
    print(f"[ours] Loaded {len(embs)} embeddings of shape {embs.shape[1]}", flush=True)

    # Load prompts
    with open(p_args.path.prompt_dir / "prompts.json") as f:
        prompts = json.load(f)
    with open(p_args.path.prompt_dir / "sys_prompt.txt") as f:
        sys_prompt = f.read()

    # Decode src tokens to text and re-encode answers (for GT recording only)
    gt_texts = []
    gt_id_lists = []
    for row in src_tokens:
        ids = [int(t) for t in row if int(t) != src_tok.pad_token_id]
        text = src_tok.decode(ids, skip_special_tokens=True).strip()
        gt_texts.append(text)
        gt_id_lists.append(tokenizer.encode(text, add_special_tokens=False))

    # Build prefix once (use same prompt for all to make eval deterministic)
    user_prompt = prompts[0]
    prefix_str = build_prefix(tokenizer, sys_prompt, user_prompt)
    prefix_ids = tokenizer.encode(prefix_str, add_special_tokens=False)
    prefix_t = torch.tensor(prefix_ids, dtype=torch.long, device=input_device)
    prefix_attn = torch.ones_like(prefix_t)
    print(f"[ours] Prefix length: {len(prefix_ids)}", flush=True)

    gen_max = int(getattr(p_args.data, "gen_max_len", 32))
    print(f"[ours] gen_max_len = {gen_max}, batch_size = {BATCH_SIZE}", flush=True)

    decoded_results = []
    model.eval()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    for i in tqdm(range(0, len(embs), BATCH_SIZE), desc="ours-gen", file=sys.stdout, mininterval=2.0):
        batch_emb = torch.tensor(embs[i:i + BATCH_SIZE], dtype=torch.bfloat16, device=input_device)
        batch_emb = torch.nn.functional.normalize(batch_emb, p=2, dim=-1)
        bsz = batch_emb.size(0)

        input_ids = prefix_t.unsqueeze(0).expand(bsz, -1).contiguous()
        attn_mask = prefix_attn.unsqueeze(0).expand(bsz, -1).contiguous()

        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                hidden_embeds=batch_emb,
                attention_mask=attn_mask,
                do_sample=False,
                num_beams=1,
                max_new_tokens=gen_max,
                pad_token_id=pad_id,
                use_cache=True,
            )
        gen_ids = out[:, input_ids.size(1):]                # (B, gen_max)

        for b in range(bsz):
            idx = i + b
            g_ids = gen_ids[b].tolist()
            # truncate at EOS if present
            if tokenizer.eos_token_id in g_ids:
                g_ids = g_ids[:g_ids.index(tokenizer.eos_token_id)]
            g_text = tokenizer.decode(g_ids, skip_special_tokens=True)
            decoded_results.append({
                "gt_ids":     gt_id_lists[idx][:gen_max],
                "greedy_ids": g_ids,
                "gt_text":    gt_texts[idx],
                "greedy_text": g_text,
            })

        if i % (BATCH_SIZE * 10) == 0:
            print(f"  [ours] sample {i + bsz}/{len(embs)} done", flush=True)

    out_path = OUT_DIR / "val_einv_decoded.json"
    with open(out_path, "w") as f:
        json.dump(decoded_results, f, indent=2, ensure_ascii=False)
    print(f"[ours] Saved {len(decoded_results)} results → {out_path}", flush=True)


if __name__ == "__main__":
    main()
