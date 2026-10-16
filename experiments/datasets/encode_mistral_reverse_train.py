"""GPU step: forward Mistral-7B over the cached REVERSED token shards and
save layer-10 last-token embeddings.

Reads:  tmp/embeddings_reverse/mistral7B/l10t16/tokens_r{R}_b{B}_l16_o0.npz
Writes: tmp/embeddings_reverse/mistral7B/l10t16/embeddings_r{R}_b{B}_l16_o0.npz

Token shards must be produced first by
  experiments.datasets.reverse_mistral_tokens

Sample order is preserved -> existing pretrain_*_index_*.pkl indexes onto these
shards directly.

Usage:
  python -m experiments.datasets.encode_mistral_reverse_train \
      [--shard-start S] [--shard-end E] [--batch B]
"""
import argparse
import os
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

PROJ = Path(__file__).resolve().parents[2]
SHARD_DIR = PROJ / "tmp/embeddings_reverse/mistral7B/l10t16"

EMB_MODEL = "mistralai/Mistral-7B-v0.1"
LAYER = 10
MAX_TOKENS = 16
HIDDEN_DIM = 4096


def list_shards():
    pat = re.compile(r"tokens_r(\d+)_b(\d+)_l16_o0\.npz")
    out = []
    for p in sorted(SHARD_DIR.iterdir()):
        m = pat.match(p.name)
        if not m:
            continue
        out.append((int(m[1]), int(m[2])))
    out.sort()
    return out


def encode_shard(rank, batch_id, model, device, batch_size, pad_id):
    tok_path = SHARD_DIR / f"tokens_r{rank}_b{batch_id}_l16_o0.npz"
    emb_path = SHARD_DIR / f"embeddings_r{rank}_b{batch_id}_l16_o0.npz"
    if emb_path.exists():
        return f"skip r{rank}_b{batch_id} (exists)"

    tokens = np.load(tok_path)["tokens"]  # (N, 16)
    n = tokens.shape[0]

    out_embs = np.zeros((n, HIDDEN_DIM), dtype=np.float32)
    attn_mask = (tokens != pad_id).astype(np.int64)

    for i in tqdm(range(0, n, batch_size), desc=f"r{rank}_b{batch_id}", leave=False):
        end = min(i + batch_size, n)
        ids = torch.from_numpy(tokens[i:end].astype(np.int64)).to(device)
        am = torch.from_numpy(attn_mask[i:end]).to(device)
        with torch.no_grad():
            out = model(
                input_ids=ids,
                attention_mask=am,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = out.hidden_states[LAYER + 1]
        seq_lens = am.shape[1] - 1 - am.flip(dims=(1,)).argmax(dim=1)
        last = hidden[torch.arange(end - i, device=device), seq_lens]
        out_embs[i:end] = last.float().cpu().numpy()

    tmp = emb_path.with_suffix(".tmp.npz")
    np.savez(tmp, embeddings=out_embs)
    tmp.replace(emb_path)
    return f"done r{rank}_b{batch_id} (n={n})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--shard-end", type=int, default=-1, help="exclusive; -1 = all")
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(PROJ / "tmp/huggingface"))

    shards = list_shards()
    if args.shard_end == -1:
        args.shard_end = len(shards)
    shards = shards[args.shard_start:args.shard_end]
    print(f"Processing {len(shards)} shard(s): {shards}")

    tok = AutoTokenizer.from_pretrained(
        EMB_MODEL, use_fast=True, cache_dir=os.environ["HF_HOME"],
        token=os.environ.get("HF_API_KEY"),
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id

    print(f"Loading {EMB_MODEL} ...")
    model = AutoModelForCausalLM.from_pretrained(
        EMB_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        cache_dir=os.environ["HF_HOME"],
        token=os.environ.get("HF_API_KEY"),
    )
    model.eval()
    device = next(model.parameters()).device

    for (r, b) in shards:
        msg = encode_shard(r, b, model, device, args.batch, pad_id)
        print(msg, flush=True)


if __name__ == "__main__":
    main()
