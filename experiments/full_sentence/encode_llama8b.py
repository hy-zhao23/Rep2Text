"""Encode the 500 sentence-aligned 16-token texts with Llama-3.1-8B layer 10.

Input:  results/llama8B/l10t16p16/full_sentence/sentences_500.json (gt_truncated)
Output: tmp/embeddings/full_sentence/llama8b/{embeddings,tokens}_r0_b0_l16_o0.npz
  - embeddings: (N, 4096) float32 — layer-10 hidden state at last real token position
  - tokens:     (N, 16)   int32   — padded token ids (llama pad)
"""
import json
import os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

PROJ = Path(__file__).resolve().parents[2]
EMB_MODEL = "meta-llama/Llama-3.1-8B"
LAYER = 10
MAX_TOKENS = 16

SENT_PATH = PROJ / "results/llama8B/l10t16p16/full_sentence/sentences_500.json"
OUT_DIR = PROJ / "tmp/embeddings/full_sentence/llama8b"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    os.environ.setdefault("HF_HOME", str(PROJ / "tmp/huggingface"))

    with open(SENT_PATH) as f:
        sentences = json.load(f)
    texts = [s["gt_truncated"] for s in sentences]
    print(f"Loaded {len(texts)} sentences from {SENT_PATH}")

    tok = AutoTokenizer.from_pretrained(
        EMB_MODEL, use_fast=True, cache_dir=os.environ["HF_HOME"],
        token=os.environ.get("HF_API_KEY"),
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

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
    hidden_dim = model.config.hidden_size

    embs = np.zeros((len(texts), hidden_dim), dtype=np.float32)
    toks = np.full((len(texts), MAX_TOKENS), tok.pad_token_id, dtype=np.int32)

    BATCH = 8
    for i in tqdm(range(0, len(texts), BATCH), desc="Encoding"):
        batch = texts[i:i + BATCH]
        enc = tok(
            batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_TOKENS,
            add_special_tokens=False,
        ).to(device)

        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
        # hidden_states[L+1] = output of L-th block (matches the encode_mistral convention)
        hidden = out.hidden_states[LAYER + 1]                              # (B, T, D)
        mask = enc["attention_mask"]
        seq_lens = mask.shape[1] - 1 - mask.flip(dims=(1,)).argmax(dim=1)  # (B,)
        last = hidden[torch.arange(len(batch), device=device), seq_lens]   # (B, D)

        embs[i:i + len(batch)] = last.float().cpu().numpy()
        toks[i:i + len(batch)] = enc["input_ids"].cpu().numpy().astype(np.int32)

    emb_path = OUT_DIR / f"embeddings_r0_b0_l{MAX_TOKENS}_o0.npz"
    tok_path = OUT_DIR / f"tokens_r0_b0_l{MAX_TOKENS}_o0.npz"
    np.savez(emb_path, embeddings=embs)
    np.savez(tok_path, tokens=toks)
    print(f"Saved embeddings: {emb_path}  shape={embs.shape}")
    print(f"Saved tokens:     {tok_path}  shape={toks.shape}")


if __name__ == "__main__":
    main()
