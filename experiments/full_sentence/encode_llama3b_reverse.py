"""Encode the 500 long sentences after reversing word order.

Pipeline per sentence:
  1. word-split on whitespace
  2. reverse the word list
  3. join back with spaces -> re-tokenize with Llama-3.2-3B tokenizer
  4. encode with Llama-3.2-3B layer 10, take last non-pad position
"""
import json
import os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

PROJ = Path(__file__).resolve().parents[2]
EMB_MODEL = "meta-llama/Llama-3.2-3B"
LAYER = 10
MAX_TOKENS = 16

SENT_PATH = PROJ / "datasets/full_sentence_3b_long_500.json"
OUT_DIR = PROJ / "tmp/embeddings/full_sentence/llama3b_reverse"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def reverse_words(text: str) -> str:
    return " ".join(text.split()[::-1])


def main():
    os.environ.setdefault("HF_HOME", str(PROJ / "tmp/huggingface"))

    with open(SENT_PATH) as f:
        sentences = json.load(f)
    orig_texts = [s["gt_truncated"] for s in sentences]
    rev_texts = [reverse_words(t) for t in orig_texts]
    print(f"Loaded {len(orig_texts)} sentences from {SENT_PATH}")
    print(f"orig[0]: {orig_texts[0]}")
    print(f"rev[0] : {rev_texts[0]}")

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
    print(f"hidden_dim={hidden_dim}")

    embs = np.zeros((len(rev_texts), hidden_dim), dtype=np.float32)
    toks = np.full((len(rev_texts), MAX_TOKENS), tok.pad_token_id, dtype=np.int32)

    BATCH = 8
    for i in tqdm(range(0, len(rev_texts), BATCH), desc="Encoding (reversed)"):
        batch = rev_texts[i:i + BATCH]
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
        hidden = out.hidden_states[LAYER + 1]
        mask = enc["attention_mask"]
        seq_lens = mask.shape[1] - 1 - mask.flip(dims=(1,)).argmax(dim=1)
        last = hidden[torch.arange(len(batch), device=device), seq_lens]

        embs[i:i + len(batch)] = last.float().cpu().numpy()
        toks[i:i + len(batch)] = enc["input_ids"].cpu().numpy().astype(np.int32)

    emb_path = OUT_DIR / f"embeddings_r0_b0_l{MAX_TOKENS}_o0.npz"
    tok_path = OUT_DIR / f"tokens_r0_b0_l{MAX_TOKENS}_o0.npz"
    txt_path = OUT_DIR / "reversed_texts.json"
    np.savez(emb_path, embeddings=embs)
    np.savez(tok_path, tokens=toks)
    with open(txt_path, "w") as f:
        json.dump(
            [{"orig": o, "reversed": r} for o, r in zip(orig_texts, rev_texts)],
            f, indent=2, ensure_ascii=False,
        )
    print(f"Saved embeddings: {emb_path}  shape={embs.shape}")
    print(f"Saved tokens:     {tok_path}  shape={toks.shape}")
    print(f"Saved texts:      {txt_path}")


if __name__ == "__main__":
    main()
