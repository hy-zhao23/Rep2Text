#!/usr/bin/env python3
"""
Unified vec2text evaluation script.

Modes
-----
  wikipedia  (default)  In-distribution evaluation on the Wikipedia val split.
                        Loads pre-computed embeddings directly from NPZ files
                        (no GPU needed for embedding).
  clinical              OOD evaluation on clinical notes.
                        Embeds texts on-the-fly with Mistral-7B.

Usage
-----
    # in-distribution (Wikipedia val)
    python test/scripts/eval_vec2text.py --mode wikipedia \\
        [--checkpoint test/results/mistral7b_l10t32_20260303_154531] \\
        [--npz_dir    $PROJ/tmp/embeddings/mistral7B/l10t32] \\
        [--max_samples 1000] [--batch_size 32] [--output results.json]

    # OOD (clinical notes)
    python test/scripts/eval_vec2text.py --mode clinical \\
        [--checkpoint test/results/mistral7b_l10t32_20260303_154531] \\
        [--data       $PROJ/datasets/wikipedia/clinical_notes/clinical_notes_filtered.json] \\
        [--field summary|note] [--max_samples 200] [--batch_size 8] \\
        [--max_tokens 32] [--output results.json]
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from tqdm import tqdm

# ── path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent   # test/scripts/
VEC2TEXT_DIR = SCRIPT_DIR.parents[1]           # scripts/vec2text/
PROJ = VEC2TEXT_DIR.parents[1]                 # embedding/

if str(VEC2TEXT_DIR) not in sys.path:
    sys.path.insert(0, str(VEC2TEXT_DIR))

os.environ.setdefault("HF_HOME", str(PROJ / "tmp" / "huggingface"))

# ── constants ─────────────────────────────────────────────────────────────────
EMBEDDER_MODEL = "mistralai/Mistral-7B-v0.1"
MAX_TOKENS = 32
DEFAULT_EMBEDDINGS_FROM_LAYER_N = 11

DEFAULT_CHECKPOINT = str(
    VEC2TEXT_DIR / "test" / "results" / "mistral7b_l10t32_20260303_154531"
)
DEFAULT_NPZ_DIR = str(PROJ / "tmp" / "embeddings" / "mistral7B" / "l10t32")
DEFAULT_CLINICAL_DATA = str(
    PROJ / "datasets" / "wikipedia" / "clinical_notes" / "clinical_notes_filtered.json"
)


# ── wikipedia: NPZ val-split loader ──────────────────────────────────────────

def load_val_samples(npz_dir: str, tokenizer_name: str, max_samples: int,
                     validation_fraction: float = 0.1, test_fraction: float = 0.1):
    """
    Replicate the same split logic as LazyNPZInversionDataset to get val samples.
    Returns (embeddings np.ndarray [N, D], texts list[str]).
    """
    emb_files = sorted(glob.glob(os.path.join(npz_dir, "embeddings_*.npz")))
    tok_files = [
        os.path.join(npz_dir, os.path.basename(f).replace("embeddings_", "tokens_"))
        for f in emb_files
    ]
    pairs = [(e, t) for e, t in zip(emb_files, tok_files) if os.path.exists(t)]
    shard_sizes = [np.load(e, mmap_mode="r")["embeddings"].shape[0] for e, _ in pairs]

    total = sum(shard_sizes)
    n_val = max(1, int(total * validation_fraction))
    n_test = max(1, int(total * test_fraction))
    n_train = total - n_val - n_test
    val_start = n_train
    val_end = n_train + n_val

    print(f"  Total NPZ samples : {total:,}")
    print(f"  Train             : {n_train:,}")
    print(f"  Val               : {n_val:,}  [{val_start:,} – {val_end:,})")
    print(f"  Using first       : {min(max_samples, n_val):,} val samples")

    src_tok = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name, cache_dir=os.environ["HF_HOME"], use_fast=True
    )
    if src_tok.pad_token is None:
        src_tok.pad_token = src_tok.eos_token

    embeddings_list, texts_list = [], []
    needed = min(max_samples, n_val)
    collected = 0
    cumulative = 0

    for (emb_path, tok_path), shard_size in zip(pairs, shard_sizes):
        shard_start = cumulative
        shard_end = cumulative + shard_size
        cumulative = shard_end

        overlap_start = max(shard_start, val_start)
        overlap_end = min(shard_end, val_end)
        if overlap_start >= overlap_end:
            continue

        local_start = overlap_start - shard_start
        take = min(overlap_end - overlap_start, needed - collected)

        emb_npz = np.load(emb_path, mmap_mode="r")
        tok_npz = np.load(tok_path, mmap_mode="r")
        embeddings_list.append(emb_npz["embeddings"][local_start: local_start + take])
        for row in tok_npz["tokens"][local_start: local_start + take]:
            text = src_tok.decode(
                row.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            texts_list.append(text.strip())

        collected += take
        if collected >= needed:
            break

    return np.concatenate(embeddings_list, axis=0), texts_list


# ── clinical: Mistral-7B last-token embedder ─────────────────────────────────

def get_last_token_embeddings(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    texts: list,
    batch_size: int = 8,
    max_len: int = MAX_TOKENS,
    embeddings_from_layer_n: int = DEFAULT_EMBEDDINGS_FROM_LAYER_N,
) -> torch.Tensor:
    """
    Extract last real-token hidden state from the selected embedder layer.
    Returns a float32 CPU tensor of shape (N, 4096).
    """
    device = next(model.parameters()).device
    all_embs = []
    model.eval()

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding (Mistral)"):
        batch = texts[i: i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
            add_special_tokens=False,
        ).to(device)

        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )

        hidden = out.hidden_states[embeddings_from_layer_n]                # (B, S, D)
        mask = enc["attention_mask"].to(hidden.device)
        seq_lens = mask.shape[1] - 1 - mask.flip(dims=(1,)).argmax(dim=1)  # (B,)
        rows = torch.arange(len(batch), device=hidden.device)
        embs = hidden[rows, seq_lens]                                     # (B, D)
        all_embs.append(embs.float().cpu())

    return torch.cat(all_embs, dim=0)


# ── shared metric helpers ─────────────────────────────────────────────────────

def _token_f1(pred: str, ref: str) -> float:
    p_toks = pred.lower().split()
    r_toks = ref.lower().split()
    if not p_toks or not r_toks:
        return 0.0
    common = set(p_toks) & set(r_toks)
    if not common:
        return 0.0
    prec = len(common) / len(p_toks)
    rec = len(common) / len(r_toks)
    return 2 * prec * rec / (prec + rec)


def _compute_bleu(preds: list, refs: list):
    """Return (mean_score, per_sample_scores) where scores are in [0, 100]."""
    try:
        from sacrebleu.metrics import BLEU
        bleu = BLEU(effective_order=True)
        scores = [bleu.sentence_score(p, [r]).score for p, r in zip(preds, refs)]
        return float(np.mean(scores)), scores
    except ImportError:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        sf = SmoothingFunction().method3
        scores = [
            sentence_bleu([r.split()], p.split(), smoothing_function=sf) * 100
            for p, r in zip(preds, refs)
        ]
        return float(np.mean(scores)), scores


def _compute_rouge(preds: list, refs: list):
    """Return mean ROUGE-1, ROUGE-2, ROUGE-L scores (0-100) and per-sample dicts."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
        r1, r2, rl, per = [], [], [], []
        for p, r in zip(preds, refs):
            s = scorer.score(r, p)
            r1.append(s["rouge1"].fmeasure * 100)
            r2.append(s["rouge2"].fmeasure * 100)
            rl.append(s["rougeL"].fmeasure * 100)
            per.append({"rouge1": s["rouge1"].fmeasure * 100,
                        "rouge2": s["rouge2"].fmeasure * 100,
                        "rougeL": s["rougeL"].fmeasure * 100})
        return float(np.mean(r1)), float(np.mean(r2)), float(np.mean(rl)), per
    except ImportError:
        print("  [WARN] rouge_score not installed; skipping ROUGE")
        empty = [{"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}] * len(preds)
        return 0.0, 0.0, 0.0, empty


def _compute_emb_cosine(
    preds: list,
    frozen_embeddings: torch.Tensor,
    corrector,
    device: str,
    t5_tokenizer,
    config,
    batch_size: int,
) -> tuple:
    """Re-embed predictions with Mistral and compute cosine similarity to target."""
    pred_embs = []
    for i in range(0, len(preds), batch_size):
        batch = preds[i: i + batch_size]
        enc = t5_tokenizer(
            batch,
            max_length=config.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            emb = corrector.embed_generated_hypothesis(input_ids=enc["input_ids"])
        pred_embs.append(emb.float().cpu())
    pred_embs = torch.cat(pred_embs, dim=0)           # (N, D)
    target = frozen_embeddings.float().cpu()           # (N, D)
    cos = torch.nn.functional.cosine_similarity(pred_embs, target, dim=1)  # (N,)
    scores = cos.tolist()
    return float(cos.mean().item()), scores


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained vec2text InversionModel (wikipedia val or clinical OOD)"
    )
    parser.add_argument(
        "--mode", choices=["wikipedia", "clinical"], default="wikipedia",
        help="Evaluation mode: 'wikipedia' (in-dist, NPZ) or 'clinical' (OOD, JSON + Mistral)",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Path to trained vec2text checkpoint directory")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", default=None,
                        help="Path to save results JSON (default: <checkpoint>/results_<mode>.json)")

    # wikipedia-only args
    parser.add_argument("--npz_dir", default=DEFAULT_NPZ_DIR,
                        help="[wikipedia] Directory containing embeddings_*.npz + tokens_*.npz")

    # clinical-only args
    parser.add_argument("--data", default=DEFAULT_CLINICAL_DATA,
                        help="[clinical] Pre-filtered eval JSON")
    parser.add_argument("--field", default="summary", choices=["summary", "note"],
                        help="[clinical] JSON field to read (default: summary)")
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS,
                        help="[clinical] Max Mistral tokens for embedding truncation")
    parser.add_argument(
        "--embeddings_from_layer_n",
        type=int,
        default=DEFAULT_EMBEDDINGS_FROM_LAYER_N,
        help=(
            "[clinical] HF hidden_states layer index to embed from. "
            "Use 11 to match TransformerLens blocks.10.hook_resid_post / l10t32."
        ),
    )
    parser.add_argument(
        "--embedder_model", default=EMBEDDER_MODEL,
        help="[clinical] HF model id for the embedder (default: %(default)s)",
    )
    parser.add_argument(
        "--frozen_embeddings", default=None,
        help="[clinical] Pre-computed embeddings .npz (key 'emb_last_token' or 'embeddings'); "
             "skips loading the embedder model.",
    )

    # corrector / iterative refinement
    parser.add_argument("--corrector", default=None,
                        help="Path to CorrectorEncoderModel checkpoint; enables iterative refinement")
    parser.add_argument("--num_steps", type=int, default=20,
                        help="Number of corrector refinement steps (only with --corrector)")
    parser.add_argument("--base_results", default=None,
                        help="Path to pre-computed base inversion results JSON; "
                             "skips re-running the base model when --corrector is set")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mode_label = "Wikipedia Val (in-dist)" if args.mode == "wikipedia" else "Clinical Notes (OOD)"
    print(f"\n{'='*60}")
    print(f"  vec2text Evaluation — {mode_label}")
    print(f"{'='*60}")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  max_samples: {args.max_samples}")
    print(f"  batch_size : {args.batch_size}")
    print(f"  device     : {device}")
    if args.mode == "wikipedia":
        print(f"  npz_dir    : {args.npz_dir}")
    else:
        print(f"  data       : {args.data}")
        print(f"  field      : {args.field}")
        print(f"  max_tokens : {args.max_tokens}")
        print(f"  emb layer  : hidden_states[{args.embeddings_from_layer_n}]")
    if args.corrector:
        print(f"  corrector  : {args.corrector}")
        print(f"  num_steps  : {args.num_steps}")
        if args.base_results:
            print(f"  base_results (pre-computed): {args.base_results}")
    print()

    # ── Step 1: Load data & embeddings ────────────────────────────────────────
    if args.mode == "wikipedia":
        print("[1/3] Loading val split from NPZ ...")
        embeddings_np, texts = load_val_samples(
            npz_dir=args.npz_dir,
            tokenizer_name=args.embedder_model,
            max_samples=args.max_samples,
        )
        frozen_embeddings = torch.from_numpy(embeddings_np)
        print(f"      embeddings: {frozen_embeddings.shape}  texts: {len(texts)}")

    else:  # clinical
        print(f"[1/3] Loading clinical data (embedder: {args.embedder_model}) ...")
        with open(args.data) as f:
            records = json.load(f)
        texts = [r[args.field].strip() for r in records if r.get(args.field, "").strip()]
        texts = texts[: args.max_samples]
        print(f"      {len(texts)} samples selected")

        if args.frozen_embeddings:
            print(f"      Loading frozen embeddings from {args.frozen_embeddings}")
            _npz = np.load(args.frozen_embeddings, allow_pickle=True)
            key = "emb_last_token" if "emb_last_token" in _npz.files else "embeddings"
            frozen_embeddings = torch.from_numpy(_npz[key][: args.max_samples]).float()
            assert frozen_embeddings.shape[0] == len(texts), (
                f"Frozen embeddings rows {frozen_embeddings.shape[0]} != texts {len(texts)}"
            )
        else:
            src_tokenizer = transformers.AutoTokenizer.from_pretrained(
                args.embedder_model, cache_dir=os.environ["HF_HOME"], use_fast=True
            )
            if src_tokenizer.pad_token is None:
                src_tokenizer.pad_token = src_tokenizer.eos_token

            emb_model = transformers.AutoModelForCausalLM.from_pretrained(
                args.embedder_model,
                cache_dir=os.environ["HF_HOME"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            frozen_embeddings = get_last_token_embeddings(
                emb_model, src_tokenizer, texts,
                batch_size=args.batch_size,
                max_len=args.max_tokens,
                embeddings_from_layer_n=args.embeddings_from_layer_n,
            )
        print(f"      Embeddings: {frozen_embeddings.shape}  dtype={frozen_embeddings.dtype}")
        if not args.frozen_embeddings:
            del emb_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Step 2: Load checkpoint ───────────────────────────────────────────────
    print(f"\n[2/3] Loading tokeniser & InversionModel from checkpoint ...")
    t5_tokenizer = transformers.AutoTokenizer.from_pretrained(args.checkpoint)
    from vec2text.models.config import InversionConfig
    from vec2text.models import InversionModel

    config = InversionConfig.from_pretrained(args.checkpoint)
    inv_model = InversionModel.from_pretrained(
        args.checkpoint, config=config, ignore_mismatched_sizes=False,
    ).to(device)
    inv_model.eval()
    print(f"      {sum(p.numel() for p in inv_model.parameters())/1e6:.1f}M params  "
          f"max_seq_length={config.max_seq_length}")

    corrector = None
    if args.corrector:
        print(f"      Loading CorrectorEncoderModel from {args.corrector} ...")
        from vec2text.models import CorrectorEncoderModel
        from vec2text.api import load_corrector
        corrector_config = InversionConfig.from_pretrained(args.corrector)
        corrector_model = CorrectorEncoderModel.from_pretrained(
            args.corrector, config=corrector_config,
        )
        corrector = load_corrector(inv_model, corrector_model)
        corrector.inversion_trainer.model.to(device).eval()
        corrector.model.to(device).eval()
        print(f"      Corrector loaded — will run {args.num_steps} refinement steps")

    # ── Step 3: Generate & evaluate ───────────────────────────────────────────
    print(f"\n[3/3] Generating reconstructed texts ...")
    gen_kwargs = {
        "early_stopping": False,
        "num_beams": 1,
        "do_sample": False,
        "no_repeat_ngram_size": 0,
        "max_length": config.max_seq_length + 8,
    }

    # ── Pre-load base results if provided (avoids re-running base model) ─────
    preloaded_base_texts = None
    if args.corrector and args.base_results:
        print(f"      Loading pre-computed base results from {args.base_results} ...")
        with open(args.base_results) as f:
            _base_data = json.load(f)
        preloaded_base_texts = [s["prediction"] for s in _base_data["samples"]]
        preloaded_base_texts = preloaded_base_texts[: len(texts)]
        print(f"      Loaded {len(preloaded_base_texts)} base predictions")

    generated_texts = []
    base_texts = []          # step-0 hypothesis (base inversion), only populated when corrector is used
    import copy as _copy
    for i in tqdm(range(0, len(texts), args.batch_size), desc="  Generating", ncols=80):
        batch_emb = frozen_embeddings[i: i + args.batch_size].to(device)
        if corrector is not None:
            cg = _copy.copy(gen_kwargs)
            cg["min_length"] = 1
            corrector.inversion_trainer.model.eval()
            corrector.model.eval()
            corrector.return_best_hypothesis = False
            with torch.no_grad():
                if preloaded_base_texts is not None:
                    # ── Use pre-computed base texts as hypothesis (no base model call) ──
                    batch_base = preloaded_base_texts[i: i + args.batch_size]
                    base_texts.extend(batch_base)
                    # Tokenize with T5 tokenizer to get hypothesis_input_ids
                    hyp_enc = t5_tokenizer(
                        batch_base,
                        max_length=config.max_seq_length,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                    ).to(device)
                    hyp_ids  = hyp_enc["input_ids"]
                    hyp_mask = hyp_enc["attention_mask"]
                    # Embed hypothesis with Mistral via corrector's embedder
                    hyp_emb = corrector.embed_generated_hypothesis(input_ids=hyp_ids)
                    out_ids = corrector.generate(
                        inputs={
                            "frozen_embeddings":      batch_emb,
                            "hypothesis_input_ids":   hyp_ids,
                            "hypothesis_attention_mask": hyp_mask,
                            "hypothesis_embedding":   hyp_emb,
                        },
                        generation_kwargs=cg,
                        num_recursive_steps=args.num_steps,
                        sequence_beam_width=1,
                    )
                else:
                    # ── Run base model for step 0, then corrector ─────────────
                    base_ids = corrector.inversion_trainer.model.generate(
                        inputs={"frozen_embeddings": batch_emb},
                        generation_kwargs={
                            "early_stopping": False,
                            "num_beams": 1,
                            "do_sample": False,
                            "no_repeat_ngram_size": 0,
                            "max_length": config.max_seq_length,
                        },
                    )
                    base_texts.extend(t5_tokenizer.batch_decode(base_ids, skip_special_tokens=True))
                    hyp_mask = (base_ids != corrector.model.encoder_decoder.config.pad_token_id).int()
                    hyp_emb  = corrector.embed_generated_hypothesis(input_ids=base_ids)
                    out_ids = corrector.generate(
                        inputs={
                            "frozen_embeddings":         batch_emb,
                            "hypothesis_input_ids":      base_ids,
                            "hypothesis_attention_mask": hyp_mask,
                            "hypothesis_embedding":      hyp_emb,
                        },
                        generation_kwargs=cg,
                        num_recursive_steps=args.num_steps,
                        sequence_beam_width=1,
                    )
            generated_texts.extend(t5_tokenizer.batch_decode(out_ids, skip_special_tokens=True))
        else:
            with torch.no_grad():
                out_ids = inv_model.generate(
                    inputs={"frozen_embeddings": batch_emb},
                    generation_kwargs=gen_kwargs,
                )
            generated_texts.extend(t5_tokenizer.batch_decode(out_ids, skip_special_tokens=True))

    # ── Metrics ───────────────────────────────────────────────────────────────
    bleu, bleu_scores = _compute_bleu(generated_texts, texts)
    f1_scores = [_token_f1(p, r) for p, r in zip(generated_texts, texts)]
    exact = sum(p.strip() == r.strip() for p, r in zip(generated_texts, texts))
    split_label = "wikipedia_val" if args.mode == "wikipedia" else f"clinical_{args.field}"
    if args.corrector:
        split_label += f"_corrector{args.num_steps}steps"

    print(f"\n{'='*60}")
    print(f"  RESULTS  (n={len(texts)}, split='{split_label}')")
    print(f"{'='*60}")
    print(f"  BLEU         : {bleu:.2f}")
    print(f"  Token F1     : {np.mean(f1_scores)*100:.2f}%")
    print(f"  Exact Match  : {exact}/{len(texts)} ({exact/len(texts)*100:.2f}%)")
    print(f"  (Run compute_extra_metrics.py for ROUGE / BERTScore / GPT scores)")

    # ── Save results JSON ─────────────────────────────────────────────────────
    base_dir = Path(args.corrector) if args.corrector else Path(args.checkpoint)
    eval_dir = base_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output or str(eval_dir / f"results_{split_label}.json")
    with open(out_path, "w") as f:
        json.dump({
            "metrics": {
                "bleu": bleu,
                "token_f1": float(np.mean(f1_scores)) * 100,
                "exact_match": exact,
                "n": len(texts),
                "split": split_label,
            },
            "samples": [
                {"index": i, "ground_truth": gt, "prediction": pred, "token_f1": fi, "bleu": bi}
                for i, (gt, pred, fi, bi) in enumerate(zip(texts, generated_texts, f1_scores, bleu_scores))
            ],
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {out_path}")

    # ── Also save base inversion results (step 0) when corrector was used ────
    if base_texts:
        base_split_label = "wikipedia_val" if args.mode == "wikipedia" else f"clinical_{args.field}"
        base_bleu, base_bleu_scores = _compute_bleu(base_texts, texts)
        base_f1_scores = [_token_f1(p, r) for p, r in zip(base_texts, texts)]
        base_exact = sum(p.strip() == r.strip() for p, r in zip(base_texts, texts))

        print(f"\n{'='*60}")
        print(f"  BASE INVERSION RESULTS  (step 0, n={len(texts)}, split='{base_split_label}')")
        print(f"{'='*60}")
        print(f"  BLEU         : {base_bleu:.2f}")
        print(f"  Token F1     : {np.mean(base_f1_scores)*100:.2f}%")
        print(f"  Exact Match  : {base_exact}/{len(texts)} ({base_exact/len(texts)*100:.2f}%)")
        print(f"  (Run compute_extra_metrics.py for ROUGE / BERTScore / GPT scores)")

        base_eval_dir = Path(args.checkpoint) / "eval"
        base_eval_dir.mkdir(parents=True, exist_ok=True)
        base_out_path = str(base_eval_dir / f"results_{base_split_label}.json")
        with open(base_out_path, "w") as f:
            json.dump({
                "metrics": {
                    "bleu": base_bleu,
                    "token_f1": float(np.mean(base_f1_scores)) * 100,
                    "exact_match": base_exact,
                    "n": len(texts),
                    "split": base_split_label,
                },
                "samples": [
                    {"index": i, "ground_truth": gt, "prediction": pred,
                     "token_f1": fi, "bleu": bi}
                    for i, (gt, pred, fi, bi) in enumerate(
                        zip(texts, base_texts, base_f1_scores, base_bleu_scores))
                ],
            }, f, indent=2, ensure_ascii=False)
        print(f"  Base results saved to: {base_out_path}")

    print("\n  [Sample outputs]")
    for i in range(min(5, len(texts))):
        print(f"\n  [{i}] GT  : {texts[i][:100].replace(chr(10), ' ')}")
        if base_texts:
            print(f"       Base: {base_texts[i][:100].replace(chr(10), ' ')}")
        print(f"       Pred: {generated_texts[i][:100].replace(chr(10), ' ')}")
    print()


if __name__ == "__main__":
    main()
