"""Run vec2text base / corrector inverters on the 500 full-sentence embeddings.

Loads NPZ from tmp/embeddings/full_sentence/mistral7b/, drives vec2text
generation (base or with corrector at given num_steps), and saves results
to results/mistral7B/l10t32p32/full_sentence/{vec2text_base, vec2text_corrector50}/
"""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
import transformers
from tqdm import tqdm
import copy as _copy

PROJ = Path(__file__).resolve().parents[2]
VEC2TEXT_DIR = PROJ / "experiments" / "vec2text"
sys.path.insert(0, str(VEC2TEXT_DIR))
os.environ.setdefault("HF_HOME", str(PROJ / "tmp/huggingface"))

NPZ_DIR = PROJ / "tmp/embeddings/full_sentence/mistral7b"
SENT_PATH = PROJ / "results/mistral7B/l10t32p32/full_sentence/sentences.json"

V2T_BASE_CKPT = PROJ / "experiments/vec2text/test/results/mistral7b_l10t32_20260303_154531"
V2T_CORR_CKPT = PROJ / "experiments/vec2text/test/results/corrector_mistral7b_l10t32_20260307_145238"

EMBEDDER = "mistralai/Mistral-7B-v0.1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "corrector50"], required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load 500 truncated GT texts (these are the references)
    with open(SENT_PATH) as f:
        sentences = json.load(f)
    refs = [s["gt_truncated"] for s in sentences]

    # Load NPZ embeddings (must match order in sentences.json)
    embs_np = np.load(NPZ_DIR / "embeddings_r0_b0_l32_o0.npz")["embeddings"]
    assert embs_np.shape[0] == len(refs), f"NPZ has {embs_np.shape[0]} samples but sentences has {len(refs)}"
    frozen = torch.from_numpy(embs_np)
    print(f"Loaded {len(refs)} embeddings: {frozen.shape}")

    # Load base inversion model
    print(f"Loading vec2text base from {V2T_BASE_CKPT}")
    t5_tok = transformers.AutoTokenizer.from_pretrained(str(V2T_BASE_CKPT))
    from vec2text.models.config import InversionConfig
    from vec2text.models import InversionModel
    base_cfg = InversionConfig.from_pretrained(str(V2T_BASE_CKPT))
    inv_model = InversionModel.from_pretrained(
        str(V2T_BASE_CKPT), config=base_cfg, ignore_mismatched_sizes=False,
    ).to(device).eval()

    corrector = None
    if args.mode == "corrector50":
        from vec2text.models import CorrectorEncoderModel
        from vec2text.api import load_corrector
        corr_cfg = InversionConfig.from_pretrained(str(V2T_CORR_CKPT))
        corr_model = CorrectorEncoderModel.from_pretrained(str(V2T_CORR_CKPT), config=corr_cfg)
        corrector = load_corrector(inv_model, corr_model)
        corrector.inversion_trainer.model.to(device).eval()
        corrector.model.to(device).eval()

    gen_kwargs = {
        "early_stopping": False,
        "num_beams": 1,
        "do_sample": False,
        "no_repeat_ngram_size": 0,
        "max_length": base_cfg.max_seq_length + 8,
    }

    preds = []
    base_preds = []  # only used in corrector mode
    for i in tqdm(range(0, len(refs), args.batch_size), desc=f"vec2text-{args.mode}"):
        batch_emb = frozen[i: i + args.batch_size].to(device)
        with torch.no_grad():
            if corrector is None:
                out_ids = inv_model.generate(
                    inputs={"frozen_embeddings": batch_emb},
                    generation_kwargs=gen_kwargs,
                )
                preds.extend(t5_tok.batch_decode(out_ids, skip_special_tokens=True))
            else:
                # step 0
                base_ids = corrector.inversion_trainer.model.generate(
                    inputs={"frozen_embeddings": batch_emb},
                    generation_kwargs={
                        "early_stopping": False,
                        "num_beams": 1,
                        "do_sample": False,
                        "no_repeat_ngram_size": 0,
                        "max_length": base_cfg.max_seq_length,
                    },
                )
                base_preds.extend(t5_tok.batch_decode(base_ids, skip_special_tokens=True))
                hyp_mask = (base_ids != corrector.model.encoder_decoder.config.pad_token_id).int()
                hyp_emb = corrector.embed_generated_hypothesis(input_ids=base_ids)
                cg = _copy.copy(gen_kwargs)
                cg["min_length"] = 1
                corrector.return_best_hypothesis = False
                out_ids = corrector.generate(
                    inputs={
                        "frozen_embeddings": batch_emb,
                        "hypothesis_input_ids": base_ids,
                        "hypothesis_attention_mask": hyp_mask,
                        "hypothesis_embedding": hyp_emb,
                    },
                    generation_kwargs=cg,
                    num_recursive_steps=50,
                    sequence_beam_width=1,
                )
                preds.extend(t5_tok.batch_decode(out_ids, skip_special_tokens=True))

    method_name = "vec2text_base" if args.mode == "base" else "vec2text_corrector50"
    out_dir = PROJ / f"results/mistral7B/l10t32p32/full_sentence/{method_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "predictions.json"
    with open(out_file, "w") as f:
        json.dump([
            {"index": i, "ground_truth": r, "prediction": p}
            for i, (r, p) in enumerate(zip(refs, preds))
        ], f, indent=2, ensure_ascii=False)
    print(f"Saved {out_file}")

    if base_preds:
        out_dir2 = PROJ / "results/mistral7B/l10t32p32/full_sentence/vec2text_base"
        out_dir2.mkdir(parents=True, exist_ok=True)
        out_file2 = out_dir2 / "predictions.json"
        with open(out_file2, "w") as f:
            json.dump([
                {"index": i, "ground_truth": r, "prediction": p}
                for i, (r, p) in enumerate(zip(refs, base_preds))
            ], f, indent=2, ensure_ascii=False)
        print(f"Also saved base step-0 predictions to {out_file2}")


if __name__ == "__main__":
    main()
