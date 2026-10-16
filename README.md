# Rep2Text: Decoding Full Text from a Single LLM Token Representation

[Paper](https://openreview.net/forum?id=OET7WYchkH) · [Checkpoints](https://huggingface.co/adfjasigageadfd/rep2text-checkpoints)

Rep2Text reconstructs a text sequence from a single vector: the residual-stream activation of a
frozen LLM at one layer and the last token position. A small projector maps this vector to a set
of soft tokens, and a frozen decoder LLM generates the original text from them. The projector is
the only trained module. An optional second stage also fine-tunes the decoder with LoRA.

```
text ──► frozen target LLM ──► hidden state at layer L, last token ──► projector ──► N soft tokens ──► frozen decoder ──► text
```

## Installation

```bash
conda create -n rep2text python=3.12 && conda activate rep2text
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

The tokenization step additionally needs `mpi4py`, which must be built against your local MPI
(`conda install -c conda-forge mpi4py openmpi`). `requirements-lock.txt` pins the exact package
versions we used.

Before running the code, set the following environment variables:

```bash
export PROJ=/path/to/Rep2Text           # repository root; all paths in the configs are relative to it
export EINV_CONFIG=configs/setup.yaml   # setup config: models, layer, sequence length, projector
export HF_API_KEY=hf_...                # Llama and Gemma are gated on Hugging Face
export OPENAI_API_KEY=sk-...            # only needed for the GPT-judge evaluation
export WANDB_MODE=disabled              # or set WANDB_API_KEY to log to Weights & Biases
```

Each experiment is defined by two configs. The setup config, selected through `EINV_CONFIG`,
specifies the target and decoder models, the layer, the sequence length and the projector shape;
the data preparation scripts read only this file. The training config, passed with `--config`,
specifies the optimization settings.

## Training and evaluation

The commands below are the SLURM versions we ran. Without SLURM, use `mpirun -n <N> python` for
the tokenization step and `python` for the embedding step.

First, build the training set from the English Wikipedia subset of `ArmelR/the-pile-splitted` and
cache one hidden-state vector per sequence:

```bash
srun python experiments/datasets/tokenize_sequences.py
python experiments/precache.py --model meta-llama/Llama-3.1-8B
srun python experiments/datasets/token_to_embeddings.py
```

Run the tokenization step with a single process to reproduce our training split, since the
sampling depends on the number of processes. Then train the projector, and optionally fine-tune
the decoder with LoRA:

```bash
accelerate launch --num_processes $NUM_GPU --multi_gpu --mixed_precision bf16 \
    experiments/train/pretrain.py --config configs/train.yaml

accelerate launch --num_processes $NUM_GPU --multi_gpu --mixed_precision bf16 \
    experiments/train/finetune.py --config configs/finetune.yaml
```

Finally, evaluate generation quality, the GPT-judge scores, and the out-of-distribution clinical set:

```bash
accelerate launch --num_processes $NUM_GPU --multi_gpu --mixed_precision bf16 \
    experiments/evaluation/eval_gen.py --model_type einv --eval val --config configs/train.yaml
python experiments/evaluation/eval_gpt.py --model_type einv --config configs/train.yaml
python experiments/evaluation/summary.py --model_type einv --config configs/train.yaml
python experiments/test/ood_clinical_summary.py --config configs/train.yaml
```

Use `--model_type einv_lora` to evaluate the LoRA-fine-tuned model.

The configs for the experiments in the paper are under `configs/`. `layer/` and `layer_sweep/`
vary the layer that is inverted. `tokens/` varies the sequence length from 8 to 256 tokens.
`n_vecs/` varies the number of soft tokens from 1 to 32. `expansion/` varies the projector width
from 0.5 to 8. `models/` and `transfer/` cover the target–decoder combinations, the training data
size ablation, and the reverse and random controls.

## Released checkpoints

We release the trained projector for every run in the paper at
[adfjasigageadfd/rep2text-checkpoints](https://huggingface.co/adfjasigageadfd/rep2text-checkpoints).
The target models are Llama-3.1-8B, Llama-3.2-3B, Mistral-7B and Gemma-7B, and the decoders are
Llama-3.1-8B, Llama-3.2-3B and Qwen2.5-14B/32B/72B. Each run folder contains `projector.bin`.
The LoRA run in `full_finetune/` additionally contains the decoder adapter and the projector
trained jointly with it (`lora_projector.bin`). The base models are not included and are
downloaded from their original Hugging Face repositories.

```bash
hf download adfjasigageadfd/rep2text-checkpoints \
    --include "table1/llama3.1-8b-L10_to_llama3.1-8b/*" --local-dir checkpoints
```

To evaluate a released projector, point `EINV_CONFIG` to the matching setup config and copy
`projector.bin` into `weights/<model>/l<layer>t<length>p<n_vecs>/`, which is where `eval_gen.py`
loads it from. If `TRAIN_TAG` is set, the file goes in a subfolder with that name.

## Evaluation data

`datasets/test/wikipedia/` contains the 1,000 held-out Wikipedia sequences used for each target
model and sequence length. `datasets/test/clinical/clinical_summaries.jsonl` contains the 1,000
out-of-distribution clinical summaries, which we generated with Qwen2.5-32B from
[Sadaftb/clinical-nlp-patient-notes](https://huggingface.co/datasets/Sadaftb/clinical-nlp-patient-notes).

## Vec2Text baseline

`experiments/vec2text/` contains our baseline, a copy of
[vec2text](https://github.com/jxmorris12/vec2text) (Morris et al.) modified to train on the hidden
states cached by our pipeline. To train it, run `vec2text/run.py` with
`--dataset_name local_npz_embeddings` and set `VEC2TEXT_LAZY_NPZ_DIR` to an embedding directory,
e.g. `tmp/embeddings/llama8B/l10t32`.

## License

This project is released under the [MIT License](LICENSE).

## Citation

```bibtex
@inproceedings{zhao2026rep2text,
  title={Rep2Text: Decoding Full Text from a Single LLM Token Representation},
  author={Zhao, Haiyan and He, Zirui and Tang, Yiming and Yang, Fan and Payani, Ali and Liu, Dianbo and Du, Mengnan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```
