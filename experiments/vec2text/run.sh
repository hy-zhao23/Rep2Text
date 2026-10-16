#!/bin/bash

IS_SOURCED=0
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    IS_SOURCED=1
fi

if [[ "$IS_SOURCED" -eq 1 ]]; then
    set -uo pipefail
else
    set -euo pipefail
fi

fail() {
    local code="${1:-1}"
    if [[ "$IS_SOURCED" -eq 1 ]]; then
        return "$code"
    fi
    exit "$code"
}

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate vec2text
else
    echo "WARN: conda not found; using current Python environment"
fi

export PROJ=/workspace/embedding
export VEC2TEXT_DIR=$PROJ/scripts/vec2text
export PYTHONPATH=$VEC2TEXT_DIR:$PROJ:${PYTHONPATH:-}
export HF_HOME=$PROJ/tmp/huggingface
export WANDB_API_KEY="c42a2de45508b901c9a3c5bb5f07bc6c1bb3f027"
export WANDB_MODE=online
export GPT_ID="gpt-4.1-mini"

DATASET_NAME=${DATASET_NAME:-local_npz_embeddings}
USE_LESS_DATA=${USE_LESS_DATA:--1}
LOCAL_DATASET_PATH=${LOCAL_DATASET_PATH:-/workspace/embedding/tmp/embeddings/mistral7B/l10t32}
LOCAL_TEXT_COLUMN=${LOCAL_TEXT_COLUMN:-text}
LOCAL_NPZ_EMBEDDINGS_GLOB=${LOCAL_NPZ_EMBEDDINGS_GLOB:-embeddings_r*_b*_l32_o0.npz}
LOCAL_NPZ_FILE_LIMIT=${LOCAL_NPZ_FILE_LIMIT:--1}
NUM_REPEAT_TOKENS=${NUM_REPEAT_TOKENS:-16}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-32}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_STEPS=${MAX_STEPS:-3000}
EVAL_STEPS=${EVAL_STEPS:-50}
SAVE_STEPS=${SAVE_STEPS:-100}
LOGGING_STEPS=${LOGGING_STEPS:-20}
WARMUP_STEPS=${WARMUP_STEPS:-100}
USE_WANDB=${USE_WANDB:-True}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJ/logs/vec2text/wiki_l10t32}
EMBEDDER_MODEL_NAME=${EMBEDDER_MODEL_NAME:-mistralai/Mistral-7B-v0.1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NUM_GPUS=${NUM_GPUS:-1}
CLEAN_OUTPUT_DIR=${CLEAN_OUTPUT_DIR:-True}

case "$DATASET_NAME" in
    nq|luar_reddit|msmarco|one_million_instructions|one_million_paired_instructions|local|local_npz_embeddings)
        ;;
    *)
        echo "ERROR: 不支持的数据集 DATASET_NAME=$DATASET_NAME"
        echo "可选: nq, luar_reddit, msmarco, one_million_instructions, one_million_paired_instructions, local, local_npz_embeddings"
        fail 1
        ;;
esac

if [[ "$DATASET_NAME" == "local" && -z "$LOCAL_DATASET_PATH" ]]; then
    echo "ERROR: DATASET_NAME=local 时必须设置 LOCAL_DATASET_PATH"
    fail 1
fi

if [[ "$DATASET_NAME" == "local_npz_embeddings" && -z "$LOCAL_DATASET_PATH" ]]; then
    echo "ERROR: DATASET_NAME=local_npz_embeddings 时必须设置 LOCAL_DATASET_PATH (目录)"
    fail 1
fi

if [[ -n "$LOCAL_DATASET_PATH" && "$DATASET_NAME" != "local" && "$DATASET_NAME" != "local_npz_embeddings" ]]; then
    echo "WARN: 已设置 LOCAL_DATASET_PATH，但 DATASET_NAME=$DATASET_NAME，不会读取本地数据。"
fi

if [[ "$USE_WANDB" == "True" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: USE_WANDB=True 但未设置 WANDB_API_KEY。"
    echo "请先执行: export WANDB_API_KEY=你的key"
    fail 1
fi

cd "$PROJ"

echo "========== vec2text effective config =========="
echo "DATASET_NAME=$DATASET_NAME"
echo "LOCAL_DATASET_PATH=$LOCAL_DATASET_PATH"
echo "LOCAL_NPZ_EMBEDDINGS_GLOB=$LOCAL_NPZ_EMBEDDINGS_GLOB"
echo "LOCAL_NPZ_FILE_LIMIT=$LOCAL_NPZ_FILE_LIMIT"
echo "EMBEDDER_MODEL_NAME=$EMBEDDER_MODEL_NAME"
echo "MAX_SEQ_LENGTH=$MAX_SEQ_LENGTH"
echo "NUM_REPEAT_TOKENS=$NUM_REPEAT_TOKENS"
echo "MAX_STEPS=$MAX_STEPS"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "NUM_GPUS=$NUM_GPUS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "==============================================="

mkdir -p "$PROJ/logs" "$PROJ/logs/cpu" "$PROJ/logs/gpu" "$HF_HOME"
if [[ "$CLEAN_OUTPUT_DIR" == "True" ]]; then
    rm -rf "$OUTPUT_DIR"
fi
rm -rf ~/.cache/torch_extensions


if [[ "$NUM_GPUS" -gt 1 ]]; then
    LAUNCH_CMD=(torchrun --standalone --nproc_per_node "$NUM_GPUS" -m vec2text.run)
else
    LAUNCH_CMD=(python -u -m vec2text.run)
fi

"${LAUNCH_CMD[@]}" \
    --experiment inversion \
    --model_name_or_path t5-base \
    --dataset_name "$DATASET_NAME" \
    --local_dataset_path "$LOCAL_DATASET_PATH" \
    --local_text_column "$LOCAL_TEXT_COLUMN" \
    --local_npz_embeddings_glob "$LOCAL_NPZ_EMBEDDINGS_GLOB" \
    --local_npz_file_limit "$LOCAL_NPZ_FILE_LIMIT" \
    --use_less_data "$USE_LESS_DATA" \
    --embedder_model_name "$EMBEDDER_MODEL_NAME" \
    --num_repeat_tokens "$NUM_REPEAT_TOKENS" \
    --embedder_no_grad True \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size "$BATCH_SIZE" \
    --max_eval_samples 100 \
    --num_train_epochs 5 \
    --max_steps "$MAX_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --logging_steps "$LOGGING_STEPS" \
    --logging_first_step True \
    --warmup_steps "$WARMUP_STEPS" \
    --learning_rate 0.001 \
    --output_dir "$OUTPUT_DIR" \
    --use_wandb "$USE_WANDB" \
    --bf16 False \
    --do_train \
    --use_frozen_embeddings_as_input True \
    --lr_scheduler_type constant_with_warmup

status=$?
if [[ "$status" -ne 0 ]]; then
    echo "run.sh failed with exit code $status"
    fail "$status"
fi
exit $status
