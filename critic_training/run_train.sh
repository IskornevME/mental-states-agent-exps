#!/usr/bin/env bash
set -Eeuo pipefail


# -----------------------------------------------------------------------------
# Required inputs
# -----------------------------------------------------------------------------

if [[ -z "${BACKBONE_PATH:-}" ]]; then
    echo "[ERROR] BACKBONE_PATH must point to a HuggingFace causal-LM checkpoint." >&2
    exit 1
fi

if [[ -z "${Q_DATA_PATH:-}" ]]; then
    echo "[ERROR] Q_DATA_PATH must point to q_dataset.json." >&2
    exit 1
fi

if [[ ! -f "${Q_DATA_PATH}" ]]; then
    echo "[ERROR] Q dataset not found: ${Q_DATA_PATH}" >&2
    exit 1
fi


# -----------------------------------------------------------------------------
# Model / tokenizer
# -----------------------------------------------------------------------------

TOKENIZER_PATH="${TOKENIZER_PATH:-${BACKBONE_PATH}}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-False}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
APPLY_SIGMOID="${APPLY_SIGMOID:-False}"


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

BACKBONE_NAME="$(basename "${BACKBONE_PATH%/}")"
RUN_NAME="${RUN_NAME:-${BACKBONE_NAME}-qnet}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/critic_training/${RUN_NAME}}"


# -----------------------------------------------------------------------------
# GPU / batch size
# -----------------------------------------------------------------------------

TRAIN_PYTHON="${TRAIN_PYTHON:-python}"

GPU_LIST="${GPU_LIST:-0,1}"

IFS=',' read -ra GPU_ARRAY <<< "${GPU_LIST}"

NUM_GPUS="${#GPU_ARRAY[@]}"

if (( NUM_GPUS < 1 )); then
    echo "[ERROR] GPU_LIST must contain at least one GPU." >&2
    exit 1
fi

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
DENOM=$((NUM_GPUS * MICRO_BATCH_SIZE))

if (( GLOBAL_BATCH_SIZE % DENOM != 0 )); then
    echo "[ERROR] GLOBAL_BATCH_SIZE must be divisible by NUM_GPUS * MICRO_BATCH_SIZE." >&2
    exit 1
fi

GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / DENOM))


# -----------------------------------------------------------------------------
# Training hyperparameters
# -----------------------------------------------------------------------------

MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-8192}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
SEED="${SEED:-42}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

# -1 means that NUM_EPOCHS controls training.
# Set e.g. TRAIN_MAX_STEPS=2 for a very short smoke test.
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:--1}"

MASTER_PORT="${MASTER_PORT:-20001}"


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo "[CONFIG] Backbone:          ${BACKBONE_PATH}"
echo "[CONFIG] Tokenizer:         ${TOKENIZER_PATH}"
echo "[CONFIG] Q data:            ${Q_DATA_PATH}"
echo "[CONFIG] Output:            ${OUTPUT_DIR}"
echo "[CONFIG] GPUs:              ${GPU_LIST}"
echo "[CONFIG] Num GPUs:          ${NUM_GPUS}"
echo "[CONFIG] Global batch:      ${GLOBAL_BATCH_SIZE}"
echo "[CONFIG] Micro batch:       ${MICRO_BATCH_SIZE}"
echo "[CONFIG] Grad accumulation: ${GRAD_ACCUM}"
echo "[CONFIG] Max length:        ${MODEL_MAX_LENGTH}"
echo "[CONFIG] Epochs:            ${NUM_EPOCHS}"
echo "[CONFIG] Max train steps:   ${TRAIN_MAX_STEPS}"
echo "[CONFIG] Learning rate:     ${LEARNING_RATE}"
echo "[CONFIG] Apply sigmoid:     ${APPLY_SIGMOID}"
echo "[CONFIG] Seed:              ${SEED}"


# -----------------------------------------------------------------------------
# Common train.py arguments
# -----------------------------------------------------------------------------

TRAIN_ARGS=(
    critic_training/train.py

    --model_name_or_path "${BACKBONE_PATH}"
    --tokenizer_name_or_path "${TOKENIZER_PATH}"
    --data_path "${Q_DATA_PATH}"
    --output_dir "${OUTPUT_DIR}"

    --model_max_length "${MODEL_MAX_LENGTH}"
    --apply_sigmoid "${APPLY_SIGMOID}"
    --trust_remote_code "${TRUST_REMOTE_CODE}"

    --bf16 True
    --tf32 True

    --num_train_epochs "${NUM_EPOCHS}"
    --max_steps "${TRAIN_MAX_STEPS}"

    --per_device_train_batch_size "${MICRO_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRAD_ACCUM}"

    --optim adamw_torch
    --learning_rate "${LEARNING_RATE}"
    --weight_decay "${WEIGHT_DECAY}"
    --warmup_ratio "${WARMUP_RATIO}"
    --lr_scheduler_type cosine

    --save_strategy steps
    --save_steps "${SAVE_STEPS}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}"

    --logging_steps "${LOGGING_STEPS}"
    --dataloader_num_workers 2

    --remove_unused_columns False
    --save_safetensors False
    --seed "${SEED}"
    --report_to none
)

if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
    TRAIN_ARGS+=(
        --attn_implementation "${ATTN_IMPLEMENTATION}"
    )
fi


# -----------------------------------------------------------------------------
# Single-GPU or FSDP multi-GPU
# -----------------------------------------------------------------------------

if (( NUM_GPUS == 1 )); then
    echo "[INFO] Starting single-GPU QNet training."

    CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
        "${TRAIN_PYTHON}" \
        "${TRAIN_ARGS[@]}"

else
    echo "[INFO] Starting ${NUM_GPUS}-GPU FSDP QNet training."

    TRAIN_ARGS+=(
        --fsdp "full_shard auto_wrap"
        --fsdp_config critic_training/fsdp_config.json
    )

    CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
        "${TRAIN_PYTHON}" -m torch.distributed.run \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port="${MASTER_PORT}" \
        "${TRAIN_ARGS[@]}"
fi

echo "[INFO] QNet training completed."
echo "[INFO] Checkpoint: ${OUTPUT_DIR}"