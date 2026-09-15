#!/usr/bin/env bash
set -Eeuo pipefail


# =============================================================================
# Input / output
# =============================================================================

DATA_DIR="${DATA_DIR:?Set DATA_DIR to a critic-data collection directory, e.g. outputs/critic_data/alfworld/run0}"
OUTPUT_PATH="${OUTPUT_PATH:-${DATA_DIR}/q_dataset.json}"


# =============================================================================
# Q-target settings
# =============================================================================

GAMMA="${GAMMA:-0.9}"
UPPER_NUM="${UPPER_NUM:-300}"
SEED="${SEED:-42}"
PYTHON="${PYTHON:-python}"


# =============================================================================
# Configuration summary
# =============================================================================

echo "[CONFIG] Input dir:   ${DATA_DIR}"
echo "[CONFIG] Output:      ${OUTPUT_PATH}"
echo "[CONFIG] Gamma:       ${GAMMA}"
echo "[CONFIG] Upper num:   ${UPPER_NUM}"
echo "[CONFIG] Seed:        ${SEED}"


# =============================================================================
# Build QNet dataset
# =============================================================================

"${PYTHON}" critic_data/build_dataset.py \
    --input-dir "${DATA_DIR}" \
    --output "${OUTPUT_PATH}" \
    --gamma "${GAMMA}" \
    --upper-num "${UPPER_NUM}" \
    --seed "${SEED}"


echo "[INFO] QNet dataset saved to: ${OUTPUT_PATH}"