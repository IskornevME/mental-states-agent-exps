#!/usr/bin/env bash
set -Eeuo pipefail


# -----------------------------------------------------------------------------
# Experiment
# -----------------------------------------------------------------------------

BENCHMARK="${BENCHMARK:-alfworld}"
PYTHON="${PYTHON:-python}"

# Human mode is intentionally limited to a few random tasks.
MAX_TASKS="${MAX_TASKS:-3}"
TASK_SEED="${TASK_SEED:-42}"

RUN_ID="${RUN_ID:-0}"
OUT_DIR="${OUT_DIR:-outputs/human_${BENCHMARK}_run${RUN_ID}}"


case "${BENCHMARK}" in
    alfworld)
        EXP_CONFIG="configs/experiments/qwen3_4b_alfworld.yaml"
        METRICS_SCRIPT="scripts/calc_results_alfworld.py"
        ;;
    sciworld)
        EXP_CONFIG="configs/experiments/qwen3_4b_sciworld.yaml"
        METRICS_SCRIPT="scripts/calc_results_sciworld.py"
        ;;
    *)
        echo "[ERROR] Unsupported benchmark: ${BENCHMARK}" >&2
        exit 1
        ;;
esac


# -----------------------------------------------------------------------------
# Benchmark-specific prerequisites
# -----------------------------------------------------------------------------

if [[ "${BENCHMARK}" == "sciworld" ]]; then
    if ! command -v java >/dev/null 2>&1; then
        echo "[ERROR] ScienceWorld requires Java, but 'java' is not in PATH." >&2
        echo "[ERROR] With Conda you can install it with:" >&2
        echo "        conda install -c conda-forge openjdk=17" >&2
        exit 1
    fi

    if [[ ! -f "envs/scienceworld/scienceworld.jar" ]]; then
        echo "[ERROR] Missing envs/scienceworld/scienceworld.jar" >&2
        exit 1
    fi
fi


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

EXPERIMENT_LOG="${LOG_DIR}/experiment.log"


echo "[CONFIG] Actor:       human"
echo "[CONFIG] Benchmark:   ${BENCHMARK}"
echo "[CONFIG] Tasks:       ${MAX_TASKS}"
echo "[CONFIG] Task seed:   ${TASK_SEED}"
echo "[CONFIG] Output dir:  ${OUT_DIR}"
echo
echo "[INFO] Starting interactive human experiment."

CUDA_VISIBLE_DEVICES="" \
    "${PYTHON}" scripts/run_experiment.py \
    --config "${EXP_CONFIG}" \
    --human \
    --max-tasks "${MAX_TASKS}" \
    --task-seed "${TASK_SEED}" \
    --num-trajectories 1 \
    --output-dir "${OUT_DIR}" \
    2>&1 | tee "${EXPERIMENT_LOG}"

echo "[INFO] Human experiment completed."
echo "[INFO] Calculating metrics."


"${PYTHON}" "${METRICS_SCRIPT}" --input "${OUT_DIR}" --output "${OUT_DIR}/metrics.json"

echo "[INFO] Finished."
echo "[INFO] Trajectories: ${OUT_DIR}/trajectories.jsonl"
echo "[INFO] Metrics:      ${OUT_DIR}/metrics.json"
echo "[INFO] Metadata:     ${OUT_DIR}/run_metadata.json"
echo "[INFO] Log:          ${EXPERIMENT_LOG}"
