#!/usr/bin/env bash
set -Eeuo pipefail

BENCHMARK="${BENCHMARK:-alfworld}"

case "${BENCHMARK}" in
    alfworld)
        ENV_CONFIG="configs/envs/alfworld.yaml"
        EXPERT_DATA="data/expert/alfworld_sft.json"
        # Reference QLASS data-collection value.
        DEFAULT_MAX_STEPS=25
        ;;

    sciworld)
        ENV_CONFIG="configs/envs/sciworld.yaml"
        EXPERT_DATA="data/expert/sciworld_sft.json"
        DEFAULT_MAX_STEPS=40
        ;;

    webshop)
        ENV_CONFIG="configs/envs/webshop.yaml"
        EXPERT_DATA="data/expert/webshop_sft.json"
        DEFAULT_MAX_STEPS=5
        ;;

    *)
        echo "[ERROR] Unsupported benchmark: ${BENCHMARK}" >&2
        exit 1
        ;;
esac


# =============================================================================
# Python / actor
# =============================================================================

PYTHON="${PYTHON:-python}"

AGENT_CONFIG="${AGENT_CONFIG:-configs/agents/qwen3_4b.yaml}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"


# =============================================================================
# Search hyperparameters
# =============================================================================

NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_DEPTH="${MAX_DEPTH:-5}"
MIN_PRUNE_DEPTH="${MIN_PRUNE_DEPTH:-3}"
SAMPLES_PER_DEPTH="${SAMPLES_PER_DEPTH:-2}"
POSITIVE_REWARD_THRESHOLD="${POSITIVE_REWARD_THRESHOLD:-0.01}"
HISTORY_LENGTH="${HISTORY_LENGTH:-50}"
MAX_STEPS="${MAX_STEPS:-${DEFAULT_MAX_STEPS}}"

# Optional limit applied independently to every worker.
MAX_TASKS_PER_WORKER="${MAX_TASKS_PER_WORKER:-}"


# =============================================================================
# Smoke-test mode
# =============================================================================

SMOKE_TEST="${SMOKE_TEST:-0}"
SMOKE_TASKS="${SMOKE_TASKS:-2}"

if [[ "${SMOKE_TEST}" == "1" ]]; then
    echo "[INFO] Smoke-test mode enabled."

    NUM_WORKERS=1
    MAX_TASKS_PER_WORKER="${SMOKE_TASKS}"
fi


# =============================================================================
# Output
# =============================================================================

RUN_ID="${RUN_ID:-0}"
OUT_DIR="${OUT_DIR:-outputs/critic_data/${BENCHMARK}/run${RUN_ID}}"
LOG_DIR="${OUT_DIR}/logs"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"


# =============================================================================
# SGLang server
# =============================================================================

SERVER_GPU="${SERVER_GPU:-5}"
SGLANG_PORT="${SGLANG_PORT:-21003}"
SERVER_ADDRESS="http://127.0.0.1:${SGLANG_PORT}"
TP_SIZE="${TP_SIZE:-1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
SERVER_READY_ATTEMPTS="${SERVER_READY_ATTEMPTS:-120}"
SERVER_READY_SLEEP_SECONDS="${SERVER_READY_SLEEP_SECONDS:-2}"
SERVER_LOG="${LOG_DIR}/sglang_server.log"


# =============================================================================
# Existing output behavior
# =============================================================================

OVERWRITE="${OVERWRITE:-0}"
OVERWRITE_ARGS=()

if [[ "${OVERWRITE}" == "1" ]]; then
    OVERWRITE_ARGS+=(--overwrite)
fi


# =============================================================================
# Configuration summary
# =============================================================================

echo "[CONFIG] Benchmark:                  ${BENCHMARK}"
echo "[CONFIG] Actor config:               ${AGENT_CONFIG}"
echo "[CONFIG] Model:                      ${MODEL_PATH}"
echo "[CONFIG] Expert data:                ${EXPERT_DATA}"
echo "[CONFIG] Workers:                    ${NUM_WORKERS}"
echo "[CONFIG] Max depth:                  ${MAX_DEPTH}"
echo "[CONFIG] Min prune depth:            ${MIN_PRUNE_DEPTH}"
echo "[CONFIG] Samples per depth:          ${SAMPLES_PER_DEPTH}"
echo "[CONFIG] Positive reward threshold:  ${POSITIVE_REWARD_THRESHOLD}"
echo "[CONFIG] History length:             ${HISTORY_LENGTH}"
echo "[CONFIG] Max steps:                  ${MAX_STEPS}"
echo "[CONFIG] Max tasks per worker:       ${MAX_TASKS_PER_WORKER:-all}"
echo "[CONFIG] SGLang GPU:                 ${SERVER_GPU}"
echo "[CONFIG] SGLang address:             ${SERVER_ADDRESS}"
echo "[CONFIG] Output dir:                 ${OUT_DIR}"
echo "[CONFIG] Smoke test:                 ${SMOKE_TEST}"


# =============================================================================
# Process cleanup
# =============================================================================

SERVER_PID=""
WORKER_PIDS=()


cleanup() {
    exit_code=$?

    trap - EXIT INT TERM

    # Stop unfinished workers first.
    for pid in "${WORKER_PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done

    for pid in "${WORKER_PIDS[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done

    # Then stop SGLang.
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[INFO] Stopping SGLang server PID=${SERVER_PID}."

        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi

    exit "${exit_code}"
}


trap cleanup EXIT INT TERM


# =============================================================================
# Start SGLang
# =============================================================================

# Avoid accidentally connecting collection to a stale server/model.
if curl -fsS --max-time 2 "${SERVER_ADDRESS}/v1/models" >/dev/null 2>&1; then
    echo "[ERROR] Another server already responds at ${SERVER_ADDRESS}." >&2
    echo "[ERROR] Stop it or choose another SGLANG_PORT." >&2
    exit 1
fi

echo "[INFO] Starting SGLang server."

CUDA_VISIBLE_DEVICES="${SERVER_GPU}" \
    "${PYTHON}" -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port "${SGLANG_PORT}" \
    --tp-size "${TP_SIZE}" \
    --context-length "${CONTEXT_LENGTH}" \
    > "${SERVER_LOG}" 2>&1 &

SERVER_PID=$!


# =============================================================================
# Wait for SGLang readiness
# =============================================================================

SERVER_READY=0

for attempt in $(seq 1 "${SERVER_READY_ATTEMPTS}"); do
    if curl -fsS "${SERVER_ADDRESS}/v1/models" >/dev/null 2>&1; then
        SERVER_READY=1

        echo "[INFO] SGLang server is ready."

        break
    fi

    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[ERROR] SGLang server terminated during startup." >&2

        tail -n 100 "${SERVER_LOG}" >&2 || true

        exit 1
    fi

    sleep "${SERVER_READY_SLEEP_SECONDS}"
done

if [[ "${SERVER_READY}" != "1" ]]; then
    echo "[ERROR] SGLang server did not become ready." >&2

    tail -n 100 "${SERVER_LOG}" >&2 || true

    exit 1
fi


# =============================================================================
# Build optional per-worker arguments
# =============================================================================

MAX_TASK_ARGS=()

if [[ -n "${MAX_TASKS_PER_WORKER}" ]]; then
    MAX_TASK_ARGS+=(
        --max-tasks "${MAX_TASKS_PER_WORKER}"
    )
fi


# =============================================================================
# Start collection workers
# =============================================================================

for ((worker_idx=0; worker_idx<NUM_WORKERS; worker_idx++)); do
    WORKER_LOG="${LOG_DIR}/worker_${worker_idx}.log"

    WORKER_OUTPUT="${OUT_DIR}/worker_${worker_idx}_trees.jsonl"

    echo \
        "[INFO] Starting worker ${worker_idx}/${NUM_WORKERS}: " \
        "${WORKER_OUTPUT}"

    # The worker process itself does not need CUDA.
    # Actor inference happens on the SGLang server.
    CUDA_VISIBLE_DEVICES="" \
        "${PYTHON}" critic_data/collect.py \
        --benchmark "${BENCHMARK}" \
        --agent-config "${AGENT_CONFIG}" \
        --env-config "${ENV_CONFIG}" \
        --expert-data "${EXPERT_DATA}" \
        --server-address "${SERVER_ADDRESS}" \
        --model-name "${MODEL_PATH}" \
        --split train \
        --worker-idx "${worker_idx}" \
        --num-workers "${NUM_WORKERS}" \
        --max-depth "${MAX_DEPTH}" \
        --min-prune-depth "${MIN_PRUNE_DEPTH}" \
        --samples-per-depth "${SAMPLES_PER_DEPTH}" \
        --positive-reward-threshold "${POSITIVE_REWARD_THRESHOLD}" \
        --history-length "${HISTORY_LENGTH}" \
        --max-steps "${MAX_STEPS}" \
        --output "${WORKER_OUTPUT}" \
        "${MAX_TASK_ARGS[@]}" \
        "${OVERWRITE_ARGS[@]}" \
        > "${WORKER_LOG}" 2>&1 &

    WORKER_PIDS+=("$!")
done


# =============================================================================
# Wait for all workers
# =============================================================================

for idx in "${!WORKER_PIDS[@]}"; do
    pid="${WORKER_PIDS[idx]}"

    if ! wait "${pid}"; then
        echo "[ERROR] Collection worker ${idx} failed." >&2
        echo "[ERROR] See ${LOG_DIR}/worker_${idx}.log" >&2

        exit 1
    fi

    echo "[INFO] Worker ${idx} completed successfully."
done


echo "[INFO] All collection workers completed successfully."
echo "[INFO] Trees: ${OUT_DIR}"
echo "[INFO] Logs:  ${LOG_DIR}"
echo
echo "[INFO] Next step:"
echo "       DATA_DIR=${OUT_DIR} bash critic_data/run_build_dataset.sh"