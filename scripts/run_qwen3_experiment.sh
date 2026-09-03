#!/usr/bin/env bash
set -Eeuo pipefail

# -----------------------------------------------------------------------------
# Experiment
# -----------------------------------------------------------------------------

BENCHMARK="${BENCHMARK:-alfworld}"

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


# Use the currently activated Python environment by default.
PYTHON="${PYTHON:-python}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"

N_TRAJS="${N_TRAJS:-3}"
MAX_TASKS="${MAX_TASKS:-}"

RUN_ID="${RUN_ID:-0}"
OUT_DIR="${OUT_DIR:-outputs/qwen3_4b_${BENCHMARK}_run${RUN_ID}}"


# -----------------------------------------------------------------------------
# SGLang server
# -----------------------------------------------------------------------------

SERVER_GPU="${SERVER_GPU:-5}"
SGLANG_PORT="${SGLANG_PORT:-21003}"
SERVER_ADDRESS="http://127.0.0.1:${SGLANG_PORT}"

TP_SIZE="${TP_SIZE:-1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"

SERVER_READY_ATTEMPTS=120
SERVER_READY_SLEEP_SECONDS=2


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

SERVER_LOG="${LOG_DIR}/sglang_server.log"
EXPERIMENT_LOG="${LOG_DIR}/experiment.log"


# -----------------------------------------------------------------------------
# Build optional runner arguments
# -----------------------------------------------------------------------------

RUNNER_ARGS=(
    scripts/run_experiment.py
    --config "${EXP_CONFIG}"
    --output-dir "${OUT_DIR}"
    --server-address "${SERVER_ADDRESS}"
    --model-name "${MODEL_PATH}"
    --num-trajectories "${N_TRAJS}"
)

if [[ -n "${MAX_TASKS}" ]]; then
    if ! [[ "${MAX_TASKS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] MAX_TASKS must be a positive integer." >&2
        exit 1
    fi

    RUNNER_ARGS+=(--max-tasks "${MAX_TASKS}")
fi


# -----------------------------------------------------------------------------
# Configuration summary
# -----------------------------------------------------------------------------

echo "[CONFIG] Benchmark:       ${BENCHMARK}"
echo "[CONFIG] Model:           ${MODEL_PATH}"
echo "[CONFIG] Server GPU:      ${SERVER_GPU}"
echo "[CONFIG] Server address:  ${SERVER_ADDRESS}"
echo "[CONFIG] Context length:  ${CONTEXT_LENGTH}"
echo "[CONFIG] Trajectories:    ${N_TRAJS}"
echo "[CONFIG] Max tasks:       ${MAX_TASKS:-all}"
echo "[CONFIG] Output dir:      ${OUT_DIR}"


# -----------------------------------------------------------------------------
# SGLang lifecycle
# -----------------------------------------------------------------------------

SERVER_PID=""


cleanup() {
    exit_code=$?

    trap - EXIT INT TERM

    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[INFO] Stopping SGLang server PID=${SERVER_PID}."
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi

    exit "${exit_code}"
}


trap cleanup EXIT INT TERM


# Protect against accidentally using a server left from another experiment.
if curl -fsS --max-time 2 "${SERVER_ADDRESS}/v1/models" >/dev/null 2>&1; then
    echo "[ERROR] Another server already responds at ${SERVER_ADDRESS}." >&2
    echo "[ERROR] Stop it or select another SGLANG_PORT." >&2
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


# -----------------------------------------------------------------------------
# Wait until SGLang is ready
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Run actor-only experiment
# -----------------------------------------------------------------------------

echo "[INFO] Starting ${BENCHMARK} experiment."

# The client itself does not need a GPU: the actor is served by SGLang.
CUDA_VISIBLE_DEVICES="" \
    "${PYTHON}" "${RUNNER_ARGS[@]}" \
    2>&1 | tee "${EXPERIMENT_LOG}"

echo "[INFO] Experiment completed."


# -----------------------------------------------------------------------------
# Calculate benchmark metrics
# -----------------------------------------------------------------------------

echo "[INFO] Calculating metrics."

"${PYTHON}" "${METRICS_SCRIPT}" --input "${OUT_DIR}" \
    --output "${OUT_DIR}/metrics.json"

echo "[INFO] Finished."
echo "[INFO] Trajectories: ${OUT_DIR}/trajectories.jsonl"
echo "[INFO] Metrics:      ${OUT_DIR}/metrics.json"
echo "[INFO] Metadata:     ${OUT_DIR}/run_metadata.json"
echo "[INFO] Logs:         ${LOG_DIR}"
