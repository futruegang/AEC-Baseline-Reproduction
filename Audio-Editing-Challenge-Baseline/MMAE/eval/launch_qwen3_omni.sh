#!/usr/bin/env bash
# Qwen3-Omni-30B-A3B-Instruct: 2x tp=4 instances on 8 GPUs
# Instance 1: GPU 0-3, port 8001
# Instance 2: GPU 4-7, port 8002
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_DIR="your_path_to_Qwen3-Omni-30B-A3B-Instruct"
MODEL_NAME="Qwen3Omni-Instruct"
HOST="0.0.0.0"
MAX_MODEL_LEN=32768
TP=4
VLLM="$PROJECT_DIR/.venv/bin/vllm"

INSTANCES=(
    "0,1,2,3:8001"
    "4,5,6,7:8002"
)

# --- Logging ---
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# Collect child PIDs
PIDS=()

cleanup() {
    echo ""
    echo "[*] Shutting down all instances ..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null && wait "$pid" 2>/dev/null || true
    done
    echo "[*] All instances stopped."
}
trap cleanup EXIT INT TERM

# --- Launch instances ---
export VLLM_WORKER_MULTIPROC_METHOD=spawn

for entry in "${INSTANCES[@]}"; do
    GPUS="${entry%%:*}"
    PORT="${entry##*:}"
    LOG_FILE="$LOG_DIR/qwen3omni-instruct_server_${TIMESTAMP}_port${PORT}.log"
    ln -sf "$LOG_FILE" "$LOG_DIR/qwen3omni-instruct_latest_port${PORT}.log"

    echo "[*] Starting instance: GPU=$GPUS PORT=$PORT"
    echo "    Log: $LOG_FILE"

    CUDA_VISIBLE_DEVICES="$GPUS" "$VLLM" serve "$MODEL_DIR" \
        --served-model-name "$MODEL_NAME" \
        --port "$PORT" \
        --host "$HOST" \
        --dtype bfloat16 \
        --max-model-len "$MAX_MODEL_LEN" \
        --allowed-local-media-path / \
        --tensor-parallel-size "$TP" \
        2>&1 | tee "$LOG_FILE" &
    PIDS+=($!)
done

echo ""
echo "============================================"
echo " 2x $MODEL_NAME (tp=$TP)"
echo " Instance 1: GPU 0-3  -> http://${HOST}:8001"
echo " Instance 2: GPU 4-7  -> http://${HOST}:8002"
echo " Ctrl+C to stop all"
echo "============================================"

wait
