#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Default values
MODEL="${VLLM_VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
DTYPE="${DTYPE:-auto}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --trust-remote-code)
            TRUST_REMOTE_CODE="true"
            shift
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# vllm serve command
CMD=(
    vllm serve
    "$MODEL"
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
)

# Optional args
if [ -n "$MAX_MODEL_LEN" ]; then
    CMD+=(--max-model-len "$MAX_MODEL_LEN")
fi

if [ "$TRUST_REMOTE_CODE" = "true" ]; then
    CMD+=(--trust-remote-code)
fi

if [ "$DTYPE" != "auto" ]; then
    CMD+=("--dtype" "$DTYPE")
fi

echo "Starting vLLM server with model: $MODEL"
echo "API URL: http://$HOST:$PORT/v1"

if ! command -v vllm &> /dev/null; then
    echo "Error: vLLM not found"
    exit 1
fi

trap 'echo -e "\nShutting down vLLM server..."; exit 0' INT TERM

exec "${CMD[@]}"

