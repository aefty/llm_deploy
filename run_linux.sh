#!/bin/bash

# Usage: ./run_linux.sh <model> <port>
#   ./run_linux.sh qwen_0.5b   8000
#   ./run_linux.sh apertus_8b  8000
#   ./run_linux.sh apertus_70b 8000

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_CACHE="$SCRIPT_DIR/model_cache_linux"
DOCKER_IMAGE="vllm/vllm-openai:latest"
if [ ! -d "$MODEL_CACHE" ]; then
  mkdir -p "$MODEL_CACHE"
fi

# ── Credentials ───────────────────────────────────────────────────────

if [ ! -f "$SCRIPT_DIR/API.keys" ]; then
  cat > "$SCRIPT_DIR/API.keys" << 'EOF'
# Apertus API Keys
# One key per line. Lines starting with # are ignored.
# Replace the values below with your own keys.

test_key
test_key2
EOF
  echo "created API.keys with default keys — edit the file to set your own"
fi

API_KEY=$(grep -v '^\s*#' "$SCRIPT_DIR/API.keys" 2>/dev/null | grep -v '^\s*$' | head -1 | tr -d '[:space:]')
if [ -z "$API_KEY" ]; then
  echo "error: API.keys is empty — add at least one key to the file"
  exit 1
fi

if [ ! -f "$SCRIPT_DIR/HF.token" ]; then
  cat > "$SCRIPT_DIR/HF.token" << 'EOF'
# HuggingFace Read Token
# Replace this line with your token from https://huggingface.co/settings/tokens
# Make sure you have accepted the model license on HuggingFace before running.

hf_REPLACE_WITH_YOUR_TOKEN
EOF
  echo "created HF.token — edit the file and replace with your HuggingFace token"
  exit 1
fi

HF_TOKEN=$(grep -v '^\s*#' "$SCRIPT_DIR/HF.token" 2>/dev/null | grep -v '^\s*$' | head -1 | tr -d '[:space:]')
if [ -z "$HF_TOKEN" ] || [ "$HF_TOKEN" = "hf_REPLACE_WITH_YOUR_TOKEN" ]; then
  echo "error: HF.token not set — edit the file and replace with your HuggingFace token"
  exit 1
fi

# ── Args ──────────────────────────────────────────────────────────────

if [ -z "${1}" ] || [ -z "${2}" ]; then
  echo "usage: ./run_linux.sh <model> <port>"
  echo "       models: qwen_0.5b | apertus_8b | apertus_70b"
  exit 1
fi

PORT="${2}"

# ── Model ─────────────────────────────────────────────────────────────

case "${1}" in
  qwen_0.5b)
    MODEL=Qwen/Qwen2.5-0.5B-Instruct
    MAX_MODEL_LEN=4096
    SWAP_SPACE=4
    KV_CACHE_DTYPE=auto
    TENSOR_PARALLEL_SIZE=1
    GPU_MEMORY_UTILIZATION=0.5
    ;;
  apertus_8b)
    MODEL=swiss-ai/Apertus-8B-Instruct-2509
    MAX_MODEL_LEN=8192
    SWAP_SPACE=16
    KV_CACHE_DTYPE=auto
    TENSOR_PARALLEL_SIZE=1
    GPU_MEMORY_UTILIZATION=0.9
    ;;
  apertus_70b)
    MODEL=swiss-ai/Apertus-70B-Instruct-2509
    MAX_MODEL_LEN=32768
    SWAP_SPACE=128
    KV_CACHE_DTYPE=fp8
    TENSOR_PARALLEL_SIZE=2
    GPU_MEMORY_UTILIZATION=0.95
    ;;
  *)
    echo "error: unknown model '${1}'"
    echo "       models: qwen_0.5b | apertus_8b | apertus_70b"
    exit 1
    ;;
esac

# ── Docker image ──────────────────────────────────────────────────────

if ! docker image inspect "$DOCKER_IMAGE" > /dev/null 2>&1; then
  echo "pulling $DOCKER_IMAGE..."
  docker pull "$DOCKER_IMAGE" || exit 1
fi

# ── Run ───────────────────────────────────────────────────────────────

echo ""
echo "  backend  vllm (docker)"
echo "  model    $MODEL"
echo "  port     $PORT"
echo "  api key  $API_KEY"
echo "  cache    $MODEL_CACHE"
echo "  context  $MAX_MODEL_LEN tokens"
echo "  swap     $SWAP_SPACE GB"
echo "  kv type  $KV_CACHE_DTYPE"
echo "  gpu mem  $GPU_MEMORY_UTILIZATION"
echo "  gpus     $TENSOR_PARALLEL_SIZE"
echo ""

docker run --gpus all --ipc=host -p $PORT:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  -v $MODEL_CACHE:/root/.cache/huggingface \
  "$DOCKER_IMAGE" \
  --model $MODEL \
  --api-key $API_KEY \
  --max-model-len $MAX_MODEL_LEN \
  --swap-space $SWAP_SPACE \
  --kv-cache-dtype $KV_CACHE_DTYPE \
  --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION
