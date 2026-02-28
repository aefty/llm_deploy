#!/bin/bash

# Usage: ./run_mac.sh <model> <port>
#   ./run_mac.sh qwen_0.5b   8000
#   ./run_mac.sh apertus_8b  8000
#   ./run_mac.sh apertus_70b 8000
#
# Requires: pip install mlx_lm

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_CACHE="$SCRIPT_DIR/model_cache_mac"
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

# ── Args ──────────────────────────────────────────────────────────────

if [ -z "${1}" ] || [ -z "${2}" ]; then
  echo "usage: ./run_mac.sh <model> <port>"
  echo "       models: qwen_0.5b | apertus_8b | apertus_8b_8bit | apertus_8b_6bit | apertus_8b_4bit | meta_llama_8b"
  exit 1
fi

PORT="${2}"

# ── Model ─────────────────────────────────────────────────────────────

case "${1}" in
  qwen_0.5b)
    MODEL=mlx-community/Qwen2.5-0.5B-Instruct-4bit
    MAX_MODEL_LEN=4096
    ;;
  apertus_8b)
    MODEL=mlx-community/Apertus-8B-Instruct-2509-bf16
    MAX_MODEL_LEN=4096
    ;;
  apertus_8b_8bit)
    MODEL=mlx-community/Apertus-8B-Instruct-2509-8bit
    MAX_MODEL_LEN=8192
    ;;
  apertus_8b_6bit)
    MODEL=mlx-community/Apertus-8B-Instruct-2509-6bit
    MAX_MODEL_LEN=8192
    ;;
  apertus_8b_4bit)
    MODEL=mlx-community/Apertus-8B-Instruct-2509-4bit
    MAX_MODEL_LEN=8192
    ;;
  meta_llama_8b)
    MODEL=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
    MAX_MODEL_LEN=8192
    ;;
  *)
    echo "error: unknown model '${1}'"
    exit 1
    ;;
esac

# ── Check ─────────────────────────────────────────────────────────────

if ! command -v mlx_lm.server > /dev/null 2>&1; then
  echo "error: mlx_lm not installed — pip install mlx_lm"
  exit 1
fi

# ── Run ───────────────────────────────────────────────────────────────

echo ""
echo "  backend  mlx_lm (mac native)"
echo "  model    $MODEL"
echo "  port     $PORT"
echo "  api key  $API_KEY"
echo "  cache    $MODEL_CACHE"
echo "  context  $MAX_MODEL_LEN tokens"
echo ""

HF_HOME=$MODEL_CACHE mlx_lm.server \
  --model $MODEL \
  --host 127.0.0.1 \
  --port $PORT \
  --max-tokens $MAX_MODEL_LEN
