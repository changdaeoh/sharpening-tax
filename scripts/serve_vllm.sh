#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Serve one model with vLLM behind an OpenAI-compatible API.
#
# Usage:
#   bash scripts/serve_vllm.sh <hugging-face-model-id> <gpus> <port>
#
# Examples:
#   bash scripts/serve_vllm.sh google/gemma-4-31B    0,1 8000    # base model
#   bash scripts/serve_vllm.sh google/gemma-4-31B-it 2,3 8001    # post-trained model
#
# Base models are queried through /v1/completions with our text harness, so they need no chat
# template. Post-trained models are queried through /v1/chat/completions with their own chat
# template and tool-call parser, which this script picks from the model name.
#
# Optional settings (environment variables):
#   MAX_MODEL_LEN       context length in tokens (default 16384). Long multi-turn transcripts,
#                       e.g. post-trained models on BFCL, may need 32768.
#   GPU_UTIL            fraction of GPU memory vLLM may use (default 0.85)
#   TP                  tensor-parallel size (default: the number of GPUs given)
#   CHAT_TEMPLATE_FILE  a chat template to use instead of the model's own. To query a base
#                       model without the harness, serve it with its post-trained sibling's
#                       template.
#   EXTRA_ARGS          more vLLM flags, separated by spaces (a value cannot contain a space)
#
# Tested with vLLM 0.19.0; gemma-4-12B and gemma-4-12B-it need 0.23 or newer.

set -eo pipefail

if [ $# -lt 3 ] || [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
  echo "Usage: bash scripts/serve_vllm.sh <hugging-face-model-id> <gpus> <port>" >&2
  echo "  e.g. bash scripts/serve_vllm.sh google/gemma-4-31B-it 0,1 8000" >&2
  exit 1
fi

MODEL=$1
GPUS=$2     # comma-separated GPU ids, e.g. 0 or 0,1
PORT=$3

IFS=, read -r -a GPU_LIST <<< "$GPUS"
TP=${TP:-${#GPU_LIST[@]}}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
GPU_UTIL=${GPU_UTIL:-0.85}


# ---------------------------------------------------------------------------------------------
# Per-family settings
#   TOOL_FLAGS   let the chat endpoint turn the model's output into tool calls
#   MODEL_FLAGS  anything else the family needs
# ---------------------------------------------------------------------------------------------
DTYPE=bfloat16
TOOL_FLAGS=()
MODEL_FLAGS=()

case "$MODEL" in
  google/gemma-4-*)
    TOOL_FLAGS=(--enable-auto-tool-choice --tool-call-parser gemma4)
    ;;

  mistralai/Ministral-3-*Instruct*)
    # Mistral's own checkpoint format. The Instruct checkpoints are FP8, so vLLM picks the dtype.
    MODEL_FLAGS=(--tokenizer-mode mistral --config-format mistral --load-format mistral)
    TOOL_FLAGS=(--enable-auto-tool-choice --tool-call-parser mistral)
    DTYPE=auto
    ;;

  mistralai/Ministral-3-*Base*)
    MODEL_FLAGS=(--tokenizer-mode mistral --config-format mistral --load-format mistral)
    # Served without a tool-call parser unless a borrowed chat template is given (to query it
    # without the harness). The other families keep their parser for base models too; the completions
    # endpoint that the harness uses never calls it.
    if [ -n "${CHAT_TEMPLATE_FILE:-}" ]; then
      TOOL_FLAGS=(--enable-auto-tool-choice --tool-call-parser mistral)
    fi
    ;;

  Qwen/Qwen2.5-*)
    TOOL_FLAGS=(--enable-auto-tool-choice --tool-call-parser hermes)
    ;;

  Qwen/Qwen3.5-*)
    TOOL_FLAGS=(--enable-auto-tool-choice --tool-call-parser qwen3_xml)
    # Qwen3.5 checkpoints are vision-language models; we use them text-only.
    MODEL_FLAGS=(--limit-mm-per-prompt '{"image": 0, "video": 0}')
    ;;

  *)
    echo "No serving recipe for $MODEL." >&2
    echo "Supported: google/gemma-4-*, mistralai/Ministral-3-*-Base-*, mistralai/Ministral-3-*-Instruct-*," >&2
    echo "           Qwen/Qwen2.5-*, Qwen/Qwen3.5-*" >&2
    exit 1
    ;;
esac

if [ -n "${CHAT_TEMPLATE_FILE:-}" ]; then
  MODEL_FLAGS+=(--chat-template "$CHAT_TEMPLATE_FILE")
fi

EXTRA_FLAGS=()
if [ -n "${EXTRA_ARGS:-}" ]; then
  read -r -a EXTRA_FLAGS <<< "$EXTRA_ARGS"
fi


# ---------------------------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------------------------
echo "Serving $MODEL on GPU $GPUS at http://127.0.0.1:$PORT/v1 (TP=$TP, context $MAX_MODEL_LEN)"

export CUDA_VISIBLE_DEVICES=$GPUS
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enable-prefix-caching \
  --trust-remote-code \
  "${MODEL_FLAGS[@]}" \
  "${TOOL_FLAGS[@]}" \
  "${EXTRA_FLAGS[@]}"
