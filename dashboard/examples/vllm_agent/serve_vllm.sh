#!/usr/bin/env bash
# Start vLLM's OpenAI-compatible server for the ATL vLLM adapter
# (agentictrading.integrations.vllm). Runs on the NVIDIA GPU host, inside the
# environment from requirements-vllm-server.txt. Every value comes from the
# environment with a safe default; nothing machine-specific is hard-coded.
#
#   VLLM_MODEL_ID                 Qwen/Qwen2.5-3B-Instruct (also the served name)
#   VLLM_MODEL_REVISION           pinned Hugging Face commit (weights + tokenizer)
#   VLLM_HOST / VLLM_PORT         127.0.0.1 / 8000 (loopback: reach it over SSH)
#   VLLM_API_KEY                  read by vLLM itself from the environment, so it
#                                 never appears on the command line (`ps`)
#   VLLM_DTYPE                    bfloat16; use float16 on pre-Ampere GPUs (T4, V100)
#   VLLM_MAX_MODEL_LEN            8192 (prompts are ~1-5k tokens)
#   VLLM_GPU_MEMORY_UTILIZATION   0.85
#   VLLM_SERVE_EXTRA_ARGS         extra flags, word-split (e.g. LoRA, Stage 2)
#
# Stop it with Ctrl-C (foreground) or `kill <pid>`; vLLM shuts down cleanly on
# SIGINT/SIGTERM and frees the GPU.
set -euo pipefail

: "${VLLM_MODEL_ID:=Qwen/Qwen2.5-3B-Instruct}"
: "${VLLM_MODEL_REVISION:=aa8e72537993ba99e69dfaafa59ed015b17504d1}"
: "${VLLM_HOST:=127.0.0.1}"
: "${VLLM_PORT:=8000}"
: "${VLLM_DTYPE:=bfloat16}"
: "${VLLM_MAX_MODEL_LEN:=8192}"
: "${VLLM_GPU_MEMORY_UTILIZATION:=0.85}"

case "$VLLM_HOST" in
  127.0.0.1|localhost|::1) ;;
  *)
    # A non-loopback bind is reachable by anything that can route to this
    # host; vLLM's key guards /v1 only, so never open it without one, and
    # never expose it to the public internet even with one.
    if [[ -z "${VLLM_API_KEY:-}" ]]; then
      echo "serve_vllm.sh: refusing to bind $VLLM_HOST without VLLM_API_KEY" >&2
      exit 2
    fi
    ;;
esac
if [[ -n "${VLLM_API_KEY:-}" ]]; then export VLLM_API_KEY; fi

command -v vllm >/dev/null || { echo "serve_vllm.sh: vllm not on PATH (pip install -r requirements-vllm-server.txt)" >&2; exit 3; }
command -v nvidia-smi >/dev/null || { echo "serve_vllm.sh: no nvidia-smi; vLLM CUDA serving needs an NVIDIA GPU" >&2; exit 4; }

args=(
  serve "$VLLM_MODEL_ID"
  --revision "$VLLM_MODEL_REVISION"
  --tokenizer-revision "$VLLM_MODEL_REVISION"
  --served-model-name "$VLLM_MODEL_ID"
  --dtype "$VLLM_DTYPE"
  --max-model-len "$VLLM_MAX_MODEL_LEN"
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --seed 0
  # Ignore the model's generation_config.json (Qwen ships top_k=20,
  # repetition_penalty=1.05): only the adapter's explicit request parameters
  # shape decoding, so the frozen config is the whole sampling config.
  --generation-config vllm
  --host "$VLLM_HOST"
  --port "$VLLM_PORT"
)
# shellcheck disable=SC2206  # intentional word-splitting of extra flags
if [[ -n "${VLLM_SERVE_EXTRA_ARGS:-}" ]]; then args+=($VLLM_SERVE_EXTRA_ARGS); fi

echo "serve_vllm.sh: vllm $(vllm --version 2>/dev/null || echo '?') serving $VLLM_MODEL_ID@$VLLM_MODEL_REVISION on $VLLM_HOST:$VLLM_PORT (key: $([[ -n "${VLLM_API_KEY:-}" ]] && echo set || echo none))" >&2
exec vllm "${args[@]}"
