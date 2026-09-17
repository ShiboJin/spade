#!/usr/bin/env bash
# Source this inside the prepared runtime. All paths can be overridden.
ENVDUELS_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export MODEL_CONFIG="${MODEL_CONFIG:-$ENVDUELS_PROJECT/cmd/models/qwen3.8-27B.sh}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-$ENVDUELS_PROJECT/checkpoints/Qwen3.8-27B}"
export REF_CHECKPOINT="${REF_CHECKPOINT:-$ENVDUELS_PROJECT/checkpoints/Qwen3.8-27B_torch_dist}"
export ENVDUELS_EXPORT_DIR="${ENVDUELS_EXPORT_DIR:-$ENVDUELS_PROJECT/../exports/duel_harness_004_rl}"
export MEGATRON_DIR="${MEGATRON_DIR:-/root/Megatron-LM}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ENVDUELS_PROJECT/outputs/qwen38-envduels-$(date -u +%Y%m%dT%H%M%SZ)}"
# Initial single-node dense-model layout, to be measured on A100 at first run.
export NUM_GPUS="${NUM_GPUS:-8}" TP="${TP:-2}" PP="${PP:-2}" CP="${CP:-1}"
export ROLLOUT_TP="${ROLLOUT_TP:-2}" CPU_OFFLOAD="${CPU_OFFLOAD:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}" GROUP_SIZE="${GROUP_SIZE:-8}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-2}" MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-8192}"
export ACTOR_MAX_TOKENS="${ACTOR_MAX_TOKENS:-1024}" MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-2048}"
export THINKING="${THINKING:-false}"
