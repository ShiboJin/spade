#!/usr/bin/env bash
# Command preparation defaults to dry-run. Uses an explicitly supplied Ray head.
set -euo pipefail

MODE="${1:---dry-run}"
if [[ "$MODE" != --dry-run && "$MODE" != --run && "$MODE" != --check-args ]] || (( $# > 1 )); then
    echo 'Usage: bash cmd/games/train_envduels.sh [--dry-run|--check-args|--run]' >&2
    exit 2
fi
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
: "${MODEL_CONFIG:?Set MODEL_CONFIG to a trusted architecture shell config defining MODEL_ARGS}"
: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the intended local HF checkpoint directory}"
: "${REF_CHECKPOINT:?Set REF_CHECKPOINT to its converted Megatron checkpoint directory}"
: "${ENVDUELS_EXPORT_DIR:?Set ENVDUELS_EXPORT_DIR to an export directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new run directory}"
: "${MEGATRON_DIR:?Set MEGATRON_DIR to the intended Megatron-LM checkout}"

NUM_GPUS="${NUM_GPUS:-8}"
TP="${TP:-2}"
PP="${PP:-1}"
CP="${CP:-1}"
ROLLOUT_TP="${ROLLOUT_TP:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
GROUP_SIZE="${GROUP_SIZE:-8}"
NUM_ROLLOUT="${NUM_ROLLOUT:-2}"
MAX_TURNS="${MAX_TURNS:-24}"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-8192}"
ACTOR_MAX_TOKENS="${ACTOR_MAX_TOKENS:-1024}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-2048}"
THINKING="${THINKING:-false}"
for name in NUM_GPUS TP PP CP ROLLOUT_TP GLOBAL_BATCH_SIZE GROUP_SIZE NUM_ROLLOUT MAX_TURNS MAX_CONTEXT_LENGTH ACTOR_MAX_TOKENS MAX_TOKENS_PER_GPU; do
    value="${!name}"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]] || (( ${#value} > 8 )); then
        echo "Invalid positive integer: $name=$value" >&2; exit 2
    fi
done
if (( NUM_GPUS % (TP * PP * CP) || NUM_GPUS % ROLLOUT_TP )); then
    echo 'GPU count must be divisible by TP*PP*CP and by ROLLOUT_TP' >&2; exit 2
fi
DP=$((NUM_GPUS / (TP * PP * CP)))
if (( GROUP_SIZE < 2 || GLOBAL_BATCH_SIZE % GROUP_SIZE || GLOBAL_BATCH_SIZE % DP )); then
    echo 'Batch must be divisible by GROUP_SIZE (>=2) and data parallel size' >&2; exit 2
fi
if [[ "$THINKING" != true && "$THINKING" != false ]]; then
    echo 'THINKING must be true or false' >&2; exit 2
fi
if (( MAX_CONTEXT_LENGTH <= ACTOR_MAX_TOKENS + 64 )); then
    echo 'Context must leave room for prompt and generation budget' >&2; exit 2
fi
[[ -f "$MODEL_CONFIG" ]] || { echo "Missing MODEL_CONFIG: $MODEL_CONFIG" >&2; exit 2; }
# Only source a reviewed model architecture config, never server credentials.
source "$MODEL_CONFIG"
(( ${#MODEL_ARGS[@]} > 0 )) || { echo 'MODEL_ARGS is empty' >&2; exit 2; }

# Read metadata only; this does not import or execute any exported environment.
"$PYTHON_BIN" -c '
import sys
from spade.core.envs.envduels_adapter import EnvDuelsAdapter
a = EnvDuelsAdapter(sys.argv[1], env_ids_file=sys.argv[2] or None)
print("Selected environments:", len(a.list_environments()))
' "$ENVDUELS_EXPORT_DIR" "${ENVDUELS_IDS_FILE:-}"

TRAIN=("$PYTHON_BIN" -m train_spade_slime
    --actor-num-nodes 1 --actor-num-gpus-per-node "$NUM_GPUS" --num-gpus-per-node "$NUM_GPUS" --colocate
    "${MODEL_ARGS[@]}"
    --hf-checkpoint "$HF_CHECKPOINT" --ref-load "$REF_CHECKPOINT"
    --save "$OUTPUT_DIR/checkpoints" --save-interval "${SAVE_INTERVAL:-1}"
    --data-source-path spade.slime.data_source.SpadeDataSource
    --rollout-function-path spade.slime.fixed_env_rollout.spade_fixed_env_rollout
    --spade-mode fixed_env --spade-fixed-env-source "envduels:$ENVDUELS_EXPORT_DIR"
    --spade-envduels-seed "${ENVDUELS_SEED:-42}" --spade-fixed-env-same-problem
    --spade-fixed-pool-size 0 --spade-game-regeneration-interval 1
    --spade-trajectories-per-game "$GROUP_SIZE" --spade-reward-normalization grpo
    --spade-max-turns "$MAX_TURNS" --spade-max-context-length "$MAX_CONTEXT_LENGTH"
    --spade-actor-max-tokens "$ACTOR_MAX_TOKENS" --spade-actor-temperature "${TEMPERATURE:-1.0}"
    --num-rollout "$NUM_ROLLOUT" --rollout-batch-size "$((GLOBAL_BATCH_SIZE / GROUP_SIZE))"
    --n-samples-per-prompt 1 --global-batch-size "$GLOBAL_BATCH_SIZE"
    --use-dynamic-global-batch-size --balance-data
    --rollout-max-response-len "$ACTOR_MAX_TOKENS" --rollout-temperature "${TEMPERATURE:-1.0}"
    --apply-chat-template --apply-chat-template-kwargs "{\"enable_thinking\":$THINKING,\"preserve_thinking\":true}"
    --skip-eval-before-train
    --advantage-estimator grpo --disable-grpo-std-normalization --disable-rewards-normalization
    --kl-loss-coef 0.0 --entropy-coef 0.0 --eps-clip 0.2 --eps-clip-high 0.28 --use-tis
    --optimizer adam --lr "${LR:-1e-6}" --lr-decay-style constant
    --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
    --tensor-model-parallel-size "$TP" --pipeline-model-parallel-size "$PP"
    --context-parallel-size "$CP" --use-dynamic-batch-size
    --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU"
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --rollout-num-gpus-per-engine "$ROLLOUT_TP" --sglang-mem-fraction-static "${ROLLOUT_MEMORY_FRACTION:-0.7}"
    --sglang-attention-backend triton --sglang-context-length "$MAX_CONTEXT_LENGTH"
    --sglang-disable-cuda-graph
    --attention-dropout 0.0 --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash)
(( TP <= 1 )) || TRAIN+=(--sequence-parallel)
[[ -z "${ENVDUELS_IDS_FILE:-}" ]] || TRAIN+=(--spade-envduels-ids-file "$ENVDUELS_IDS_FILE")
[[ -z "${LOAD_DIR:-}" ]] || TRAIN+=(--load "$LOAD_DIR")
if [[ "${CPU_OFFLOAD:-0}" == 1 ]]; then
    TRAIN+=(--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer)
fi

RUNTIME_JSON="$("$PYTHON_BIN" -c '
import json, sys
print(json.dumps({"env_vars": {"PYTHONPATH": ":".join(sys.argv[1:]),
    "CUDA_DEVICE_MAX_CONNECTIONS": "1", "PYTHONUNBUFFERED": "1"}}))
' "$PROJECT_ROOT" "$PROJECT_ROOT/slime" "$MEGATRON_DIR")"
COMMAND=(ray job submit --address "${RAY_ADDRESS:-http://127.0.0.1:8265}"
    --runtime-env-json "$RUNTIME_JSON" -- "${TRAIN[@]}")
echo "mode=$MODE GPUs=$NUM_GPUS TP=$TP PP=$PP CP=$CP DP=$DP rollout_TP=$ROLLOUT_TP"
echo "Training batch: $((GLOBAL_BATCH_SIZE / GROUP_SIZE)) problems x $GROUP_SIZE episodes; rollout overprovisions for failures."
printf '%q ' "${COMMAND[@]}"
printf '\n'
if [[ "$MODE" == --dry-run ]]; then
    echo 'Dry-run only. No runtime or checkpoint compatibility has been verified.'
    exit 0
fi
if [[ "$MODE" == --check-args ]]; then
    exec "${TRAIN[@]}" --validate-only
fi
: "${RAY_ADDRESS:?For --run explicitly set the address of your dedicated Ray cluster}"
[[ -f "$HF_CHECKPOINT/config.json" && -d "$REF_CHECKPOINT" && -d "$MEGATRON_DIR/megatron" ]] || {
    echo 'Missing HF config, Megatron checkpoint directory, or Megatron source' >&2; exit 2;
}
[[ ! -e "$OUTPUT_DIR" ]] || { echo 'OUTPUT_DIR must be new; use LOAD_DIR to resume into a new output directory' >&2; exit 2; }
[[ -f "$REF_CHECKPOINT/latest_checkpointed_iteration.txt" ]] || {
    echo 'Missing Megatron checkpoint tracker: run conversion first' >&2; exit 2;
}
command -v ray >/dev/null
exec "${COMMAND[@]}"
