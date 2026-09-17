#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/cmd/games/qwen38_envduels_profile.sh"
source "$MODEL_CONFIG"
MODE="${1:---dry-run}"
[[ "$MODE" == --dry-run || "$MODE" == --run ]] || { echo 'Expected --dry-run or --run' >&2; exit 2; }
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/slime:$MEGATRON_DIR${PYTHONPATH:+:$PYTHONPATH}"
CONVERT_GPUS="${CONVERT_GPUS:-8}"
[[ "$CONVERT_GPUS" =~ ^[1-9][0-9]*$ ]] || exit 2
CMD=(torchrun --standalone --nproc-per-node "$CONVERT_GPUS"
    "$PROJECT_ROOT/slime/tools/convert_hf_to_torch_dist.py"
    "${MODEL_ARGS[@]}" --hf-checkpoint "$HF_CHECKPOINT" --save "$REF_CHECKPOINT")
printf '%q ' "${CMD[@]}"; printf '\n'
[[ "$MODE" == --run ]] || exit 0
[[ ! -e "$REF_CHECKPOINT" ]] || { echo 'Refusing to overwrite existing converted checkpoint' >&2; exit 2; }
python "$PROJECT_ROOT/scripts/preflight_envduels.py" --runtime
exec "${CMD[@]}"
