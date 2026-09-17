#!/usr/bin/env bash
# One isolated Docker container per operation; no host Ray management.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-plan}"
IMAGE="${ENVDUELS_IMAGE:-envduels-spade:qwen38-v1}"
if [[ "$IMAGE" == envduels-spade:spruce-cu124-foundation ]]; then
    echo 'The spruce cu124 foundation is not a training image; see docker/envduels/SPRUCE_CU124.md.' >&2
    exit 2
fi
case "$ACTION" in
    build)
        exec docker build -f "$ROOT/docker/envduels/Dockerfile" -t "$IMAGE" "$ROOT" ;;
    plan|check|convert|smoke|train) ;;
    *) echo 'Usage: bash scripts/envduels_runtime.sh {build|plan|check|convert|smoke|train}' >&2; exit 2 ;;
esac
RUN=(docker run --rm --init --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864
    --user "$(id -u):$(id -g)"
    --mount "type=bind,src=$ROOT,dst=/workspace/envduels/spade"
    --mount "type=bind,src=${ENVDUELS_EXPORT_DIR:-$ROOT/../exports/duel_harness_004_rl},dst=/workspace/envduels/exports/duel_harness_004_rl,readonly"
    --workdir /workspace/envduels/spade
    --env PYTHONPATH=/workspace/envduels/spade:/workspace/envduels/spade/slime:/root/Megatron-LM
    --env XDG_CACHE_HOME=/tmp/envduels-cache --env HF_HOME=/tmp/envduels-hf
    --env TRITON_CACHE_DIR=/tmp/envduels-triton --env MPLCONFIGDIR=/tmp/envduels-mpl
    --env TORCHINDUCTOR_CACHE_DIR=/tmp/envduels-inductor --env USER
    --env FLASHINFER_WORKSPACE_BASE=/tmp/envduels-flashinfer
    --env NUM_GPUS="${NUM_GPUS:-8}" --env CONVERT_GPUS="${NUM_GPUS:-8}")
for name in TP PP CP ROLLOUT_TP GLOBAL_BATCH_SIZE GROUP_SIZE MAX_TURNS MAX_CONTEXT_LENGTH ACTOR_MAX_TOKENS MAX_TOKENS_PER_GPU THINKING CPU_OFFLOAD ENVDUELS_SEED LR TEMPERATURE; do
    [[ ! -v "$name" ]] || RUN+=(--env "$name=${!name}")
done
# Selection/resume files should be inside the mounted checkout; supply container paths.
for name in ENVDUELS_IDS_FILE LOAD_DIR OUTPUT_DIR; do
    [[ ! -v "$name" ]] || RUN+=(--env "$name=${!name}")
done
case "$ACTION" in
    plan)
        CMD='source cmd/games/qwen38_envduels_profile.sh; bash cmd/games/train_envduels.sh --dry-run' ;;
    check)
        CMD='set -e; source cmd/games/qwen38_envduels_profile.sh; python scripts/preflight_envduels.py --runtime; bash cmd/games/train_envduels.sh --check-args' ;;
    convert)
        : "${GPU_IDS:?Set GPU_IDS to the allocated GPU indices, e.g. 0,1,3,4,6,7,8,9}"
        RUN+=(--gpus "\"device=$GPU_IDS\"")
        CMD='bash scripts/convert_qwen38.sh --run' ;;
    smoke|train)
        : "${GPU_IDS:?Set GPU_IDS to the allocated GPU indices}"
        RUN+=(--gpus "\"device=$GPU_IDS\"")
        if [[ "$ACTION" == smoke ]]; then
            RUN+=(--env NUM_ROLLOUT=2)
        else
            : "${NUM_ROLLOUT:?Set NUM_ROLLOUT explicitly for training}"
            RUN+=(--env "NUM_ROLLOUT=$NUM_ROLLOUT")
        fi
        CMD='set -euo pipefail
source cmd/games/qwen38_envduels_profile.sh
python scripts/preflight_envduels.py --runtime
ray start --head --node-ip-address=127.0.0.1 --num-gpus="$NUM_GPUS" --disable-usage-stats --dashboard-host=127.0.0.1 --dashboard-port=8265
export RAY_ADDRESS=http://127.0.0.1:8265
bash cmd/games/train_envduels.sh --run' ;;
esac
if [[ "$ACTION" == convert || "$ACTION" == smoke || "$ACTION" == train ]]; then
    [[ "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'GPU_IDS must be comma-separated numeric indices' >&2; exit 2; }
    IFS=, read -r -a SELECTED_GPUS <<< "$GPU_IDS"
    (( ${#SELECTED_GPUS[@]} == ${NUM_GPUS:-8} )) || { echo 'GPU_IDS count differs from NUM_GPUS' >&2; exit 2; }
    declare -A SEEN_GPUS=()
    for gpu in "${SELECTED_GPUS[@]}"; do
        [[ ! -v "SEEN_GPUS[$gpu]" ]] || { echo "Duplicate GPU index: $gpu" >&2; exit 2; }
        SEEN_GPUS[$gpu]=1
    done
    if [[ -r /proc/driver/nvidia/version && "$IMAGE" == envduels-spade:qwen38-v1 ]]; then
        DRIVER_TEXT="$(</proc/driver/nvidia/version)"
        if [[ "$DRIVER_TEXT" == *' 555.'* ]]; then
            echo 'Blocked: host driver 555 is outside this CUDA 12.9 image compatibility conditions. See cmd/games/ENVDUELS_RUN.md; do not bypass NVIDIA_REQUIRE_CUDA.' >&2
            exit 2
        fi
    fi
fi
# No --network=host, no published ports, no Docker socket mount. Ray stays isolated.
if [[ "$ACTION" == convert || "$ACTION" == smoke || "$ACTION" == train ]]; then
    mkdir -p "$ROOT/outputs/logs"
    LOG_FILE="$ROOT/outputs/logs/$ACTION-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
    echo "Persistent console log: $LOG_FILE"
    "${RUN[@]}" "$IMAGE" bash -c "$CMD" 2>&1 | tee "$LOG_FILE"
    exit "${PIPESTATUS[0]}"
fi
exec "${RUN[@]}" "$IMAGE" bash -c "$CMD"
