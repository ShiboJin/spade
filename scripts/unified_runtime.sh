#!/usr/bin/env bash
# Build and run the CUDA 12.4 ms-swift/vLLM image without host Python deps.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-plan}"
IMAGE="${ENVDUELS_IMAGE:-envduels-unified:cu124}"
EXPORT_DIR="$(realpath -e "${ENVDUELS_EXPORT_DIR:-$ROOT/../exports/duel_harness_004_rl}")"
DATASET="$ROOT/data/envduels/fixed90-swift.jsonl"
TRAIN_CONFIG="${TRAIN_CONFIG:-$ROOT/configs/train_qwen38_envduels_lora.json}"

case "$ACTION" in
    build)
        exec docker build \
            --network host \
            --build-arg HTTP_PROXY --build-arg HTTPS_PROXY \
            --build-arg ALL_PROXY --build-arg NO_PROXY \
            --build-arg "BUILD_JOBS=${BUILD_JOBS:-64}" \
            --build-arg "GIT_FETCH_JOBS=${GIT_FETCH_JOBS:-2}" \
            --build-arg "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0;8.9}" \
            -f "$ROOT/docker/unified-cu124/Dockerfile" -t "$IMAGE" "$ROOT"
        ;;
    prepare)
        exec python "$ROOT/scripts/prepare_envduels_swift.py" \
            --export-dir "$EXPORT_DIR" --output "$DATASET" \
            --seed "${TRAIN_SEED:-42}"
        ;;
    plan)
        exec python3 "$ROOT/scripts/run_train.py" --config "$TRAIN_CONFIG" --dry-run
        ;;
    smoke)
        exec python3 "$ROOT/scripts/run_train.py" --config "$TRAIN_CONFIG" --smoke
        ;;
    train)
        exec python3 "$ROOT/scripts/run_train.py" --config "$TRAIN_CONFIG"
        ;;
    *) echo 'Usage: bash scripts/unified_runtime.sh {build|prepare|plan|smoke|train}' >&2; exit 2 ;;
esac
