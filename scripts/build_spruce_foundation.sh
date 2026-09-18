#!/usr/bin/env bash
# No build is started unless --build is explicitly supplied. Never mounts GPUs.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:---plan}"
[[ $# -le 1 && ( "$MODE" == --plan || "$MODE" == --build ) ]] || {
    echo 'Usage: bash scripts/build_spruce_foundation.sh [--plan|--build]' >&2; exit 2;
}
BUILD_JOBS="${BUILD_JOBS:-4}"
[[ "$BUILD_JOBS" =~ ^([1-9]|1[0-6])$ ]] || { echo 'BUILD_JOBS must be 1..16' >&2; exit 2; }
CMD=(docker build --target foundation --build-arg "BUILD_JOBS=$BUILD_JOBS"
    -f "$ROOT/docker/envduels/Dockerfile.spruce-cu124"
    -t "envduels-spade:spruce-cu124-foundation" "$ROOT/docker/envduels")
echo 'Experimental PyTorch/CUDA foundation only; not a training runtime.'
printf '%q ' "${CMD[@]}"; printf '\n'
[[ "$MODE" == --build ]] || exit 0
mkdir -p "$ROOT/outputs/logs"
LOG_FILE="$ROOT/outputs/logs/spruce-foundation-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
echo "Build log: $LOG_FILE"
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
