#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${BENCHMARK_EVAL_CONFIG:-${REPO_ROOT}/configs/qwen38_benchmarks.json}"
TMUX_SESSION="${TMUX_SESSION:-qwen38-vllm-benchmarks}"
LAUNCH_LOG_DIR="${REPO_ROOT}/outputs/evaluation"
LAUNCH_LOG="${LAUNCH_LOG_DIR}/${TMUX_SESSION}-$(date +%Y%m%d_%H%M%S).log"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

command -v tmux >/dev/null || die "tmux is not installed"
[[ -f "${CONFIG}" ]] || die "configuration not found: ${CONFIG}"
tmux has-session -t "${TMUX_SESSION}" 2>/dev/null && \
    die "tmux session already exists: ${TMUX_SESSION}"

mkdir -p "${LAUNCH_LOG_DIR}"
cd "${REPO_ROOT}"
python3 scripts/run_eval.py --config "${CONFIG}" --dry-run >/dev/null

printf -v command \
    'cd %q && exec python3 scripts/run_eval.py --config %q >%q 2>&1' \
    "${REPO_ROOT}" "${CONFIG}" "${LAUNCH_LOG}"
tmux new-session -d -s "${TMUX_SESSION}" "${command}"

printf 'Started tmux session: %s\n' "${TMUX_SESSION}"
printf 'Attach: tmux attach -t %s\n' "${TMUX_SESSION}"
printf 'Launcher log: %s\n' "${LAUNCH_LOG}"
