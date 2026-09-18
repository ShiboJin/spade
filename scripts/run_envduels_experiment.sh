#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ $# -le 2 ]] || { echo 'Usage: run_envduels_experiment.sh ACTION [config.yaml]' >&2; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python3
exec "$PYTHON_BIN" scripts/run_envduels_experiment.py "${1:-plan}" \
    --config "${2:-$ROOT/configs/envduels90_aime26.yaml}"
