#!/usr/bin/env bash
# Qwen3.8 retains the Qwen3.5-27B architecture. Model identity comes from
# --hf-checkpoint, never from this reused architecture filename.
QWEN38_CONFIG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$QWEN38_CONFIG_DIR/../../slime/scripts/models/qwen3.5-27B.sh"
