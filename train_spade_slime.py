#!/usr/bin/env python3
"""SPADE training wrapper for Slime.

This script wraps Slime's training loop with SPADE-specific argument handling.
It uses Slime's extension points to add SPADE arguments without modifying Slime's core.
"""

import importlib.util
import sys
from pathlib import Path


def load_train():
    # The pinned Slime checkout ships train.py at its repository root, not
    # as slime.train in the installed Python package.
    entry = Path(__file__).resolve().parent / "slime" / "train.py"
    spec = importlib.util.spec_from_file_location("spade_slime_train", entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.train


def main():
    """Parse arguments with SPADE extensions and run training."""
    from slime.utils.arguments import parse_args
    validate_only = "--validate-only" in sys.argv
    if validate_only:
        sys.argv.remove("--validate-only")
        # Megatron's full validator queries CUDA device properties even before
        # model allocation. This mode deliberately stops at CPU parsing/HF checks.
        from spade.slime.arguments import add_spade_arguments
        from slime.utils.arguments import get_slime_extra_args_provider
        from slime.backends.megatron_utils.arguments import megatron_parse_args
        from slime.backends.sglang_utils.arguments import sglang_parse_args
        sglang_parse_args()
        megatron_parse_args(extra_args_provider=get_slime_extra_args_provider(add_spade_arguments))
        print("CPU argument parsing and HF configuration checks passed; CUDA-dependent validation deferred.")
        return

    from spade.slime.arguments import add_spade_arguments

    args = parse_args(add_custom_arguments=add_spade_arguments)

    load_train()(args)


if __name__ == "__main__":
    main()
