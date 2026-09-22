"""Compatibility wrapper for the merged evaluation launcher.

Use ``scripts/run_eval.py`` for new commands. Existing commands which name
this file continue to work and accept the same unified JSON configuration.
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from scripts.run_eval import (  # noqa: F401
    ROOT,
    checkpoint_mount,
    child_environment,
    docker_command,
    load_config,
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
