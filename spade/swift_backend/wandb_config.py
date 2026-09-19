"""Attach SPADE's resolved experiment configuration to the Trainer W&B run."""
from __future__ import annotations

import json
import os
from pathlib import Path

from swift.callbacks import TrainerCallback, callbacks_map


class EnvDuelsWandbConfigCallback(TrainerCallback):
    """Upload launcher-only settings after Transformers initializes W&B."""

    def on_train_begin(self, args, state, control, **kwargs):
        del kwargs
        if not state.is_world_process_zero or "wandb" not in args.report_to:
            return control

        import wandb

        if wandb.run is None:
            raise RuntimeError("W&B run was not initialized before the SPADE config callback")
        config_path = os.environ.get("SPADE_RESOLVED_CONFIG")
        if not config_path:
            raise RuntimeError("SPADE_RESOLVED_CONFIG is required when W&B reporting is enabled")
        document = json.loads(Path(config_path).read_text(encoding="utf-8"))
        experiment = document.get("config")
        if not isinstance(experiment, dict):
            raise RuntimeError("Resolved training report is missing its config mapping")
        wandb.config.update({"experiment": experiment}, allow_val_change=True)
        return control


callbacks_map["envduels_wandb_config"] = EnvDuelsWandbConfigCallback
