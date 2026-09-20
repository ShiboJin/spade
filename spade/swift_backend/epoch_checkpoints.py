"""Keep complete epoch checkpoints outside Trainer's rolling checkpoint set."""
import math
from pathlib import Path
import re
import shutil
import tempfile

import torch.distributed as dist
from swift.callbacks import TrainerCallback, callbacks_map
from swift.utils import get_logger


def completed_epoch(state):
    if state.epoch is None or state.global_step <= 0:
        return None
    epoch = round(state.epoch)
    return epoch if epoch > 0 and math.isclose(state.epoch, epoch, abs_tol=1e-8, rel_tol=0) else None


class EpochCheckpointCallback(TrainerCallback):
    def __init__(self, args=None, trainer=None):
        # Swift's registry passes (args, trainer); direct HF use passes neither.
        super().__init__(args, trainer)
        self.last_saved_step = None

    def on_epoch_end(self, args, state, control, **kwargs):
        # Trainer also calls this on an early stop partway through an epoch.
        if completed_epoch(state) is not None and self.last_saved_step != state.global_step:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        self.last_saved_step = state.global_step
        epoch = completed_epoch(state)
        if epoch is None:
            return control
        distributed = dist.is_available() and dist.is_initialized()
        if distributed:
            # Each rank writes its own RNG state before reaching on_save.
            dist.barrier()
        error = [None]
        if state.is_world_process_zero:
            try:
                source = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                archive_root = Path(args.output_dir) / "epoch_checkpoints"
                archive_root.mkdir(exist_ok=True)
                target = archive_root / f"epoch-{epoch:03d}-step-{state.global_step}"
                if not target.exists():
                    # Copy independent files, then publish atomically. Rolling
                    # checkpoint deletion must never remove archive contents.
                    with tempfile.TemporaryDirectory(prefix=".staging-", dir=archive_root) as temporary:
                        staged = Path(temporary) / "checkpoint"
                        shutil.copytree(source, staged)
                        staged.rename(target)
                    get_logger().info("Preserved complete epoch checkpoint: %s", target)
                # Publish the new complete copy before removing the old one.
                # Only manage our exact directory names, never arbitrary files.
                archives = []
                for path in archive_root.iterdir():
                    match = re.fullmatch(r"epoch-(\d+)-step-(\d+)", path.name)
                    if match and path.is_dir() and not path.is_symlink():
                        archives.append(((int(match[1]), int(match[2])), path))
                for _, obsolete in sorted(archives)[:-1]:
                    shutil.rmtree(obsolete)
                    get_logger().info("Removed superseded epoch checkpoint: %s", obsolete)
            except Exception as exc:
                error[0] = f"Could not archive epoch {epoch} checkpoint: {exc}"
        if distributed:
            # Hold peers until the copy finishes, and propagate I/O failures.
            dist.broadcast_object_list(error, src=0)
        if error[0] is not None:
            raise RuntimeError(error[0])
        return control


callbacks_map["envduels_epoch_checkpoints"] = EpochCheckpointCallback
