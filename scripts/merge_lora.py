"""Merge a PEFT LoRA into a separate Hugging Face checkpoint using ms-swift.

The source model and adapter are mounted read-only.  The destination must not
already exist, so this command can never overwrite the base checkpoint.

Plan this run's checkpoint-24 merge:
  python scripts/merge_lora.py

Run the merge:
  python scripts/merge_lora.py --run

Merge another checkpoint:
  python scripts/merge_lora.py --adapter outputs/training/RUN/checkpoint/VERSION/checkpoint-N \
    --output checkpoints/my-merged-model --run
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "checkpoints/Qwen3.8-27B"
DEFAULT_ADAPTER = (
    ROOT
    / "outputs/training/20260922T102058.777696Z/checkpoint/"
    "v0-20260922-102143/checkpoint-24"
)
DEFAULT_OUTPUT = ROOT / "checkpoints/Qwen3.8-27B-spade-merged-ckpt24"
DEFAULT_IMAGE = "envduels-unified:cu124-wandb"
GIB = 1024**3


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def staging_path(output: Path) -> Path:
    return output.parent / f".{output.name}.merge-tmp"


def validate_inputs(base: Path, adapter: Path, output: Path) -> None:
    if output == base or output == adapter or output.is_relative_to(base) or output.is_relative_to(adapter):
        raise ValueError("Output must be separate from the base and adapter")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if staging_path(output).exists():
        raise FileExistsError(
            f"Previous merge staging directory exists: {staging_path(output)}"
        )
    if not (base / "config.json").is_file():
        raise FileNotFoundError(f"Base model config not found: {base / 'config.json'}")
    if not (base / "model.safetensors.index.json").is_file():
        raise FileNotFoundError("Base model must be a sharded safetensors checkpoint")
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Adapter config not found: {adapter}")
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Adapter weights not found: {adapter}")

    adapter_cfg = json.loads((adapter / "adapter_config.json").read_text())
    if adapter_cfg.get("peft_type") != "LORA":
        raise ValueError("Adapter is not a PEFT LoRA checkpoint")
    base_cfg = json.loads((base / "config.json").read_text())
    if base_cfg.get("model_type") != "qwen3_5":
        raise ValueError("Expected the Qwen3.5/Qwen3.8 base checkpoint")

    required = directory_size(base) + 10 * GIB
    disk_path = next((parent for parent in (output.parent, *output.parents) if parent.exists()), None)
    if disk_path is None:
        raise FileNotFoundError(f"No existing parent directory for output: {output}")
    available = shutil.disk_usage(disk_path).free
    if available < required:
        raise OSError(
            f"Insufficient disk space: need at least {required / GIB:.1f} GiB, "
            f"have {available / GIB:.1f} GiB"
        )


def docker_command(
    base: Path,
    adapter: Path,
    output: Path,
    image: str,
    memory_gib: int,
) -> list[str]:
    staging = staging_path(output)
    return [
        "docker", "run", "--rm", "--init", "--network", "none",
        "--memory", f"{memory_gib}g", "--memory-swap", f"{memory_gib}g",
        "--memory-swappiness", "0", "--shm-size", "8g",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--mount", f"type=bind,src={base},dst=/model,readonly",
        "--mount", f"type=bind,src={adapter},dst=/adapter,readonly",
        "--mount", f"type=bind,src={staging},dst=/output",
        "--env", "HOME=/tmp/home", "--env", "USER=envduels",
        "--env", "HF_HOME=/tmp/hf",
        "--env", "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor",
        "--entrypoint", "swift", image, "export",
        "--model", "/model", "--model_type", "qwen3_5", "--template", "qwen3_8",
        "--adapters", "/adapter",
        "--load_args", "false", "--load_data_args", "false",
        "--merge_lora", "true", "--torch_dtype", "bfloat16",
        "--device_map", "cpu", "--safe_serialization", "true",
        "--max_shard_size", "4GB", "--exist_ok", "true",
        "--output_dir", "/output/merged",
    ]


def validate_output(output: Path) -> None:
    config = output / "config.json"
    index_path = output / "model.safetensors.index.json"
    if not config.is_file() or not index_path.is_file():
        raise RuntimeError(f"Merged checkpoint is incomplete: {output}")
    index = json.loads(index_path.read_text())
    shards = {output / name for name in index.get("weight_map", {}).values()}
    missing = sorted(str(path) for path in shards if not path.is_file())
    if not shards or missing:
        raise RuntimeError(f"Merged checkpoint has missing weight shards: {missing}")
    if (output / "adapter_model.safetensors").exists():
        raise RuntimeError("Output still contains adapter weights; merge did not complete")
    print(
        f"Merged checkpoint validated: {output} "
        f"({len(shards)} shards, {directory_size(output) / GIB:.1f} GiB)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--memory-gib", type=int, default=96)
    parser.add_argument("--sudo-docker", action="store_true", help="Run Docker through passwordless sudo")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    base = args.base.expanduser().resolve()
    adapter = args.adapter.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if args.memory_gib < 64:
        raise ValueError("Merging this 27B BF16 model requires at least 64 GiB")
    validate_inputs(base, adapter, output)
    command = docker_command(base, adapter, output, args.image, args.memory_gib)
    if args.sudo_docker:
        command = ["sudo", "-n", *command]
    print("Command:")
    print(" ".join(command))
    if not args.run:
        print("Plan only; add --run to merge the checkpoint.")
        return 0
    staging = staging_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        subprocess.run(command, check=True)
    except BaseException:
        # Remove only a still-empty directory.  Any partial checkpoint is kept
        # for inspection and will never be overwritten on the next invocation.
        try:
            staging.rmdir()
        except OSError:
            pass
        raise
    merged = staging / "merged"
    validate_output(merged)
    merged.rename(output)
    staging.rmdir()
    print(f"Final merged checkpoint: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
