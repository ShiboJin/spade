"""Run config-driven EnvDuels LoRA GRPO in the unified Docker image."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_guard import (
    DEFAULT_RESERVE_GIB, docker_memory_args, guarded_wait, task_resources, validate_limits,
)
CONTAINER_ROOT = Path("/workspace/envduels/spade")
CONTAINER_EXPORT = Path("/workspace/envduels/exports/duel_harness_004_rl")

FIELDS = {
    "image", "gpu_ids", "model", "dataset", "export_dir", "output_dir",
    "fixed_pool_seed", "fixed_pool_epochs", "max_steps",
    "num_games_per_rollout", "trajectories_per_game", "batch_size",
    "per_device_train_batch_size", "num_substeps", "dataset_shuffle",
    "remove_constant_reward_groups", "max_rollout_attempts",
    "learning_rate", "warmup_ratio", "lr_scheduler_type",
    "adam_beta1", "adam_beta2", "adam_epsilon", "weight_decay",
    "kl_penalty_coef", "loss_type", "reward_normalization",
    "ppo_clip_low", "ppo_clip_high", "seed",
    "lora_rank", "lora_alpha", "max_turns", "max_context_length",
    "actor_max_tokens", "enable_thinking", "preserve_thinking",
    "vllm_tensor_parallel", "vllm_gpu_memory_utilization", "move_model_batches",
    "actor_temperature", "actor_top_p", "actor_top_k",
    "overlong_filter", "rollout_json_export",
    "wandb_enabled", "wandb_mode", "wandb_project", "wandb_entity",
    "wandb_run_name", "resume_from_checkpoint",
    "save_every", "save_total_limit",
}
MEMORY_DEFAULTS = {"memory_limit_gib": 160, "host_memory_reserve_gib": DEFAULT_RESERVE_GIB}
ROLLOUT_DEFAULTS = {
    "sage_hint_resampling": True,
    "min_valid_groups": 4,
    "vllm_enforce_eager": True,
    "sleep_level": 2,
    "offload_model": True,
    "offload_optimizer": True,
}
# Preserve the validated low-memory behavior for existing configuration files.
GRPO_MEMORY_DEFAULTS = {
    "grpo_chunked_logps": True,
    "grpo_logps_chunk_size": 128,
    "grpo_decoder_checkpointing": True,
    "grpo_cpu_activation_offload": True,
    "grpo_checkpoint_delta_rule": True,
}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_path(value: str) -> Path:
    return (ROOT / Path(value).expanduser()).resolve()


def container_path(path: Path) -> str:
    try:
        relative = path.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"{path} must be inside the SPADE checkout") from exc
    return str(CONTAINER_ROOT / relative)


def fixed_seed(master_seed: int, env_id: str) -> int:
    material = f"{master_seed}\0{env_id}\0{0}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 2**63


def load_config(path: Path, max_steps: int | None = None, epochs: int | None = None) -> dict:
    if path.suffix.lower() != ".json":
        raise ValueError("Training configuration must be a .json file")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {"training", "accelerate"}:
        raise ValueError("Config must contain exactly 'training' and 'accelerate' mappings")
    if not isinstance(document["training"], dict) or not isinstance(document["accelerate"], dict):
        raise ValueError("training and accelerate must be mappings")
    cfg = {**MEMORY_DEFAULTS, **GRPO_MEMORY_DEFAULTS, **ROLLOUT_DEFAULTS, **document["training"]}
    missing = FIELDS - cfg.keys()
    unknown = cfg.keys() - (FIELDS | MEMORY_DEFAULTS.keys() | GRPO_MEMORY_DEFAULTS.keys() | ROLLOUT_DEFAULTS.keys())
    if missing:
        raise ValueError(f"Missing training settings: {sorted(missing)}")
    if unknown:
        raise ValueError(f"Unknown training settings: {sorted(unknown)}")
    validate_limits(cfg["memory_limit_gib"], cfg["host_memory_reserve_gib"])
    if max_steps is not None and epochs is not None:
        raise ValueError("max_steps and epochs overrides are mutually exclusive")
    if max_steps is not None:
        cfg["max_steps"] = max_steps
    elif epochs is not None:
        cfg["max_steps"] = None
        cfg["fixed_pool_epochs"] = epochs

    for key in ("image", "model", "dataset", "export_dir", "output_dir"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if cfg["resume_from_checkpoint"] is not None and (
        not isinstance(cfg["resume_from_checkpoint"], str)
        or not cfg["resume_from_checkpoint"].strip()
    ):
        raise ValueError("resume_from_checkpoint must be null or a nonempty path")
    ids = cfg["gpu_ids"]
    if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("gpu_ids must contain nonnegative integers")
    if len(set(ids)) != len(ids):
        raise ValueError("gpu_ids must not contain duplicates")

    positive_ints = (
        "fixed_pool_epochs", "num_games_per_rollout", "trajectories_per_game",
        "batch_size", "per_device_train_batch_size", "num_substeps",
        "max_rollout_attempts", "min_valid_groups", "lora_rank", "lora_alpha", "max_turns",
        "max_context_length", "actor_max_tokens", "vllm_tensor_parallel",
        "actor_top_k", "save_every", "save_total_limit", "move_model_batches", "grpo_logps_chunk_size",
    )
    for key in positive_ints:
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["max_steps"] is not None and (type(cfg["max_steps"]) is not int or cfg["max_steps"] < 1):
        raise ValueError("max_steps must be null or a positive integer")
    if type(cfg["fixed_pool_seed"]) is not int or not 0 <= cfg["fixed_pool_seed"] < 2**63:
        raise ValueError("fixed_pool_seed must be an integer in [0, 2**63)")
    for key in ("sage_hint_resampling", "dataset_shuffle", "remove_constant_reward_groups", "enable_thinking",
                "preserve_thinking", "overlong_filter", "rollout_json_export",
                "wandb_enabled", "grpo_chunked_logps", "grpo_decoder_checkpointing",
                "grpo_cpu_activation_offload", "grpo_checkpoint_delta_rule",
                "vllm_enforce_eager", "offload_model", "offload_optimizer"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    if cfg["sage_hint_resampling"] and cfg["num_substeps"] != 1:
        raise ValueError("sage_hint_resampling requires num_substeps=1 for whole-window skipping")
    if cfg["sage_hint_resampling"] and cfg["min_valid_groups"] > cfg["num_games_per_rollout"]:
        raise ValueError("min_valid_groups must not exceed num_games_per_rollout")
    if type(cfg["sleep_level"]) is not int or cfg["sleep_level"] not in (0, 1, 2):
        raise ValueError("sleep_level must be an integer in {0, 1, 2}")
    if cfg["grpo_cpu_activation_offload"] and not cfg["grpo_decoder_checkpointing"]:
        raise ValueError("grpo_cpu_activation_offload requires grpo_decoder_checkpointing")
    if cfg["wandb_mode"] not in ("online", "offline"):
        raise ValueError("wandb_mode must be online or offline")
    if not isinstance(cfg["wandb_project"], str) or not cfg["wandb_project"].strip():
        raise ValueError("wandb_project must be a nonempty string")
    for key in ("wandb_entity", "wandb_run_name"):
        if cfg[key] is not None and (not isinstance(cfg[key], str) or not cfg[key].strip()):
            raise ValueError(f"{key} must be null or a nonempty string")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    for key in ("learning_rate", "warmup_ratio", "adam_beta1", "adam_beta2",
                "adam_epsilon", "weight_decay", "kl_penalty_coef", "ppo_clip_low",
                "ppo_clip_high", "vllm_gpu_memory_utilization", "actor_temperature",
                "actor_top_p"):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]):
            raise ValueError(f"{key} must be a finite number")
    if cfg["learning_rate"] <= 0 or cfg["adam_epsilon"] <= 0 or cfg["weight_decay"] < 0:
        raise ValueError("Invalid optimizer setting")
    if cfg["kl_penalty_coef"] < 0:
        raise ValueError("kl_penalty_coef must be nonnegative")
    if not 0 <= cfg["warmup_ratio"] < 1 or not 0 < cfg["actor_top_p"] <= 1:
        raise ValueError("warmup_ratio or actor_top_p is outside its valid range")
    if not 0 <= cfg["adam_beta1"] < 1 or not 0 <= cfg["adam_beta2"] < 1:
        raise ValueError("Adam beta values must be in [0, 1)")
    if not 0 <= cfg["ppo_clip_low"] < 1 or cfg["ppo_clip_high"] < cfg["ppo_clip_low"]:
        raise ValueError("Invalid PPO clipping range")
    if not 0 < cfg["vllm_gpu_memory_utilization"] < 1:
        raise ValueError("vllm_gpu_memory_utilization must be in (0, 1)")
    if not 0 <= cfg["actor_temperature"] <= 2:
        raise ValueError("actor_temperature must be in [0, 2]")
    if cfg["lr_scheduler_type"] not in ("constant", "linear", "cosine"):
        raise ValueError("Unsupported lr_scheduler_type")
    if cfg["loss_type"] != "grpo":
        raise ValueError("This launcher requires loss_type=grpo")
    if cfg["reward_normalization"] not in ("grpo", "grpo_no_std"):
        raise ValueError("reward_normalization must be grpo or grpo_no_std")
    if cfg["vllm_tensor_parallel"] > len(ids) or len(ids) % cfg["vllm_tensor_parallel"]:
        raise ValueError("vllm_tensor_parallel must divide the selected GPU count")
    if cfg["trajectories_per_game"] < 2:
        raise ValueError("trajectories_per_game must be at least 2 for GRPO")
    if cfg["batch_size"] != cfg["num_games_per_rollout"] * cfg["trajectories_per_game"]:
        raise ValueError("batch_size must equal num_games_per_rollout * trajectories_per_game")
    global_micro_batch = cfg["per_device_train_batch_size"] * len(ids)
    if cfg["batch_size"] % global_micro_batch:
        raise ValueError("batch_size must be divisible by per-device batch times GPU count")
    gradient_accumulation_steps = cfg["batch_size"] // global_micro_batch
    if cfg["max_context_length"] <= cfg["actor_max_tokens"] + 64:
        raise ValueError("max_context_length must leave room for prompts and history")

    for key in ("model", "dataset", "export_dir", "output_dir"):
        cfg[key] = resolve_path(cfg[key])
    if cfg["resume_from_checkpoint"] is not None:
        cfg["resume_from_checkpoint"] = resolve_path(cfg["resume_from_checkpoint"])
    for key in ("model", "dataset", "output_dir"):
        container_path(cfg[key])
    if cfg["resume_from_checkpoint"] is not None:
        container_path(cfg["resume_from_checkpoint"])
        if not cfg["resume_from_checkpoint"].is_dir():
            raise ValueError(f"Missing resume checkpoint: {cfg['resume_from_checkpoint']}")
    if not (cfg["model"] / "config.json").is_file():
        raise ValueError(f"Missing Hugging Face model: {cfg['model']}")
    if not cfg["dataset"].is_file():
        raise ValueError(f"Missing training dataset: {cfg['dataset']}")
    if not (cfg["export_dir"] / "manifest.json").is_file():
        raise ValueError(f"Missing EnvDuels export: {cfg['export_dir']}")
    manifest = json.loads((cfg["export_dir"] / "manifest.json").read_text(encoding="utf-8"))
    manifest_ids = [item.get("id") for item in manifest.get("environments", [])]
    if not manifest_ids or any(not isinstance(env_id, str) or not env_id for env_id in manifest_ids):
        raise ValueError("EnvDuels manifest has invalid environment IDs")
    if len(manifest_ids) != len(set(manifest_ids)):
        raise ValueError("EnvDuels manifest has duplicate environment IDs")
    if cfg["sage_hint_resampling"]:
        from spade.core.envduels_hints import load_hint_levels
        for row in manifest["environments"]:
            if not load_hint_levels(cfg["export_dir"], row):
                raise ValueError(f"SAGE training requires an author hint: {row['id']}")
    accelerate = document["accelerate"]
    if accelerate.get("distributed_type") != "FSDP":
        raise ValueError("Accelerate config must use FSDP")
    if accelerate.get("num_processes") != len(ids):
        raise ValueError("Accelerate num_processes must equal the selected GPU count")
    if accelerate.get("mixed_precision") != "bf16" or accelerate.get("use_cpu") is not False:
        raise ValueError("Accelerate config must use GPU BF16")
    fsdp = accelerate.get("fsdp_config")
    if not isinstance(fsdp, dict) or fsdp.get("fsdp_version") != 2:
        raise ValueError("Accelerate config must use FSDP2")
    cfg["accelerate"] = accelerate

    rows = 0
    environment_counts = Counter()
    environment_seeds = set()
    with cfg["dataset"].open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                env = row["env_config"]
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid training row {number}: {exc}") from exc
            if env.get("name") != "envduels" or env.get("export_dir") != str(CONTAINER_EXPORT):
                raise ValueError(f"Training row {number} has an incompatible EnvDuels config")
            env_id, seed = env.get("env_id"), env.get("seed")
            if not isinstance(env_id, str) or not env_id or type(seed) is not int:
                raise ValueError(f"Training row {number} requires an environment ID and integer seed")
            if (env_id, seed) in environment_seeds:
                raise ValueError(f"Training row {number} duplicates an environment/seed pair")
            if seed != fixed_seed(cfg["fixed_pool_seed"], env_id):
                raise ValueError(
                    f"Training row {number} seed does not match fixed_pool_seed={cfg['fixed_pool_seed']}"
                )
            environment_seeds.add((env_id, seed))
            environment_counts[env_id] += 1
            rows += 1
    if set(environment_counts) != set(manifest_ids):
        raise ValueError("Fixed dataset must contain every exported environment exactly once")
    if any(count != 1 for count in environment_counts.values()):
        raise ValueError("Fixed dataset must contain exactly one row per environment")
    if rows % cfg["num_games_per_rollout"]:
        raise ValueError("Environment count must be divisible by num_games_per_rollout")
    cfg["dataset_rows"] = rows
    steps_per_dataset_pass = rows // cfg["num_games_per_rollout"] * cfg["num_substeps"]
    cfg["derived"] = {
        "dataset_environments": len(environment_counts),
        "fixed_instances_per_environment": 1,
        "rollouts_per_environment_per_pool_epoch": cfg["trajectories_per_game"],
        "rollouts_per_environment_total": (
            cfg["trajectories_per_game"] * cfg["fixed_pool_epochs"]
            if cfg["max_steps"] is None else None
        ),
        "global_micro_batch_rollouts": global_micro_batch,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "generation_batch_rollouts": cfg["batch_size"],
        "environments_per_generation_batch": cfg["num_games_per_rollout"],
        "environments_per_optimizer_step": cfg["num_games_per_rollout"],
        "optimizer_steps_per_dataset_pass": steps_per_dataset_pass,
        "rollouts_per_dataset_pass": rows * cfg["trajectories_per_game"],
        "configured_dataset_passes": (
            cfg["max_steps"] / steps_per_dataset_pass
            if cfg["max_steps"] is not None
            else cfg["fixed_pool_epochs"]
        ),
    }
    return cfg


def docker_command(cfg: dict, run_dir: Path, container_name: str) -> tuple[list[str], Path]:
    checkpoint_dir = run_dir / "checkpoint"
    online_wandb = cfg["wandb_enabled"] and cfg["wandb_mode"] == "online"
    command = [
        "docker", "run", "--rm", "--init", "--name", container_name,
        "--gpus", '"device=' + ",".join(map(str, cfg["gpu_ids"])) + '"',
        "--network", "bridge" if online_wandb else "none", "--shm-size", "16g",
        "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--mount", f"type=bind,src={ROOT},dst={CONTAINER_ROOT}",
        "--mount", f"type=bind,src={cfg['export_dir']},dst={CONTAINER_EXPORT},readonly",
        "--workdir", str(CONTAINER_ROOT),
    ]
    command += docker_memory_args(cfg["memory_limit_gib"])
    command += ["--label", "spade.gpu-concurrent=true"]
    environment = {
        # Numeric host UIDs need not exist in the image's /etc/passwd.
        # Inductor calls getpass.getuser() even with TORCHINDUCTOR_CACHE_DIR set.
        "USER": "envduels",
        "SPADE_MEMORY_LIMIT_GIB": cfg["memory_limit_gib"],
        "NUM_GPUS": len(cfg["gpu_ids"]),
        "MODEL": container_path(cfg["model"]),
        "DATASET": container_path(cfg["dataset"]),
        "OUTPUT_DIR": container_path(checkpoint_dir),
        "ACCELERATE_CONFIG": container_path(run_dir / "accelerate_config.json"),
        "NUM_GENERATIONS": cfg["trajectories_per_game"],
        "PER_DEVICE_TRAIN_BATCH_SIZE": cfg["per_device_train_batch_size"],
        "GRADIENT_ACCUMULATION_STEPS": cfg["derived"]["gradient_accumulation_steps"],
        "GENERATION_BATCH_SIZE": cfg["batch_size"],
        "NUM_ITERATIONS": cfg["num_substeps"],
        "DATASET_SHUFFLE": str(cfg["dataset_shuffle"]).lower(),
        "LEARNING_RATE": cfg["learning_rate"],
        "WARMUP_RATIO": cfg["warmup_ratio"],
        "BETA": cfg["kl_penalty_coef"],
        "TRAIN_SEED": cfg["seed"],
        "LORA_RANK": cfg["lora_rank"],
        "LORA_ALPHA": cfg["lora_alpha"],
        "SPADE_GRPO_CHUNKED_LOGPS": str(cfg["grpo_chunked_logps"]).lower(),
        "SPADE_GRPO_LOGPS_CHUNK_SIZE": cfg["grpo_logps_chunk_size"],
        "SPADE_GRPO_DECODER_CHECKPOINTING": str(cfg["grpo_decoder_checkpointing"]).lower(),
        "SPADE_GRPO_CPU_ACTIVATION_OFFLOAD": str(cfg["grpo_cpu_activation_offload"]).lower(),
        "SPADE_GRPO_CHECKPOINT_DELTA_RULE": str(cfg["grpo_checkpoint_delta_rule"]).lower(),
        "MAX_TURNS": cfg["max_turns"],
        "MAX_LENGTH": cfg["max_context_length"],
        "MAX_COMPLETION_LENGTH": cfg["actor_max_tokens"],
        "VLLM_TP": cfg["vllm_tensor_parallel"],
        "VLLM_ENFORCE_EAGER": str(cfg["vllm_enforce_eager"]).lower(),
        "SLEEP_LEVEL": cfg["sleep_level"],
        "OFFLOAD_MODEL": str(cfg["offload_model"]).lower(),
        "OFFLOAD_OPTIMIZER": str(cfg["offload_optimizer"]).lower(),
        "MOVE_MODEL_BATCHES": cfg["move_model_batches"],
        "VLLM_GPU_MEMORY_UTILIZATION": cfg["vllm_gpu_memory_utilization"],
        "TEMPERATURE": cfg["actor_temperature"],
        "TOP_P": cfg["actor_top_p"],
        "TOP_K": cfg["actor_top_k"],
        "ENABLE_THINKING": str(cfg["enable_thinking"]).lower(),
        "PRESERVE_THINKING": str(cfg["preserve_thinking"]).lower(),
        "SCALE_REWARDS": "none" if cfg["reward_normalization"] == "grpo_no_std" else "group",
        "LOSS_TYPE": cfg["loss_type"],
        "PPO_CLIP_LOW": cfg["ppo_clip_low"],
        "PPO_CLIP_HIGH": cfg["ppo_clip_high"],
        # SAGE owns group refill, so disable the separate DAPO resampling loop.
        "DYNAMIC_SAMPLE": str(cfg["remove_constant_reward_groups"] and not cfg["sage_hint_resampling"]).lower(),
        "SPADE_SAGE_HINT_RESAMPLING": str(cfg["sage_hint_resampling"]).lower(),
        "SPADE_MAX_ROLLOUT_ATTEMPTS": cfg["max_rollout_attempts"],
        "SPADE_MIN_VALID_GROUPS": cfg["min_valid_groups"],
        "MAX_RESAMPLE_TIMES": cfg["max_rollout_attempts"],
        "OVERLONG_FILTER": str(cfg["overlong_filter"]).lower(),
        "LOG_COMPLETIONS": str(cfg["rollout_json_export"]).lower(),
        "WEIGHT_DECAY": cfg["weight_decay"],
        "ADAM_BETA1": cfg["adam_beta1"],
        "ADAM_BETA2": cfg["adam_beta2"],
        "ADAM_EPSILON": cfg["adam_epsilon"],
        "LR_SCHEDULER_TYPE": cfg["lr_scheduler_type"],
        "SAVE_STEPS": cfg["save_every"],
        "SAVE_TOTAL_LIMIT": cfg["save_total_limit"],
        "REPORT_TO": "tensorboard,wandb" if cfg["wandb_enabled"] else "tensorboard",
        "RUN_NAME": cfg["wandb_run_name"] or container_name,
        "HF_HUB_OFFLINE": 1,
        "TRANSFORMERS_OFFLINE": 1,
        "HF_HOME": "/tmp/hf",
        "HF_DATASETS_CACHE": "/tmp/hf/datasets",
        "HUGGINGFACE_HUB_CACHE": "/tmp/hf/hub",
        "MODELSCOPE_CACHE": "/tmp/modelscope",
        "XDG_CACHE_HOME": "/tmp/cache",
        "XDG_CONFIG_HOME": "/tmp/config",
        "VLLM_CACHE_ROOT": "/tmp/vllm",
        "VLLM_CONFIG_ROOT": "/tmp/vllm-config",
        "VLLM_NO_USAGE_STATS": 1,
        "FLASHINFER_WORKSPACE_BASE": "/tmp/flashinfer",
        "CUDA_CACHE_PATH": "/tmp/cuda",
        "TRITON_CACHE_DIR": "/tmp/triton",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor",
        "PYTHONUNBUFFERED": 1,
        # Pinned vLLM temporarily disables this inside its sleep memory pool.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    if cfg["wandb_enabled"]:
        environment.update({
            "WANDB_MODE": cfg["wandb_mode"],
            "WANDB_PROJECT": cfg["wandb_project"],
            "WANDB_DIR": container_path(run_dir / "wandb"),
            "WANDB_DATA_DIR": container_path(run_dir / "wandb" / "data"),
            "WANDB_CACHE_DIR": container_path(run_dir / "wandb" / "cache"),
            "WANDB_LOG_MODEL": "false",
            "WANDB_WATCH": "false",
            "SPADE_RESOLVED_CONFIG": container_path(run_dir / "resolved_config.json"),
        })
        if cfg["wandb_entity"] is not None:
            environment["WANDB_ENTITY"] = cfg["wandb_entity"]
    if cfg["max_steps"] is not None:
        environment["MAX_STEPS"] = cfg["max_steps"]
    else:
        environment["NUM_TRAIN_EPOCHS"] = cfg["fixed_pool_epochs"]
    if cfg["resume_from_checkpoint"] is not None:
        environment["RESUME_FROM_CHECKPOINT"] = container_path(cfg["resume_from_checkpoint"])
    for key, value in environment.items():
        command += ["--env", f"{key}={value}"]
    if online_wandb:
        # Let Docker copy the host value without exposing the secret in launch.json.
        command += ["--env", "WANDB_API_KEY"]
    command += [cfg["image"], "bash", "cmd/games/train_envduels_lora_swift.sh"]
    return command, checkpoint_dir


def report_config(cfg: dict) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in cfg.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train_qwen38_envduels_lora.json")
    parser.add_argument("--smoke", action="store_true", help="Run exactly one optimizer step")
    parser.add_argument("--max-steps", type=int, help="Override training.max_steps")
    parser.add_argument("--epochs", type=int, help="Override training.fixed_pool_epochs")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the Docker command")
    args = parser.parse_args()
    if args.smoke and (args.max_steps is not None or args.epochs is not None):
        parser.error("--smoke cannot be combined with --max-steps or --epochs")
    if args.max_steps is not None and args.epochs is not None:
        parser.error("--max-steps and --epochs cannot be used together")
    cfg = load_config(args.config, 1 if args.smoke else args.max_steps, args.epochs)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = cfg["output_dir"] / stamp
    container_name = "spade-train-" + stamp.lower()
    command, checkpoint_dir = docker_command(cfg, run_dir, container_name)
    report = {
        "config": report_config(cfg),
        "mode": "smoke" if args.smoke else "train",
        "output": str(run_dir),
        "checkpoint_output": str(checkpoint_dir),
        "container": container_name,
        "command": command,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("Dry run only; no container started or GPU allocated.")
        return
    if cfg["wandb_enabled"] and cfg["wandb_mode"] == "online" and not os.environ.get("WANDB_API_KEY"):
        parser.error("online W&B requires WANDB_API_KEY in the host environment")

    with task_resources(cfg, ROOT):
        run_training(cfg, run_dir, container_name, command, report)


def run_training(cfg, run_dir, container_name, command, report):
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", cfg["image"], "--format", "{{.Id}}"], text=True
    ).strip()
    run_dir.mkdir(parents=True, exist_ok=False)
    if cfg["wandb_enabled"]:
        for subdirectory in ("data", "cache"):
            (run_dir / "wandb" / subdirectory).mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "accelerate_config.json", cfg["accelerate"])
    write_json(run_dir / "resolved_config.json", report)
    write_json(run_dir / "launch.json", {"image_id": image_id, "command": command})
    write_json(run_dir / "status.json", {"status": "running"})
    print(f"Running; follow progress with: tail -f {run_dir / 'console.log'}", flush=True)
    try:
        with (run_dir / "console.log").open("w") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            code = guarded_wait(process, container_name, cfg["host_memory_reserve_gib"])
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        write_json(run_dir / "status.json", {"status": status, "error": str(exc)})
        raise
    write_json(run_dir / "status.json", {
        "status": "completed" if code == 0 else "failed", "exit_code": code
    })
    print(f"Exit {code}; results/logs: {run_dir}", flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
