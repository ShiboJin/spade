"""Read data-only YAML and pass validated values to existing runtime scripts."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_data import load_benchmark, validate_benchmark
DEFAULT_CONFIG = ROOT / "configs/envduels90_aime26.yaml"
# YAML field -> existing runtime variable and validation kind.
FIELDS = {
    "hardware": {
        "gpu_ids": ("GPU_IDS", "gpus"), "num_gpus": ("NUM_GPUS", "positive"),
        "tensor_parallel": ("TP", "positive"), "pipeline_parallel": ("PP", "positive"),
        "context_parallel": ("CP", "positive"), "rollout_tensor_parallel": ("ROLLOUT_TP", "positive"),
        "optimizer_cpu_offload": ("CPU_OFFLOAD", "bool01"),
    },
    "environments": {"count": ("ENVDUELS_EXPECTED_COUNT", "count"), "seed": ("ENVDUELS_SEED", "seed")},
    "training": {
        "learning_rate": ("LR", "positive_float"), "global_batch_size": ("GLOBAL_BATCH_SIZE", "positive"),
        "group_size": ("GROUP_SIZE", "positive"), "num_rollouts": ("NUM_ROLLOUT", "positive"),
        "save_interval": ("SAVE_INTERVAL", "positive"), "seed": ("TRAIN_SEED", "seed"),
        "temperature": ("TEMPERATURE", "nonnegative_float"), "max_turns": ("MAX_TURNS", "positive"),
        "max_tokens_per_turn": ("ACTOR_MAX_TOKENS", "positive"),
        "max_context_length": ("MAX_CONTEXT_LENGTH", "positive"),
        "max_tokens_per_gpu": ("MAX_TOKENS_PER_GPU", "positive"), "thinking": ("THINKING", "bool"),
        "rollout_memory_fraction": ("ROLLOUT_MEMORY_FRACTION", "fraction"),
    },
    "evaluation": {
        "benchmark": ("EVAL_BENCHMARK", "benchmark"),
        "data": ("EVAL_DATA", "path"), "hf_checkpoint": ("EVAL_HF_CHECKPOINT", "path"),
        "samples_per_problem": ("EVAL_N_SAMPLES", "positive"), "temperature": ("EVAL_TEMPERATURE", "nonnegative_float"),
        "top_p": ("EVAL_TOP_P", "fraction"), "top_k": ("EVAL_TOP_K", "top_k"),
        "max_tokens": ("EVAL_MAX_TOKENS", "positive"), "context_length": ("EVAL_CONTEXT_LENGTH", "positive"),
        "thinking": ("EVAL_THINKING", "bool"), "seed": ("EVAL_SEED", "seed"),
        "tensor_parallel": ("EVAL_TP", "positive"), "max_concurrent": ("EVAL_MAX_CONCURRENT", "positive"),
    },
}


class UniqueSafeLoader(yaml.SafeLoader):
    """Reject duplicate keys rather than silently accepting the last value."""
    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise ValueError(f"Invalid or duplicate YAML key: {key!r}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def encode(value, kind, from_env=False):
    if kind == "benchmark":
        return validate_benchmark(value)
    if kind.startswith("bool"):
        if from_env and value in ("true", "false", "1", "0"):
            value = value in ("true", "1")
        if type(value) is not bool:
            raise ValueError("expected YAML true/false")
        return str(int(value)) if kind == "bool01" else str(value).lower()
    if kind == "gpus":
        if from_env:
            value = [] if not value else value.split(",")
        if not isinstance(value, list):
            raise ValueError("gpu_ids must be a list")
        ids = [encode(v, "seed", from_env) for v in value]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate GPU IDs")
        return ",".join(ids)
    if kind in ("positive", "seed", "count", "top_k"):
        if from_env and isinstance(value, str) and value.lstrip("-").isdigit():
            value = int(value)
        if type(value) is not int:
            raise ValueError("expected integer")
        if kind == "count" and value != 90:
            raise ValueError("this experiment requires exactly 90 environments")
        if kind == "top_k":
            if value != -1 and value < 1:
                raise ValueError("top_k must be -1 or positive")
        elif value < (0 if kind == "seed" else 1):
            raise ValueError("integer out of range")
        return str(value)
    if kind == "path":
        if not isinstance(value, str) or not value or "\0" in value:
            raise ValueError("expected a nonempty path")
        if Path(value).is_absolute() or not (ROOT / value).resolve().is_relative_to(ROOT):
            raise ValueError("use a path relative to the SPADE checkout, without escaping it")
        return value
    if isinstance(value, bool):
        raise ValueError("expected number, not boolean")
    number = float(value)  # Accept scientific notation strings parsed by YAML 1.1.
    if not math.isfinite(number) or number < 0:
        raise ValueError("expected a finite nonnegative number")
    if kind in ("positive_float", "fraction") and number <= 0:
        raise ValueError("must be positive")
    if kind == "fraction" and number > 1:
        raise ValueError("must be <= 1")
    return str(number)


def load_environment(path, overrides=None):
    overrides = os.environ if overrides is None else overrides
    cfg = yaml.load(Path(path).read_text(), Loader=UniqueSafeLoader)
    if not isinstance(cfg, dict) or set(cfg) != set(FIELDS):
        raise ValueError(f"Required sections: {', '.join(FIELDS)}; unknown sections are rejected")
    result = {}
    for section, fields in FIELDS.items():
        if not isinstance(cfg[section], dict) or set(cfg[section]) != set(fields):
            raise ValueError(f"{section}: missing or unknown fields; expected {', '.join(fields)}")
        for key, (name, kind) in fields.items():
            try:
                result[name] = encode(cfg[section][key], kind)
                if name in overrides and name != "ENVDUELS_EXPECTED_COUNT":
                    result[name] = encode(overrides[name], kind, from_env=True)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{section}.{key}: {exc}") from exc
    result["ENVDUELS_IDS_FILE"] = ""  # Do not inherit an earlier split selection.
    group, batch = int(result["GROUP_SIZE"]), int(result["GLOBAL_BATCH_SIZE"])
    if group < 2 or batch % group:
        raise ValueError("global_batch_size must be divisible by group_size >= 2")
    if int(result["MAX_CONTEXT_LENGTH"]) <= int(result["ACTOR_MAX_TOKENS"]) + 64:
        raise ValueError("training context must leave room for the prompt")
    if int(result["EVAL_CONTEXT_LENGTH"]) <= int(result["EVAL_MAX_TOKENS"]):
        raise ValueError("evaluation context must leave room for the question")
    return result


def validate_hardware(values, action):
    n = int(values["NUM_GPUS"])
    if values["GPU_IDS"] and len(values["GPU_IDS"].split(",")) != n:
        raise ValueError("num_gpus must match the number of gpu_ids")
    if action in ("eval", "eval-plan"):
        if int(values["EVAL_TP"]) > n:
            raise ValueError("evaluation.tensor_parallel exceeds num_gpus")
    elif action in ("plan", "check", "smoke", "train"):
        model_parallel = int(values["TP"]) * int(values["PP"]) * int(values["CP"])
        if n % model_parallel or n % int(values["ROLLOUT_TP"]):
            raise ValueError("num_gpus must be divisible by TP*PP*CP and rollout_tensor_parallel")
        if int(values["GLOBAL_BATCH_SIZE"]) % (n // model_parallel):
            raise ValueError("global_batch_size must be divisible by the data-parallel size")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", nargs="?", default="plan",
                   choices=["config", "plan", "check", "convert", "smoke", "train", "prepare-eval", "eval-plan", "eval"])
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = p.parse_args()
    try:
        values = load_environment(args.config)
        validate_hardware(values, args.action)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        p.error(str(exc))
    if args.action == "config":
        print(json.dumps(values, indent=2))
        return
    env = {**os.environ, **values}
    if args.action == "prepare-eval":
        if values["EVAL_BENCHMARK"] != "aime2026":
            rows, digest = load_benchmark(ROOT / values["EVAL_DATA"], values["EVAL_BENCHMARK"])
            print(json.dumps(dict(benchmark=values["EVAL_BENCHMARK"], n_problems=len(rows), sha256=digest)))
            return
        cmd = [sys.executable, "scripts/prepare_aime26.py", "--output", values["EVAL_DATA"]]
    elif args.action == "eval-plan":
        cmd = [sys.executable, "scripts/eval_benchmark.py", "--plan"]
    else:
        cmd = ["bash", "scripts/envduels_runtime.sh", args.action]
    raise SystemExit(subprocess.call(cmd, cwd=ROOT, env=env))


if __name__ == "__main__":
    main()
