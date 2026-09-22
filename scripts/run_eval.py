"""Run one or more evaluations against one local vLLM server.

The configuration may describe the original single JSONL evaluation or an
ordered ``evaluations`` list mixing JSONL tasks (for example AIME) and
``eval_offline`` suites (for example GEM and ACEBench).

Run: python scripts/run_eval.py --config configs/evaluation.json
Validate without Docker/GPU access: add --dry-run.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_data import grade
from scripts.memory_guard import (
    DEFAULT_RESERVE_GIB,
    docker_memory_args,
    guarded_wait,
    task_resources,
    validate_limits,
    verify_container_limits,
)


DEFAULTS = dict(
    output_dir="outputs/evaluation",
    benchmark="boxed_integer",
    prompt_key="problem",
    answer_key="answer",
    id_key="id",
    prompt_suffix="\nPlease reason step by step, and put your final answer within \\boxed{}.",
    samples_per_problem=1,
    avg_at=[],
    max_problems=None,
    chat_template_kwargs={},
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0.0,
    presence_penalty=0.0,
    repetition_penalty=1.0,
    max_tokens=8192,
    seed=42,
    image="envduels-unified:cu124",
    gpu_ids=[4, 5, 6, 7],
    tensor_parallel=4,
    data_parallel=1,
    rootless_docker=False,
    dtype="auto",
    context_length=12288,
    gpu_memory_utilization=0.9,
    max_num_seqs=16,
    max_num_batched_tokens=4096,
    enforce_eager=True,
    limit_mm_per_prompt={},
    max_concurrent_problems=1,
    max_concurrent=16,
    request_timeout_seconds=1800,
    startup_timeout_seconds=1200,
    port=8000,
    reasoning_parser=None,
    tool_call_parser=None,
    served_model_name="evaluation-model",
    hf_cache_dir="../spade-workspace/hf",
    wandb_enabled=False,
    wandb_project="spade",
    wandb_group="evaluation",
    wandb_name="evaluation",
    lora=None,
    memory_limit_gib=96,
    host_memory_reserve_gib=DEFAULT_RESERVE_GIB,
)

JSONL_KEYS = {
    "benchmark", "prompt_key", "answer_key", "id_key", "prompt_suffix",
    "samples_per_problem", "avg_at", "max_problems", "chat_template_kwargs",
    "temperature", "top_p", "top_k", "min_p", "presence_penalty",
    "repetition_penalty", "max_tokens", "seed", "max_concurrent_problems",
    "request_timeout_seconds", "context_length",
}
SUITE_KEYS = {
    "config", "suites", "max_concurrent", "request_timeout_seconds",
    "context_length", "max_tokens",
}
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def resolve_path(value: str) -> Path:
    return (ROOT / Path(value).expanduser()).resolve()


def _nonempty_string(cfg: dict, key: str) -> None:
    if not isinstance(cfg.get(key), str) or not cfg[key].strip():
        raise ValueError(f"{key} must be a nonempty string")


def _positive_int(cfg: dict, key: str) -> None:
    if type(cfg.get(key)) is not int or cfg[key] < 1:
        raise ValueError(f"{key} must be a positive integer")


def _validate_jsonl(entry: dict) -> None:
    for key in ("data", "prompt_key", "answer_key"):
        _nonempty_string(entry, key)
    if entry["id_key"] is not None and (
        not isinstance(entry["id_key"], str) or not entry["id_key"]
    ):
        raise ValueError("id_key must be a nonempty string or null")
    if entry["benchmark"] not in ("boxed_integer", "boxed_exact_match"):
        raise ValueError("benchmark must be boxed_integer or boxed_exact_match")
    if not isinstance(entry["prompt_suffix"], str):
        raise ValueError("prompt_suffix must be a string")
    for key in (
        "samples_per_problem", "max_tokens", "max_concurrent_problems",
        "request_timeout_seconds", "context_length",
    ):
        _positive_int(entry, key)
    if entry["max_problems"] is not None and (
        type(entry["max_problems"]) is not int or entry["max_problems"] < 1
    ):
        raise ValueError("max_problems must be null or a positive integer")
    if entry["max_tokens"] >= entry["context_length"]:
        raise ValueError(
            f"Evaluation {entry['name']!r}: context_length must leave room "
            "for the prompt beyond max_tokens"
        )
    if type(entry["seed"]) is not int or not 0 <= entry["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if type(entry["top_k"]) is not int or (
        entry["top_k"] != -1 and entry["top_k"] < 1
    ):
        raise ValueError("top_k must be -1 or a positive integer")
    for key in (
        "temperature", "top_p", "min_p", "presence_penalty",
        "repetition_penalty",
    ):
        if type(entry[key]) not in (int, float) or not math.isfinite(entry[key]):
            raise ValueError(f"{key} must be a finite number")
    if not 0 <= entry["temperature"] <= 2 or not 0 < entry["top_p"] <= 1:
        raise ValueError("temperature must be in [0, 2]; top_p must be in (0, 1]")
    if entry["temperature"] == 0 and entry["samples_per_problem"] > 1:
        raise ValueError("Multiple samples require temperature > 0")
    if not -2 <= entry["presence_penalty"] <= 2 or not 0 <= entry["min_p"] <= 1:
        raise ValueError("Invalid presence_penalty or min_p")
    if entry["repetition_penalty"] <= 0:
        raise ValueError("repetition_penalty must be positive")
    if not isinstance(entry["avg_at"], list) or any(
        type(k) is not int or k < 1 or k > entry["samples_per_problem"]
        for k in entry["avg_at"]
    ):
        raise ValueError("avg_at must contain integers between 1 and samples_per_problem")
    entry["avg_at"] = sorted(set(entry["avg_at"] + [entry["samples_per_problem"]]))
    if not isinstance(entry["chat_template_kwargs"], dict):
        raise ValueError("chat_template_kwargs must be a mapping")


def _normalise_evaluations(
    cfg: dict, supplied: dict, forced_jsonl: dict, selected: list[str] | None,
) -> tuple[list[dict], bool]:
    raw = supplied.get("evaluations")
    suite_configs = supplied.get("suite_configs")
    if raw is not None and suite_configs is not None:
        raise ValueError("Use evaluations or legacy suite_configs, not both")

    legacy_single = raw is None and suite_configs is None
    if raw is None and suite_configs is not None:
        if not isinstance(suite_configs, list) or not suite_configs:
            raise ValueError("suite_configs must be a nonempty list")
        raw = []
        for group in suite_configs:
            if not isinstance(group, dict) or set(group) != {"config", "suites"}:
                raise ValueError("Each suite_configs entry requires only config and suites")
            suites = group.get("suites")
            name = "-".join(suites) if isinstance(suites, list) else "suite"
            raw.append({"name": name, "type": "suite", **group})
    elif raw is None:
        if "data" not in cfg:
            raise ValueError("Single evaluation config requires data")
        raw = [{"name": "evaluation", "type": "jsonl", "data": cfg["data"]}]

    if not isinstance(raw, list) or not raw:
        raise ValueError("evaluations must be a nonempty list")

    entries: list[dict] = []
    names: set[str] = set()
    for index, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise ValueError(f"evaluations[{index}] must be a mapping")
        kind = item.get("type") or ("suite" if "config" in item else "jsonl")
        name = item.get("name")
        if not isinstance(name, str) or not NAME_RE.fullmatch(name):
            raise ValueError(
                f"evaluations[{index}].name must match {NAME_RE.pattern}"
            )
        if name in names:
            raise ValueError(f"Duplicate evaluation name: {name}")
        names.add(name)

        if kind == "jsonl":
            unknown = set(item) - ({"name", "type", "data"} | JSONL_KEYS)
            if unknown:
                raise ValueError(
                    f"Unknown settings for JSONL evaluation {name!r}: {sorted(unknown)}"
                )
            entry = {key: deepcopy(cfg[key]) for key in JSONL_KEYS}
            entry.update(item)
            entry.update(forced_jsonl)
            entry.update(name=name, type="jsonl")
            _validate_jsonl(entry)
            entry["data"] = str(resolve_path(entry["data"]))
        elif kind == "suite":
            unknown = set(item) - ({"name", "type"} | SUITE_KEYS)
            if unknown:
                raise ValueError(
                    f"Unknown settings for suite evaluation {name!r}: {sorted(unknown)}"
                )
            entry = {
                "name": name,
                "type": "suite",
                "max_concurrent": cfg["max_concurrent"],
                "request_timeout_seconds": cfg["request_timeout_seconds"],
                "context_length": cfg["context_length"],
                "max_tokens": cfg["max_tokens"],
                **item,
            }
            _nonempty_string(entry, "config")
            _positive_int(entry, "max_concurrent")
            _positive_int(entry, "request_timeout_seconds")
            _positive_int(entry, "context_length")
            _positive_int(entry, "max_tokens")
            if entry["max_tokens"] >= entry["context_length"]:
                raise ValueError(
                    f"Suite evaluation {name!r}: context_length must leave room "
                    "for the prompt beyond max_tokens"
                )
            suites = entry.get("suites")
            if not isinstance(suites, list) or not suites or any(
                not isinstance(value, str) or not value.strip() for value in suites
            ):
                raise ValueError(f"Suite evaluation {name!r} requires nonempty suites")
            if len(set(suites)) != len(suites):
                raise ValueError(f"Suite evaluation {name!r} repeats a suite")
            config_path = resolve_path(entry["config"])
            try:
                config_path.relative_to(ROOT)
            except ValueError as exc:
                raise ValueError("Suite configs must be inside the repository") from exc
            entry["config"] = str(config_path)
        else:
            raise ValueError(f"Evaluation {name!r} has unsupported type {kind!r}")
        entries.append(entry)

    if selected:
        if len(set(selected)) != len(selected):
            raise ValueError("--evals must not repeat names")
        by_name = {entry["name"]: entry for entry in entries}
        missing = [name for name in selected if name not in by_name]
        if missing:
            raise ValueError(
                f"Unknown --evals names {missing}; available: {list(by_name)}"
            )
        entries = [by_name[name] for name in selected]
        legacy_single = False
    return entries, legacy_single


def load_config(
    path: Path | str,
    overrides: dict | None = None,
    selected: list[str] | None = None,
) -> dict:
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError("Evaluation configuration must be a .json file")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("evaluation"), dict):
        raise ValueError("Config must contain an 'evaluation' mapping")
    supplied = document["evaluation"]
    allowed = set(DEFAULTS) | {
        "checkpoint", "data", "evaluations", "suite_configs", "_legacy_single",
    }
    unknown = supplied.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown evaluation settings: {sorted(unknown)}")

    overrides = overrides or {}
    cfg = {**deepcopy(DEFAULTS), **supplied, **overrides}
    for key in (
        "checkpoint", "output_dir", "image", "dtype", "served_model_name",
        "hf_cache_dir", "wandb_project", "wandb_group", "wandb_name",
    ):
        _nonempty_string(cfg, key)
    if cfg["lora"] is not None and (
        not isinstance(cfg["lora"], str) or not cfg["lora"].strip()
    ):
        raise ValueError("lora must be a nonempty path or null")
    for key in (
        "tensor_parallel", "data_parallel", "context_length", "max_num_seqs",
        "max_num_batched_tokens", "startup_timeout_seconds", "port",
        "memory_limit_gib", "host_memory_reserve_gib", "max_concurrent",
    ):
        _positive_int(cfg, key)
    if cfg["port"] > 65535:
        raise ValueError("port must be at most 65535")
    ids = cfg["gpu_ids"]
    if not isinstance(ids, list) or not ids or any(
        type(value) is not int or value < 0 for value in ids
    ) or len(ids) != len(set(ids)):
        raise ValueError("gpu_ids must contain unique nonnegative integers")
    if cfg["tensor_parallel"] * cfg["data_parallel"] != len(ids):
        raise ValueError("tensor_parallel * data_parallel must equal the GPU count")
    if cfg["dtype"] not in ("auto", "float16", "bfloat16", "float32"):
        raise ValueError("Unsupported dtype")
    if any(type(cfg[key]) is not bool for key in (
        "enforce_eager", "rootless_docker", "wandb_enabled"
    )):
        raise ValueError("enforce_eager, rootless_docker and wandb_enabled must be booleans")
    if type(cfg["gpu_memory_utilization"]) not in (int, float) or not (
        math.isfinite(cfg["gpu_memory_utilization"])
        and 0 < cfg["gpu_memory_utilization"] < 1
    ):
        raise ValueError("gpu_memory_utilization must be between 0 and 1")
    if not isinstance(cfg["limit_mm_per_prompt"], dict) or any(
        not isinstance(key, str) or type(value) is not int or value < 0
        for key, value in cfg["limit_mm_per_prompt"].items()
    ):
        raise ValueError("limit_mm_per_prompt requires nonnegative integer limits")
    for key in ("reasoning_parser", "tool_call_parser"):
        if cfg[key] is not None and (
            not isinstance(cfg[key], str) or not cfg[key].strip()
        ):
            raise ValueError(f"{key} must be a nonempty string or null")
    validate_limits(cfg["memory_limit_gib"], cfg["host_memory_reserve_gib"])

    forced_jsonl = {key: value for key, value in overrides.items() if key in JSONL_KEYS}
    entries, legacy_single = _normalise_evaluations(
        cfg, supplied, forced_jsonl, selected
    )
    if "data" in overrides and not (
        len(entries) == 1 and entries[0]["type"] == "jsonl"
    ):
        raise ValueError("--data can only override a single JSONL evaluation")
    if "data" in overrides:
        entries[0]["data"] = str(resolve_path(overrides["data"]))
    cfg["evaluations"] = entries
    cfg["_legacy_single"] = legacy_single
    cfg["checkpoint"] = str(resolve_path(cfg["checkpoint"]))
    cfg["output_dir"] = str(resolve_path(cfg["output_dir"]))
    cfg["hf_cache_dir"] = str(resolve_path(cfg["hf_cache_dir"]))
    if cfg["lora"] is not None:
        cfg["lora"] = str(resolve_path(cfg["lora"]))
    if legacy_single:
        cfg["data"] = entries[0]["data"]
        for key in JSONL_KEYS:
            cfg[key] = entries[0][key]
    return cfg


def load_dataset(cfg: dict) -> tuple[list[dict], str, int]:
    raw = Path(cfg["data"]).read_bytes()
    rows, ids = [], set()
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            row_id = item[cfg["id_key"]] if cfg["id_key"] is not None else line_number
            prompt, answer = item[cfg["prompt_key"]], item[cfg["answer_key"]]
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid JSONL record/fields at line {line_number}: {exc}") from exc
        if type(row_id) not in (str, int) or not str(row_id).strip() or str(row_id) in ids:
            raise ValueError(f"Line {line_number}: ID must be a unique nonempty string or integer")
        ids.add(str(row_id))
        if type(answer) not in (str, int) or not str(answer).strip():
            raise ValueError(f"Line {line_number}: answer must be a nonempty string or integer")
        if cfg["benchmark"] == "boxed_integer" and not grade(
            "\\boxed{" + str(answer).strip() + "}", answer, "boxed_integer"
        )[2]:
            raise ValueError(f"Line {line_number}: boxed_integer requires an integer label")
        if isinstance(prompt, str) and prompt.strip():
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and prompt:
            if any(
                not isinstance(message, dict)
                or message.get("role") not in ("system", "user", "assistant")
                or not isinstance(message.get("content"), str)
                or not message["content"].strip()
                for message in prompt
            ):
                raise ValueError(f"Line {line_number}: invalid text chat messages")
            messages = [
                {"role": message["role"], "content": message["content"]}
                for message in prompt
            ]
        else:
            raise ValueError(
                f"Line {line_number}: prompt must be text or a text chat message list"
            )
        if messages[-1]["role"] != "user":
            raise ValueError(f"Line {line_number}: prompt must end with a user message")
        suffix = cfg["prompt_suffix"]
        if suffix and not messages[-1]["content"].rstrip().endswith(suffix.strip()):
            messages[-1]["content"] += suffix
        rows.append({"id": row_id, "messages": messages, "answer": answer})
    if not rows:
        raise ValueError("Evaluation dataset is empty")
    total = len(rows)
    return rows[: cfg["max_problems"]], hashlib.sha256(raw).hexdigest(), total


async def score_dataset(
    client, rows: list[dict], cfg: dict, output, checkpoint_path: Path | None = None,
) -> dict:
    """Compute Avg@k and observed pass@k from one ordered sample set."""
    semaphore = asyncio.Semaphore(cfg["max_concurrent_problems"])
    k = cfg["samples_per_problem"]
    completed = {}

    async def problem(index, row):
        async with semaphore:
            completions = await client.chat(
                messages=row["messages"], n=k, temperature=cfg["temperature"],
                top_p=cfg["top_p"], max_tokens=cfg["max_tokens"],
                extra_body={
                    "top_k": cfg["top_k"], "min_p": cfg["min_p"],
                    "presence_penalty": cfg["presence_penalty"],
                    "repetition_penalty": cfg["repetition_penalty"],
                    "seed": (cfg["seed"] + index + 1) % 2**32,
                    "chat_template_kwargs": cfg["chat_template_kwargs"],
                },
            )
            if len(completions) != k:
                raise RuntimeError(
                    f"Problem {row['id']}: expected {k} samples, got {len(completions)}"
                )
            correct_flags = []
            invalid_count = length_count = 0
            for sample, completion in enumerate(completions):
                predicted, correct, valid = grade(
                    completion.text, row["answer"], cfg["benchmark"]
                )
                output.write(json.dumps({
                    "id": row["id"], "sample": sample, "answer": row["answer"],
                    "predicted": predicted, "correct": correct,
                    "valid_answer": valid, "response": completion.text,
                    "finish_reason": completion.finish_reason, "raw": completion.raw,
                }, ensure_ascii=False) + "\n")
                correct_flags.append(bool(correct))
                invalid_count += not valid
                length_count += completion.finish_reason == "length"
            output.flush()
            completed[str(row["id"])] = {
                "samples": k, "correct": sum(correct_flags),
                "invalid": invalid_count, "length_stops": length_count,
            }
            if checkpoint_path is not None:
                write_json(checkpoint_path, {
                    "status": "running", "completed_problems": len(completed),
                    "total_problems": len(rows),
                    "completed_samples": len(completed) * k,
                    "total_samples": len(rows) * k, "problems": completed,
                })
            print(f"Problem {row['id']}: {sum(correct_flags)}/{k} correct", flush=True)
            return correct_flags, invalid_count, length_count

    tasks = [asyncio.create_task(problem(i, row)) for i, row in enumerate(rows)]
    try:
        counts = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    n = len(rows) * k
    metrics = {}
    for report_k in cfg["avg_at"]:
        correct = sum(sum(count[0][:report_k]) for count in counts)
        passed = sum(any(count[0][:report_k]) for count in counts)
        metrics[f"avg_at_{report_k}"] = correct / (len(rows) * report_k)
        metrics[f"pass_at_{report_k}"] = passed / len(rows)
    return {
        "n_problems": len(rows), "n_samples_per_problem": k,
        "n_completions": n, "sample_accuracy": metrics[f"avg_at_{k}"],
        "fraction_problems_any_correct": metrics[f"pass_at_{k}"],
        "invalid_answer_fraction": sum(count[1] for count in counts) / n,
        "length_stop_fraction": sum(count[2] for count in counts) / n,
        **metrics,
    }


def request_model_name(cfg: dict) -> str:
    return "evaluation-model" if cfg["lora"] is not None else cfg["served_model_name"]


def build_server_command(cfg: dict) -> list[str]:
    base_name = "evaluation-base" if cfg["lora"] is not None else cfg["served_model_name"]
    command = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", cfg["checkpoint"], "--served-model-name", base_name,
        "--host", "127.0.0.1", "--port", str(cfg["port"]),
        "--tensor-parallel-size", str(cfg["tensor_parallel"]),
        "--data-parallel-size", str(cfg["data_parallel"]),
        "--dtype", cfg["dtype"], "--max-model-len", str(cfg["context_length"]),
        "--gpu-memory-utilization", str(cfg["gpu_memory_utilization"]),
        "--max-num-seqs", str(cfg["max_num_seqs"]),
        "--max-num-batched-tokens", str(cfg["max_num_batched_tokens"]),
        "--seed", str(cfg["seed"]), "--trust-remote-code",
    ]
    if cfg["enforce_eager"]:
        command.append("--enforce-eager")
    if cfg["limit_mm_per_prompt"]:
        command += ["--limit-mm-per-prompt", json.dumps(cfg["limit_mm_per_prompt"])]
    if cfg["reasoning_parser"]:
        command += ["--reasoning-parser", cfg["reasoning_parser"]]
    if cfg["tool_call_parser"]:
        command += [
            "--enable-auto-tool-choice", "--tool-call-parser", cfg["tool_call_parser"],
        ]
    if cfg["lora"] is not None:
        adapter_config = json.loads(
            (Path(cfg["lora"]) / "adapter_config.json").read_text()
        )
        rank = adapter_config.get("r")
        if type(rank) is not int or rank < 1:
            raise ValueError("LoRA adapter_config.json requires a positive integer r")
        command += [
            "--enable-lora", "--max-lora-rank", str(rank),
            "--lora-modules", f"evaluation-model={cfg['lora']}",
        ]
    return command


@contextmanager
def serve(cfg: dict, out: Path, label: str = "server"):
    command = build_server_command(cfg)
    write_json(out / f"{label}_command.json", command)
    log_path = out / f"{label}.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            deadline = time.monotonic() + cfg["startup_timeout_seconds"]
            health = f"http://127.0.0.1:{cfg['port']}/health"
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"vLLM exited ({proc.returncode}); see {log_path.name}"
                    )
                try:
                    with urlopen(health, timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"vLLM startup timed out; see {log_path.name}"
                    )
                time.sleep(2)
            yield f"http://127.0.0.1:{cfg['port']}"
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


def _evaluate_jsonl(cfg: dict, entry: dict, out: Path, base_url: str) -> dict:
    from eval_offline.client import OfflineClient

    rows, digest, total = load_dataset(entry)
    checkpoint_path = out / "evaluation_checkpoint.json"
    write_json(checkpoint_path, {
        "status": "running", "completed_problems": 0,
        "total_problems": len(rows), "completed_samples": 0,
        "total_samples": len(rows) * entry["samples_per_problem"], "problems": {},
    })

    async def run() -> dict:
        client = OfflineClient(
            base_url, request_model_name(cfg), model_path=cfg["checkpoint"],
            max_concurrent=entry["max_concurrent_problems"], max_retries=3,
            request_timeout_seconds=entry["request_timeout_seconds"],
        )
        try:
            with (out / "responses.jsonl").open("x", encoding="utf-8") as stream:
                return await score_dataset(client, rows, entry, stream, checkpoint_path)
        finally:
            await client.openai_client.close()

    metrics = asyncio.run(run())
    metrics.update(
        benchmark=entry["benchmark"], dataset_total_problems=total,
        is_subset=len(rows) < total, data_sha256=digest,
    )
    write_json(out / "runtime.json", {
        "data_sha256": digest, "dataset_total_problems": total,
        "selected_ids": [row["id"] for row in rows],
    })
    write_json(out / "scores.json", metrics)
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint.update(status="completed", scores=metrics)
    write_json(checkpoint_path, checkpoint)
    return metrics


def _evaluate_suite(cfg: dict, entry: dict, out: Path, base_url: str) -> dict:
    import yaml

    source = Path(entry["config"])
    document = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    available = document.get("suites") or {}
    selected = entry["suites"]
    missing = [suite for suite in selected if suite not in available]
    if missing:
        raise ValueError(f"{source} has no suites: {missing}")
    # An ordered run must stay ordered even when a source YAML marks a suite
    # as background/concurrent. Concurrency inside a suite is unchanged.
    ordered_suites = {}
    for suite in selected:
        suite_config = deepcopy(available[suite])
        if isinstance(suite_config, dict):
            suite_config["concurrent"] = False
            # A single top-level cap applies to every evaluation type.  GEM
            # expands this override to every task after loading its task file;
            # ACEBench consumes it directly.
            suite_config["max_tokens"] = entry["max_tokens"]
        ordered_suites[suite] = suite_config
    combined_path = out / "selected_suites.yaml"
    combined_path.write_text(
        yaml.safe_dump({"suites": ordered_suites}, sort_keys=False),
        encoding="utf-8",
    )

    child_env = os.environ.copy()
    child_env["SPADE_EVAL_REQUEST_TIMEOUT_SECONDS"] = str(
        entry["request_timeout_seconds"]
    )
    if "acebench" in selected:
        ace_runtime = out / "acebench-runtime"
        shutil.copytree("/opt/benchmarks/ACEBench", ace_runtime)
        child_env["ACEBENCH_DIR"] = str(ace_runtime)
        child_env["ACEBENCH_PYTHON"] = "/opt/benchmarks/acebench-venv/bin/python"

    command = [
        sys.executable, "-m", "eval_offline.run_offline_eval",
        "--base-url", base_url, "--served-model-name", request_model_name(cfg),
        "--ckpt", cfg["checkpoint"], "--config", str(combined_path),
        "--output-dir", str(out), "--suites", ",".join(selected),
        "--tp", str(cfg["tensor_parallel"]), "--dp", str(cfg["data_parallel"]),
        "--max-concurrent", str(entry["max_concurrent"]), "--no-wandb",
    ]
    write_json(out / "driver_command.json", command)
    code = subprocess.run(command, env=child_env).returncode
    if code:
        raise RuntimeError(
            f"Evaluation {entry['name']!r} failed with exit code {code}; see {out}"
        )
    results_path = out / "results.json"
    if not results_path.is_file():
        raise RuntimeError(f"Evaluation {entry['name']!r} produced no results.json")
    return json.loads(results_path.read_text(encoding="utf-8"))


def _wandb_start(cfg: dict):
    if not cfg["wandb_enabled"]:
        return None
    try:
        import wandb
        return wandb.init(
            project=cfg["wandb_project"],
            entity=os.environ.get("WANDB_ENTITY") or None,
            group=cfg["wandb_group"], name=cfg["wandb_name"],
            config={key: value for key, value in cfg.items() if not key.startswith("_")},
            reinit=True,
        )
    except Exception as exc:
        print(f"W&B init failed; continuing without it: {exc}", flush=True)
        return None


def _numeric_metrics(prefix: str, value: object) -> dict[str, int | float]:
    if type(value) in (int, float) and math.isfinite(value):
        return {prefix: value}
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(_numeric_metrics(f"{prefix}/{key}", child))
        return result
    return {}


def _server_groups(entries: list[dict]) -> list[list[tuple[int, dict]]]:
    """Group adjacent evaluations that can reuse the same vLLM context."""
    groups: list[list[tuple[int, dict]]] = []
    for index, entry in enumerate(entries, 1):
        if not groups or groups[-1][0][1]["context_length"] != entry["context_length"]:
            groups.append([])
        groups[-1].append((index, entry))
    return groups


def evaluate(cfg: dict, out: Path) -> None:
    import torch
    import transformers
    import vllm

    write_json(out / "runtime.json", {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": transformers.__version__, "vllm": vllm.__version__,
    })
    sequence = []
    started = time.time()
    wandb_run = _wandb_start(cfg)
    current_name = None
    try:
        groups = _server_groups(cfg["evaluations"])
        for group_index, group in enumerate(groups, 1):
            server_cfg = {**cfg, "context_length": group[0][1]["context_length"]}
            label = "server" if len(groups) == 1 else (
                f"server-{group_index:02d}-ctx{server_cfg['context_length']}"
            )
            print(
                f"Starting vLLM segment {group_index}/{len(groups)} with "
                f"max-model-len={server_cfg['context_length']}", flush=True,
            )
            with serve(server_cfg, out, label) as base_url:
                for index, entry in group:
                    name = entry["name"]
                    current_name = name
                    step_out = out if cfg["_legacy_single"] else out / f"{index:02d}-{name}"
                    step_out.mkdir(parents=True, exist_ok=cfg["_legacy_single"])
                    write_json(out / "sequence_status.json", {
                        "status": "running", "current_index": index,
                        "current_evaluation": name,
                        "context_length": entry["context_length"],
                        "max_tokens": entry["max_tokens"],
                        "completed": [item["name"] for item in sequence],
                    })
                    print(
                        f"[{index}/{len(cfg['evaluations'])}] Starting {name} "
                        f"({entry['type']}, max_tokens={entry['max_tokens']})",
                        flush=True,
                    )
                    step_started = time.time()
                    if entry["type"] == "jsonl":
                        result = _evaluate_jsonl(cfg, entry, step_out, base_url)
                    else:
                        result = _evaluate_suite(cfg, entry, step_out, base_url)
                    elapsed = time.time() - step_started
                    sequence.append({
                        "name": name, "type": entry["type"],
                        "context_length": entry["context_length"],
                        "max_tokens": entry["max_tokens"],
                        "elapsed_sec": elapsed, "output_dir": str(step_out),
                        "result": result,
                    })
                    write_json(out / "sequence_results.json", {
                        "elapsed_sec": time.time() - started, "evaluations": sequence,
                    })
                    if wandb_run:
                        try:
                            wandb_run.log(
                                _numeric_metrics(f"eval/{name}", result)
                                | {f"eval/{name}/elapsed_sec": elapsed}
                            )
                        except Exception as exc:
                            print(f"W&B log failed for {name}: {exc}", flush=True)
                    print(
                        f"[{index}/{len(cfg['evaluations'])}] Finished {name}",
                        flush=True,
                    )
                    current_name = None
    except BaseException as exc:
        write_json(out / "sequence_status.json", {
            "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            "current_evaluation": current_name,
            "completed": [item["name"] for item in sequence], "error": str(exc),
        })
        raise
    finally:
        if wandb_run:
            try:
                wandb_run.summary["total_elapsed_sec"] = time.time() - started
                wandb_run.summary["num_evaluations_completed"] = len(sequence)
                wandb_run.finish()
            except Exception as exc:
                print(f"W&B finalize failed: {exc}", flush=True)
    write_json(out / "sequence_status.json", {
        "status": "completed", "completed": [item["name"] for item in sequence],
    })


def _read_secret(path: Path) -> str | None:
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def child_environment() -> tuple[dict[str, str], list[str]]:
    """Load local credentials without putting secret values in the command."""
    env = os.environ.copy()
    sources = {
        "HF_TOKEN": ROOT / "api-keys/hf.txt",
        "OPENROUTER_API_KEY": ROOT / "api-keys/openrouter.txt",
        "WANDB_ENTITY": ROOT / "api-keys/w&b-entity.txt",
        "WANDB_API_KEY": ROOT / "api-keys/w&b.txt",
    }
    for variable, source in sources.items():
        if not env.get(variable):
            value = _read_secret(source)
            if value:
                env[variable] = value
    forwarded = [name for name in (*sources, "OPENAI_API_KEY") if env.get(name)]
    return env, forwarded


def checkpoint_mount(checkpoint: Path) -> tuple[Path, str, str]:
    """Keep Hugging Face snapshot symlinks valid inside the container."""
    if checkpoint.parent.name == "snapshots":
        repository = checkpoint.parent.parent
        if (repository / "blobs").is_dir():
            hub = repository.parent
            container_root = "/model-hub"
            return (
                hub, container_root,
                f"{container_root}/{repository.name}/snapshots/{checkpoint.name}",
            )
    return checkpoint, "/model", "/model"


def _dataset_target(cfg: dict, index: int, entry: dict) -> str:
    if cfg["_legacy_single"]:
        return "/dataset/test.jsonl"
    return f"/datasets/{index:02d}-{entry['name']}.jsonl"


def docker_command(
    cfg: dict, out: Path, name: str, forwarded: list[str] | None = None,
) -> list[str]:
    has_suites = any(entry["type"] == "suite" for entry in cfg["evaluations"])
    needs_network = has_suites or cfg["wandb_enabled"]
    checkpoint_source, checkpoint_target, _ = checkpoint_mount(Path(cfg["checkpoint"]))
    mounts = [
        (ROOT, "/workspace/spade", True),
        (checkpoint_source, checkpoint_target, True),
        (out, "/results", False),
    ]
    for index, entry in enumerate(cfg["evaluations"], 1):
        if entry["type"] == "jsonl":
            mounts.append((Path(entry["data"]), _dataset_target(cfg, index, entry), True))
    if has_suites:
        mounts.append((Path(cfg["hf_cache_dir"]), "/cache/hf", False))
    if needs_network:
        netrc = Path.home() / ".netrc"
        if netrc.is_file():
            mounts.append((netrc, "/tmp/home/.netrc", True))
    if cfg["lora"] is not None:
        mounts.append((Path(cfg["lora"]), "/lora", True))

    command = [
        "docker", "run", "--rm", "--init", "--name", name,
        "--gpus", '"device=' + ",".join(map(str, cfg["gpu_ids"])) + '"',
    ]
    if not needs_network:
        command += ["--network", "none"]
    if not cfg["rootless_docker"] and not has_suites:
        command += ["--user", f"{os.getuid()}:{os.getgid()}"]
    command += ["--shm-size", "16g", "--label", "spade.gpu-concurrent=true"]
    command += docker_memory_args(cfg["memory_limit_gib"])
    for source, target, readonly in mounts:
        if "," in str(source):
            raise ValueError("Docker mount paths must not contain commas")
        command += [
            "--mount",
            f"type=bind,src={source},dst={target}" + (",readonly" if readonly else ""),
        ]
    command += ["--workdir", "/workspace/spade"]
    settings = {
        "HOME": "/tmp/home", "USER": "envduels", "PYTHONUNBUFFERED": "1",
        "XDG_CACHE_HOME": "/tmp/cache", "XDG_CONFIG_HOME": "/tmp/config",
        "VLLM_CACHE_ROOT": "/tmp/vllm", "VLLM_CONFIG_ROOT": "/tmp/vllm-config",
        "VLLM_NO_USAGE_STATS": "1", "FLASHINFER_WORKSPACE_BASE": "/tmp/flashinfer",
        "CUDA_CACHE_PATH": "/tmp/cuda", "GLOO_SOCKET_IFNAME": "lo",
        "NCCL_SOCKET_IFNAME": "lo", "TRITON_CACHE_DIR": "/tmp/triton",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor",
    }
    if has_suites:
        settings.update(HF_HOME="/cache/hf", HF_DATASETS_CACHE="/cache/hf/datasets")
    else:
        settings.update(HF_HOME="/tmp/hf", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for key, value in settings.items():
        command += ["--env", f"{key}={value}"]
    for variable in forwarded or []:
        command += ["--env", variable]
    return command + [
        "--entrypoint", "python3", cfg["image"], "scripts/run_eval.py",
        "--inside-container", "--config", "/results/container_config.json",
    ]


def _container_config(cfg: dict) -> dict:
    result = deepcopy(cfg)
    _, _, result["checkpoint"] = checkpoint_mount(Path(cfg["checkpoint"]))
    result["output_dir"] = "/results"
    result["hf_cache_dir"] = "/cache/hf"
    result["lora"] = "/lora" if cfg["lora"] is not None else None
    for index, entry in enumerate(result["evaluations"], 1):
        if entry["type"] == "jsonl":
            entry["data"] = _dataset_target(cfg, index, entry)
        else:
            relative = Path(entry["config"]).relative_to(ROOT)
            entry["config"] = "/workspace/spade/" + str(relative)
    if cfg["_legacy_single"]:
        result["data"] = result["evaluations"][0]["data"]
    return result


def _validate_host_inputs(cfg: dict) -> list[dict]:
    checkpoint = Path(cfg["checkpoint"])
    if not (checkpoint / "config.json").is_file() or not (
        any(checkpoint.glob("*.safetensors"))
        or any(checkpoint.glob("pytorch_model*.bin"))
    ):
        raise ValueError(
            "checkpoint must be a complete local Hugging Face model directory, "
            "not an adapter-only or Megatron checkpoint"
        )
    if cfg["lora"] is not None:
        lora = Path(cfg["lora"])
        if not (lora / "adapter_config.json").is_file() or not (
            (lora / "adapter_model.safetensors").is_file()
            or (lora / "adapter_model.bin").is_file()
        ):
            raise ValueError(
                "lora must be a PEFT adapter directory with adapter config and weights"
            )
    if any(entry["type"] == "suite" for entry in cfg["evaluations"]):
        cache = Path(cfg["hf_cache_dir"])
        if not cache.is_dir():
            raise FileNotFoundError(f"Hugging Face cache directory not found: {cache}")
    plan = []
    for entry in cfg["evaluations"]:
        if entry["type"] == "jsonl":
            if not Path(entry["data"]).is_file():
                raise FileNotFoundError(
                    f"Evaluation {entry['name']!r} data not found: {entry['data']}"
                )
            rows, digest, total = load_dataset(entry)
            plan.append({
                "name": entry["name"], "type": "jsonl",
                "context_length": entry["context_length"],
                "max_tokens": entry["max_tokens"],
                "n_problems": len(rows), "dataset_total_problems": total,
                "n_completions": len(rows) * entry["samples_per_problem"],
                "data_sha256": digest, "selected_ids": [row["id"] for row in rows],
            })
        else:
            if not Path(entry["config"]).is_file():
                raise FileNotFoundError(
                    f"Evaluation {entry['name']!r} config not found: {entry['config']}"
                )
            plan.append({
                "name": entry["name"], "type": "suite",
                "context_length": entry["context_length"],
                "max_tokens": entry["max_tokens"],
                "suites": entry["suites"], "config": entry["config"],
            })
    return plan


def run_evaluation(
    cfg: dict, out: Path, name: str, command: list[str], report: dict,
    checkpoint: Path, child_env: dict[str, str],
) -> int:
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", cfg["image"], "--format", "{{.Id}}"],
        text=True,
    ).strip()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "resolved_config.json", report)
    write_json(out / "config.json", {"evaluation": cfg})
    write_json(out / "container_config.json", {"evaluation": _container_config(cfg)})
    write_json(out / "launch.json", {
        "command": command, "image_id": image_id, "container": name,
    })
    manifest = checkpoint / "download_manifest.json"
    if manifest.is_file():
        write_json(
            out / "model_download_manifest.json",
            json.loads(manifest.read_text(encoding="utf-8")),
        )
    write_json(out / "status.json", {"status": "running"})
    print(f"Running; follow progress with: tail -f {out / 'console.log'}", flush=True)
    try:
        with (out / "console.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env=child_env,
            )
            code = guarded_wait(process, name, cfg["host_memory_reserve_gib"])
    except BaseException as exc:
        write_json(out / "status.json", {
            "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            "error": str(exc),
        })
        raise
    write_json(out / "status.json", {
        "status": "completed" if code == 0 else "failed", "exit_code": code,
    })
    print(f"Exit {code}; results/logs: {out}", flush=True)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/evaluation.json")
    parser.add_argument("--checkpoint", help="Override evaluation.checkpoint")
    parser.add_argument("--data", help="Override data for a single JSONL evaluation")
    parser.add_argument("--output-dir", help="Override output parent directory")
    parser.add_argument("--image", help="Override evaluation.image")
    parser.add_argument("--lora", help="Optional Hugging Face/PEFT LoRA adapter directory")
    parser.add_argument("--max-problems", type=int, help="Limit every selected JSONL evaluation")
    parser.add_argument("--samples-per-problem", type=int, help="Override k for every JSONL evaluation")
    parser.add_argument("--gpu-ids", type=int, nargs="+", help="Physical GPU indices")
    parser.add_argument("--memory-limit-gib", type=int, help="Container host-RAM budget")
    parser.add_argument(
        "--evals", nargs="+", metavar="NAME",
        help="Run named evaluations in exactly this order, e.g. aime25 aime26 gem",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inside-container", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.inside_container:
        cfg = json.loads(args.config.read_text(encoding="utf-8"))["evaluation"]
        verify_container_limits(cfg["memory_limit_gib"])
        evaluate(cfg, Path(cfg["output_dir"]))
        return 0

    overrides = {
        key: getattr(args, key)
        for key in (
            "checkpoint", "data", "output_dir", "max_problems",
            "samples_per_problem", "image", "lora", "gpu_ids", "memory_limit_gib",
        )
        if getattr(args, key) is not None
    }
    if args.samples_per_problem is not None:
        overrides["avg_at"] = []
    if args.gpu_ids is not None:
        overrides["tensor_parallel"] = len(args.gpu_ids)
        overrides["data_parallel"] = 1
    cfg = load_config(args.config.resolve(), overrides, args.evals)
    steps = _validate_host_inputs(cfg)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = Path(cfg["output_dir"]) / stamp
    name = "spade-eval-" + stamp.lower()
    child_env, forwarded = child_environment()
    command = docker_command(cfg, out, name, forwarded)
    checkpoint = Path(cfg["checkpoint"])
    report = {
        "config": cfg, "evaluations": steps, "output": str(out),
        "container": name, "forwarded_credentials": forwarded, "command": command,
        "model_config_sha256": hashlib.sha256(
            (checkpoint / "config.json").read_bytes()
        ).hexdigest(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("Dry run only; no container started or GPU allocated.")
        return 0
    with task_resources(cfg, ROOT):
        code = run_evaluation(
            cfg, out, name, command, report, checkpoint, child_env
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
