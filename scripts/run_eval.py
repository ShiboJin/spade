"""Config-driven local HF checkpoint evaluation using an isolated vLLM container.

Run: python scripts/run_eval.py --config configs/evaluation.json
Validate without Docker/GPU access: add --dry-run.
"""
import argparse
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_data import grade
from scripts.memory_guard import (
    DEFAULT_RESERVE_GIB, docker_memory_args, guarded_wait, launch_lock, preflight,
    validate_limits, verify_container_limits,
)

DEFAULTS = dict(
    output_dir="outputs/evaluation", benchmark="boxed_integer", prompt_key="problem",
    answer_key="answer", id_key="id",
    prompt_suffix="\nPlease reason step by step, and put your final answer within \\boxed{}.",
    samples_per_problem=1, max_problems=None, chat_template_kwargs={}, temperature=0.7,
    top_p=0.8, top_k=20, presence_penalty=0.0, max_tokens=8192, seed=42,
    image="envduels-unified:cu124", gpu_ids=[4, 5, 6, 7], tensor_parallel=4,
    dtype="auto", context_length=12288, gpu_memory_utilization=0.9,
    max_num_seqs=16, max_num_batched_tokens=4096, enforce_eager=True,
    limit_mm_per_prompt={}, max_concurrent_problems=1,
    request_timeout_seconds=1800, startup_timeout_seconds=1200, lora=None,
    memory_limit_gib=96, host_memory_reserve_gib=DEFAULT_RESERVE_GIB,
)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_path(value):
    return (ROOT / Path(value).expanduser()).resolve()


def load_config(path, overrides=None):
    if path.suffix.lower() != ".json":
        raise ValueError("Evaluation configuration must be a .json file")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("evaluation"), dict):
        raise ValueError("Config must contain an 'evaluation' mapping")
    supplied = document["evaluation"]
    unknown = supplied.keys() - (DEFAULTS.keys() | {"checkpoint", "data"})
    if unknown:
        raise ValueError(f"Unknown evaluation settings: {sorted(unknown)}")
    cfg = {**DEFAULTS, **supplied}
    cfg.update(overrides or {})
    validate_limits(cfg["memory_limit_gib"], cfg["host_memory_reserve_gib"])
    for key in ("checkpoint", "data", "output_dir", "image", "prompt_key", "answer_key"):
        if not isinstance(cfg.get(key), str) or not cfg[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if cfg["id_key"] is not None and (not isinstance(cfg["id_key"], str) or not cfg["id_key"]):
        raise ValueError("id_key must be a nonempty string or null")
    if cfg["lora"] is not None and (not isinstance(cfg["lora"], str) or not cfg["lora"].strip()):
        raise ValueError("lora must be a nonempty path or null")
    if cfg["benchmark"] not in ("boxed_integer", "boxed_exact_match"):
        raise ValueError("benchmark must be boxed_integer or boxed_exact_match")
    if not isinstance(cfg["prompt_suffix"], str):
        raise ValueError("prompt_suffix must be a string")
    ids = cfg["gpu_ids"]
    if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("gpu_ids must contain unique nonnegative integers")
    for key in ("tensor_parallel", "samples_per_problem", "max_tokens", "context_length",
                "max_num_seqs", "max_num_batched_tokens", "max_concurrent_problems",
                "request_timeout_seconds", "startup_timeout_seconds"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["tensor_parallel"] != len(ids):
        raise ValueError("tensor_parallel must equal the number of selected GPUs")
    if cfg["max_problems"] is not None and (type(cfg["max_problems"]) is not int or cfg["max_problems"] < 1):
        raise ValueError("max_problems must be null or a positive integer")
    if cfg["max_tokens"] >= cfg["context_length"]:
        raise ValueError("context_length must leave room for the prompt beyond max_tokens")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if type(cfg["top_k"]) is not int or (cfg["top_k"] != -1 and cfg["top_k"] < 1):
        raise ValueError("top_k must be -1 or a positive integer")
    for key in ("temperature", "top_p", "presence_penalty", "gpu_memory_utilization"):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]):
            raise ValueError(f"{key} must be a finite number")
    if not 0 <= cfg["temperature"] <= 2 or not 0 < cfg["top_p"] <= 1:
        raise ValueError("temperature must be in [0, 2]; top_p must be in (0, 1]")
    if cfg["temperature"] == 0 and cfg["samples_per_problem"] > 1:
        raise ValueError("Multiple samples require temperature > 0")
    if not -2 <= cfg["presence_penalty"] <= 2 or not 0 < cfg["gpu_memory_utilization"] < 1:
        raise ValueError("Invalid presence_penalty or gpu_memory_utilization")
    if type(cfg["enforce_eager"]) is not bool:
        raise ValueError("enforce_eager must be a boolean")
    if cfg["dtype"] not in ("auto", "float16", "bfloat16", "float32"):
        raise ValueError("Unsupported dtype")
    if not isinstance(cfg["chat_template_kwargs"], dict) or not isinstance(cfg["limit_mm_per_prompt"], dict):
        raise ValueError("chat_template_kwargs and limit_mm_per_prompt must be mappings")
    if any(not isinstance(k, str) or type(v) is not int or v < 0 for k, v in cfg["limit_mm_per_prompt"].items()):
        raise ValueError("limit_mm_per_prompt requires nonnegative integer limits")
    for key in ("checkpoint", "data", "output_dir"):
        cfg[key] = str(resolve_path(cfg[key]))
    if cfg["lora"] is not None:
        cfg["lora"] = str(resolve_path(cfg["lora"]))
    return cfg


def load_dataset(cfg):
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
        if cfg["benchmark"] == "boxed_integer":
            if not grade("\\boxed{" + str(answer).strip() + "}", answer, "boxed_integer")[2]:
                raise ValueError(f"Line {line_number}: boxed_integer requires an integer label")
        if isinstance(prompt, str) and prompt.strip():
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and prompt:
            if any(not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant")
                   or not isinstance(m.get("content"), str) or not m["content"].strip() for m in prompt):
                raise ValueError(f"Line {line_number}: invalid text chat messages")
            messages = [{"role": m["role"], "content": m["content"]} for m in prompt]
        else:
            raise ValueError(f"Line {line_number}: prompt must be text or a text chat message list")
        if messages[-1]["role"] != "user":
            raise ValueError(f"Line {line_number}: prompt must end with a user message")
        suffix = cfg["prompt_suffix"]
        if suffix and not messages[-1]["content"].rstrip().endswith(suffix.strip()):
            messages[-1]["content"] += suffix
        rows.append(dict(id=row_id, messages=messages, answer=answer))
    if not rows:
        raise ValueError("Evaluation dataset is empty")
    total = len(rows)
    return rows[:cfg["max_problems"]], hashlib.sha256(raw).hexdigest(), total


async def score_dataset(client, rows, cfg, output):
    """Compute observed pass@k using exactly k samples for each selected problem."""
    semaphore = asyncio.Semaphore(cfg["max_concurrent_problems"])
    k = cfg["samples_per_problem"]

    async def problem(index, row):
        async with semaphore:
            completions = await client.chat(
                messages=row["messages"], n=k, temperature=cfg["temperature"],
                top_p=cfg["top_p"], max_tokens=cfg["max_tokens"],
                extra_body=dict(top_k=cfg["top_k"], presence_penalty=cfg["presence_penalty"],
                                seed=(cfg["seed"] + index + 1) % 2**32,
                                chat_template_kwargs=cfg["chat_template_kwargs"]))
            if len(completions) != k:
                raise RuntimeError(f"Problem {row['id']}: expected {k} samples, got {len(completions)}")
            correct_count = invalid_count = length_count = 0
            for sample, completion in enumerate(completions):
                predicted, correct, valid = grade(completion.text, row["answer"], cfg["benchmark"])
                record = dict(id=row["id"], sample=sample, answer=row["answer"], predicted=predicted,
                              correct=correct, valid_answer=valid, response=completion.text,
                              finish_reason=completion.finish_reason, raw=completion.raw)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                correct_count += correct
                invalid_count += not valid
                length_count += completion.finish_reason == "length"
            output.flush()
            print(f"Problem {row['id']}: {correct_count}/{k} correct", flush=True)
            return correct_count, invalid_count, length_count

    tasks = [asyncio.create_task(problem(i, row)) for i, row in enumerate(rows)]
    try:
        counts = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    n = len(rows) * k
    avg = sum(c[0] for c in counts) / n
    passed = sum(c[0] > 0 for c in counts) / len(rows)
    return dict(n_problems=len(rows), n_samples_per_problem=k, n_completions=n,
                sample_accuracy=avg, fraction_problems_any_correct=passed,
                invalid_answer_fraction=sum(c[1] for c in counts) / n,
                length_stop_fraction=sum(c[2] for c in counts) / n,
                **{f"avg_at_{k}": avg, f"pass_at_{k}": passed})


def build_server_command(cfg):
    base_model_name = "evaluation-base" if cfg["lora"] is not None else "evaluation-model"
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", cfg["checkpoint"],
               "--served-model-name", base_model_name, "--host", "127.0.0.1", "--port", "8000"]
    for flag, key in (("tensor-parallel-size", "tensor_parallel"), ("dtype", "dtype"),
                      ("max-model-len", "context_length"), ("gpu-memory-utilization", "gpu_memory_utilization"),
                      ("max-num-seqs", "max_num_seqs"), ("max-num-batched-tokens", "max_num_batched_tokens"),
                      ("seed", "seed")):
        command += ["--" + flag, str(cfg[key])]
    if cfg["enforce_eager"]:
        command.append("--enforce-eager")
    if cfg["limit_mm_per_prompt"]:
        command += ["--limit-mm-per-prompt", json.dumps(cfg["limit_mm_per_prompt"])]
    if cfg["lora"] is not None:
        adapter_config = json.loads((Path(cfg["lora"]) / "adapter_config.json").read_text())
        rank = adapter_config.get("r")
        if type(rank) is not int or rank < 1:
            raise ValueError("LoRA adapter_config.json requires a positive integer r")
        command += ["--enable-lora", "--max-lora-rank", str(rank),
                    "--lora-modules", f"evaluation-model={cfg['lora']}"]
    return command


@contextmanager
def serve(cfg, out):
    command = build_server_command(cfg)
    write_json(out / "server_command.json", command)
    with (out / "server.log").open("w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + cfg["startup_timeout_seconds"]
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"vLLM exited ({proc.returncode}); see server.log")
                try:
                    with urlopen("http://127.0.0.1:8000/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("vLLM startup timed out; see server.log")
                time.sleep(2)
            yield
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


def evaluate(cfg, out):
    from eval_offline.client import OfflineClient
    import torch
    import transformers
    import vllm

    rows, digest, total = load_dataset(cfg)
    write_json(out / "runtime.json", dict(torch=torch.__version__, cuda=torch.version.cuda,
               transformers=transformers.__version__, vllm=vllm.__version__, data_sha256=digest))

    async def run():
        client = OfflineClient("http://127.0.0.1:8000", "evaluation-model",
                               max_concurrent=cfg["max_concurrent_problems"], max_retries=3)
        client.openai_client.timeout = float(cfg["request_timeout_seconds"])
        try:
            with (out / "responses.jsonl").open("x", encoding="utf-8") as stream:
                return await score_dataset(client, rows, cfg, stream)
        finally:
            await client.openai_client.close()

    with serve(cfg, out):
        metrics = asyncio.run(run())
    metrics.update(benchmark=cfg["benchmark"], dataset_total_problems=total,
                   is_subset=len(rows) < total, data_sha256=digest)
    write_json(out / "scores.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)


def docker_command(cfg, out, name):
    mounts = [(ROOT, "/workspace/spade", True), (Path(cfg["checkpoint"]), "/model", True),
              (Path(cfg["data"]), "/dataset/test.jsonl", True), (out, "/results", False)]
    if cfg["lora"] is not None:
        mounts.append((Path(cfg["lora"]), "/lora", True))
    command = ["docker", "run", "--rm", "--init", "--name", name,
               "--gpus", '"device=' + ','.join(map(str, cfg["gpu_ids"])) + '"',
               "--network", "none", "--shm-size", "16g", "--user", f"{os.getuid()}:{os.getgid()}"]
    command += docker_memory_args(cfg["memory_limit_gib"])
    for source, target, readonly in mounts:
        if "," in str(source):
            raise ValueError("Docker mount paths must not contain commas")
        command += ["--mount", f"type=bind,src={source},dst={target}" + (",readonly" if readonly else "")]
    command += ["--workdir", "/workspace/spade"]
    for setting in ("USER=envduels", "PYTHONUNBUFFERED=1", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1",
                    "XDG_CACHE_HOME=/tmp/cache", "XDG_CONFIG_HOME=/tmp/config", "HF_HOME=/tmp/hf",
                    "VLLM_CACHE_ROOT=/tmp/vllm", "VLLM_CONFIG_ROOT=/tmp/vllm-config", "VLLM_NO_USAGE_STATS=1",
                    "FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer", "CUDA_CACHE_PATH=/tmp/cuda",
                    "GLOO_SOCKET_IFNAME=lo", "NCCL_SOCKET_IFNAME=lo", "TRITON_CACHE_DIR=/tmp/triton",
                    "TORCHINDUCTOR_CACHE_DIR=/tmp/inductor"):
        command += ["--env", setting]
    return command + ["--entrypoint", "python3", cfg["image"], "scripts/run_eval.py", "--inside-container",
                      "--config", "/results/container_config.json"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/evaluation.json")
    parser.add_argument("--checkpoint", help="Override evaluation.checkpoint")
    parser.add_argument("--data", help="Override evaluation.data (local JSONL file)")
    parser.add_argument("--output-dir", help="Override output parent directory; a new timestamp subdirectory is created")
    parser.add_argument("--image", help="Override evaluation.image")
    parser.add_argument("--lora", help="Optional Hugging Face/PEFT LoRA adapter directory")
    parser.add_argument("--max-problems", type=int, help="Evaluate only the first N problems")
    parser.add_argument("--samples-per-problem", type=int, help="Override k in pass@k")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print plan without Docker/GPU access")
    parser.add_argument("--inside-container", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    cfg = load_config(args.config, {k: getattr(args, k) for k in
                      ("checkpoint", "data", "output_dir", "max_problems", "samples_per_problem", "image", "lora")
                      if getattr(args, k) is not None})
    if args.inside_container:
        verify_container_limits(cfg["memory_limit_gib"])
        evaluate(cfg, Path(cfg["output_dir"]))
        return
    checkpoint = Path(cfg["checkpoint"])
    if not (checkpoint / "config.json").is_file() or not (
            any(checkpoint.glob("*.safetensors")) or any(checkpoint.glob("pytorch_model*.bin"))):
        raise ValueError("checkpoint must be a complete local Hugging Face model directory, not an adapter-only or Megatron checkpoint")
    if cfg["lora"] is not None:
        lora = Path(cfg["lora"])
        if not (lora / "adapter_config.json").is_file() or not (
                (lora / "adapter_model.safetensors").is_file() or (lora / "adapter_model.bin").is_file()):
            raise ValueError("lora must be a PEFT adapter directory with adapter_config.json and adapter weights")
    rows, digest, total = load_dataset(cfg)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = Path(cfg["output_dir"]) / stamp
    name = "spade-eval-" + stamp.lower()
    command = docker_command(cfg, out, name)
    report = dict(config=cfg, n_problems=len(rows), dataset_total_problems=total,
                  n_completions=len(rows) * cfg["samples_per_problem"], data_sha256=digest,
                  selected_ids=[r["id"] for r in rows], output=str(out), command=command,
                  model_config_sha256=hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest())
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("Dry run only; no container started or GPU allocated.")
        return
    with launch_lock(ROOT / ".spade-memory.lock"):
        preflight(cfg["memory_limit_gib"], cfg["host_memory_reserve_gib"])
        run_evaluation(cfg, out, name, command, report, checkpoint)


def run_evaluation(cfg, out, name, command, report, checkpoint):
    image_id = subprocess.check_output(["docker", "image", "inspect", cfg["image"], "--format", "{{.Id}}"], text=True).strip()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "resolved_config.json", report)
    write_json(out / "config.json", {"evaluation": cfg})
    write_json(out / "container_config.json", {"evaluation": {**cfg, "checkpoint": "/model",
               "data": "/dataset/test.jsonl", "output_dir": "/results",
               "lora": "/lora" if cfg["lora"] is not None else None}})
    write_json(out / "launch.json", dict(command=command, image_id=image_id, container=name))
    manifest = checkpoint / "download_manifest.json"
    if manifest.is_file():
        write_json(out / "model_download_manifest.json", json.loads(manifest.read_text()))
    write_json(out / "status.json", dict(status="running"))
    print(f"Running; follow progress with: tail -f {out / 'console.log'}", flush=True)
    # Keep Docker attached to the host runner, but stop only our own container on Ctrl-C.
    try:
        with (out / "console.log").open("w") as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            code = guarded_wait(proc, name, cfg["host_memory_reserve_gib"])
    except BaseException as exc:
        write_json(out / "status.json", dict(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc)))
        raise
    write_json(out / "status.json", dict(status="completed" if code == 0 else "failed", exit_code=code))
    print(f"Exit {code}; results/logs: {out}", flush=True)
    if code == 0:
        print((out / "scores.json").read_text(), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
