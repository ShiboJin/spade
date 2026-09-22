"""Evaluate a local checkpoint as an unhinted EnvDuels solver and rank it.

The host process validates inputs and launches one isolated Docker container on
the requested GPUs.  Inside the container, a local vLLM OpenAI server is started
and every exported environment is played on its frozen benchmark seeds.

Results are append-only and resumable: one summary row and one full trajectory
are written per environment/seed pair.  Ranking uses the frozen source-run
episodes as the baseline and reports both the canonical self-author-excluded
score and a same-environment direct solver comparison.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
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
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.memory_guard import (  # noqa: E402
    DEFAULT_RESERVE_GIB,
    docker_memory_args,
    guarded_wait,
    task_resources,
    validate_limits,
    verify_container_limits,
)
SYSTEM_PROMPT = (
    "You are playing an interactive language game. Make one valid action per turn. "
    "Reason carefully and put the action for that turn inside \\boxed{}."
)
TERMINAL = {"success", "failure"}
DEFAULTS = {
    "checkpoint": "checkpoints/Qwen3.8-27B-spade-merged-ckpt19",
    "export_dir": "../exports/duel_harness_004_rl",
    "baseline_run": "../exports/duel_harness_004_push",
    "output_dir": "outputs/envduels_solver_eval/qwen38-spade-ckpt19",
    "image": "envduels-unified:cu124",
    "gpu_ids": [0, 1],
    "tensor_parallel": 2,
    "dtype": "bfloat16",
    "context_length": 16384,
    "gpu_memory_utilization": 0.9,
    "max_num_seqs": 64,
    "max_num_batched_tokens": 8192,
    "enforce_eager": True,
    "served_model_name": "qwen38-spade-ckpt19",
    "solver_name": "qwen3.8-27b-spade-ckpt19",
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "max_tokens_per_turn": 1024,
    "enable_thinking": False,
    "max_concurrent_episodes": 64,
    "request_timeout_seconds": 900,
    "request_retries": 3,
    "startup_timeout_seconds": 1800,
    "seed": 42,
    "max_environments": None,
    "seeds_per_environment": None,
    "port": 31038,
    "memory_limit_gib": 96,
    "host_memory_reserve_gib": DEFAULT_RESERVE_GIB,
}


def extract_action(response: str) -> str | None:
    """Extract the final balanced ``\\boxed{...}`` without importing ML deps."""
    marker = response.rfind("\\boxed{")
    if marker < 0:
        return None
    start = marker + len("\\boxed{")
    depth = 1
    for index in range(start, len(response)):
        if response[index] == "{":
            depth += 1
        elif response[index] == "}":
            depth -= 1
            if depth == 0:
                return response[start:index].strip() or None
    return None


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str) -> Path:
    return (ROOT / Path(value).expanduser()).resolve()


def load_config(path: Path, overrides: dict | None = None) -> dict:
    document = read_json(path)
    raw = document.get("evaluation", document)
    if not isinstance(raw, dict):
        raise ValueError("Configuration must contain an evaluation mapping")
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown evaluation setting(s): {sorted(unknown)}")
    cfg = {**DEFAULTS, **raw, **(overrides or {})}
    for key in ("checkpoint", "export_dir", "baseline_run", "output_dir"):
        cfg[key] = str(resolve_path(cfg[key]))
    validate_config(cfg)
    return cfg


def positive_int(cfg: dict, key: str) -> None:
    if type(cfg.get(key)) is not int or cfg[key] < 1:
        raise ValueError(f"{key} must be a positive integer")


def validate_config(cfg: dict) -> None:
    for key in (
        "tensor_parallel", "context_length", "max_num_seqs",
        "max_num_batched_tokens", "max_tokens_per_turn",
        "max_concurrent_episodes", "request_timeout_seconds",
        "request_retries", "startup_timeout_seconds", "port",
        "memory_limit_gib", "host_memory_reserve_gib",
    ):
        positive_int(cfg, key)
    if not isinstance(cfg["gpu_ids"], list) or not cfg["gpu_ids"] or any(
        type(gpu) is not int or gpu < 0 for gpu in cfg["gpu_ids"]
    ):
        raise ValueError("gpu_ids must contain nonnegative integers")
    if len(set(cfg["gpu_ids"])) != len(cfg["gpu_ids"]):
        raise ValueError("gpu_ids cannot contain duplicates")
    if cfg["tensor_parallel"] != len(cfg["gpu_ids"]):
        raise ValueError("tensor_parallel must equal len(gpu_ids)")
    for key in ("temperature", "top_p", "min_p", "repetition_penalty", "gpu_memory_utilization"):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]):
            raise ValueError(f"{key} must be finite")
    if not 0 <= cfg["temperature"] <= 2 or not 0 < cfg["top_p"] <= 1:
        raise ValueError("Invalid temperature or top_p")
    if not 0 <= cfg["min_p"] <= 1 or cfg["repetition_penalty"] <= 0:
        raise ValueError("Invalid min_p or repetition_penalty")
    if not 0 < cfg["gpu_memory_utilization"] <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    if cfg["max_tokens_per_turn"] >= cfg["context_length"]:
        raise ValueError("max_tokens_per_turn must be below context_length")
    for key in ("max_environments", "seeds_per_environment"):
        if cfg[key] is not None and (type(cfg[key]) is not int or cfg[key] < 1):
            raise ValueError(f"{key} must be null or a positive integer")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    for key in ("served_model_name", "solver_name", "image", "dtype"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    validate_limits(cfg["memory_limit_gib"], cfg["host_memory_reserve_gib"])


def validate_inputs(cfg: dict) -> dict:
    checkpoint = Path(cfg["checkpoint"])
    export = Path(cfg["export_dir"])
    baseline = Path(cfg["baseline_run"])
    if not (checkpoint / "config.json").is_file() or not any(checkpoint.glob("*.safetensors")):
        raise ValueError("checkpoint must be a complete local safetensors model")
    if not (export / "manifest.json").is_file():
        raise FileNotFoundError(f"EnvDuels manifest not found: {export}")
    if not (baseline / "run.json").is_file():
        raise FileNotFoundError(f"Baseline run.json not found: {baseline}")
    cases, manifest_hash = load_cases(cfg)
    return {
        "environment_count": len({case["env_id"] for case in cases}),
        "episode_count": len(cases),
        "manifest_sha256": manifest_hash,
    }


def load_cases(cfg: dict) -> tuple[list[dict], str]:
    root = Path(cfg["export_dir"])
    raw = (root / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    rows = manifest.get("environments", [])
    if cfg["max_environments"] is not None:
        rows = rows[: cfg["max_environments"]]
    cases = []
    for row in rows:
        metadata = read_json(root / row["metadata"])
        seeds = metadata.get("benchmark_seeds")
        if not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds):
            raise ValueError(f"Missing benchmark seeds for {row['id']}")
        if cfg["seeds_per_environment"] is not None:
            seeds = seeds[: cfg["seeds_per_environment"]]
        for episode, seed in enumerate(seeds):
            cases.append({
                "env_id": row["id"], "author": row["author"],
                "domain": row["domain"], "episode": episode, "seed": seed,
                "source_sha256": row["source_sha256"],
                "max_turns": row["max_turns"],
            })
    if not cases:
        raise ValueError("No evaluation cases selected")
    return cases, hashlib.sha256(raw).hexdigest()


def episode_key(case: dict) -> str:
    return f"{case['env_id']}|{case['seed']}|without_hint"


def trajectory_name(case: dict) -> str:
    digest = hashlib.sha256(episode_key(case).encode()).hexdigest()[:16]
    safe = case["env_id"].replace("/", "__")
    return f"{safe}__seed-{case['seed']}__{digest}.json"


class LocalOpenAIClient:
    def __init__(self, cfg: dict, base_url: str):
        self.cfg = cfg
        self.url = base_url.rstrip("/") + "/v1/chat/completions"

    def _request(self, messages: list[dict], request_seed: int) -> dict:
        payload = {
            "model": self.cfg["served_model_name"],
            "messages": messages,
            "temperature": self.cfg["temperature"],
            "top_p": self.cfg["top_p"],
            "max_tokens": self.cfg["max_tokens_per_turn"],
            "seed": request_seed,
            "top_k": self.cfg["top_k"],
            "min_p": self.cfg["min_p"],
            "repetition_penalty": self.cfg["repetition_penalty"],
            "chat_template_kwargs": {"enable_thinking": self.cfg["enable_thinking"]},
        }
        request = Request(
            self.url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.cfg["request_timeout_seconds"]) as response:
                return json.loads(response.read())
        except HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body[:2000]}") from exc

    async def chat(self, messages: list[dict], request_seed: int) -> dict:
        last = None
        for attempt in range(self.cfg["request_retries"]):
            try:
                result = await asyncio.to_thread(self._request, messages, request_seed)
                choice = result["choices"][0]
                content = choice["message"].get("content") or ""
                return {
                    "text": content,
                    "finish_reason": choice.get("finish_reason"),
                    "usage": result.get("usage", {}),
                }
            except (OSError, URLError, RuntimeError, KeyError, IndexError, json.JSONDecodeError) as exc:
                last = exc
                if attempt + 1 < self.cfg["request_retries"]:
                    await asyncio.sleep(min(8, 2**attempt))
        raise RuntimeError(f"Model request failed after retries: {last}")


async def play_case(cfg: dict, adapter: EnvDuelsAdapter, client: LocalOpenAIClient,
                    case: dict, out: Path) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    steps = []
    instance = None
    status = "incomplete"
    stop_reason = "not_started"
    error = None
    total_usage = defaultdict(int)
    try:
        instance = adapter.create_instance_with_seed(case["env_id"], case["seed"])
        observation, reset_info = instance.reset()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Observation: {observation}\n\n"
                "Respond with exactly one action for this turn inside \\boxed{}."
            )},
        ]
        for turn in range(1, case["max_turns"] + 1):
            request_seed = int.from_bytes(hashlib.sha256(
                f"{cfg['seed']}|{episode_key(case)}|{turn}".encode()
            ).digest()[:4], "big")
            response = await client.chat(messages, request_seed)
            text = response["text"]
            for key, value in response["usage"].items():
                if isinstance(value, int):
                    total_usage[key] += value
            next_observation, reward, terminated, truncated, info = instance.step(text)
            steps.append({
                "turn": turn, "observation": observation, "response": text,
                "parsed_action": extract_action(text), "reward": reward,
                "terminated": terminated, "truncated": truncated,
                "info": info, "finish_reason": response["finish_reason"],
                "usage": response["usage"], "request_seed": request_seed,
            })
            if terminated or truncated:
                status = "success" if reward > 0 else "failure"
                stop_reason = "environment_terminated" if terminated else "max_turns"
                break
            messages.extend([
                {"role": "assistant", "content": text},
                {"role": "user", "content": next_observation},
            ])
            observation = next_observation
        else:
            status, stop_reason = "failure", "max_turns"
        reset = reset_info
    except Exception as exc:  # Preserve enough evidence to distinguish model/runtime/env faults.
        text = str(exc)
        error = f"{type(exc).__name__}: {text}"
        if "maximum context" in text.lower() or "context length" in text.lower():
            status, stop_reason = "failure", "context_length"
        else:
            status, stop_reason = "incomplete", "error"
        reset = None
    finally:
        if instance is not None:
            instance.close()
    finished = datetime.now(timezone.utc).isoformat()
    summary = {
        **case, "key": episode_key(case), "solver": cfg["solver_name"],
        "condition": "without_hint", "trial": 0, "status": status,
        "turns": len(steps), "stop_reason": stop_reason, "error": error,
        "usage": dict(total_usage), "started_at": started, "finished_at": finished,
        "trajectory": f"trajectories/{trajectory_name(case)}",
    }
    trajectory = {
        "protocol": "envduels_local_solver_v1", "summary": summary,
        "system_prompt": SYSTEM_PROMPT, "reset_info": reset, "steps": steps,
    }
    write_json(out / summary["trajectory"], trajectory)
    return summary


def load_existing(path: Path) -> dict[str, dict]:
    rows = {}
    if not path.is_file():
        return rows
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row["key"] in rows:
            raise ValueError(f"Duplicate result key on line {number}: {row['key']}")
        rows[row["key"]] = row
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def domain_score(env_scores: dict[str, float], env_meta: dict[str, dict],
                 allowed: set[str] | None = None, excluded_author: str | None = None) -> tuple[float | None, dict]:
    domains = defaultdict(list)
    for env_id, score in env_scores.items():
        if allowed is not None and env_id not in allowed:
            continue
        meta = env_meta.get(env_id)
        if meta is None or (excluded_author is not None and meta["author"] == excluded_author):
            continue
        domains[meta["domain"]].append(score)
    per_domain = {key: mean(values) for key, values in sorted(domains.items())}
    return mean(per_domain.values()), per_domain


def baseline_scores(root: Path) -> tuple[list[str], dict, dict]:
    matrix = read_json(root / "run.json")
    models = [row["name"] for row in matrix["models"]]
    env_meta, scores = {}, {model: {} for model in models}
    for item in matrix["environments"]:
        env_root = root / item["path"]
        relative = item.get("final_env")
        if not isinstance(relative, str):
            continue
        folder = (env_root / relative).parent
        report_path = folder / "run.json"
        if not report_path.is_file():
            continue
        report = read_json(report_path)
        env_id = str(folder.relative_to(root))
        env_meta[env_id] = {"author": item["designer"], "domain": item["domain"]}
        grouped = defaultdict(list)
        for episode in report.get("episodes", []):
            if episode.get("condition", "without_hint") != "without_hint":
                continue
            if episode.get("status") in TERMINAL and episode.get("breaker") in scores:
                grouped[episode["breaker"]].append(episode["status"] == "success")
        for model, values in grouped.items():
            scores[model][env_id] = mean(values)
    return models, env_meta, scores


def build_ranking(cfg: dict, result_rows: dict[str, dict], out: Path) -> dict:
    models, env_meta, scores = baseline_scores(Path(cfg["baseline_run"]))
    new_grouped = defaultdict(list)
    for row in result_rows.values():
        if row["status"] in TERMINAL:
            new_grouped[row["env_id"]].append(row["status"] == "success")
    scores[cfg["solver_name"]] = {env_id: mean(values) for env_id, values in new_grouped.items()}
    all_models = models + [cfg["solver_name"]]
    common = set(env_meta)
    for model in all_models:
        common &= set(scores[model])

    ranking = []
    for model in all_models:
        author = model if model in models else None
        canonical, canonical_domains = domain_score(
            scores[model], env_meta, excluded_author=author,
        )
        direct, direct_domains = domain_score(scores[model], env_meta, allowed=common)
        ranking.append({
            "model": model, "canonical_autonomous_solve": canonical,
            "canonical_domain_scores": canonical_domains,
            "canonical_environment_count": sum(
                env_id in env_meta and (author is None or env_meta[env_id]["author"] != author)
                for env_id in scores[model]
            ),
            "direct_common_panel_solve": direct,
            "direct_common_panel_domain_scores": direct_domains,
            "direct_common_panel_environment_count": len(common),
        })
    for field, rank_field in (
        ("canonical_autonomous_solve", "canonical_rank"),
        ("direct_common_panel_solve", "direct_common_panel_rank"),
    ):
        ordered = sorted(ranking, key=lambda row: (
            row[field] is None, -(row[field] or 0), row["model"],
        ))
        for rank, row in enumerate(ordered, 1):
            row[rank_field] = rank if row[field] is not None else None
    ranking.sort(key=lambda row: (row["direct_common_panel_rank"] or 10**9, row["model"]))
    payload = {
        "protocol": "envduels_local_solver_ranking_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "solver": cfg["solver_name"], "baseline_run": cfg["baseline_run"],
        "ranking_notes": {
            "canonical": "Domain-balanced no-hint mean; original solvers exclude self-authored environments.",
            "direct_common_panel": "All solvers evaluated on the same environment intersection; self plays retained.",
            "seed_aggregation": "Available terminal trials average within environment before domains receive equal weight.",
        },
        "common_environment_count": len(common),
        "common_environments": sorted(common), "ranking": ranking,
    }
    write_json(out / "ranking_solver.json", payload)
    return payload


def write_progress(cfg: dict, cases: list[dict], rows: dict[str, dict], out: Path) -> None:
    counts = defaultdict(int)
    for row in rows.values():
        counts[row["status"]] += 1
    terminal = sum(counts[key] for key in TERMINAL)
    payload = {
        "status": "completed" if len(rows) == len(cases) and not counts["incomplete"] else "running",
        "total_episodes": len(cases), "recorded_episodes": len(rows),
        "terminal_episodes": terminal, "counts": dict(counts),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out / "progress.json", payload)


async def evaluate_cases(cfg: dict, out: Path, base_url: str, retry_errors: bool) -> None:
    # Import only in the runtime image.  The host-side planner intentionally
    # needs no numpy/torch environment.
    from spade.core.envs.envduels_adapter import EnvDuelsAdapter

    cases, _ = load_cases(cfg)
    result_path = out / "episodes.jsonl"
    existing = load_existing(result_path)
    if retry_errors:
        existing = {key: row for key, row in existing.items() if row["status"] in TERMINAL}
        with result_path.open("w", encoding="utf-8") as stream:
            for row in existing.values():
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    adapter = EnvDuelsAdapter(cfg["export_dir"])
    client = LocalOpenAIClient(cfg, base_url)
    semaphore = asyncio.Semaphore(cfg["max_concurrent_episodes"])
    write_lock = asyncio.Lock()

    async def one(case: dict) -> None:
        key = episode_key(case)
        if key in existing:
            return
        async with semaphore:
            row = await play_case(cfg, adapter, client, case, out)
        async with write_lock:
            append_jsonl(result_path, row)
            existing[key] = row
            write_progress(cfg, cases, existing, out)
            done = len(existing)
            print(
                f"[{done}/{len(cases)}] {row['status']:<10} "
                f"{row['env_id']} seed={row['seed']} turns={row['turns']}",
                flush=True,
            )

    tasks = [asyncio.create_task(one(case)) for case in cases]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    write_progress(cfg, cases, existing, out)
    ranking = build_ranking(cfg, existing, out)
    new = next(row for row in ranking["ranking"] if row["model"] == cfg["solver_name"])
    print(
        f"Solver ranking: direct={new['direct_common_panel_rank']}, "
        f"canonical={new['canonical_rank']}, "
        f"direct score={100*(new['direct_common_panel_solve'] or 0):.2f}%",
        flush=True,
    )


def build_server_command(cfg: dict) -> list[str]:
    command = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", cfg["checkpoint"], "--served-model-name", cfg["served_model_name"],
        "--host", "127.0.0.1", "--port", str(cfg["port"]),
        "--tensor-parallel-size", str(cfg["tensor_parallel"]),
        "--dtype", cfg["dtype"], "--max-model-len", str(cfg["context_length"]),
        "--gpu-memory-utilization", str(cfg["gpu_memory_utilization"]),
        "--max-num-seqs", str(cfg["max_num_seqs"]),
        "--max-num-batched-tokens", str(cfg["max_num_batched_tokens"]),
        "--seed", str(cfg["seed"]), "--trust-remote-code",
        "--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0}),
    ]
    if cfg["enforce_eager"]:
        command.append("--enforce-eager")
    return command


@contextmanager
def serve(cfg: dict, out: Path):
    command = build_server_command(cfg)
    write_json(out / "server_command.json", command)
    with (out / "server.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + cfg["startup_timeout_seconds"]
            health = f"http://127.0.0.1:{cfg['port']}/health"
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"vLLM exited ({process.returncode}); see server.log")
                try:
                    with urlopen(health, timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("vLLM startup timed out; see server.log")
                time.sleep(2)
            yield f"http://127.0.0.1:{cfg['port']}"
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def container_config(cfg: dict) -> dict:
    result = deepcopy(cfg)
    result.update(
        checkpoint="/model", export_dir="/envduels-export",
        baseline_run="/baseline-run", output_dir="/results",
    )
    return result


def docker_command(cfg: dict, out: Path, name: str, retry_errors: bool) -> list[str]:
    mounts = [
        (ROOT, "/workspace/spade", True),
        (Path(cfg["checkpoint"]), "/model", True),
        (Path(cfg["export_dir"]), "/envduels-export", True),
        (Path(cfg["baseline_run"]), "/baseline-run", True),
        (out, "/results", False),
    ]
    command = [
        "docker", "run", "--rm", "--init", "--name", name,
        "--gpus", '"device=' + ",".join(map(str, cfg["gpu_ids"])) + '"',
        "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}",
        "--shm-size", "16g", "--label", "spade.gpu-concurrent=true",
        *docker_memory_args(cfg["memory_limit_gib"]),
    ]
    for source, target, readonly in mounts:
        command += ["--mount", f"type=bind,src={source},dst={target}" + (",readonly" if readonly else "")]
    settings = {
        "HOME": "/tmp/home", "USER": "envduels", "PYTHONUNBUFFERED": "1",
        "HF_HOME": "/tmp/hf", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "XDG_CACHE_HOME": "/tmp/cache", "VLLM_CACHE_ROOT": "/tmp/vllm",
        "VLLM_CONFIG_ROOT": "/tmp/vllm-config", "VLLM_NO_USAGE_STATS": "1",
        "FLASHINFER_WORKSPACE_BASE": "/tmp/flashinfer", "CUDA_CACHE_PATH": "/tmp/cuda",
        "GLOO_SOCKET_IFNAME": "lo", "NCCL_SOCKET_IFNAME": "lo",
        "TRITON_CACHE_DIR": "/tmp/triton", "TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor",
    }
    for key, value in settings.items():
        command += ["--env", f"{key}={value}"]
    command += [
        "--workdir", "/workspace/spade", "--entrypoint", "python3", cfg["image"],
        "scripts/run_envduels_solver_eval.py", "--inside-container",
        "--config", "/results/container_config.json",
    ]
    if retry_errors:
        command.append("--retry-errors")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen38_envduels_solver_eval.json")
    parser.add_argument("--gpu-ids", type=int, nargs="+", help="Physical GPU indices")
    parser.add_argument("--max-environments", type=int, help="Smoke-test prefix of environments")
    parser.add_argument("--seeds-per-environment", type=int, help="Smoke-test prefix of benchmark seeds")
    parser.add_argument("--output-dir", help="Reusable result directory (not a timestamped parent)")
    parser.add_argument("--retry-errors", action="store_true", help="Retry prior incomplete/error episodes")
    parser.add_argument("--rank-only", action="store_true", help="Rebuild ranking from saved episodes")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inside-container", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    overrides = {}
    for key in ("max_environments", "seeds_per_environment", "output_dir"):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    if args.gpu_ids is not None:
        overrides.update(gpu_ids=args.gpu_ids, tensor_parallel=len(args.gpu_ids))
    cfg = load_config(args.config.resolve(), overrides)
    out = Path(cfg["output_dir"])

    if args.inside_container:
        verify_container_limits(cfg["memory_limit_gib"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "trajectories").mkdir(exist_ok=True)
        if args.rank_only:
            build_ranking(cfg, load_existing(out / "episodes.jsonl"), out)
            return 0
        with serve(cfg, out) as base_url:
            asyncio.run(evaluate_cases(cfg, out, base_url, args.retry_errors))
        return 0

    plan = validate_inputs(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "trajectories").mkdir(exist_ok=True)
    name_hash = hashlib.sha256(str(out).encode()).hexdigest()[:10]
    name = f"spade-envduels-solver-{name_hash}"
    write_json(out / "container_config.json", {"evaluation": container_config(cfg)})
    command = docker_command(cfg, out, name, args.retry_errors)
    report = {"config": cfg, **plan, "container": name, "command": command}
    write_json(out / "resolved_config.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.rank_only:
        ranking = build_ranking(cfg, load_existing(out / "episodes.jsonl"), out)
        print(json.dumps(ranking["ranking"], ensure_ascii=False, indent=2))
        return 0
    if args.dry_run:
        print("Dry run only; no container or GPU was started.")
        return 0
    write_json(out / "status.json", {"status": "running", "updated_at": datetime.now(timezone.utc).isoformat()})
    try:
        with task_resources(cfg, ROOT):
            with (out / "console.log").open("a", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                code = guarded_wait(process, name, cfg["host_memory_reserve_gib"])
    except BaseException as exc:
        write_json(out / "status.json", {
            "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            "error": str(exc), "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        raise
    write_json(out / "status.json", {
        "status": "completed" if code == 0 else "failed", "exit_code": code,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    print(f"Exit {code}; results: {out}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
