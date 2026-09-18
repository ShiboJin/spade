"""Local benchmark evaluation (legacy filename). Default plan does not use GPUs."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_data import load_benchmark, grade, validate_benchmark


def inside_checkout(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Evaluation data/checkpoint paths must be inside the SPADE checkout")
    return path


def configuration():
    cfg = {"benchmark": validate_benchmark(os.environ.get("EVAL_BENCHMARK", "aime2026"))}
    for key, default in dict(N_SAMPLES=32, TOP_K=20, MAX_TOKENS=32768, CONTEXT_LENGTH=40960,
                             SEED=42, TP=2, MAX_CONCURRENT=2).items():
        cfg[key.lower()] = int(os.environ.get("EVAL_" + key, default))
    for key, default in dict(TEMPERATURE=0.7, TOP_P=0.8).items():
        cfg[key.lower()] = float(os.environ.get("EVAL_" + key, default))
    thinking = os.environ.get("EVAL_THINKING", "false")
    if thinking not in ("true", "false"):
        raise ValueError("EVAL_THINKING must be true or false")
    cfg["thinking"] = thinking == "true"
    if any(cfg[k] < 1 for k in ("n_samples", "max_tokens", "context_length", "tp", "max_concurrent")):
        raise ValueError("Evaluation counts/lengths must be positive")
    if not math.isfinite(cfg["temperature"]) or cfg["temperature"] < 0 or not 0 < cfg["top_p"] <= 1:
        raise ValueError("Invalid sampling parameters")
    if cfg["top_k"] != -1 and cfg["top_k"] < 1:
        raise ValueError("top_k must be -1 or positive")
    if cfg["context_length"] <= cfg["max_tokens"]:
        raise ValueError("Evaluation context must leave room for the question")
    return cfg


def score_answer(text, answer):
    return grade(text, answer, "aime2026")[:2]


async def score_dataset(client, rows, cfg, output):
    # Fail on API errors/missing samples rather than silently shrink the denominator.
    benchmark = cfg.get("benchmark", "aime2026")
    async def problem(index, row):
        prompt = row["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        completions = await client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg["temperature"], top_p=cfg["top_p"], max_tokens=cfg["max_tokens"],
            n=cfg["n_samples"], extra_body={"top_k": cfg["top_k"], "seed": cfg["seed"] + index + 1,
                "chat_template_kwargs": {"enable_thinking": cfg["thinking"], "preserve_thinking": True}})
        if len(completions) != cfg["n_samples"]:
            raise RuntimeError(f"Problem {row['id']}: missing completions")
        results = []
        for i, c in enumerate(completions):
            predicted, correct, valid = grade(c.text, row["answer"], benchmark)
            result = dict(id=row["id"], sample=i, answer=row["answer"], predicted=predicted,
                          correct=correct, valid_answer=valid, response=c.text, raw=c.raw,
                          finish_reason=c.finish_reason, completion_tokens=c.completion_tokens)
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            results.append(result)
        output.flush()
        print(f"{benchmark} problem {row['id']}: {sum(r['correct'] for r in results)}/{len(results)}", flush=True)
        return results

    groups = await asyncio.gather(*(problem(index, row) for index, row in enumerate(rows)))
    samples = [r for group in groups for r in group]
    return {"n_problems": len(rows), "n_samples_per_problem": cfg["n_samples"],
            "n_completions": len(samples),
            "sample_accuracy": sum(r["correct"] for r in samples) / len(samples),
            "fraction_problems_any_correct": sum(any(r["correct"] for r in g) for g in groups) / len(groups),
            "invalid_answer_fraction": sum(not r["valid_answer"] for r in samples) / len(samples),
            "length_stop_fraction": sum(r["finish_reason"] == "length" for r in samples) / len(samples)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.run and args.plan:
        parser.error("Choose --run or --plan")
    cfg = configuration()
    data = inside_checkout(os.environ.get("EVAL_DATA", os.environ.get("AIME26_DATA", "data/aime26/aime2026.jsonl")))
    checkpoint = inside_checkout(os.environ.get("EVAL_HF_CHECKPOINT", "checkpoints/Qwen3.8-27B"))
    rows, digest = load_benchmark(data, cfg["benchmark"])
    if not (checkpoint / "config.json").is_file() or not any(checkpoint.glob("*.safetensors")):
        raise ValueError("EVAL_HF_CHECKPOINT must be a local HF weights directory, not a Megatron checkpoint")
    report = dict(protocol=cfg, data=str(data), data_sha256=digest, checkpoint=str(checkpoint),
                  n_problems=len(rows), metric="mean sample correctness (Avg@N), not any-correct pass@N")
    print(json.dumps(report, indent=2))
    if not args.run:
        print("Plan only; no model or GPU allocated.")
        return
    # Heavy/runtime imports only after all input checks.
    from eval_offline.server import sglang_server
    from eval_offline.client import OfflineClient
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = ROOT / "outputs" / "evaluation" / cfg["benchmark"] / stamp
    out.mkdir(parents=True, exist_ok=False)
    (out / "resolved_config.json").write_text(json.dumps(report, indent=2))
    with sglang_server(checkpoint, tp=cfg["tp"], dp=1, tool_call_parser=None,
                       log_path=out / "server.log", startup_timeout=1200,
                       extra_args=["--attention-backend", "triton", "--disable-cuda-graph",
                                   "--context-length", str(cfg["context_length"])]) as (url, model):
        client = OfflineClient(url, model, max_concurrent=cfg["max_concurrent"])
        async def evaluate():
            try:
                with (out / "responses.jsonl").open("x") as stream:
                    return await score_dataset(client, rows, cfg, stream)
            finally:
                await client.openai_client.close()
        metrics = asyncio.run(evaluate())
    (out / "scores.json").write_text(json.dumps(metrics, indent=2))
    print(f"Results: {out}\n{json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
