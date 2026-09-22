"""GEM suite adapter for the retained Reasoning-Gym evaluations.

Reads the same YAML schema as eval_configs/gem_eval_*.yaml, builds an
OfflineModelAdapter, calls GemEvaluator.evaluate_all, writes scores.json.

Config (either form):
    suites:
      gem:
        defaults: {...}      # inline gem_eval schema (preferred: single-file
        tasks: [...]         # protocol, nothing hidden in an include)
    suites:
      gem:
        config_path: path/to/gem_eval.yaml   # external include, still supported

Output:
    <out>/scores.json   — flat metric dict (gem_eval/... keys)
    <out>/raw_result.json — full GemEvalResult (per-task results, errors)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

logger = logging.getLogger("eval_offline.suites.gem")


def _resolve_config_path(p: str) -> Path:
    if not p:
        raise ValueError("gem suite needs `config_path`.")
    pp = Path(p)
    if pp.is_absolute() and pp.is_file():
        return pp
    for root in ("/workspace", os.getcwd()):
        candidate = Path(root) / p
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"gem config_path {p!r} not found")


def _load_prior_result(path: Path):
    from spade.core.eval.gem_evaluator import GemEvalResult, GemTaskResult

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["per_task_results"] = [
        GemTaskResult(**item) for item in raw.get("per_task_results", [])
    ]
    return GemEvalResult(**raw)


def _aggregate_task_results(per_task_results):
    """Recompute GEM rollups after replacing failed episodes during resume."""
    from spade.core.eval.gem_evaluator import GemEvalResult
    from spade.core.eval.gem_tasks import rg_golden_goose_category

    total_episodes = sum(item.num_episodes for item in per_task_results)
    total_wins = sum(item.num_wins for item in per_task_results)
    weighted_reward = sum(
        item.mean_reward * item.num_episodes for item in per_task_results
    )
    weighted_turns = sum(
        item.mean_turns * item.num_episodes for item in per_task_results
    )

    category_wins: dict[str, int] = defaultdict(int)
    category_episodes: dict[str, int] = defaultdict(int)
    category_rewards: dict[str, list[float]] = defaultdict(list)
    gg_wins: dict[str, int] = defaultdict(int)
    gg_episodes: dict[str, int] = defaultdict(int)
    gg_rewards: dict[str, list[float]] = defaultdict(list)
    for item in per_task_results:
        category_wins[item.category] += item.num_wins
        category_episodes[item.category] += item.num_episodes
        category_rewards[item.category].append(item.mean_reward)
        gg = rg_golden_goose_category(item.task_id)
        if gg is not None and item.num_episodes:
            key = f"rg_{gg}"
            gg_wins[key] += item.num_wins
            gg_episodes[key] += item.num_episodes
            gg_rewards[key].append(item.mean_reward)

    category_metrics: dict[str, dict[str, float]] = {}
    for category, episodes in category_episodes.items():
        if episodes:
            category_metrics[category] = {
                "win_rate": category_wins[category] / episodes,
                "mean_reward": sum(category_rewards[category]) / len(category_rewards[category]),
                "num_tasks": float(len(category_rewards[category])),
                "num_episodes": float(episodes),
            }
    for category, episodes in gg_episodes.items():
        if episodes:
            category_metrics[category] = {
                "win_rate": gg_wins[category] / episodes,
                "mean_reward": sum(gg_rewards[category]) / len(gg_rewards[category]),
                "num_tasks": float(len(gg_rewards[category])),
                "num_episodes": float(episodes),
            }

    return GemEvalResult(
        overall_win_rate=total_wins / total_episodes if total_episodes else 0.0,
        overall_mean_reward=weighted_reward / total_episodes if total_episodes else 0.0,
        overall_mean_turns=weighted_turns / total_episodes if total_episodes else 0.0,
        total_episodes=total_episodes,
        total_tasks=len(per_task_results),
        per_task_results=per_task_results,
        per_category_metrics=category_metrics,
        errors=sum(item.errors for item in per_task_results),
    )


def _merge_retry_result(prior, retry, task_specs):
    """Replace prior failed episode slots with retry aggregates."""
    from spade.core.eval.gem_evaluator import GemTaskResult

    prior_by_key = {
        (item.task_id, item.metric_suffix): item for item in prior.per_task_results
    }
    retry_by_key = {
        (item.task_id, item.metric_suffix): item for item in retry.per_task_results
    }
    merged = []
    for spec in task_specs:
        key = (spec.task_id, spec.metric_suffix)
        old = prior_by_key.get(key)
        new = retry_by_key.get(key)
        if old is None:
            if new is not None:
                merged.append(new)
            continue
        if new is None:
            merged.append(old)
            continue
        episodes = old.num_episodes + new.num_episodes
        merged.append(GemTaskResult(
            task_id=old.task_id,
            category=old.category,
            num_episodes=episodes,
            num_wins=old.num_wins + new.num_wins,
            win_rate=(old.num_wins + new.num_wins) / episodes if episodes else 0.0,
            mean_reward=(
                old.mean_reward * old.num_episodes
                + new.mean_reward * new.num_episodes
            ) / episodes if episodes else 0.0,
            mean_turns=(
                old.mean_turns * old.num_episodes
                + new.mean_turns * new.num_episodes
            ) / episodes if episodes else 0.0,
            # Old errors are the slots just retried. Only retry failures remain.
            errors=new.errors,
            elapsed_sec=old.elapsed_sec + new.elapsed_sec,
            metric_suffix=old.metric_suffix,
        ))
    return _aggregate_task_results(merged)


def run(client, cfg: dict, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    from spade.core.eval.gem_evaluator import GemEvaluator
    from spade.core.eval.gem_tasks import load_gem_eval_config
    from eval_offline.model_adapter_shim import OfflineModelAdapter

    if cfg.get("tasks"):
        # Inline form: the suite config IS the gem_eval schema. Materialise it
        # to a temp file so the shared loader stays the single parsing path.
        import tempfile, yaml as _yaml
        inline = {"gem_eval": {k: cfg[k] for k in ("defaults", "tasks") if k in cfg}}
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
            _yaml.safe_dump(inline, tf)
            config_path = Path(tf.name)
        logger.info("[gem] using inline task config (%d tasks)", len(cfg["tasks"]))
    else:
        config_path = _resolve_config_path(cfg.get("config_path", ""))
        logger.info("[gem] loading config from %s", config_path)
        # A resume wrapper may point at the normal offline suite YAML rather
        # than duplicating its full 102-task protocol.
        import yaml as _yaml
        source = _yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        nested = (source.get("suites") or {}).get("gem")
        if isinstance(nested, dict) and nested.get("tasks"):
            import tempfile
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
                _yaml.safe_dump({"gem_eval": nested}, tf)
                config_path = Path(tf.name)
    defaults, task_specs = load_gem_eval_config(str(config_path))

    max_tokens = cfg.get("max_tokens")
    if max_tokens is not None:
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("gem max_tokens must be a positive integer")
        task_specs = [replace(spec, max_tokens=max_tokens) for spec in task_specs]
        logger.info("[gem] overriding every task max_tokens=%d", max_tokens)

    prefixes = cfg.get("include_task_prefixes", [])
    if not isinstance(prefixes, list) or any(
        not isinstance(prefix, str) or not prefix for prefix in prefixes
    ):
        raise ValueError("gem include_task_prefixes must be a list of nonempty strings")
    if prefixes:
        task_specs = [
            spec for spec in task_specs
            if any(spec.task_id.startswith(prefix) for prefix in prefixes)
        ]
        logger.info("[gem] selected task prefixes: %s", ", ".join(prefixes))

    excluded = cfg.get("exclude_tasks", [])
    if not isinstance(excluded, list) or any(
        not isinstance(task_id, str) or not task_id.strip()
        for task_id in excluded
    ):
        raise ValueError("gem exclude_tasks must be a list of nonempty task IDs")
    if len(set(excluded)) != len(excluded):
        raise ValueError("gem exclude_tasks must not contain duplicates")
    if excluded:
        available_ids = {spec.task_id for spec in task_specs}
        unknown = sorted(set(excluded) - available_ids)
        if unknown:
            raise ValueError(f"gem exclude_tasks contains unknown task IDs: {unknown}")
        excluded_ids = set(excluded)
        task_specs = [spec for spec in task_specs if spec.task_id not in excluded_ids]
        logger.info(
            "[gem] excluded %d configured task(s): %s",
            len(excluded_ids), ", ".join(excluded),
        )

    if not task_specs:
        logger.warning("[gem] no tasks in config — nothing to run")
        return {"gem_n_tasks": 0}

    prior_result = None
    resume_value = cfg.get("resume_raw_result")
    all_task_specs = task_specs
    if resume_value:
        resume_path = _resolve_config_path(str(resume_value))
        prior_result = _load_prior_result(resume_path)
        prior_by_key = {
            (item.task_id, item.metric_suffix): item
            for item in prior_result.per_task_results
        }
        task_specs = [
            replace(
                spec,
                episodes=(
                    prior_by_key[(spec.task_id, spec.metric_suffix)].errors
                    if (spec.task_id, spec.metric_suffix) in prior_by_key
                    else spec.episodes
                ),
            )
            for spec in task_specs
            if (
                (spec.task_id, spec.metric_suffix) not in prior_by_key
                or prior_by_key[(spec.task_id, spec.metric_suffix)].errors > 0
            )
        ]
        logger.info(
            "[gem] resume loaded %d prior tasks; retrying %d failed episode(s) across %d task(s)",
            len(prior_result.per_task_results),
            sum(spec.episodes for spec in task_specs),
            len(task_specs),
        )

    # Fail fast on missing external prerequisites. The asset-backed envs load
    # their data in __init__ (GPQA pulls the gated Idavidrein/gpqa dataset,
    # LiveCodeBench needs LCB_OFFICIAL_DIR and the release_v6 jsonl), and a
    # task whose episodes all error would otherwise be scored as a
    # real-looking 0.0 win rate.
    import gem as _gem
    for task_id in dict.fromkeys(spec.task_id for spec in task_specs):
        try:
            _gem.make(task_id)
        except Exception as e:
            raise RuntimeError(
                f"[gem] cannot construct env {task_id!r}: {e}. Fix the missing "
                "prerequisite (eval_offline/README.md, Data setup) and rerun."
            ) from e

    tokenizer_path = getattr(client, "model_path", None) or client.model
    adapter = OfflineModelAdapter(client, model_path=tokenizer_path)

    configured_concurrent = cfg.get("max_concurrent", defaults.max_concurrent)
    # Do not let the outer GEM worker pool exceed the HTTP client's admission
    # limit. Otherwise workers can spend most of their generation timeout
    # waiting for an inner semaphore slot rather than generating.
    max_concurrent = min(configured_concurrent, client.max_concurrent)
    if max_concurrent != configured_concurrent:
        logger.warning(
            "[gem] capping max_concurrent from %d to client limit %d",
            configured_concurrent,
            client.max_concurrent,
        )
    evaluator = GemEvaluator(
        model=adapter,
        max_concurrent=max_concurrent,
        generation_timeout_seconds=client.request_timeout_seconds,
    )

    logger.info(
        "[gem] running %d tasks (defaults episodes=%d max_turns=%d, "
        "max_concurrent=%d, generation_timeout=%.0fs)",
        len(task_specs), defaults.episodes, defaults.max_turns, max_concurrent,
        client.request_timeout_seconds,
    )
    retry_result = asyncio.run(evaluator.evaluate_all(task_specs))
    if prior_result is not None:
        (out_dir / "retry_raw_result.json").write_text(
            json.dumps(asdict(retry_result), indent=2, default=str)
        )
        eval_result = _merge_retry_result(prior_result, retry_result, all_task_specs)
    else:
        eval_result = retry_result

    try:
        (out_dir / "raw_result.json").write_text(
            json.dumps(asdict(eval_result), indent=2, default=str)
        )
    except Exception as e:  # pragma: no cover
        logger.warning("[gem] could not serialize raw result: %s", e)

    # A task with zero completed episodes has no score; refusing to emit one
    # keeps a runtime failure (dead endpoint, mid-run asset loss) from
    # rendering as 0.0 in the paper table. raw_result.json above still holds
    # the per-episode errors for debugging.
    failed = [
        f"{r.task_id}{r.metric_suffix} ({r.errors} errors)"
        for r in eval_result.per_task_results
        if r.num_episodes == 0
    ]
    if failed:
        raise RuntimeError(
            "[gem] every episode failed for: " + ", ".join(failed)
            + ". No scores.json written; see raw_result.json for details."
        )

    metrics: dict[str, Any] = dict(eval_result.to_metrics_dict(prefix="gem_eval"))
    (out_dir / "scores.json").write_text(json.dumps(metrics, indent=2))

    return metrics
