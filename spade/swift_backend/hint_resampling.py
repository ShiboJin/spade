"""Whole-group EnvDuels hint escalation, independent of Torch/ms-swift imports.

All ranks enter every generation round, including ranks with no local retries.
Selection happens before reward scoring, encoding and policy log-prob computation.
The selected samples retain their complete hinted message histories and token IDs.
"""
from copy import deepcopy
import hashlib
import json
import uuid


class RefillUnavailable(RuntimeError):
    """No other fixed-pool environments are available for this refill."""


class SkipHintBatch(RuntimeError):
    """Bounded SAGE sampling could not fill a complete trainable window."""


def env_config(sample):
    extra = sample.extra
    # Match OnPolicySample.to_rollout_request: nested data_dict takes precedence.
    nested = extra.get("data_dict") or {}
    return nested.get("env_config", extra.get("env_config"))


def set_hint_level(sample, level):
    config = env_config(sample)
    if not isinstance(config, dict) or config.get("name", "envduels") != "envduels":
        raise ValueError("Hint resampling requires EnvDuels env_config on every sample")
    config["hint_level"] = level


def reward(sample):
    value = sample.rollout_infos.get("total_reward")
    if value not in (0, 1):
        raise ValueError(f"Expected unchanged binary EnvDuels terminal reward, got {value!r}")
    return int(value)


def resample_with_hints(samples, *, generate, gather, group_size, hints_for, metrics=None, audit=None):
    """Retry an all-zero group once with the first author hint; retain whole groups.

    ``gather`` follows accelerate.gather_object's rank-ordered list concatenation.
    ``generate`` accepts uneven/empty local lists and returns one sample per input.
    """
    if type(group_size) is not int or group_size < 2:
        raise ValueError("Hint resampling requires group_size >= 2")
    originals = deepcopy(samples)
    descriptors = []
    # Gather validation errors too, so one rank cannot leave others in collectives.
    for sample in originals:
        try:
            set_hint_level(sample, 0)
            config = env_config(sample)
            levels = tuple(hints_for(config))
            if not levels or any(not isinstance(h, str) or not h.strip() for h in levels):
                raise ValueError("Every training environment must provide author hint(s)")
            descriptors.append(dict(
                request_id=sample.request_id,
                context=json.dumps([config, sample.messages], sort_keys=True),
                hints=[hashlib.sha256(h.encode()).hexdigest() for h in levels],
                env_id=config["env_id"], seed=config["seed"], error=None))
        except Exception as exc:
            descriptors.append(dict(error=str(exc)))
    global_rows = gather(descriptors)
    errors = [r["error"] for r in global_rows if r["error"]]
    if errors:
        raise ValueError(f"Invalid hint-resampling inputs: {errors[0]}")
    if not global_rows or len(global_rows) % group_size:
        raise ValueError("Rollout batch must contain complete GRPO groups")
    ids = [r["request_id"] for r in global_rows]
    if any(not rid for rid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Rollout requests require unique nonempty IDs")
    groups = [global_rows[i:i + group_size] for i in range(0, len(global_rows), group_size)]
    for group in groups:
        if any((r["context"], r["hints"]) != (group[0]["context"], group[0]["hints"]) for r in group):
            raise ValueError("Each GRPO group must share the same env, seed, prompt and author hints")
    group_for = {r["request_id"]: g for g, group in enumerate(groups) for r in group}
    local_groups = [group_for[s.request_id] for s in originals]
    # The dataset uses the same placeholder prompt for every environment; explicitly
    # identify real groups so request-aware GRPO never merges different games/levels.
    for sample, g in zip(originals, local_groups):
        sample.prompt_id = f"envduels_group_{g}_hint_0"

    def run_round(inputs, slots, level):
        generated = generate(inputs)  # All ranks, even when inputs == [].
        local = []
        error = None
        try:
            if len(generated) != len(inputs):
                raise ValueError("Hint resampling requires one trajectory per input")
            for slot, expected, actual in zip(slots, inputs, generated):
                if actual.request_id != expected.request_id:
                    raise ValueError("Rollout changed request order/identity")
                if env_config(actual) != env_config(expected):
                    raise ValueError("Rollout changed the selected env/seed/hint context")
                if level:
                    hint = hints_for(env_config(expected))[level - 1]
                    if not any(m.get("role") == "user" and isinstance(m.get("content"), str)
                               and "\nPlayer hint:\n" + hint + "\n" in m["content"] for m in actual.messages):
                        raise ValueError("Hint missing from selected training messages")
                local.append(dict(group=local_groups[slot], reward=reward(actual)))
        except Exception as exc:
            error = str(exc)
        gathered = gather([dict(error=error, rows=local)])
        errors = [r["error"] for r in gathered if r["error"]]
        if errors:
            raise ValueError(f"Invalid hint rollout: {errors[0]}")
        by_group = {}
        for rank in gathered:
            for row in rank["rows"]:
                by_group.setdefault(row["group"], []).append(row["reward"])
        if any(len(values) != group_size for values in by_group.values()):
            raise ValueError("A hint retry produced an incomplete group")
        return generated, by_group

    selected, initial_rewards = run_round(deepcopy(originals), list(range(len(originals))), 0)
    latest = dict(initial_rewards)
    selected_levels = [0] * len(groups)
    attempts = [[dict(hint_level=0, rewards=initial_rewards[g])] for g in range(len(groups))]
    # One hinted retry, even when the export also provides a stronger hint_2.
    for level in (1,):
        pending = {g for g in latest if not any(latest[g]) and level <= len(groups[g][0]["hints"])}
        if not pending:
            break
        slots = [i for i, g in enumerate(local_groups) if g in pending]
        retry = []
        for i in slots:
            sample = deepcopy(originals[i])  # Fresh env and empty trajectory; same seed.
            set_hint_level(sample, level)
            sample.request_id = "chatcmpl-" + uuid.uuid4().hex
            sample.prompt_id = f"envduels_group_{local_groups[i]}_hint_{level}"
            retry.append(sample)
        replacements, results = run_round(retry, slots, level)
        if set(results) != pending:
            raise ValueError("Hint retry did not cover exactly the selected groups")
        for i, sample in zip(slots, replacements):
            selected[i] = sample
        for g, values in results.items():
            latest[g] = values
            selected_levels[g] = level
            attempts[g].append(dict(hint_level=level, rewards=values))

    for sample, g in zip(selected, local_groups):
        sample.rollout_infos.update(sage_hint_level=selected_levels[g],
                                   sage_no_hint_rewards=initial_rewards[g],
                                   sage_attempts=attempts[g])
    count = len(groups)
    stats = {
        "no_hint_success_fraction": sum(any(v) for v in initial_rewards.values()) / count,
        "hinted_group_fraction": sum(level > 0 for level in selected_levels) / count,
        "rescued_group_fraction": sum(not any(initial_rewards[g]) and any(latest[g]) for g in latest) / count,
        "rescued_trainable_group_fraction": sum(
            not any(initial_rewards[g]) and 0 < sum(latest[g]) < group_size for g in latest) / count,
        "constant_group_fraction": sum(len(set(v)) == 1 for v in latest.values()) / count,
        "exhausted_group_fraction": sum(not any(v) for v in latest.values()) / count,
        "selected_hint_level_mean": sum(selected_levels) / count,
        "rollout_groups_per_selected_group": sum(map(len, attempts)) / count,
    }
    if metrics is not None:
        metrics(stats)
    if audit is not None:
        audit([dict(env_id=group[0]["env_id"], seed=group[0]["seed"],
                    group_size=group_size, hint_sha256=group[0]["hints"],
                    selected_hint_level=selected_levels[g], attempts=attempts[g])
               for g, group in enumerate(groups)])
    return selected


def refill_constant_groups(samples, *, sage_generate, gather, group_size, max_attempts,
                           refill, process_index=0, metrics=None, min_valid_groups=None, audit=None):
    """Fill the original batch slots with complete nonconstant-reward groups.

    Each attempt is a no-hint-first SAGE pass. max_attempts includes the initial
    pass. At exhaustion, accept at least min_valid_groups (default: all groups).
    Preserve physical slots, marking invalid groups with zero training weight.
    Below the threshold, raise SkipHintBatch on every rank. No cross-window cache.
    ``refill`` receives all failed group IDs and current group env IDs, and returns
    one fresh sample prototype per failed group, identically on every rank.
    """
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_rollout_attempts must be positive")
    lengths = gather([len(samples)])
    if not sum(lengths) or sum(lengths) % group_size:
        raise ValueError("Cannot refill incomplete GRPO groups")
    total_groups = sum(lengths) // group_size
    if min_valid_groups is None:
        min_valid_groups = total_groups
    if type(min_valid_groups) is not int or not 1 <= min_valid_groups <= total_groups:
        raise ValueError("min_valid_groups must be between 1 and the requested group count")
    offset = sum(lengths[:process_index])
    groups = [(offset + i) // group_size for i in range(len(samples))]
    pending = set(range(sum(lengths) // group_size))
    selected = [None] * len(samples)
    inputs = deepcopy(samples)
    discarded = 0

    def finish(reason, attempt):
        valid_groups = total_groups - len(pending)
        accepted = valid_groups >= min_valid_groups
        if metrics is not None:
            metrics(dict(refill_attempts=attempt, discarded_groups=discarded,
                         valid_groups=valid_groups, effective_trajectories=valid_groups * group_size if accepted else 0,
                         masked_groups=len(pending), partial_batch=float(accepted and bool(pending))))
        if audit is not None:
            audit(dict(event="batch_selection", valid_groups=valid_groups, requested_groups=total_groups,
                       min_valid_groups=min_valid_groups, valid_group_ids=sorted(set(range(total_groups)) - pending),
                       masked_group_ids=sorted(pending), accepted=accepted, reason=reason,
                       refill_attempt=attempt))
        if not accepted:
            raise SkipHintBatch(f"{reason} Valid groups={valid_groups}/{total_groups}; "
                                f"min_valid_groups={min_valid_groups}.")
        for sample, g in zip(selected, groups):
            sample.prompt_id = f"envduels_selected_group_{g}_hint_{env_config(sample).get('hint_level', 0)}"
            sample.rollout_infos.update(sage_valid_group=g not in pending,
                                       sage_valid_groups=valid_groups, sage_requested_groups=total_groups)
        return selected

    for attempt in range(max_attempts):
        slots = [i for i, g in enumerate(groups) if g in pending]
        outputs = sage_generate(inputs, attempt)
        if len(outputs) != len(slots):
            raise ValueError("SAGE changed local rollout batch size")
        for i, sample in zip(slots, outputs):
            selected[i] = sample
        summaries = gather([dict(group=g, reward=reward(sample),
                                 env_id=env_config(sample)["env_id"],
                                 seed=env_config(sample)["seed"],
                                 level=env_config(sample).get("hint_level", 0))
                            for g, sample in zip(groups, selected)])
        grouped = {}
        for row in summaries:
            grouped.setdefault(row["group"], []).append(row)
        for rows in grouped.values():
            if len(rows) != group_size or len({(r["env_id"], r["seed"], r["level"]) for r in rows}) != 1:
                raise ValueError("Refill mixed environments, seeds or hint levels in a GRPO group")
        pending = {g for g, rows in grouped.items() if len({r["reward"] for r in rows}) == 1}
        discarded += len(pending)
        if not pending:
            return finish("All requested groups are valid.", attempt)
        if attempt + 1 == max_attempts:
            return finish(
                f"EnvDuels hint resampling exhausted max_rollout_attempts={max_attempts}; "
                f"{len(pending)} constant-reward groups remain.", attempt)
        current_envs = {g: rows[0]["env_id"] for g, rows in grouped.items()}
        # The callback reads only fixed-pool rows; it never edits/removes them.
        try:
            prototypes = refill(sorted(pending), current_envs, attempt + 1)
        except RefillUnavailable as exc:
            return finish(str(exc), attempt)
        inputs = []
        for i, g in enumerate(groups):
            if g in pending:
                sample = deepcopy(prototypes[g])
                sample.request_id = "chatcmpl-" + uuid.uuid4().hex
                inputs.append(sample)
    raise AssertionError("unreachable")
