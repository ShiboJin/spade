"""Install whole-group hint rescue and bounded fixed-pool refill for ms-swift."""
from copy import deepcopy
from functools import wraps
import json
import logging
import os
from pathlib import Path
import random
import weakref

from spade.swift_backend.hint_resampling import (
    RefillUnavailable, SkipHintBatch, env_config, refill_constant_groups, resample_with_hints, set_hint_level,
)
from spade.swift_backend.envduels_sage_loss import install_effective_batch_loss


_SKIP_UPDATE_OWNERS = weakref.WeakKeyDictionary()


def has_teacher_input(sample):
    # Match OnPolicySample.build_teacher_view without mutating the trajectory.
    return bool(getattr(sample, "teacher_prompt", None)) or any(
        getattr(sample, key, None) is not None for key in ("teacher_images", "teacher_messages"))


def runtime_conflicts(trainer, is_global_inputs=False):
    """Report actual incompatible settings, not Swift's teacher capability flag."""
    checks = [
        ("is_global_inputs", is_global_inputs, False),
        ("async_generate", getattr(trainer, "async_generate", False), False),
        ("dynamic_num_samples", getattr(trainer, "dynamic_num_samples", False), False),
        ("dynamic_sample", getattr(trainer, "dynamic_sample", False), False),
        ("num_iterations", trainer.num_iterations, 1),
        ("steps_per_generation", trainer.args.steps_per_generation, trainer.args.gradient_accumulation_steps),
        ("loss_type", trainer.loss_type, "grpo"),
        ("use_liger_loss", trainer.use_liger_loss, False),
        ("kl_in_reward", trainer.kl_in_reward, False),
        ("chord_sft_iterator", trainer.chord_sft_iterator is not None, False),
        ("sequence_parallel_size", trainer.template.sequence_parallel_size, 1),
        ("use_gym_env", trainer.use_gym_env, True),
        ("vllm_mode", trainer.vllm_mode, "colocate"),
    ]
    conflicts = [f"{name}={actual!r} (required {expected!r})"
                 for name, actual, expected in checks if actual != expected]
    if trainer.scale_rewards not in ("none", "group"):
        conflicts.append(f"scale_rewards={trainer.scale_rewards!r} (required 'none' or 'group')")
    # Recent Swift sets _has_teacher=True even for plain reward GRPO: it means
    # dynamic OPSD is available, and only activates when samples carry teacher input.
    explicit = getattr(trainer, "_has_teacher_explicit", None)
    if (explicit is not None and explicit()) or any(
            getattr(trainer, key, None) is not None for key in ("_teacher_model", "teacher_model_server")) or (
            getattr(trainer, "_teacher_use_disable_adapter", False) or getattr(trainer, "use_teacher_api", False)):
        conflicts.append("explicit_teacher=True (required False)")
    return conflicts


def _guard_update_step(component, trainer):
    if component is None:
        return
    _SKIP_UPDATE_OWNERS[component] = weakref.ref(trainer)
    cls = type(component)
    original = cls.step
    if getattr(original, "_envduels_sage_skip", False):
        return

    @wraps(original)
    def step(self, *args, **kwargs):
        owner = _SKIP_UPDATE_OWNERS.get(self)
        trainer = owner() if owner is not None else None
        if trainer is not None and getattr(trainer, "_sage_skip_update", False):
            return None
        return original(self, *args, **kwargs)

    # Patch the method, not instance attributes: PyTorch scheduler.state_dict()
    # serializes instance attributes, which must not capture a Trainer or closure.
    step._envduels_sage_skip = True
    cls.step = step


def install_skip_updates(trainer_class):
    """Consume failed rollout windows without backward, AdamW or scheduler steps."""
    original = trainer_class.training_step
    if getattr(original, "_envduels_sage_skip", False):
        return

    @wraps(original)
    def training_step(self, model, inputs, *args, **kwargs):
        import torch

        # Trainer/Accelerate have prepared these objects by the first microstep.
        # Returning zero loss alone would still execute AdamW decay and advance LR.
        for component in (self.optimizer, self.lr_scheduler):
            _guard_update_step(component, self)

        if getattr(self, "_sage_skip_remaining", 0):
            self._sage_skip_remaining -= 1
            self._step += 1
            return torch.zeros((), device=self.accelerator.device)
        self._sage_skip_update = False
        try:
            result = original(self, model, inputs, *args, **kwargs)
        except SkipHintBatch as exc:
            accumulation = self.args.gradient_accumulation_steps
            if (self._step % accumulation or
                    getattr(self, "current_gradient_accumulation_steps", accumulation) != accumulation):
                raise RuntimeError("Cannot skip a partially accumulated SAGE window") from exc
            model.zero_grad(set_to_none=True)
            self._buffered_inputs = None
            self._sage_skip_update = True
            self._sage_skip_remaining = accumulation - 1
            self._step += 1
            self._metrics["train"]["sage/skipped_update"].append(1.0)
            self._metrics["train"]["sage/applied_update"].append(0.0)
            if self.accelerator.is_main_process:
                logging.getLogger(__name__).warning("Skipping SAGE update: %s", exc)
                with (Path(self.args.output_dir) / "hint_resampling.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(global_step=self.state.global_step,
                                                 rollout_step=self._step - 1,
                                                 event="skipped_update", reason=str(exc))) + "\n")
            return torch.zeros((), device=self.accelerator.device)
        if self._step % self.args.gradient_accumulation_steps == 0:
            self._metrics["train"]["sage/skipped_update"].append(0.0)
            self._metrics["train"]["sage/applied_update"].append(1.0)
        return result

    training_step._envduels_sage_skip = True
    trainer_class.training_step = training_step


def install_hint_resampling(trainer_class, gather):
    install_skip_updates(trainer_class)
    install_effective_batch_loss(trainer_class, gather)
    original = trainer_class._infer_single_or_multi_turn
    if getattr(original, "_envduels_sage", False):
        return

    @wraps(original)
    def infer(self, samples, request_config, is_global_inputs=False):
        from spade.swift_backend.envduels_gym import _get_adapter

        def hints_for(config):
            return _get_adapter(config["export_dir"]).get_hint_levels(config["env_id"])

        # Evaluation always starts without privileged context, even if a cached
        # row carries a hint level from a prior training rollout.
        samples = deepcopy(samples)
        for sample in samples:
            set_hint_level(sample, 0)
        if not self.model.training:
            return original(self, samples, request_config, is_global_inputs)
        conflicts = runtime_conflicts(self, is_global_inputs)
        if any(has_teacher_input(sample) for sample in samples):
            conflicts.append("teacher_input=True (teacher_prompt/images/messages are unsupported)")
        # A conflict on one rank must stop all ranks before they enter rollout.
        conflicts = sorted(set(gather(conflicts)))
        if conflicts:
            raise ValueError("EnvDuels SAGE incompatible runtime settings: " + "; ".join(conflicts))

        def metrics(values):
            for key, value in values.items():
                self._metrics["train"]["sage/" + key].append(value)

        pool = None
        # Exclusions are batch-local only. Every future dataloader pass keeps the
        # full fixed pool, including environments which were just discarded.
        tried = set()

        def refill(pending, current_envs, attempt):
            nonlocal pool
            tried.update(current_envs.values())
            local_error = None
            try:
                if pool is None:
                    pool = {}
                    for i in range(len(self.train_dataset)):
                        prototype = self.to_samples([self.train_dataset[i]])[0]
                        if has_teacher_input(prototype):
                            raise ValueError("SAGE refill dataset contains unsupported teacher input")
                        cfg = env_config(prototype)
                        if not isinstance(cfg, dict) or cfg["env_id"] in pool:
                            raise ValueError("SAGE refill requires one fixed dataset row per environment")
                        pool[cfg["env_id"]] = prototype
                candidates = sorted(set(pool) - tried)
                if len(candidates) < len(pending):
                    # Reuse earlier failed environments only after trying the rest
                    # of the pool, but never duplicate a current batch environment.
                    candidates = sorted(set(pool) - set(current_envs.values()))
                if len(candidates) < len(pending):
                    raise ValueError("Not enough other fixed-pool environments to refill this batch")
                random.Random(f"{self.args.seed}:{self.state.global_step}:{self._step}:{attempt}").shuffle(candidates)
                chosen = dict(zip(pending, candidates))
            except Exception as exc:
                local_error = str(exc)
                chosen = {}
            records = gather([dict(error=local_error, chosen=chosen)])
            if any(r["error"] for r in records):
                message = [r['error'] for r in records if r['error']][0]
                if message.startswith("Not enough other fixed-pool"):
                    raise RefillUnavailable(message)
                raise ValueError(f"SAGE refill failed: {message}")
            if any(r["chosen"] != chosen for r in records):
                raise ValueError("Fixed-pool refill choices differ across ranks")
            return {g: deepcopy(pool[env_id]) for g, env_id in chosen.items()}

        def sage_generate(inputs, attempt):
            def audit(records):
                if not self.accelerator.is_main_process:
                    return
                out = Path(self.args.output_dir) / "hint_resampling.jsonl"
                with out.open("a", encoding="utf-8") as stream:
                    for record in records:
                        final = record["attempts"][-1]["rewards"]
                        record.update(global_step=self.state.global_step,
                                      rollout_step=self._step, refill_attempt=attempt,
                                      discarded=len(set(final)) == 1)
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")

            return resample_with_hints(
                inputs, generate=lambda batch: original(self, batch, request_config, False),
                gather=gather, group_size=self.num_generations, hints_for=hints_for,
                metrics=metrics, audit=audit)

        def selection_audit(record):
            if self.accelerator.is_main_process:
                record.update(global_step=self.state.global_step, rollout_step=self._step)
                with (Path(self.args.output_dir) / "hint_resampling.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")

        return refill_constant_groups(
            samples, sage_generate=sage_generate, gather=gather, group_size=self.num_generations,
            max_attempts=int(os.environ.get("SPADE_MAX_ROLLOUT_ATTEMPTS", "2")),
            min_valid_groups=int(os.environ.get("SPADE_MIN_VALID_GROUPS", "4")),
            refill=refill, process_index=self.accelerator.process_index, metrics=metrics, audit=selection_audit)

    infer._envduels_sage = True
    trainer_class._infer_single_or_multi_turn = infer


if os.environ.get("SPADE_SAGE_HINT_RESAMPLING", "false").lower() == "true":
    from accelerate.utils import gather_object
    from swift.rlhf_trainers.grpo_trainer import GRPOTrainer

    install_hint_resampling(GRPOTrainer, gather_object)
