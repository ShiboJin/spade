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
        if (is_global_inputs or getattr(self, "async_generate", False)
                or getattr(self, "dynamic_num_samples", False)
                or getattr(self, "dynamic_sample", False)
                or self.num_iterations != 1
                or self.args.steps_per_generation != self.args.gradient_accumulation_steps
                or self.loss_type != "grpo" or self.scale_rewards not in ("none", "group")
                or self.use_liger_loss or self.kl_in_reward or self._has_teacher
                or self.chord_sft_iterator is not None
                or self.template.sequence_parallel_size != 1
                or not self.use_gym_env or self.vllm_mode != "colocate"):
            raise ValueError("EnvDuels SAGE requires synchronous colocated Gym GRPO, fixed group sizes, "
                             "sequence_parallel_size=1, dynamic_sample=false, num_iterations=1 and "
                             "steps_per_generation=gradient_accumulation_steps; masked effective batches "
                             "require standard GRPO with group/none reward scaling and no teacher/CHORD/Liger/KL-in-reward")

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
