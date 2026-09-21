"""Attach SAGE diagnostics while preserving Swift's full-batch GRPO objective."""
from functools import wraps

from spade.swift_backend.hint_resampling import env_config
from spade.swift_backend.logps_diagnostics import install_logps_diagnostics


def install_effective_batch_loss(trainer_class, gather):
    # Retain the installer name for callers; no mask or loss scaling is installed.
    install_logps_diagnostics(trainer_class)
    original_postprocess = trainer_class._postprocess_batch
    if getattr(original_postprocess, "_envduels_sage_diagnostics", False):
        return

    @wraps(original_postprocess)
    def postprocess(self, samples, batches):
        original_postprocess(self, samples, batches)
        if not self.model.training:
            return
        chunks = self.split_by_mini_batches(samples)
        if len(chunks) != len(batches):
            raise ValueError("SAGE diagnostics do not match Swift micro-batches")
        for chunk, batch in zip(chunks, batches):
            grpo = batch["grpo_batch"]
            grpo._spade_diagnostic_samples = [dict(request_id=s.request_id,
                                                  env_config=env_config(s),
                                                  reward=s.rollout_infos["total_reward"],
                                                  valid=s.rollout_infos["sage_valid_group"])
                                              for s in chunk]
        rewards = gather([s.rollout_infos["total_reward"] for s in samples])
        self._metrics["train"]["sage/effective_reward"].append(sum(rewards) / len(rewards))

    postprocess._envduels_sage_diagnostics = True
    trainer_class._postprocess_batch = postprocess
