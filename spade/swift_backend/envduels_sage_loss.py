"""Keep Swift/FSDP batch shapes while averaging GRPO over valid trajectories."""
from functools import wraps


def install_effective_batch_loss(trainer_class, gather):
    original_postprocess = trainer_class._postprocess_batch
    if getattr(original_postprocess, "_envduels_sage_mask", False):
        return
    original_loss = trainer_class.compute_loss

    @wraps(original_postprocess)
    def postprocess(self, samples, batches):
        original_postprocess(self, samples, batches)
        if not self.model.training:
            return
        import torch

        rows = gather([dict(valid=s.rollout_infos["sage_valid_group"],
                            valid_groups=s.rollout_infos["sage_valid_groups"],
                            requested_groups=s.rollout_infos["sage_requested_groups"],
                            reward=s.rollout_infos["total_reward"]) for s in samples])
        total = len(rows)
        effective = sum(r["valid"] for r in rows)
        if (not effective or any(r["valid_groups"] * self.num_generations != effective
                                 or r["requested_groups"] * self.num_generations != total for r in rows)):
            raise ValueError("SAGE effective batch counts disagree across ranks")
        for start in range(0, total, self.num_generations):
            if len({r["valid"] for r in rows[start:start + self.num_generations]}) != 1:
                raise ValueError("SAGE must mask whole groups identically on every rank")
        chunks = self.split_by_mini_batches(samples)
        if len(chunks) != len(batches):
            raise ValueError("SAGE mask does not match Swift micro-batches")
        for chunk, batch in zip(chunks, batches):
            grpo = batch["grpo_batch"]
            valid = torch.tensor([s.rollout_infos["sage_valid_group"] for s in chunk],
                                 device=grpo.completion_mask.device, dtype=torch.bool)
            if len(valid) != grpo.completion_mask.shape[0]:
                raise ValueError("SAGE mask does not match encoded trajectories")
            grpo.completion_mask = grpo.completion_mask & valid[:, None]
            # Explicitly exclude invalid advantages as well as policy/KL tokens.
            grpo.advantages = grpo.advantages.masked_fill(~valid[:, None], 0.0)
            # Swift averages each micro-batch over its physical slots; DDP/FSDP
            # then averages ranks, and Trainer averages accumulation microsteps.
            # Multiplying by total/effective makes the result the exact mean over
            # valid trajectories, including ranks/microsteps with zero valid rows.
            batch["_sage_loss_scale"] = total / effective
        self._metrics["train"]["sage/effective_reward"].append(
            sum(r["reward"] for r in rows if r["valid"]) / effective)

    @wraps(original_loss)
    def compute_loss(self, model, inputs, *args, **kwargs):
        result = original_loss(self, model, inputs, *args, **kwargs)
        if not self.model.training:
            return result
        batch = inputs[0] if isinstance(inputs, list) and len(inputs) == 1 else inputs
        if "_sage_loss_scale" not in batch:
            raise ValueError("SAGE training batch is missing its effective-loss normalization")
        return result * batch["_sage_loss_scale"]

    postprocess._envduels_sage_mask = True
    trainer_class._postprocess_batch = postprocess
    trainer_class.compute_loss = compute_loss
