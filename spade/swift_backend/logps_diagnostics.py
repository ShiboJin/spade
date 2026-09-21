"""Diagnose old/current policy scoring without changing the training objective."""
from functools import wraps
import hashlib
import json
import logging
import os
from pathlib import Path


def _snapshot_value(value):
    """Copy only tensor/plain-data inputs so snapshots load with weights_only=True."""
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _snapshot_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_snapshot_value(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported logps snapshot input: {type(value).__name__}")


def input_fingerprint(inputs):
    """Fingerprint encoded context, masks and positions, not decoded text."""
    import torch
    digest = hashlib.sha256()
    for key in sorted(inputs):
        value = inputs[key]
        if isinstance(value, torch.Tensor):
            digest.update(str((key, tuple(value.shape), value.dtype)).encode())
            # dtype views require at least one dimension when element sizes
            # differ. Preserve the original shape above, then flatten for bytes.
            digest.update(value.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes())
        elif isinstance(value, (bool, int, float, str, type(None))):
            digest.update(repr((key, value)).encode())
    return digest.hexdigest()


def summarize_logps(current, old, advantages, mask, *, epsilon_low, epsilon_high):
    """Use the loss's native subtraction/exp dtype; exclude masked tokens."""
    import torch
    with torch.no_grad():
        delta = current.detach() - old.detach()
        ratios = delta.exp()
        policy = -torch.minimum(ratios * advantages,
                                ratios.clamp(1 - epsilon_low, 1 + epsilon_high) * advantages)
        valid = mask.bool()
        result = dict(valid_tokens=int(valid.sum()), old_dtype=str(old.dtype), current_dtype=str(current.dtype))
        for name, values in (("log_ratio", delta), ("ratio", ratios), ("advantage", advantages)):
            selected = values[valid].double()
            finite = selected[torch.isfinite(selected)]
            result[name + "_nonfinite"] = int(selected.numel() - finite.numel())
            if finite.numel():
                result[name + "_min"] = finite.min().item()
                result[name + "_max"] = finite.max().item()
                result[name + "_abs_mean"] = finite.abs().mean().item()
                result[name + "_p99"] = torch.quantile(finite, 0.99).item()
        # Preserve physical-slot averaging, matching standard GRPO before SAGE scaling.
        loss = (policy.double().masked_fill(~valid, 0).sum(-1) / valid.sum(-1).clamp(min=1)).mean()
        result["policy_loss_before_sage_scale"] = loss.item() if torch.isfinite(loss) else None
        # Include the token most responsible for a large positive policy loss.
        if valid.any():
            contributions = policy.double() / valid.sum(-1).clamp(min=1)[:, None] / valid.shape[0]
            index = contributions.masked_fill(~valid, float("-inf")).flatten().argmax().item()
            row, col = divmod(index, valid.shape[1])
            result["worst_token"] = dict(row=row, completion_offset=col)
            for name, value in (("old_logp", old[row, col]), ("current_logp", current[row, col]),
                                ("advantage", advantages[row, col]), ("ratio", ratios[row, col]),
                                ("loss_contribution", contributions[row, col])):
                result["worst_token"][name] = value.item() if torch.isfinite(value) else None
        return result


def install_logps_diagnostics(trainer_class):
    """Write rank-local diagnostics during the first update; no extra forward pass."""
    original = getattr(trainer_class, "_get_per_token_logps_and_entropies", None)
    if original is None or getattr(original, "_spade_logps_diagnostics", False):
        return

    @wraps(original)
    def get_logps(self, model, model_inputs, grpo_batch, *args, **kwargs):
        result = original(self, model, model_inputs, grpo_batch, *args, **kwargs)
        if not self.model.training or self.state.global_step != 0 or not getattr(self.args, "output_dir", None):
            return result
        import torch
        old = grpo_batch.old_per_token_logps
        # Old scoring happens before postprocessing, with no gradients.
        if old is None:
            grpo_batch._spade_old_input_hash = input_fingerprint(model_inputs)
            # Opt-in debugging only: retain exact encoded inputs, not a lossy
            # re-tokenization of logged prompt/completion strings.
            if os.environ.get("SPADE_LOGPS_SAVE_INPUTS", "false").lower() == "true":
                index = getattr(self, "_spade_logps_snapshot_index", 0)
                self._spade_logps_snapshot_index = index + 1
                directory = Path(self.args.output_dir) / "logps_inputs"
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"rank{self.accelerator.process_index}.{index:04d}.pt"
                torch.save(dict(model_inputs=_snapshot_value(model_inputs),
                                logits_to_keep=grpo_batch.logits_to_keep,
                                old_logps=_snapshot_value(result[0]),
                                input_hash=grpo_batch._spade_old_input_hash,
                                temperature=self.temperature), path)
                grpo_batch._spade_logps_snapshot_path = str(path)
        elif torch.is_grad_enabled():
            mask = grpo_batch.completion_mask.clone()
            if self.overlong_filter and grpo_batch.truncated_mask is not None:
                mask &= ~grpo_batch.truncated_mask[:, None]
            record = summarize_logps(result[0], old, grpo_batch.advantages, mask,
                                     epsilon_low=self.epsilon_low, epsilon_high=self.epsilon_high)
            old_hash = getattr(grpo_batch, "_spade_old_input_hash", None)
            record.update(global_step=self.state.global_step, rollout_step=getattr(self, "_step", None),
                          rank=self.accelerator.process_index, beta=self.beta,
                          importance_sampling_level=self.importance_sampling_level,
                          rollout_importance_sampling_mode=self.rollout_importance_sampling_mode,
                          input_matches_old=(old_hash == input_fingerprint(model_inputs)) if old_hash else None,
                          input_snapshot=getattr(grpo_batch, "_spade_logps_snapshot_path", None),
                          samples=getattr(grpo_batch, "_spade_diagnostic_samples", []))
            path = Path(self.args.output_dir) / f"logps_diagnostics.rank{self.accelerator.process_index}.jsonl"
            with path.open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            if record.get("log_ratio_max", 0) > 2 or record["ratio_nonfinite"]:
                logging.getLogger(__name__).warning("Large first-update policy ratio; diagnostics: %s", path)
        return result

    get_logps._spade_logps_diagnostics = True
    trainer_class._get_per_token_logps_and_entropies = get_logps
