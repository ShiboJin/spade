"""Token log-probabilities without retaining sequence-by-vocabulary tensors."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _project_logps(hidden, weight, bias, targets, temperature, compute_entropy):
    logits = F.linear(hidden, weight, bias)
    logits.div_(temperature)
    # Match TRL's selective_log_softmax, including its BF16 rounding behavior.
    if logits.dtype in (torch.float32, torch.float64):
        logps = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)
    else:
        logps = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    entropy = None
    if compute_entropy:
        all_logps = F.log_softmax(logits, dim=-1)
        entropy = -(all_logps.exp() * all_logps).sum(-1)
    return logps, entropy


def chunked_linear_logps(hidden, weight, targets, *, bias=None, temperature=1.0,
                         chunk_size=128, compute_entropy=False):
    """Project and normalize complete vocabulary distributions in token chunks.

    No vocabulary approximation, token truncation, or loss change is made.
    Non-reentrant checkpointing discards each chunk's logits in training and
    recomputes them during backward. Frozen output weights remain valid inputs
    to autograd, so gradients still propagate into the LoRA-equipped backbone.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if hidden.shape[:-1] != targets.shape:
        raise ValueError("hidden and targets must have the same token dimensions")
    shape = targets.shape
    hidden = hidden.reshape(-1, hidden.shape[-1])
    targets = targets.reshape(-1)
    logps, entropies = [], []
    needs_grad = torch.is_grad_enabled() and (
        hidden.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad))
    for start in range(0, targets.numel(), chunk_size):
        args = (hidden[start:start + chunk_size], weight, bias, targets[start:start + chunk_size],
                temperature, compute_entropy)
        if needs_grad:
            values, entropy = checkpoint(_project_logps, *args, use_reentrant=False, preserve_rng_state=False)
        else:
            values, entropy = _project_logps(*args)
        logps.append(values)
        if compute_entropy:
            entropies.append(entropy)
    if not logps:
        empty = hidden.new_empty(shape)
        return empty, empty.clone() if compute_entropy else None
    return (torch.cat(logps).reshape(shape),
            torch.cat(entropies).reshape(shape) if compute_entropy else None)
