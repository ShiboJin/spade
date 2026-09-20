"""Bound autograd storage in Transformers' PyTorch gated-delta fallback."""
import torch
from torch.utils.checkpoint import checkpoint


def checkpointed_delta_rule(rule, query, key, value, g, beta, chunk_size=64,
                            initial_state=None, output_final_state=False,
                            use_qk_l2norm_in_kernel=False, **kwargs):
    # Preserve the original algorithm's chunk boundaries and recurrent state.
    # Each segment calls the original implementation; gradients flow through
    # every state boundary (no detach / truncated backpropagation).
    segment_size = chunk_size * 4
    options = dict(chunk_size=chunk_size, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kwargs)
    if query.shape[1] <= segment_size:
        return rule(query, key, value, g, beta, initial_state=initial_state,
                    output_final_state=output_final_state, **options)

    def segment(q, k, v, decay, gate, state):
        return rule(q, k, v, decay, gate, initial_state=state, output_final_state=True, **options)

    outputs = []
    state = initial_state
    # Old-policy scoring and reentrant checkpoint forwards run without grads.
    # They still need bounded FP32 intermediates in the PyTorch fallback.
    grad_enabled = torch.is_grad_enabled()
    result = None
    for start in range(0, query.shape[1], segment_size):
        stop = start + segment_size
        args = (query[:, start:stop], key[:, start:stop], value[:, start:stop],
                g[:, start:stop], beta[:, start:stop], state)
        if grad_enabled:
            output, state = checkpoint(segment, *args, use_reentrant=False, preserve_rng_state=False)
            outputs.append(output)
        else:
            output, state = segment(*args)
            if result is None:
                result = output.new_empty((output.shape[0], query.shape[1], *output.shape[2:]))
            result[:, start:stop].copy_(output)
    if not grad_enabled:
        return result, state if output_final_state else None
    return torch.cat(outputs, dim=1), state if output_final_state else None
