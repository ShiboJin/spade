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
    if not torch.is_grad_enabled() or query.shape[1] <= segment_size:
        return rule(query, key, value, g, beta, initial_state=initial_state,
                    output_final_state=output_final_state, **options)

    def segment(q, k, v, decay, gate, state):
        return rule(q, k, v, decay, gate, initial_state=state, output_final_state=True, **options)

    outputs = []
    state = initial_state
    for start in range(0, query.shape[1], segment_size):
        stop = start + segment_size
        output, state = checkpoint(
            segment, query[:, start:stop], key[:, start:stop], value[:, start:stop],
            g[:, start:stop], beta[:, start:stop], state,
            use_reentrant=False, preserve_rng_state=False)
        outputs.append(output)
    return torch.cat(outputs, dim=1), state if output_final_state else None
