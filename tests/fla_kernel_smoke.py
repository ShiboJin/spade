#!/usr/bin/env python3
"""Exercise the Qwen3.5 CUDA kernels with a real forward/backward pass."""

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the FLA kernel smoke test")

    from causal_conv1d import causal_conv1d_fn
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    device = torch.device("cuda")
    dtype = torch.bfloat16

    x = torch.randn(2, 64, 257, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(64, 4, device=device, dtype=dtype, requires_grad=True)
    conv_out = causal_conv1d_fn(x, weight, activation="silu")
    assert conv_out.shape == x.shape
    conv_out.float().square().mean().backward()
    assert x.grad is not None and weight.grad is not None

    shape = (2, 4, 257, 64)
    q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    g = -torch.nn.functional.softplus(
        torch.randn(shape[:-1], device=device, dtype=torch.float32)
    )
    beta = torch.sigmoid(torch.randn(shape[:-1], device=device, dtype=dtype))
    delta_out, final_state = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        output_final_state=True,
    )
    assert delta_out.shape == v.shape
    assert final_state is not None
    delta_out.float().square().mean().backward()
    assert q.grad is not None and k.grad is not None and v.grad is not None

    torch.cuda.synchronize()
    print(
        "PASS",
        torch.cuda.get_device_name(),
        "causal-conv1d forward/backward",
        "FLA gated-delta forward/backward",
    )


if __name__ == "__main__":
    main()
