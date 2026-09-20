"""Replay the failing no-grad Qwen delta operation, without model loading or rollout.

Run inside the training image: python3 tests/delta_rule_memory_smoke.py
Uses real model head dimensions and synthetic inputs; not a full training test.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=16384)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--config", type=Path, default=Path("checkpoints/Qwen3.8-27B/config.json"))
    args = parser.parse_args()
    if args.length < 1:
        parser.error("length must be positive")
    import torch
    torch.cuda.set_device(args.device)
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    from spade.swift_backend.chunked_delta_rule import checkpointed_delta_rule

    config = json.loads(args.config.read_text())["text_config"]
    torch.manual_seed(31)
    shape = (1, args.length, config["linear_num_value_heads"])
    q, k = [torch.randn(*shape, config["linear_key_head_dim"], device="cuda", dtype=torch.bfloat16)
            for _ in range(2)]
    v = torch.randn(*shape, config["linear_value_head_dim"], device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(shape, device="cuda")
    beta = torch.rand(shape, device="cuda", dtype=torch.bfloat16)
    reference = None
    reports = []
    with torch.no_grad():
        for name, rule in (("original_no_grad", torch_chunk_gated_delta_rule),
                           ("segmented_no_grad", lambda *a, **kw: checkpointed_delta_rule(
                               torch_chunk_gated_delta_rule, *a, **kw))):
            torch.cuda.empty_cache()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.monotonic()
            output, state = rule(q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True)
            torch.cuda.synchronize()
            reports.append(dict(path=name, length=args.length, seconds=time.monotonic() - started,
                                extra_peak_mib=(torch.cuda.max_memory_allocated() - baseline) / 1024**2))
            actual = (output.cpu(), state.cpu())
            del output, state
            if reference is None:
                reference = actual
            else:
                for a, b in zip(actual, reference):
                    torch.testing.assert_close(a, b, rtol=0.025, atol=0.025)
    print(json.dumps({"parity": "passed", "measurements": reports}, indent=2), flush=True)


if __name__ == "__main__":
    main()
