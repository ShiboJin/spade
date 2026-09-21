"""Eight-GPU real-model 16K memory check without waiting for game rollouts.

Explicit torchrun utility, not part of unittest discovery. Synthetic tokens
exercise the same Qwen/LoRA/FSDP2 forward/backward shapes as production; this
does not replace the final real-environment checkpoint validation.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import MethodType, SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.memory_guard import verify_container_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnose-logps", action="store_true",
                        help="Compare old/new logps without applying an optimizer update")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--length", type=int, default=16384)
    parser.add_argument("--microbatches", type=int, default=3)
    parser.add_argument("--reentrant-checkpoint", action="store_true",
                        help="Simulate Swift enabling HF checkpointing after FSDP setup")
    parser.add_argument("--reserve-gib", type=float, default=1.0)
    parser.add_argument("--rank0-extra-reserve-gib", type=float, default=2.0)
    parser.add_argument("--memory-limit-gib", type=int, default=160)
    parser.add_argument("--model", default="checkpoints/Qwen3.8-27B")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.length < 2 or args.microbatches < 1 or args.reserve_gib < 0 or args.rank0_extra_reserve_gib < 0:
        parser.error("length must be >= 2, microbatches >= 1, and GPU memory reserves nonnegative")
    verify_container_limits(args.memory_limit_gib)

    import torch
    import torch.distributed as dist
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    os.environ["ACCELERATE_USE_FSDP"] = "true"
    os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "true"
    dist.init_process_group("nccl")
    try:
        from accelerate import Accelerator, FullyShardedDataParallelPlugin
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen3_5ForConditionalGeneration
        from transformers.utils.logging import disable_progress_bar
        from spade.swift_backend import fsdp_ram_loader  # noqa: F401
        from spade.swift_backend.memory_efficient_grpo import GRPOTrainer

        disable_progress_bar()
        accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=args.microbatches,
            fsdp_plugin=FullyShardedDataParallelPlugin(
                fsdp_version=2, auto_wrap_policy="transformer_based_wrap",
                transformer_cls_names_to_wrap=["Qwen3_5DecoderLayer"],
                cpu_ram_efficient_loading=True, reshard_after_forward=True,
                activation_checkpointing=True))
        torch.manual_seed(42)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
            local_files_only=True)
        modules = (r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|"
                   r"up_proj|down_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)")
        model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
                                               target_modules=modules, task_type="CAUSAL_LM"))
        model.config.use_cache = False
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-6)
        model, optimizer = accelerator.prepare(model, optimizer)
        if dist.get_rank() == 0:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
            from collections import Counter
            print("checkpointed modules:", Counter(type(m._checkpoint_wrapped_module).__name__
                  for m in model.modules() if isinstance(m, CheckpointWrapper)), flush=True)
            print("SPADE checkpointed decoder layers:", sum(bool(getattr(m, "_spade_checkpointed", False))
                  for m in model.modules()), flush=True)
        model.train()
        if args.reentrant_checkpoint:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
            model.enable_input_require_grads()
        # Include headroom for vLLM/CUDA contexts left resident after sleep.
        reserve_bytes = int((args.reserve_gib + (args.rank0_extra_reserve_gib if dist.get_rank() == 0 else 0)) * 1024**3)
        reserve = torch.empty(reserve_bytes, dtype=torch.uint8, device=accelerator.device)
        trainer = SimpleNamespace(accelerator=accelerator, args=SimpleNamespace(report_to=[]),
            is_multimodal=True, model_kwarg_keys={"use_cache", "logits_to_keep"}, temperature=args.temperature)
        trainer._get_last_hidden_state = MethodType(GRPOTrainer._get_last_hidden_state, trainer)
        tokens = torch.randint(100, 10000, (1, args.length), device=accelerator.device)
        inputs = {"input_ids": tokens, "attention_mask": torch.ones_like(tokens), "use_cache": False}
        keep = args.length - 1
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        print(f"rank {dist.get_rank()}: prepared model; starting {args.length}-token old logps", flush=True)
        from swift.trainers import disable_gradient_checkpointing
        with torch.no_grad(), disable_gradient_checkpointing(model):
            old, _ = GRPOTrainer._get_logps_via_local_forward(trainer, model, inputs, keep, tokens)
        assert torch.isfinite(old).all()
        if args.reentrant_checkpoint and not args.diagnose_logps:
            guarded = [m for m in model.modules() if getattr(m, "_spade_checkpointed", False)]
            assert guarded and all(not m.gradient_checkpointing for m in guarded)
        print(f"rank {dist.get_rank()}: old logps passed", flush=True)
        diagnostics = []
        for step in range(args.microbatches):
            with accelerator.accumulate(model):
                logps, _ = GRPOTrainer._get_logps_via_local_forward(trainer, model, inputs, keep, tokens)
                print(f"rank {dist.get_rank()}: training forward {step + 1} passed", flush=True)
                delta = logps.float() - old.float()
                ratio = delta.exp()
                diagnostic = dict(microbatch=step, old_dtype=str(old.dtype), new_dtype=str(logps.dtype),
                                  delta_abs_max=delta.abs().max().item(),
                                  delta_abs_mean=delta.abs().mean().item(), ratio_max=ratio.max().item(),
                                  ratio_mean=ratio.mean().item())
                diagnostics.append(diagnostic)
                print(f"rank {dist.get_rank()}: logps parity {diagnostic}", flush=True)
                advantage = 0.5 if dist.get_rank() % 2 else -0.5
                loss = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.28) * advantage).mean()
                assert torch.isfinite(loss)
                accelerator.backward(loss)
            print(f"rank {dist.get_rank()}: backward {step + 1}/{args.microbatches} passed", flush=True)
        nonzero = False
        for param in model.parameters():
            if param.grad is not None:
                grad = param.grad.to_local() if hasattr(param.grad, "to_local") else param.grad
                assert torch.isfinite(grad).all()
                nonzero = nonzero or bool(grad.count_nonzero())
        assert nonzero, "no nonzero adapter gradients"
        if not args.diagnose_logps:
            optimizer.step()
        torch.cuda.synchronize()
        report = {"rank": dist.get_rank(), "length": args.length, "microbatches": args.microbatches,
                  "reserved_headroom_gib": reserve.numel() / 1024**3,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                  "seconds": time.monotonic() - started, "status": "passed", "logps_diagnostics": diagnostics,
                  "optimizer_updated": not args.diagnose_logps}
        reports = [None] * dist.get_world_size()
        dist.all_gather_object(reports, report)
        if dist.get_rank() == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(reports, indent=2) + "\n")
            print(json.dumps(reports, indent=2), flush=True)
    except BaseException:
        # A failed rank cannot collectively destroy NCCL while peers are still
        # computing. Print the original failure before torchrun stops peers.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
