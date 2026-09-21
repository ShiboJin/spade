"""GPU diagnostic: cache old logps, then accumulate backwards without an update.

Unlike a single-forward parity check, this uses different inputs and lengths for
every microbatch, matching Swift's old-policy scoring order. Run with torchrun
inside the memory-guarded training image. No optimizer step or checkpoint save.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
from types import MethodType, SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Local real checkpoint; omit for a tiny random Qwen")
    parser.add_argument("--lengths", default="128,255,256,257,511,512,513,767,768,769")
    parser.add_argument("--replay-completions", type=Path,
                        help="Replay first logged prompt/completion batch (text, not an encoded-input snapshot)")
    parser.add_argument("--replay-max-length", type=int, default=16384)
    parser.add_argument("--replay-inputs", type=Path, help="Exact rank-local .pt input snapshots from logps diagnostics")
    parser.add_argument("--dropout", action="store_true", help="Control run with LoRA dropout enabled")
    parser.add_argument("--nonzero-lora", action="store_true", help="Test dropout with nonzero adapters")
    parser.add_argument("--projection", choices=("chunked", "standard"), default="chunked")
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--decoder-checkpointing", action="store_true",
                        help="Use SPADE whole-decoder checkpointing instead of Accelerate submodule wrappers")
    parser.add_argument("--fsdp-tail-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trace-layers", action="store_true")
    parser.add_argument("--trace-filter", default=".*", help="Regex selecting traced module names")
    parser.add_argument("--trace-on-gpu", action="store_true",
                        help="Avoid per-layer host synchronization while tracing")
    parser.add_argument("--repeat-old", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--rounds", type=int, default=1, help="Repeat accumulation without any parameter update")
    parser.add_argument("--max-backwards", type=int, help="Stop after this many forwards/backwards (old scoring still covers all batches)")
    parser.add_argument("--max-logp-delta", type=float, default=0.125)
    parser.add_argument("--memory-limit-gib", type=int, default=160)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]
    if not lengths or min(lengths) < 64 or args.max_logp_delta < 0:
        parser.error("lengths must be >=64 and max-logp-delta nonnegative")
    if args.rounds < 1:
        parser.error("rounds must be positive")
    if args.max_backwards is not None and args.max_backwards < 1:
        parser.error("max-backwards must be positive")
    if args.replay_completions and (not args.model or args.replay_max_length < 2):
        parser.error("replay requires --model and replay-max-length >=2")
    if args.replay_inputs and (not args.model or args.replay_completions):
        parser.error("replay-inputs requires --model and cannot be combined with replay-completions")
    from scripts.memory_guard import verify_container_limits
    verify_container_limits(args.memory_limit_gib)

    # Default to the A100 production path; explicit flags select ablations.
    os.environ["SPADE_GRPO_CHUNKED_LOGPS"] = str(args.projection == "chunked").lower()
    os.environ["SPADE_GRPO_LOGPS_CHUNK_SIZE"] = "512"
    os.environ["SPADE_GRPO_DECODER_CHECKPOINTING"] = str(args.decoder_checkpointing).lower()
    os.environ["SPADE_GRPO_CPU_ACTIVATION_OFFLOAD"] = "false"
    os.environ["SPADE_GRPO_CHECKPOINT_DELTA_RULE"] = "false"
    os.environ["SPADE_GRPO_FSDP_TAIL_NORM"] = str(args.fsdp_tail_norm).lower()
    os.environ["ACCELERATE_USE_FSDP"] = "true"
    os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = str(bool(args.model)).lower()

    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    import torch.distributed as dist
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        from accelerate import Accelerator, FullyShardedDataParallelPlugin
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        from trl.trainer import disable_dropout_in_model
        from swift.trainers import disable_gradient_checkpointing
        from spade.swift_backend import fsdp_ram_loader  # noqa: F401
        from spade.swift_backend.memory_efficient_grpo import GRPOTrainer
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

        accelerator = Accelerator(
            mixed_precision="bf16", gradient_accumulation_steps=len(lengths),
            fsdp_plugin=FullyShardedDataParallelPlugin(
                fsdp_version=2, auto_wrap_policy="transformer_based_wrap",
                transformer_cls_names_to_wrap=["Qwen3_5DecoderLayer"],
                cpu_ram_efficient_loading=bool(args.model), reshard_after_forward=True,
                activation_checkpointing=args.activation_checkpointing))
        torch.manual_seed(42)
        if args.model:
            model = Qwen3_5ForConditionalGeneration.from_pretrained(
                args.model, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
        else:
            config = Qwen3_5Config(
                text_config=dict(
                    vocab_size=97, hidden_size=64, intermediate_size=128,
                    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                    head_dim=16, layer_types=["linear_attention", "full_attention"],
                    linear_num_key_heads=2, linear_num_value_heads=4,
                    linear_key_head_dim=16, linear_value_head_dim=16,
                    rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                     "partial_rotary_factor": 1.0, "mrope_section": [2, 3, 3]},
                    max_position_embeddings=max(lengths), pad_token_id=0),
                vision_config=dict(depth=1, hidden_size=32, intermediate_size=64,
                                   num_heads=4, out_hidden_size=64))
            config._attn_implementation = "sdpa"
            model = Qwen3_5ForConditionalGeneration(config).bfloat16()
        vocab_size = model.config.text_config.vocab_size
        modules = (r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|"
                   r"up_proj|down_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)")
        model = get_peft_model(model, LoraConfig(
            r=32 if args.model else 4, lora_alpha=64 if args.model else 8,
            lora_dropout=0.05, target_modules=modules, task_type="CAUSAL_LM"))
        if args.nonzero_lora:
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    torch.nn.init.normal_(parameter, std=0.02)
        if not args.dropout:
            disable_dropout_in_model(model)
        model.config.use_cache = False
        # FSDP2 requires an optimizer alongside the model for parameter remapping.
        # It is never stepped; zero LR also prevents accidental parameter updates.
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.0)
        model, optimizer = accelerator.prepare(model, optimizer)
        model.train()
        active_dropout = sum(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in model.modules())
        checkpoint_wrappers = sum(isinstance(m, CheckpointWrapper) for m in model.modules())
        checkpoint_decoders = sum(bool(getattr(m, "_spade_checkpointed", False)) for m in model.modules())
        final_norm = model.base_model.model.model.language_model.norm
        head = model.get_output_embeddings()
        tail_grouped = (hasattr(final_norm, "_get_fsdp_state") and
                        final_norm._get_fsdp_state() is head._get_fsdp_state())
        if args.fsdp_tail_norm:
            assert tail_grouped, "Final norm was not grouped with the FSDP2 head"
        assert args.dropout or active_dropout == 0
        if args.activation_checkpointing:
            assert checkpoint_wrappers > 0 or checkpoint_decoders > 0, "Activation checkpointing was not installed"
        local = lambda p: p.to_local() if hasattr(p, "to_local") else p
        initial_adapters = {n: local(p).detach().cpu().clone()
                            for n, p in model.named_parameters() if p.requires_grad}
        trainer = SimpleNamespace(
            accelerator=accelerator, args=SimpleNamespace(report_to=[]), is_multimodal=True,
            model_kwarg_keys={"use_cache", "logits_to_keep"}, temperature=0.8,
            template=SimpleNamespace(padding_free=False))
        trainer._get_last_hidden_state = MethodType(GRPOTrainer._get_last_hidden_state, trainer)
        score = MethodType(GRPOTrainer._get_logps_via_local_forward, trainer)
        torch.manual_seed(100 + dist.get_rank())
        batches = []
        snapshot_logps = []
        if args.replay_inputs:
            def to_device(value):
                if isinstance(value, torch.Tensor):
                    return value.to(accelerator.device)
                if isinstance(value, dict):
                    return {key: to_device(item) for key, item in value.items()}
                if isinstance(value, (list, tuple)):
                    return type(value)(to_device(item) for item in value)
                return value
            paths = sorted(args.replay_inputs.glob(f"rank{dist.get_rank()}.*.pt"))
            assert len(paths) == len(lengths), (len(paths), len(lengths))
            for path in paths:
                saved = torch.load(path, map_location="cpu", weights_only=True)
                assert saved["temperature"] == trainer.temperature
                inputs = to_device(saved["model_inputs"])
                batches.append((inputs, int(saved["logits_to_keep"]), inputs["input_ids"]))
                snapshot_logps.append(saved["old_logps"].to(accelerator.device))
        elif args.replay_completions:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
            with args.replay_completions.open() as stream:
                saved = json.loads(next(stream))
            assert len(saved["prompt"]) == len(saved["completion"]) == len(lengths) * dist.get_world_size()
            start = dist.get_rank() * len(lengths)
            for index in range(start, start + len(lengths)):
                prompt, completion = saved["prompt"][index], saved["completion"][index]
                ids = tokenizer.encode(prompt + completion, add_special_tokens=False)
                prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
                keep = min(len(ids) - prompt_len, args.replay_max_length - 1)
                ids = ids[-args.replay_max_length:]
                assert 0 < keep < len(ids)
                tokens = torch.tensor([ids], device=accelerator.device)
                inputs = dict(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False)
                batches.append((inputs, keep, tokens))
        else:
            for length in lengths:
                # Unequal rank lengths and varying microbatches exercise FSDP state reuse.
                length -= dist.get_rank() * 3
                tokens = torch.randint(1, min(vocab_size, 10000), (1, length), device=accelerator.device)
                inputs = dict(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False)
                batches.append((inputs, length - 32, tokens))
        trace = {}
        trace_differences = []
        trace_records = []

        def flush_trace():
            for phase, (microbatch, name), maximum, mean in trace_differences:
                if maximum.item() != 0:
                    record = dict(event="layer_difference", rank=dist.get_rank(), phase=phase,
                                  microbatch=microbatch, module=name, max_delta=maximum.item(),
                                  mean_delta=mean.item())
                    trace_records.append(record)
                    print(json.dumps(record), flush=True)
            trace_differences.clear()
        trace_state = {"phase": "old", "microbatch": 0}
        if args.trace_layers:
            def capture(name):
                def hook(module, inputs, output):
                    if trace_state["phase"] == "backward":
                        return
                    value = output[0] if isinstance(output, tuple) else output
                    if not isinstance(value, torch.Tensor) or value.ndim != 3:
                        return
                    sample = value[0, [0, value.shape[1] // 2, value.shape[1] - 1]].detach().float()
                    if not args.trace_on_gpu:
                        sample = sample.cpu()
                    key = (trace_state["microbatch"], name)
                    if trace_state["phase"] == "old":
                        trace[key] = sample.clone()
                    else:
                        delta = (sample - trace[key]).abs()
                        trace_differences.append((trace_state["phase"], key, delta.max(), delta.mean()))
                return hook
            for name, module in model.named_modules():
                if re.search(args.trace_filter, name) and name.endswith((
                        "embed_tokens", "input_layernorm", "post_attention_layernorm",
                        "linear_attn", "self_attn", "mlp", "language_model.norm")):
                    module.register_forward_hook(capture(name))
        old_logps = []
        started = time.monotonic()
        print(json.dumps(dict(event="prepared", rank=dist.get_rank(), model=args.model or "tiny",
                              active_dropout=active_dropout, checkpoint_wrappers=checkpoint_wrappers,
                              checkpoint_decoders=checkpoint_decoders)), flush=True)
        print(json.dumps(dict(event="tail_group", rank=dist.get_rank(), norm_and_head_grouped=tail_grouped)), flush=True)
        with torch.no_grad(), disable_gradient_checkpointing(model):
            for i, (inputs, keep, tokens) in enumerate(batches):
                trace_state.update(phase="old", microbatch=i)
                old, _ = score(model, inputs, keep, tokens)
                old_logps.append(old.detach().clone())
                if snapshot_logps:
                    print(json.dumps(dict(event="snapshot_reference", rank=dist.get_rank(), microbatch=i,
                                          max_delta=(old.float() - snapshot_logps[i].float()).abs().max().item())), flush=True)
                print(json.dumps(dict(event="old_scored", rank=dist.get_rank(), microbatch=i)), flush=True)
            if args.repeat_old:
                for i, (inputs, keep, tokens) in enumerate(batches):
                    trace_state.update(phase="repeat_old", microbatch=i)
                    repeated, _ = score(model, inputs, keep, tokens)
                    print(json.dumps(dict(event="old_repeat", rank=dist.get_rank(), microbatch=i,
                                          max_delta=(repeated.float() - old_logps[i].float()).abs().max().item())), flush=True)
        diagnostics = []
        backward_steps = len(batches) * args.rounds
        if args.max_backwards is not None:
            backward_steps = min(backward_steps, args.max_backwards)
        for step in range(backward_steps):
            i = step % len(batches)
            (inputs, keep, tokens), old = batches[i], old_logps[i]
            with accelerator.accumulate(model):
                trace_state.update(phase="new", microbatch=i)
                current, _ = score(model, inputs, keep, tokens)
                with torch.no_grad():
                    delta = current.float() - old.float()
                    finite = bool(torch.isfinite(delta).all())
                    stats = dict(microbatch=i, round=step // len(batches), length=tokens.shape[1], old_dtype=str(old.dtype),
                                 new_dtype=str(current.dtype), finite=finite,
                                 delta_abs_max=delta.abs().max().item() if finite else None,
                                 delta_abs_mean=delta.abs().mean().item() if finite else None,
                                 ratio_max=delta.exp().max().item() if finite else None,
                                 ratio_min=delta.exp().min().item() if finite else None,
                                 first_old=old[0, 0].item(), first_new=current[0, 0].item())
                diagnostics.append(stats)
                print(json.dumps(dict(rank=dist.get_rank(), **stats)), flush=True)
                flush_trace()
                # Backpropagate a bounded objective without stepping the optimizer.
                trace_state["phase"] = "backward"
                accelerator.backward(-current.float().mean())
                del current
        flush_trace()
        unchanged = all(torch.equal(initial_adapters[n], local(p).detach().cpu())
                        for n, p in model.named_parameters() if p.requires_grad)
        finite_grads = all(bool(torch.isfinite(local(p.grad)).all())
                           for p in model.parameters() if p.grad is not None)
        nonzero_grads = any(bool(local(p.grad).count_nonzero())
                           for p in model.parameters() if p.grad is not None)
        passed = unchanged and finite_grads and nonzero_grads and all(
            d["finite"] and d["delta_abs_max"] <= args.max_logp_delta for d in diagnostics)
        report = dict(rank=dist.get_rank(), passed=passed, adapters_unchanged=unchanged,
                      finite_grads=finite_grads, nonzero_grads=nonzero_grads,
                      active_dropout=active_dropout, checkpoint_wrappers=checkpoint_wrappers,
                      checkpoint_decoders=checkpoint_decoders,
                      norm_and_head_grouped=tail_grouped,
                      layer_differences=trace_records,
                      seconds=time.monotonic() - started, diagnostics=diagnostics)
        reports = [None] * dist.get_world_size()
        dist.all_gather_object(reports, report)
        all_passed = all(r["passed"] for r in reports)
        if dist.get_rank() == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(dict(
                status="passed" if all_passed else "failed", optimizer_updated=False,
                configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                ranks=reports), indent=2) + "\n")
            print(f"Parity {'PASSED' if all_passed else 'FAILED'}: {args.output}", flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return 0 if all_passed else 1
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # A failing rank must not block in collective cleanup while its peers compute.
        os._exit(1)


if __name__ == "__main__":
    sys.exit(main())
