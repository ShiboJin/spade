# Tests

Run the offline suite from the repository root:

```bash
python -m pytest
```

Tests that need optional packages or local model assets skip when those
dependencies are unavailable. `test_actor_thinking.py` can use a local Qwen3
tokenizer through `QWEN3_TOKENIZER`. The chunked GRPO probability and gated-delta
tests include optional CUDA/BF16 cases and run their CPU cases without a GPU.

The training image supplies Torch, Transformers and ms-swift for the GRPO tests:

```bash
PYTHONPATH=.:tests python3 -m unittest test_chunked_logps test_chunked_delta_rule test_envduels_scheduler -v
torchrun --nnodes=1 --master_addr=127.0.0.1 --master_port=29671 --nproc_per_node=2 tests/fsdp_chunked_logps_smoke.py
```

`qwen_grpo_memory_smoke.py` is an explicit eight-GPU test with the local 27B
checkpoint, LoRA, FSDP2, 16K tokens and three accumulated backward passes. Run it
only in a container with the training memory guard's 160 GiB limit and swap
disabled, with the GPUs reserved for this test:

```bash
torchrun --nnodes=1 --master_addr=127.0.0.1 --master_port=29672 --nproc_per_node=8 tests/qwen_grpo_memory_smoke.py --length 16384 --microbatches 3 --output outputs/oom_validation/qwen16k_chunked.json
```

It also reserves GPU memory for resident inference contexts. This synthetic
shape/gradient check does not replace a real EnvDuels rollout and checkpoint run.
The model path, host memory limit and GPU reserves can be selected with
`--model`, `--memory-limit-gib`, `--reserve-gib` and `--rank0-extra-reserve-gib`.
The host memory limit must match the container's actual enforced limit.
The default diagnostic exercises the low-memory path; the plugin also accepts
the `SPADE_GRPO_*` environment settings described in
[`configs/TRAINING.md`](../configs/TRAINING.md).

`grpo_accumulation_parity_smoke.py` checks the A100 logprob path without an
optimizer update. It caches old logps for ten different microbatches before
accumulating their backward passes, disables dropout by default, checks adapter
weights are unchanged, and fails if any logprob difference exceeds `0.125`.
It writes rank-local measurements into one JSON report. Run in the same guarded
container with four reserved GPUs:

```bash
python3 -m torch.distributed.run --nnodes=1 --master_addr=127.0.0.1 \
  --master_port=29681 --nproc_per_node=4 tests/grpo_accumulation_parity_smoke.py \
  --output outputs/logps_validation/tiny.json
```

Add `--model checkpoints/Qwen3.8-27B` for real weights. `--lengths` selects
comma-separated sequence lengths; `--projection standard` and
`--no-activation-checkpointing` isolate those paths. `--dropout` is an explicit
control run; with freshly initialized zero LoRA B weights, dropout alone should
not alter the model outputs. This synthetic check does not cover rollout
tokenization, the full Swift trainer, or vLLM weight synchronization.

`--rounds 2` repeats accumulation without updating parameters. To localize a
failure, `--trace-layers --trace-on-gpu` samples intermediate activations without
per-layer CPU synchronization; `--trace-filter` limits module names by regex.
Instrumentation can change the timing/allocation pattern of intermittent errors,
so also verify without tracing. `--deterministic` is an explicit diagnostic
control, not a production default.

The Qwen FSDP2 final-norm discovery fix is enabled by default. The diagnostic
asserts that the nested language-model final norm and output head share the
same FSDP state. `--no-fsdp-tail-norm` disables the fix for a control run;
`--max-backwards 2` limits backward passes while still caching all old scores,
which is useful for isolating failures after the first backward.

For an exact replay, set `SPADE_LOGPS_SAVE_INPUTS=true` **inside the training
container**. First-update diagnostics save rank-local, detached CPU tensor
snapshots under `checkpoint/.../logps_inputs/`; pass that directory using
`--replay-inputs`. These contain training inputs and should be treated as private
run artifacts. `--replay-completions` instead re-tokenizes logged text, optionally
left-truncating it with `--replay-max-length`; that is not an exact input replay.

Generated games, API experiments, and one-off validation scripts belong under
`scripts/` or an ignored output directory, not in this test suite.
