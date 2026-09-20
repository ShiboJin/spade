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

Generated games, API experiments, and one-off validation scripts belong under
`scripts/` or an ignored output directory, not in this test suite.
