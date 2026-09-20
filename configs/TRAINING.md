# Qwen EnvDuels training profiles

Run from the repository root using the unified training image, the local model
and the EnvDuels export configured in the JSON. Online W&B requires
`WANDB_API_KEY` in the host environment. Set `wandb_mode` to `offline` in the
configuration if online tracking is not needed.

```bash
# Existing low-memory profile (8 x RTX 4090)
python3 scripts/run_train.py --config configs/train_qwen38_envduels_lora.json

# A100 80GB starting profile (8 GPUs)
python3 scripts/run_train.py --config configs/train_qwen38_envduels_lora_a100_80gb.json
```

Both profiles retain 16K context, 24 turns, TP=8, a batch of 24 trajectories and
three gradient accumulation microbatches. They save every five optimizer steps.
The A100 profile has not been benchmarked or validated on A100 hardware. It
retains ordinary FSDP activation checkpointing and disables the additional
low-memory decoder and gated-delta recomputation paths. It does not change the
rollout model/optimizer offloading settings or assume a particular GPU topology.

| Training setting | Low-memory default | A100 profile | Plugin environment |
| --- | --- | --- | --- |
| `grpo_chunked_logps` | `true` | `true` | `SPADE_GRPO_CHUNKED_LOGPS` |
| `grpo_logps_chunk_size` | `128` | `128` | `SPADE_GRPO_LOGPS_CHUNK_SIZE` |
| `grpo_decoder_checkpointing` | `true` | `false` | `SPADE_GRPO_DECODER_CHECKPOINTING` |
| `grpo_cpu_activation_offload` | `true` | `false` | `SPADE_GRPO_CPU_ACTIVATION_OFFLOAD` |
| `grpo_checkpoint_delta_rule` | `true` | `false` | `SPADE_GRPO_CHECKPOINT_DELTA_RULE` |

These JSON settings are optional for backward compatibility; omitted settings
use the low-memory defaults. The launcher validates and passes them into Docker.
CPU activation offload requires decoder checkpointing. Gated-delta recomputation
only wraps the Transformers PyTorch fallback, not an installed native FLA kernel.
GPU capacity alone does not change which attention implementation is installed.

The plugins use the Qwen3.5, Accelerate and ms-swift interfaces in the pinned
unified image. Recheck compatibility before changing those runtime versions.
Host RAM limits are independent of GPU memory: adjust `memory_limit_gib` and
`host_memory_reserve_gib` to the destination machine's available host RAM.

For a real one-step checkpoint check, copy the chosen JSON, set `save_every` to
`1`, and pass `--max-steps 1`. Training logs and checkpoints remain under
`outputs/training/`; generated artifacts and local `/tmp` configs are not Git
inputs. A synthetic memory test does not validate rollout or checkpoint saving.
