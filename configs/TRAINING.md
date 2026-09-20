# Qwen EnvDuels training profiles

Run from the repository root using the unified training image, the local model
and the EnvDuels export configured in the JSON. Online W&B requires
`WANDB_API_KEY` in the host environment. Set `wandb_mode` to `offline` in the
configuration if online tracking is not needed.
The 4090 profile currently uses `offline` so W&B API outages do not prevent
training from starting. Metrics and completions tables are stored locally in
`outputs/training/<run>/wandb/wandb/offline-run-*`; they are not streamed to the
dashboard. Once connectivity is restored, run `wandb sync <offline-run-directory>`
in an environment with W&B installed, network access and your W&B credentials.
Set `wandb_mode` back to `online` when live tracking is required.

```bash
# Existing low-memory profile (8 x RTX 4090)
python3 scripts/run_train.py --config configs/train_qwen38_envduels_lora.json

# A100 80GB starting profile (8 GPUs)
python3 scripts/run_train.py --config configs/train_qwen38_envduels_lora_a100_80gb.json
```

Both profiles retain 16K context, 24 turns, TP=8, a batch of 24 trajectories and
three gradient accumulation microbatches. They save every five optimizer steps.
The latest three rolling `checkpoint-N` directories are retained. At each full
epoch boundary, a complete resumable copy is also kept under
`epoch_checkpoints/epoch-001-step-15` (then epoch 002/step 30 and epoch 003/step 45
for these profiles). These independent copies are not subject to
`save_total_limit`. Only the latest completed epoch archive is retained: after
the new archive has been copied successfully, older epoch archives are deleted.
Allow space for both old and new archives while the copy is in progress.
If an epoch boundary is not a regular save step, the callback requests a save.
Stopping partway through an epoch does not create a completed-epoch archive.
Set `resume_from_checkpoint` to an archive directory to resume from it.
The A100 profile has not been benchmarked or validated on A100 hardware. It
retains ordinary FSDP activation checkpointing and disables the additional
low-memory decoder and gated-delta recomputation paths. It disables enforced
eager rollout and training model/optimizer offloading, uses vLLM sleep level 1,
and increases the logps chunk size to 512. These are starting settings for an
A100 smoke run, not measured throughput improvements or an OOM-free guarantee.
Sleep level 1 still allows vLLM to offload its weights while sleeping; disabling
`offload_model` does not disable that separate vLLM mechanism.

| Training setting | Low-memory default | A100 profile | Plugin environment |
| --- | --- | --- | --- |
| `grpo_chunked_logps` | `true` | `true` | `SPADE_GRPO_CHUNKED_LOGPS` |
| `grpo_logps_chunk_size` | `128` | `512` | `SPADE_GRPO_LOGPS_CHUNK_SIZE` |
| `grpo_decoder_checkpointing` | `true` | `false` | `SPADE_GRPO_DECODER_CHECKPOINTING` |
| `grpo_cpu_activation_offload` | `true` | `false` | `SPADE_GRPO_CPU_ACTIVATION_OFFLOAD` |
| `grpo_checkpoint_delta_rule` | `true` | `false` | `SPADE_GRPO_CHECKPOINT_DELTA_RULE` |
| `vllm_enforce_eager` | `true` | `false` | `VLLM_ENFORCE_EAGER` |
| `sleep_level` | `2` | `1` | `SLEEP_LEVEL` |
| `offload_model` | `true` | `false` | `OFFLOAD_MODEL` |
| `offload_optimizer` | `true` | `false` | `OFFLOAD_OPTIMIZER` |
| `move_model_batches` | `64` | `32` | `MOVE_MODEL_BATCHES` |

The GRPO and four rollout switch settings are optional for backward compatibility;
omitted settings use the low-memory defaults. `move_model_batches` remains required.
The launcher validates and passes them into Docker. `sleep_level` accepts integers
0, 1 and 2; the eager/offload switches require JSON booleans.
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

## Fast checks after a delta-rule OOM

Inside the training image, from the checkout root:

```bash
# Single GPU; no checkpoint loading, no game rollout. Uses 27B head dimensions.
python3 tests/delta_rule_memory_smoke.py --length 16384

# Small Qwen/LoRA model; verify FSDP2 gradients with the outer checkpoint mode
# observed in Swift's training traceback. Requires two free GPUs.
CUDA_VISIBLE_DEVICES=0,3 torchrun --standalone --nproc_per_node=2 \
  tests/fsdp_chunked_logps_smoke.py --reentrant-checkpoint
```

The first check compares the original and segmented no-grad fallback's outputs,
recurrent state, elapsed GPU work and incremental peak allocated memory. It uses
synthetic tensors, not saved rollout data or the complete process memory state.
The failed process cannot resume at the exception without a previously saved
input snapshot. These checks isolate the failing operation without repeating
rollout, but do not establish the full 27B training peak or vLLM coexistence.

Long no-grad delta-rule calls must still be segmented: both old-policy scoring
and the initial forward of a reentrant checkpoint can disable gradients. The
training launcher also binds `LOCAL_RANK` before importing Swift pipelines and
enables `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The pinned vLLM build
temporarily disables expandable segments within its sleep memory pool; recheck
this compatibility if replacing the image. Early binding is a precaution, not
proof that all secondary GPU0 contexts have been eliminated.

When SPADE decoder checkpointing is enabled, a root pre-forward hook disables
HF checkpointing only on the decoder layers already managed by SPADE. Swift can
otherwise re-enable an outer reentrant checkpoint after FSDP setup, retaining
the layer inputs on GPU and bypassing SPADE's CPU offload on the initial forward.
The A100 profile, which does not install SPADE decoder checkpointing, is unaffected.

For a stronger check without game rollout, the existing real-model test can
simulate Swift re-enabling checkpointing and add explicit GPU memory pressure:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=8 tests/qwen_grpo_memory_smoke.py \
  --length 16384 --microbatches 1 --reserve-gib 2 --rank0-extra-reserve-gib 0 \
  --reentrant-checkpoint --output outputs/oom_validation/qwen16k_checkpoint_guard.json
```

Run this in a training container with the required 160 GiB host-memory limit.
It loads real weights and checks old logps, training forward/backward, and one
optimizer update. The synthetic reserve does not reproduce vLLM's allocator
history, so this still does not replace an end-to-end run.
