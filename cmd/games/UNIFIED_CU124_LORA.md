# CUDA 12.4 unified LoRA GRPO runtime

This runtime trains Qwen3.8-27B on the 90 fixed EnvDuels environments with
ms-swift LoRA GRPO, FSDP2 and colocated vLLM. The same Docker image also runs
base-model and PEFT-adapter AIME evaluation.

A Chinese field-by-field guide is available in `cmd/games/TRAIN_CONFIG_CN.md`.

The training launcher caps total container RAM at 160 GiB and requires another
48 GiB of available host RAM before starting. Evaluation defaults to a 96 GiB
cap with the same reserve. Both launchers verify kernel-enforced limits before
model loading, serialize runs in this checkout, and stop their own container
if host available memory falls below half the reserve. On this cgroup-v1 host,
swap accounting is unavailable, so `memory.swappiness=0` is also enforced and
checked. See the Chinese guide's memory-protection section for scope and limits.

The pinned Transformers loader patch prevents both checkpoint reads and the
subsequent full-parameter zero-fill on nonzero FSDP ranks. The zero-fill alone
previously exhausted host memory despite skipping worker checkpoint reads.

The numeric-UID container sets `USER=envduels` and writable configuration/cache
paths so Inductor and vLLM can initialize without a passwd entry. Rollouts use
TP=8: TP=4 leaves insufficient space for KV cache at the configured 0.45 GPU
memory budget. `move_model_batches=64` transfers decoder weights one layer at
a time, avoiding a full 27B all-gather on every GPU when synchronizing to vLLM.

The runtime bind-mounts source code, configs, datasets, checkpoints and output
directories from the host. Editing those files or replacing the 90-environment
export does not require rebuilding the image. Rebuild only after changing a
Python/system dependency, a compiled CUDA component, or the target GPU
architecture.

## Current experiment definition

The checked-in training config expresses this exact experiment:

```text
fixed environments                         90
fixed instances (seeds) per environment     1
environments in each rollout/update batch   6
independent trajectories per environment    4
GRPO group size                              4
rollout episodes per batch                  24
fixed-pool epochs                            3
optimizer steps per fixed-pool epoch        15
episodes per fixed-pool epoch              360
episodes per environment in the whole run   12
optimizer steps in the whole run            45
episodes in the whole run                 1080
```

Each environment needs one deterministic integer seed to make reset
reproducible. That seed is part of its single fixed instance; it does not create
multiple dataset rows. The dataset therefore has exactly 90 rows.

The important config relationship is:

```text
batch_size = num_games_per_rollout * trajectories_per_game
           = 6 * 4
           = 24
```

`num_games_per_rollout: 6` preserves complete four-sample GRPO groups, divides
the 90-item fixed pool, and produces a 24-sample generation batch. On eight
GPUs, the global micro-batch is eight, so the launcher derives gradient
accumulation of three. If either rollout dimension changes, `batch_size` must
change with it and remain divisible by the global micro-batch.

## How the original SPADE rollout maps to this run

The Inkling config in the experiment used:

```text
num_games_per_rollout = 8
trajectories_per_game = 4
batch_size = 32
fixed_pool_epochs = 3
```

SPADE selects games from the fixed pool and creates several independent actor
trajectories for each selected game. Rewards are normalized within the
same-game group, then the actor trajectories are sent to the policy update. In
the local fixed-env Slime implementation, failed generations are
overprovisioned and only complete same-problem groups are retained.

Our ms-swift path preserves the relevant actor-only behavior:

1. One JSONL row selects one fixed environment and its deterministic reset.
2. ms-swift expands the row into four independent Gym environment objects.
3. The four actors play the same problem for at most 25 turns.
4. Each episode receives the EnvDuels terminal reward, currently binary 0/1.
5. `grpo_no_std` subtracts the four-reward group mean without dividing by the
   group standard deviation.
6. The PPO-style clipped GRPO loss uses low/high clips 0.20/0.28 and updates
   only rank-32 LoRA parameters.

When all four rewards are equal, all four advantages are zero. Because
`remove_constant_reward_groups` is false, the group is kept and contributes
zero policy-gradient signal instead of being resampled.

`num_substeps: 1` means each generated batch is used for one update. Raising it
reuses rollout data and makes later updates more off-policy. `kl_penalty_coef:
0.0` disables an explicit reference-model KL penalty.

The original environment-generator, hint, self-judge, environment-validator,
environment-reward scaling and `train_on_env_trajectories` settings are omitted:
the 90 Python environments already exist, and this run trains only on actor
gameplay. AIME settings live in `configs/evaluation.json`. `gamma` is also
omitted because this bridge supplies one terminal episode return rather than a
discounted per-step value target. Inkling's `thinking_effort` has no direct Qwen
equivalent; Qwen uses `enable_thinking` and `preserve_thinking`.

## Prepare the fixed dataset

From the SPADE checkout:

```bash
cd /data1/shibo517/envduels/spade
bash scripts/unified_runtime.sh prepare
```

This produces `data/envduels/fixed90-swift.jsonl`: one deterministic row for
each ID in the 90-environment manifest. The training launcher rejects missing,
duplicate or extra environments and rejects a dataset whose seeds do not match
`fixed_pool_seed`.

## Inspect the resolved run without using a GPU

```bash
python3 scripts/run_train.py \
  --config configs/train_qwen38_envduels_lora.json \
  --dry-run
```

The printed `config.derived` block shows the resolved rollout count, batch
shape, optimizer steps and fixed-pool passes. The printed Docker command is the
exact command the launcher will execute.

## Run one real update

First set `training.gpu_ids` to eight idle GPUs. Then run:

```bash
python3 scripts/run_train.py \
  --config configs/train_qwen38_envduels_lora.json \
  --smoke
```

This loads the BF16 base model, creates LoRA, generates six four-trajectory
groups, accumulates three micro-batches, runs one optimizer update, synchronizes
the adapter to vLLM, and saves a PEFT checkpoint. It is the smallest end-to-end
GPU test for the configured batch shape.

## Run the configured training experiment

```bash
python3 scripts/run_train.py \
  --config configs/train_qwen38_envduels_lora.json
```

The default is three fixed-pool epochs. Each of the 90 environments gets four
rollouts in each epoch, for 12 rollouts per environment in the whole run. To
run two epochs temporarily, which gives exactly eight rollouts per environment:

```bash
python3 scripts/run_train.py \
  --config configs/train_qwen38_envduels_lora.json \
  --epochs 2
```

That override produces 30 optimizer steps, 720 episodes and eight episodes per
environment. An explicit update cap is also available:

```bash
python3 scripts/run_train.py \
  --config configs/train_qwen38_envduels_lora.json \
  --max-steps 10
```

`--max-steps` stops by updates and therefore does not promise equal coverage of
all 90 environments. It is mainly useful for debugging. `--epochs` and
`--max-steps` are mutually exclusive.

Every invocation creates a UTC-stamped directory under `outputs/training/`
containing the resolved config, exact command, status, console log and adapter
checkpoints. Ctrl-C asks Docker to stop and keeps that run directory.
Set `resume_from_checkpoint` to a previous `checkpoint-N` directory when the
optimizer, scheduler and trainer state should be resumed; leave it `null` for a
new run.

## Merge a saved LoRA adapter

`scripts/merge_lora.py` exports a separate BF16 Hugging Face checkpoint. Its
defaults point to the 2026-09-22 run's `checkpoint-24` adapter and write to
`checkpoints/Qwen3.8-27B-spade-merged-ckpt24`:

```bash
cd /data1/shibo517/envduels/spade
python3 scripts/merge_lora.py --run
```

For another checkpoint, supply `--base`, `--adapter`, and `--output` as needed.
On hosts where Docker requires passwordless sudo, add `--sudo-docker`.
Without `--run`, the script validates paths and free disk space and prints the
Docker command. It refuses to overwrite an existing output. The base and
adapter are mounted read-only, and the output is checked before its directory
gets the final name. Keep about 62 GiB free for this 27B BF16 export.

The merged directory can be used as the `checkpoint` in an evaluation config.
It is an inference checkpoint; resume training from the original adapter
checkpoint, including its optimizer and trainer state.

## Relevant Inkling-to-ms-swift config mapping

| SPADE/Inkling setting | Current setting or ms-swift behavior |
|---|---|
| `fixed_pool_seed` | deterministic seed in each of the 90 rows |
| `fixed_pool_epochs` | `num_train_epochs` |
| `num_games_per_rollout` | environments selected per generation batch |
| `trajectories_per_game` | `num_generations`, the GRPO group size |
| `batch_size` | `generation_batch_size` |
| `num_substeps` | `num_iterations` |
| `learning_rate` | `learning_rate` |
| `lora_rank` | `lora_rank` |
| `kl_penalty_coef` | `beta` |
| `reward_normalization: grpo_no_std` | `scale_rewards: none` |
| `remove_constant_reward_groups` | `dynamic_sample` |
| `ppo_clip_low/high` | `epsilon/epsilon_high` |
| `rollout_json_export` | `log_completions` |
| `actor_temperature/top_p/top_k` | rollout sampling settings |
| `actor_max_tokens` | maximum tokens generated per actor turn |
| `max_context_length` | model/vLLM total context window |

The config keeps the Inkling learning rate (`1e-6`), LoRA rank (`32`), Adam
settings, reward normalization and PPO clips. It starts with an 8192-token
total context and 1024 generated tokens per turn because 65536/8192 has not yet
been validated on 8x24GB RTX 4090. Increase these only after the one-update
smoke test and memory/latency measurements.

## Docker image

Build the base image if it does not already exist:

```bash
BUILD_JOBS=64 bash scripts/unified_runtime.sh build
```

Then add the small W&B-only layer used by the checked-in training config:

```bash
bash scripts/unified_runtime.sh build-wandb
```

The base image is `envduels-unified:cu124`; the thin tracking image is
`envduels-unified:cu124-wandb`. The second command reuses the already-built base
image and only installs the Python W&B and Qwen processor runtime packages. It
does not rebuild CUDA, Torch, vLLM or ms-swift.

The base Dockerfile pins CUDA 12.4.1, Torch, vLLM and ms-swift. Torch and vLLM
were built for SM80 and SM89, covering A100 and RTX 4090. The image has
completed CPU/import checks, and Qwen3.8-27B has loaded and generated a token
with TP=4 on RTX 4090. The eight-GPU optimizer-step smoke remains the final
validation when all eight GPUs are idle.

FSDP2 is configured in both the Accelerate launcher configuration and
`configs/ms_swift_fsdp2.json`. The latter is passed explicitly to ms-swift so
its pre-Trainer model-loading phase enables rank-0-only weight loading. Native
FSDP activation checkpointing is used instead of generic gradient
checkpointing.

## W&B tracking

The default config sends metrics to both TensorBoard and W&B. W&B receives the
Trainer and GRPO metrics (including loss, reward, learning rate, gradient norm,
completion length, KL/clipping metrics when emitted by ms-swift) and completion
tables because `rollout_json_export` is enabled. The complete resolved launcher
configuration, including fixed-pool settings and derived batch/step counts, is
stored under the W&B config key `experiment`. Model checkpoints are not uploaded
to W&B.

Before an online run, provide the key in the host environment. The launcher
passes the variable name to Docker without serializing its value into
`launch.json` or `resolved_config.json`:

```bash
read -rsp 'W&B API key: ' WANDB_API_KEY
export WANDB_API_KEY
echo
python3 scripts/run_train.py \
  --config configs/train_qwen38_envduels_lora.json \
  --smoke
```

The project is selected by `wandb_project`. Set `wandb_entity` in the JSON only
when the run should belong to a particular team. A null `wandb_run_name` uses
the launcher's unique UTC run name. Online W&B uses Docker's bridge network;
Hugging Face model and dataset access remains offline.

For a network-isolated run, set `wandb_mode` to `offline`. The launcher then
keeps `--network none`, writes the W&B run beneath the timestamped output
directory, and does not require `WANDB_API_KEY`. Upload it later with
`wandb sync` from a networked environment. Set `wandb_enabled` to false to keep
TensorBoard only.

## Evaluate base and LoRA checkpoints

Baseline AIME26 avg@8/pass@8:

```bash
python3 scripts/run_eval.py --config configs/evaluation.json
```

Evaluate a saved PEFT adapter directly:

```bash
python3 scripts/run_eval.py \
  --config configs/evaluation.json \
  --lora outputs/training/RUN/checkpoint/checkpoint-N
```

Use the same evaluation JSON for the base and RL adapter runs so the AIME data,
thinking mode, sampling parameters and token limits remain identical.
