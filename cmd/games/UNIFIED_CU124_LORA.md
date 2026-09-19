# CUDA 12.4 unified LoRA GRPO runtime

This runtime trains Qwen3.8-27B on the 90 fixed EnvDuels environments with
ms-swift LoRA GRPO, FSDP2 and colocated vLLM. The same Docker image also runs
base-model and PEFT-adapter AIME evaluation.

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
environments in each rollout/update batch   1
independent trajectories per environment    8
GRPO group size                              8
rollout episodes per batch                   8
fixed-pool epochs                            1
optimizer steps per fixed-pool epoch        90
episodes per fixed-pool epoch              720
episodes per environment in the whole run    8
```

Each environment needs one deterministic integer seed to make reset
reproducible. That seed is part of its single fixed instance; it does not create
multiple dataset rows. The dataset therefore has exactly 90 rows.

The important config relationship is:

```text
batch_size = num_games_per_rollout * trajectories_per_game
           = 1 * 8
           = 8
```

`num_games_per_rollout: 1` is the conservative 8x4090 starting point. One
complete eight-sample GRPO group is generated and updated at a time. It also
divides the 90-item fixed pool exactly. If it is later increased, it must divide
90 and `batch_size` must be changed with it. For example, 5 games and 8
trajectories means a rollout batch of 40 episodes and substantially higher
rollout memory demand.

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
2. ms-swift expands the row into eight independent Gym environment objects.
3. The eight actors play the same problem for at most 25 turns.
4. Each episode receives the EnvDuels terminal reward, currently binary 0/1.
5. `grpo_no_std` subtracts the eight-reward group mean without dividing by the
   group standard deviation.
6. The PPO-style clipped GRPO loss uses low/high clips 0.20/0.28 and updates
   only rank-32 LoRA parameters.

When all eight rewards are equal, all eight advantages are zero. Because
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

This loads the BF16 base model, creates LoRA, generates one eight-trajectory
group, runs backward and one optimizer update, synchronizes the adapter to
vLLM, and saves a PEFT checkpoint. It is the smallest end-to-end GPU test.

## Run the configured training experiment

```bash
python3 scripts/run_train.py \
  --config configs/train_qwen38_envduels_lora.json
```

The default is exactly one fixed-pool epoch: each of the 90 environments gets
eight rollouts once. To repeat the same fixed pool three times temporarily:

```bash
python3 scripts/run_train.py \
  --config configs/train_qwen38_envduels_lora.json \
  --epochs 3
```

That override produces 270 optimizer steps, 2160 episodes and 24 episodes per
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

Build, if the local image does not already exist:

```bash
BUILD_JOBS=64 bash scripts/unified_runtime.sh build
```

The image is `envduels-unified:cu124`. Its Dockerfile pins CUDA 12.4.1, Torch,
vLLM and ms-swift. Torch and vLLM were built for SM80 and SM89, covering A100
and RTX 4090. The image has completed CPU/import checks, and Qwen3.8-27B has
loaded and generated a token with TP=4 on RTX 4090. The eight-GPU optimizer-step
smoke remains the final validation when all eight GPUs are idle.

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
