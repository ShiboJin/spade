# EnvDuels fixed-environment integration

This uses SPADE's `fixed_env` adapter infrastructure, not the paper's
`train_fixed_gpt55_*` launcher. Official baseline launchers are unchanged.

The source is an EnvDuels format-version-1 export containing `manifest.json`
and its environment sources. All manifest entries are eligible by default;
the export's train/validation/test split is deliberately ignored. To choose a
subset, provide a text file containing exact manifest IDs, one per line
(blank lines and `#` comments are accepted). No environment count is fixed.

Parameters to add to a future, model-specific Slime training launcher:

```bash
--spade-mode fixed_env \
--spade-fixed-env-source "envduels:${ENVDUELS_EXPORT_DIR}" \
--spade-envduels-ids-file "${ENVDUELS_IDS_FILE}" \
--spade-envduels-seed 42 \
--spade-fixed-env-same-problem \
--spade-fixed-pool-size 0 \
--spade-reward-normalization grpo \
--spade-trajectories-per-game 8 \
--global-batch-size 32
```

These are configuration fragments, **not a complete training command**.
Omit `--spade-envduels-ids-file` to use all entries. Same-problem groups are
enforced by the adapter even without the flag. EnvDuels-only runs disable
adaptive difficulty; mixing sources is not supported in this first version.
GPU/model/optimizer configuration and the Slime GPU runtime still need separate
validation. Installing SPADE alone does not supply that runtime.

## Preparing the launcher without running training

`cmd/games/train_envduels.sh` defaults to `--dry-run`. It validates integer
parameters, GPU/group divisibility and manifest selection, then prints a
shell-escaped Ray submission command. It neither starts Ray nor executes
environments, installs dependencies, downloads weights, or creates run directories.
It sources the supplied model architecture shell script, so use a trusted config.

For example, **only to inspect a command** using the existing Qwen3-8B config:

```bash
MODEL_CONFIG="$PWD/cmd/models/qwen3-8B.sh" \
HF_CHECKPOINT=/models/Qwen3-8B \
REF_CHECKPOINT=/models/Qwen3-8B_torch_dist \
MEGATRON_DIR=/opt/Megatron-LM \
ENVDUELS_EXPORT_DIR=/absolute/path/to/duel_harness_004_rl \
OUTPUT_DIR=/runs/envduels-smoke-001 \
bash cmd/games/train_envduels.sh --dry-run
```

This example does not select Qwen3-8B as the project's target model. Set
`MODEL_CONFIG`, `HF_CHECKPOINT`, and `REF_CHECKPOINT` consistently for the
confirmed model. The available `slime/scripts/models/qwen3.5-27B.sh` is another
architecture config, but its A100 training compatibility has not been tested.

Defaults are provisional smoke-test settings: 8 GPUs, training TP=2/PP=1/CP=1,
rollout TP=2, 32 retained episodes in groups of 8, 2 iterations, context 8192,
per-turn response limit 1024. They are **not memory/performance guarantees**.
Override `NUM_GPUS`, `TP`, `PP`, `CP`, `ROLLOUT_TP`, `GLOBAL_BATCH_SIZE`,
`GROUP_SIZE`, `NUM_ROLLOUT`, `MAX_TURNS`, `MAX_CONTEXT_LENGTH`,
`ACTOR_MAX_TOKENS`, `MAX_TOKENS_PER_GPU`, `TEMPERATURE`, `THINKING`, `LR`,
`CPU_OFFLOAD`, `SAVE_INTERVAL`, `ENVDUELS_SEED`, and `ENVDUELS_IDS_FILE`
through the process environment. This initial launcher targets dense models;
MoE expert-parallel tuning is not supplied. Model-specific dimension/kernel
compatibility remains a separate check.

SPADE normalizes binary rewards per problem before Slime consumes them. The
launcher therefore disables Slime's second reward normalization and GRPO std
normalization. Initial KL coefficient is zero, without loading a KL reference
actor; `--ref-load` still supplies initial Megatron weights. In-loop evaluation
and W&B are not enabled in this preparation launcher.

Later, inside a validated GPU runtime with a dedicated Ray cluster, explicit
`--run` plus `RAY_ADDRESS=http://host:port` submits the command. Checkpoint and
Megatron paths must then exist, the output directory must be new, and the same
absolute paths and Python executable must be accessible to the Ray job. The
runtime environment includes this checkout, its pinned Slime submodule, and
`MEGATRON_DIR` on `PYTHONPATH`. No Ray lifecycle/cleanup commands are issued.
Use `LOAD_DIR` to resume a checkpoint into a new output directory (sampler
resume is still not exact). Do not submit to someone else's Ray cluster.

## Contracts

- A problem is `(env_id, seed)`. Each sampled group gets one seed and distinct
  environment objects. `reset()` reuses the bound seed; attempts to override
  it with a different seed fail. Metadata preserves `env_id`, `seed`,
  `problem_id`, and source hash. `game_file` aliases `problem_id` for existing
  SPADE reward normalization, including its fixed-pool branch.
- The adapter extracts the final balanced `\boxed{ACTION}` and passes only
  the enclosed action to `step`. Missing/malformed boxes return zero reward
  and corrective feedback, consuming one interaction turn. They are not
  environment exceptions. Each episode is bounded by the manifest's turn cap
  as well as the rollout's global limits.
- Reward is `float(terminated and raw_reward > 0)`. Nonterminal shaping rewards
  become zero. Raw rewards and format errors are retained in trajectory
  `reward_diagnostics`, without copying private environment info into prompts.
- Crashes propagate to SPADE's FAILED trajectory handling and are filtered.
  For the training batch, groups missing an episode are discarded; complete
  groups are selected together. Insufficient complete groups cause an explicit
  error instead of padding with duplicate trajectories.
- Same-problem grouping for advantages is distinct from Slime's scheduling
  group IDs. Existing token/loss-mask construction is retained.

## CPU verification

From the SPADE checkout, with its development environment active:

```bash
python -m unittest discover -s tests -p test_envduels_adapter.py -v
```

To include two mock-model episodes in the real exported RotationLock environment:

```bash
ENVDUELS_TEST_EXPORT=/absolute/path/to/duel_harness_004_rl \
  python -m unittest discover -s tests -p test_envduels_adapter.py -v
```

No model service, GPU, Slime installation, or pytest is needed for these tests.
They check episode execution and reward grouping, not a gradient update.

## Current boundaries

Only use trusted environment exports. Sources execute in-process, as in the
export registry; hash validation is integrity checking, not sandboxing. Bubblewrap
or another subprocess isolation layer is not implemented here. `privileged.json`
is not loaded. Problem RNG is reproducible on fresh runs with the same sampling
order, but exact sampler-state restoration on checkpoint resume is not implemented.
Checkpoints, export data, credentials and server-specific paths should stay outside
the code repository; configure paths per server.
