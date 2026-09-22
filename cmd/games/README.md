# Games training

| Command | Paper setting | Provenance |
|---|---|---|
| `train_spade_4b.sh` | SPADE, Qwen3-4B-Instruct-2507 | Initial defaults match; later phase requires the overrides below |
| `train_spade_8b.sh` | SPADE, Qwen3-8B | Defaults match the paper: both roles thinking, KL `0.005` |
| `train_spade_30b.sh` | SPADE, Qwen3-30B-A3B-Instruct-2507 | Strong command match |
| `train_fixed_rlve_{4b,8b,30b}.sh` | Fixed-env RLVE baselines | Reconstructed |
| `train_fixed_gpt55_{4b,8b,30b}.sh` | Fixed-env GRPO on GPT-5.5 curated games | Shared 7,872-game pool and `400/24/192` budget |

Fixed-env GRPO means actor-only GRPO on the shared GPT-5.5 curated-game pool;
it is distinct from the RLVE baseline. The three model entry points download
and verify the same pool before invoking the common trainer. Set
`HF_DATASET_REVISION` to override the pinned paper snapshot
`383e4512938c148a7bef2e72b985075eea180647`.

The adaptive SPADE launchers require `CORPUS_FILE=/path/to/games-corpus.jsonl`.
The grounding corpus is not bundled because its authoritative public snapshot,
checksum, license, and redistribution terms remain unresolved. The
`train_no_corpus_30b.sh` ablation explicitly sets `CORPUS_FILE` to an empty
value and is the only paper command that intentionally runs without it.

The paper records a later 4B phase with a 49,152-token context and plateau band
`[0.2, 0.4]`. Resume that phase with
`MAX_CONTEXT_LENGTH=49152 PLATEAU_LO=0.2 PLATEAU_HI=0.4 PLATEAU_RAMP=0.2`.
The exact transition checkpoint is not encoded in the paper or launcher, so it
remains a required external input.

Files beginning with `_` are shared implementation helpers and are not separate
experiments.

## Baseline settings

The fixed-RLVE baseline does not share the SPADE hyperparameters; it reproduces
the paper's baseline configuration, so these differences are deliberate:

| Setting | `train_spade_{4b,8b,30b}.sh` | `train_fixed_rlve_{4b,8b,30b}.sh` |
|---|---|---|
| `--rollout-batch-size` / `--global-batch-size` | 24 / 192 | 16 / 256 |
| `--spade-actor-temperature` | 0.6 | 1.0 |
| `--kl-loss-coef`, 8B | 0.005 | 0.00 |

At 4B and 30B both settings use `--kl-loss-coef 0.00`; the 8B SPADE run is the
only KL-anchored one. Rollout budget (`--num-rollout 400`) and rollout sampling
temperature (`--rollout-temperature 1.0`) match across both.

### EnvDuels author-hint GRPO

`scripts/run_train.py` keeps `training.sage_hint_resampling` disabled by default,
so each group receives only its initial rollout. Set it explicitly to `true` to
enable the optional author-hint retry described below.

The fixed pool need not divide evenly by `num_games_per_rollout`. Swift/TRL's
sampler drops the incomplete final generation batch after shuffling each pass;
for example, 90 environments with 8 groups per batch produce 11 complete windows
and omit 2 environments from that pass's initial sampling. The dataset stays
intact for future passes. With shuffling disabled, the same tail
is omitted each pass. The pool must still contain at least one full batch.
Dry-run/resolved configuration reports `sampled_environments_per_dataset_pass`
and `dropped_environments_per_dataset_pass`; step counts use complete batches.
These counts describe initial sampling, before hint rescue.

Each group has `trajectories_per_game` independent episodes with the same exported
environment and fixed seed. Sampling starts without a hint. If every terminal
reward is zero, resample the **entire group** using the author's hint from the
manifest's `privileged.json`, **once**, using the same environment and seed.
For graded exports this uses `hint_1` only; it never escalates to `hint_2`.
An unhinted all-1 group is retained without retry. A mixed 0/1 group is retained.
After the hinted retry, retain every resulting group, including all-0/all-1 groups.
There is **no replacement-environment refill**. Environments remain in the dataset
for future windows/epochs. Swift's separate DAPO `dynamic_sample` loop is disabled.

Hints are player context only; environment execution and binary terminal rewards
are unchanged. Selected hinted trajectories keep the hint in their user message
throughout old/current-policy log-prob computation. The hint itself is not an
assistant target. The adapter forwards the final action with its `\boxed{}`
envelope intact, because exported environments parse that envelope themselves.

SAGE requires `max_rollout_attempts: 1` (one environment-selection pass; its
all-zero groups may still have one hinted retry). The retired `min_valid_groups`
setting is ignored, including in older configs and environment variables.

Every batch follows the normal backward, optimizer and LR scheduler path,
including batches with zero or one mixed-reward group. There is no SAGE group
mask and no effective-batch rescaling: GRPO averages over the full batch.
With 10 groups and only one mixed group, its policy gradient is diluted by 10
relative to averaging over that group alone. Constant groups have zero centered
reward advantage; with beta=0 an entirely constant batch has no policy-gradient
signal, but optimizer momentum/weight decay and the scheduler still run.
Standard prompt/padding/overlong token masks remain in effect.

Supported runtime: synchronous ms-swift colocated Gym GRPO, fixed group sizes,
no sequence parallelism, `num_substeps=1`, and one generation batch per complete
gradient-accumulation window (`steps_per_generation=gradient_accumulation_steps`).
The integration supports standard `loss_type=grpo`, group/none reward
scaling, and no Liger, teacher, CHORD or KL-in-reward path. Ordinary KL loss is
supported for constant groups as well.
Decisions are shared across training ranks; even ranks without local retry samples
join collectives. Existing evaluation entrypoints stay no-hint, and in-training
eval explicitly resets the hint level to zero.

Inspect `checkpoint/v*/hint_resampling.jsonl` under each training run (the actual
Trainer `output_dir`) for env/seed, hint hashes, per-level reward lists, selected
levels, constant-reward groups, and `batch_selection` events with all groups
accepted. `discarded_groups`, `masked_groups`, `partial_batch`, and
`refill_attempts` are zero. `sage/valid_groups` counts mixed-reward groups for
monitoring only. `sage/effective_trajectories` counts the entire retained batch
(before standard overlong filtering), and `sage/effective_reward` averages it.
`sage/skipped_update=0` and `sage/applied_update=1` denote normal completed
training windows; they do not guarantee a nonzero reward gradient or parameter
change. Training rewards include hinted trajectories and are not no-hint eval.

CPU verification in the unified runtime (no model weights loaded):

```bash
python3 -m unittest discover -s tests -p 'test_hint_resampling.py'
python3 tests/envduels_action_smoke.py
python3 tests/sage_runtime_smoke.py
python3 tests/sage_effective_batch_smoke.py
python3 tests/sage_swift_smoke.py
```
