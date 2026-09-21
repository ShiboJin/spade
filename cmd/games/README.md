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

`scripts/run_train.py` enables `training.sage_hint_resampling` by default. Set it
explicitly to `false` to reproduce the original no-hint training protocol.

The fixed pool need not divide evenly by `num_games_per_rollout`. Swift/TRL's
sampler drops the incomplete final generation batch after shuffling each pass;
for example, 90 environments with 8 groups per batch produce 11 complete windows
and omit 2 environments from that pass's initial sampling. The dataset stays
intact for future passes and SAGE refill. With shuffling disabled, the same tail
is omitted each pass. The pool must still contain at least one full batch.
Dry-run/resolved configuration reports `sampled_environments_per_dataset_pass`
and `dropped_environments_per_dataset_pass`; step counts use complete batches.
These counts describe initial sampling, before hint rescue/refill or skips.

Each group has `trajectories_per_game` independent episodes with the same exported
environment and fixed seed. Sampling starts without a hint. If every terminal
reward is zero, resample the **entire group** using the author's hint from the
manifest's `privileged.json`. Single-hint exports have levels 0/1; graded exports
use 0, `hint_1`, then `hint_2`. Stop escalation at the first group with any success,
or after the last available hint. Hints are player context only; environment
execution and binary terminal rewards are unchanged. Selected hinted trajectories
keep the hint in their user message throughout old/current-policy log-prob
computation. The hint itself is not an assistant target.

Discard any selected constant-reward group (all 0 or all 1, including an unhinted
all-1 group), then refill its slots from other environments in the fixed dataset.
Each replacement again starts without a hint. This does not remove environments
from the pool or modify dataset rows. `max_rollout_attempts` bounds total group
selection passes per batch, **including the initial pass**; each pass may visit
all available hint levels. Swift's separate DAPO `dynamic_sample` loop is disabled
while SAGE owns this refill process.

The supplied training configs set `max_rollout_attempts: 2`: the initial pass
plus **one additional refill pass**. Set `min_valid_groups: 4` to accept a smaller
effective batch after that bounded refill (or when no other environments remain).
With 6 requested groups and 4 trajectories/group:

| Valid groups | Effective trajectories | Action |
|---|---|---|
| 6 | 24 | Update |
| 5 | 20 | Update |
| 4 | 16 | Update |
| 0–3 | 0 | Skip window |

Swift/FSDP retains the original physical batch slots and accumulation schedule.
Invalid groups have an explicit whole-group mask: their completion tokens and
advantages are masked out of the loss, including any KL term. GRPO is normalized
by the actual valid trajectory count, not the padded slot count. All ranks agree
on the selected groups; a rank containing only masked slots still participates in
forward/backward collectives with zero loss contribution. Thus a partial update
can still consume compute for the masked slots.

Below the threshold, skip the entire window: no backward, optimizer step, AdamW
weight decay/momentum update, or learning-rate scheduler step. Continue to the
next window; no cross-window cache and no constant-reward fallback. Environments
are never permanently removed from the fixed pool.

Supported runtime: synchronous ms-swift colocated Gym GRPO, fixed group sizes,
no sequence parallelism, `num_substeps=1`, and one generation batch per complete
gradient-accumulation window (`steps_per_generation=gradient_accumulation_steps`).
Effective batch masking supports standard `loss_type=grpo`, group/none reward
scaling, and no Liger, teacher, CHORD or KL-in-reward path. Ordinary KL loss is
supported and masked along with policy loss.
Decisions are shared across training ranks; even ranks without local retry samples
join collectives. Existing evaluation entrypoints stay no-hint, and in-training
eval explicitly resets the hint level to zero.

Inspect `checkpoint/v*/hint_resampling.jsonl` under each training run (the actual
Trainer `output_dir`) for env/seed, hint hashes, per-level reward lists, selected
levels, refill attempts, discarded groups, `batch_selection` events with valid
and masked group IDs, and `skipped_update` events with reasons.
`sage/*` metrics appear in normal Trainer logs and configured reporters (including
W&B). `sage/skipped_update` is 1 for a skipped window and 0 for a completed training
window; `sage/applied_update` is the complement. Log aggregation can average these
values over multiple windows. Trainer `global_step`, checkpoint cadence and the
configured training duration still count consumed windows, including skips;
they do not count only actual optimizer updates. Skips do not extend training.
`sage/valid_groups`, `sage/effective_trajectories`, `sage/masked_groups` and
`sage/partial_batch` show effective batch sizes. `sage/effective_reward` measures
only the valid trajectories used by the update. Swift's ordinary rollout/reward
logs still include the masked physical slots; use the SAGE selection records and
effective metrics to distinguish them. Neither training reward is a no-hint
evaluation score.

CPU verification in the unified runtime (no model weights loaded):

```bash
python3 -m unittest discover -s tests -p 'test_hint_resampling.py'
python3 tests/sage_runtime_smoke.py
python3 tests/sage_effective_batch_smoke.py
python3 tests/sage_swift_smoke.py
```
