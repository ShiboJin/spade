# Local checkpoint evaluation on the 90 EnvDuels environments

Run from this checkout:

```bash
python3 scripts/run_envduels_solver_eval.py --config configs/qwen38_duel_harness_004_rl_solver_eval.json
```

The default config is `configs/qwen38_duel_harness_004_rl_solver_eval.json`. It uses the local merged checkpoint 24 on GPUs 0 and 1. The checkpoint plays the 90 hardened environments from `duel_harness_004_rl`: four frozen seeds per environment, once without the hint and once with the saved author hint, for 720 episodes total. The script resumes completed episodes in the same output directory. Use `--retry-errors` to rerun incomplete episodes.

The `duel_harness_004_push` snapshot supplies the original models' per-episode results **for reporting and ranking only**. No environment is executed from push. Before starting Docker, the script checks that all 90 hardened sources, four seeds, and hints in the RL export match those in push.

```bash
# Validate the plan without starting Docker or using a GPU.
python3 scripts/run_envduels_solver_eval.py --config configs/qwen38_duel_harness_004_rl_solver_eval.json --dry-run

# Evaluate another checkpoint in an independent output directory.
python3 scripts/run_envduels_solver_eval.py \
  --config configs/qwen38_duel_harness_004_rl_solver_eval.json \
  --checkpoint checkpoints/MY-MERGED-MODEL \
  --solver-name my-model \
  --output-dir outputs/envduels_solver_eval/my-model \
  --gpu-ids 2 3

# Recompute reports from saved episodes without starting the model.
python3 scripts/run_envduels_solver_eval.py --config configs/qwen38_duel_harness_004_rl_solver_eval.json --rank-only
```

The configured output directory contains `episodes.jsonl`, full `trajectories/`, `progress.json`, `status.json`, `console.log`, `server.log`, and these reports:

- `environment_results.json`: no-hint and hinted successes, attempts, accuracy, and matched hint gain for every model on every environment.
- `solver_summary.json`: the new checkpoint's overall accuracy, hint gain, and solve rank.
- `ranking_solver.json`: all model solve ranks, hint gains, hint gain differences relative to the new checkpoint, and original author design ranks. The new solver has no design rank because it authored no environments.

Canonical solve is a domain-balanced no-hint mean and excludes an original model's self-authored environments. Direct common-panel solve uses the environment intersection shared by all models. The push snapshot has no terminal original-model outcomes on `glm-5.3/env_007/harden_01`, so the direct common panel covers at most 89 environments; the new checkpoint still plays all 90.
