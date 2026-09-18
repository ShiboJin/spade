# Configurable local evaluation

Edit `evaluation` in `configs/envduels90_aime26.yaml`. The filename is retained
for existing commands; copy the YAML to a new experiment filename if desired.
Training remains independent of the selected evaluation dataset.

```yaml
evaluation:
  benchmark: boxed_integer
  data: data/my_benchmark/test.jsonl
  # Keep the remaining checkpoint and sampling settings from the full YAML.
```

Supported protocols:

| benchmark | Data and grading |
| --- | --- |
| aime2026 | Existing pinned 30-question dataset with SHA256 validation; boxed integer 0–999 |
| boxed_integer | Any nonempty local JSONL question set; signed integer equality, ignoring leading zeros |
| boxed_exact_match | Any nonempty local JSONL question set; case-sensitive boxed string equality after trimming outer whitespace |

Generic JSONL rows use `{"id":"question-1","problem":"What is 2+2?","answer":"4"}`.
IDs must be unique strings or integers. A boxed final answer is requested in
the prompt. Exact match is NOT mathematical-equivalence grading: `1/2` and
`0.5` differ. Coding, tool-use and interactive benchmarks require dedicated
adapters, not merely a different data path. Extend `scripts/benchmark_data.py`
for additional single-turn answer protocols; other interaction types also
require changes to the evaluation runner.

```bash
python scripts/run_envduels_experiment.py prepare-eval --config configs/envduels90_aime26.yaml
python scripts/run_envduels_experiment.py eval-plan --config configs/envduels90_aime26.yaml
# Only after the GPU runtime and checkpoint are ready:
python scripts/run_envduels_experiment.py eval --config configs/envduels90_aime26.yaml
```

`prepare-eval` downloads the pinned data only for `aime2026`; for other protocols
you supply the file and this command validates it. All paths are relative to
the SPADE checkout. Evaluation is an explicit separate action, not an automatic
periodic training callback. Use an exported HF checkpoint for a trained model.
Keep baseline/post-training sampling settings identical.

Results go to `outputs/evaluation/<benchmark>/<timestamp>/`, including the
resolved protocol, dataset SHA256, per-sample responses and scores. Main score
is average sample correctness (Avg@N); any-correct fraction is reported separately.

GPU YAML parameters configure single-node parallelism, not hardware discovery or
dependency installation. GPU count must divide appropriately by TP*PP*CP and
rollout TP; batch size must divide by data-parallel size. Model dimensions,
GPU memory, driver/CUDA compatibility and compiled GPU architectures still need
separate validation. The CUDA 12.4 foundation image alone is not a full training runtime.
