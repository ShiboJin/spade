# One experiment: all 90 environments, independent AIME26 evaluation

Edit **configs/envduels90_aime26.yaml**. It is a data-only YAML file, never
executed as shell code. Sections: hardware, environments, training, evaluation.
Unknown/duplicate keys and invalid values are rejected. Existing shell environment
variables may override corresponding fields and are validated as well.
hardware.gpu_ids starts as an empty list deliberately; enter allocated GPUs.
The foundation CUDA 12.4 image is NOT the complete training/evaluation runtime.
These entrypoints do not install missing CUDA extensions or bypass driver checks.

## CPU-only preparation

```bash
# Show effective values without Docker, downloads or GPU access:
.venv/bin/python scripts/run_envduels_experiment.py config
# Python entrypoint; choose your copied YAML for a separate experiment:
.venv/bin/python scripts/run_envduels_experiment.py eval-plan \
  --config configs/envduels90_aime26.yaml
# Existing shell entrypoint remains a thin wrapper (no shell config to edit):
bash scripts/run_envduels_experiment.sh prepare-eval
bash scripts/run_envduels_experiment.sh plan
bash scripts/run_envduels_experiment.sh eval-plan
```

`plan` uses the existing container without GPU access. `eval-plan` uses the
development venv (or PYTHON_BIN) and does not load a model. All 90 manifest
entries are eligible, independently of original split labels. This means
sampling from a pool of 90, NOT visiting every environment on every update or
guaranteeing full coverage in a short run. Existing sampling logic is unchanged.
The launcher fails if the manifest selection is not exactly 90 entries.

## Once the full runtime is ready and GPUs are allocated

1. `check`: import/configuration checks.
2. `convert`: initial HF-to-Megatron conversion, once per starting model.
3. `smoke`: two rollout/update iterations, checkpoint saved each iteration.
4. `eval`: evaluate the untrained HF model to establish a baseline.
5. `train`: train with your selected hyperparameters and budget.
6. Export a selected training checkpoint to HF, then set evaluation.hf_checkpoint
   to that exported directory and run `eval` with the same protocol.

Example **after** full runtime/GPU checks, not a ready-to-run claim on spruce:

```bash
GPU_IDS=0,1,3,4,6,7,8,9 bash scripts/run_envduels_experiment.sh smoke
LR=5e-7 NUM_ROLLOUT=100 GPU_IDS=0,1,3,4,6,7,8,9 \
  bash scripts/run_envduels_experiment.sh train
# Evaluation uses one TP=2 engine, so allocate only two GPUs if desired:
GPU_IDS=0,1 NUM_GPUS=2 EVAL_TP=2 bash scripts/run_envduels_experiment.sh eval
```

The IDs are examples, not GPU reservations. Training saves the resolved command
and manifest in OUTPUT_DIR/resolved_training.json, checkpoints below that run,
and console logs in outputs/logs. Eval saves resolved protocol/data hash/model
path, raw per-sample responses, scores and server logs in outputs/aime26.
The script never feeds AIME questions or answers into EnvDuels training.

## AIME26 protocol

Source: [math-ai/aime26](https://huggingface.co/datasets/math-ai/aime26), a community
copy, **not** an official MAA distribution. Pinned revision:
`79037aebdb6580008fb960d17cb21fd3099083e3`; 30 unique questions, integer answers.
File SHA256: `52822957957a3f577d1e9706c36a66a8108a3f99b6aff424cfb72dff0094a9ee`.
Data is ignored by Git; prepare-eval downloads it on each new server.
Schema/uniqueness checks do not independently verify every answer against MAA.

Defaults: 32 samples/question, temperature .7, top_p .8, top_k 20,
32768 output tokens, 40960 total context, thinking=false. These are experiment
choices, not a claim to reproduce a Qwen paper/model-card result. Use identical
settings for baseline and RL. Seed is forwarded but GPU execution is not
guaranteed bitwise reproducible. Scoring takes the last balanced boxed answer,
requiring an integer 0..999; leading zeroes are normalized.

`sample_accuracy` is mean correctness over all 960 completions (Avg@32).
`fraction_problems_any_correct` is the fraction of questions with at least one
correct completion among those 32; it is not Avg@32. Results also report length
stops and invalid answer formats. Missing data or missing completions fail;
there is no silent partial-dataset success. Repeated hyperparameter selection
on AIME26 makes it a development benchmark, not an untouched final test set.

## Remaining validation / checkpoint export

Full CUDA 12.4 SGLang/Megatron/extension stack is still missing. No real AIME
model generation or RL update has been validated yet. EVAL_HF_CHECKPOINT must
contain HF config + safetensors; raw Megatron output is rejected.
Slime provides tools/convert_torch_dist_to_hf.py with --input-dir (iteration
directory containing common.pt and distributed checkpoint metadata), --output-dir,
--origin-hf-dir and --model-name qwen3_5. Exporting our real checkpoint, validating
text weight coverage and handling original vision/MTP weights remains a GPU
integration task; do not point evaluation at the original base model and label
its score as a trained-checkpoint result. No automatic export or periodic eval
has been enabled here.
