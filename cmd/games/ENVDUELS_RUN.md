# Qwen3.8-27B on EnvDuels: runbook

For the all-90-environments + AIME26 experiment, edit
`configs/envduels90_aime26.yaml` and use `scripts/run_envduels_experiment.py`
(or its existing `.sh` wrapper).
See `cmd/games/ENVDUELS90_AIME26.md` for the separate training/eval protocol
and the remaining runtime/checkpoint-export validation requirements.

Run all commands from the SPADE repository. The code is prepared for a single
node with 8 allocated A100 80GB GPUs out of the available 10. This is an initial
layout, not a measured throughput or memory guarantee. GPU conversion and
training have deliberately not been executed on the occupied server.

**Current spruce blocker:** the host driver is 555.42.02, while this image uses
CUDA 12.9.1. Its `NVIDIA_REQUIRE_CUDA` admits CUDA >=12.9 or specific compatible
driver branches, but does not admit 555. Do not disable this requirement.
An administrator must schedule a supported driver upgrade (e.g. a supported
driver >=575.57.08), or use another server with a compatible driver, before the
GPU commands below. No host driver changes were made. A separately built and
validated older-CUDA stack is another option, not provided by this recipe.
An experimental CUDA 12.4 source-build track is documented in
`docker/envduels/SPRUCE_CU124.md`; its foundation is not a training image.
See NVIDIA's [compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html).

## Preparation (no GPU)

```bash
bash scripts/envduels_runtime.sh build
bash scripts/envduels_runtime.sh plan
bash scripts/envduels_runtime.sh check
```

The base image digest, Slime submodule revision and main Python dependencies
are fixed in `docker/envduels/Dockerfile`. The built image includes a full
`/opt/envduels-runtime-freeze.txt` for provenance. Build context excludes weights
and the local venv. Preserve/export the built image for exact environment reuse
on another server; rebuilding still resolves transitive Python dependencies.

`plan` only prints commands. `check` mounts no GPUs and loads configuration,
tokenizer and safetensors headers, not model tensors. It checks every selected
environment source hash and imports the training runtime. Model weights live in
`checkpoints/Qwen3.8-27B`; the pinned download command is in the compatibility
report. Exports remain in the parent EnvDuels repository; override the host
`ENVDUELS_EXPORT_DIR` when moving servers.

The CPU argument check parses the real launch flags and checks the HF config.
Full Megatron argument validation queries CUDA and is deferred to GPU startup.
The base image also has pip metadata conflicts for optional stacks (FA4,
DeepGEMM/TileLang/TVM, video/NumPy, OpenAI Agents, NIXL). This profile does not
enable those backends; passing CPU imports is not a clean `pip check` or GPU
compatibility certification. Do not enable additional backends without validation.

## Once GPUs are allocated

Choose GPUs that are actually available. The IDs below are an example consistent
with this server's paired NVLinks, **not a reservation or availability check**.

```bash
export GPU_IDS=0,1,3,4,6,7,8,9
export NUM_GPUS=8
bash scripts/envduels_runtime.sh convert
bash scripts/envduels_runtime.sh smoke
```

Conversion creates `checkpoints/Qwen3.8-27B_torch_dist` and refuses to overwrite
an existing destination. It requires GPUs and must finish before training.
`smoke` runs two rollout/update iterations and saves checkpoints. Success means
finite losses, completed optimizer updates, a second rollout after weight sync,
and saved checkpoint files—not simply a server starting. Inspect the logs for
environment failures, zero-variance groups and invalid-action rates.

Then run an explicitly chosen training budget, for example:

```bash
NUM_ROLLOUT=100 bash scripts/envduels_runtime.sh train
```

The number 100 is an example budget, not a paper requirement. Outputs go under
`outputs/qwen38-envduels-<UTC timestamp>/checkpoints`. Every operation creates an
isolated container with its own network and Ray processes. Nothing stops or
attaches to the host's Ray cluster. Container processes run as the calling user's
UID/GID so checkpoint files remain user-owned. Container removal at exit does
not remove bind-mounted checkpoints.
Console logs persist under `outputs/logs/` for conversion, smoke and training.

## Experiment choices

- By default all 90 manifest entries are eligible; no 72/9/9 split is imposed.
  Put exact manifest IDs into a text file under the checkout and pass its
  **container path**, e.g.
  `ENVDUELS_IDS_FILE=/workspace/envduels/spade/configs/my_envs.txt`.
- The initial profile uses TP=2, PP=2, CP=1, DP=2, rollout TP=2, BF16 (Slime
  default), optimizer CPU offload, 32 retained episodes in groups of 8,
  8192 context tokens, 1024 response tokens per turn, thinking disabled.
  Triton attention and disabled rollout CUDA graphs are conservative initial
  settings. CPU RAM/offload throughput and A100 kernels still require the
  first GPU run. Increase context/batch only after the smoke run passes.
- `THINKING=true` retains reasoning history and trains generated reasoning
  tokens too. The initial 1024 per-turn budget is only a smoke-test budget;
  reasoning runs may need a larger budget to avoid truncation.
- `GROUP_SIZE`, `GLOBAL_BATCH_SIZE`, `LR`, `NUM_ROLLOUT`, `MAX_CONTEXT_LENGTH`,
  `ACTOR_MAX_TOKENS`, `TP`, `PP`, `CP`, `ROLLOUT_TP` are environment overrides.
- Normalized binary success is computed per `(env_id, seed)` group. Slime's
  second reward normalization is disabled. KL coefficient starts at zero.
- Invalid format consumes a turn and returns zero; environment crashes are
  excluded. Incomplete groups are discarded; insufficient complete groups stop
  the run instead of duplicating data. All-zero/all-one groups have zero GRPO
  advantage and should be monitored.
- Checkpoint resume uses `LOAD_DIR` (container path) and a new `OUTPUT_DIR`.
  Exact environment sampler-state restoration is not implemented yet.

This recipe trains the text policy. Vision/MTP training, speculative decoding,
W&B, automatic AIME evaluation and multi-node scheduling are not enabled.
Keep AIME separate from training; dataset selection and evaluation protocol have
not yet been specified. Environment source executes in-process inside the
container; use trusted exports. The export mount is read-only, but this is not
a per-environment security sandbox.

## CPU regression checks

```bash
ENVDUELS_TEST_EXPORT="$PWD/../exports/duel_harness_004_rl" \
  .venv/bin/python -m unittest discover -s tests -p 'test_envduels*.py' -v
HF_HUB_OFFLINE=1 .venv/bin/python scripts/preflight_envduels.py
```
