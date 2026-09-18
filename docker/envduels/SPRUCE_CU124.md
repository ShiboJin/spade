# Spruce driver 555: experimental CUDA 12.4 build track

Status: 16-job retry built successfully; image ID
`sha256:13574f8da610cf5bbe602cf0375f7a8eb603d9b44bcc670c5931a8050d23f71d`.
CPU import reports Torch 2.11.0 / CUDA 12.4. Still a foundation, not a training runtime.
A tiny BF16 GPU matrix multiplication passed on spruce A100 GPU 0 using this
image and the unchanged 555.42.02 driver. This does not test NCCL, Triton,
SGLang, Megatron, weight conversion or training.
The existing CUDA 12.9 Dockerfile and default launcher are unchanged.
Do not set ENVDUELS_IMAGE to the foundation image for training.

## Why this is not a dependency downgrade

The host kernel driver is 555.42.02. The candidate toolkit is CUDA 12.4.1;
the [CUDA release notes](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-toolkit-release-notes/index.html)
list 550.54.15 as the corresponding Linux toolkit driver. This is a sensible
target, not proof that the full Qwen stack works on spruce.

The [official cu124 Torch wheel index](https://download.pytorch.org/whl/cu124/torch/)
offers Torch 2.6, whereas the
[Slime-pinned SGLang requirements](https://github.com/sgl-project/sglang/blob/5a15cde858ea09b77116212a39356f2fc51b8584/python/pyproject.toml)
require Torch 2.11. Downgrading Torch alone would violate that requirement.

Candidate: keep Torch 2.11 and compile against CUDA 12.4. Its
[CMake CUDA check](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/cmake/public/cuda.cmake)
accepts CUDA >=12.0, but passing that check does not establish build support.
Compile failure may require a different approach; do not disguise it by
relaxing package metadata or disabling NVIDIA_REQUIRE_CUDA.

## First stage

`Dockerfile.spruce-cu124` pins the NVIDIA base digest and Torch source commit,
targets SM80, builds distributed/NCCL support, and checks the installed Torch
version/toolkit without allocating a GPU. cuDNN comes from the base image;
remaining apt/Python build dependencies are not fully locked yet.

```bash
bash scripts/build_spruce_foundation.sh --plan
# Long CPU/disk-intensive operation; only run after allocating build resources:
BUILD_JOBS=4 bash scripts/build_spruce_foundation.sh --build
```

The default is four compiler jobs; BUILD_JOBS accepts 1..16. The interrupted
four-job build was successfully retried with 16 jobs after checking shared host load.
This bounds compiler parallelism, not total CPU or memory. It is not a fix
for Docker connection errors. An interrupted RUN layer may need recompilation.
Build context is only docker/envduels; weights and exports are not sent.
No host driver installation, privileged container, Docker socket mount,
GPU access, host Ray access, or cleanup of unrelated data is requested.
Logs are retained in outputs/logs; build artifacts consume shared Docker disk.

## Remaining stages (not implemented or validated)

1. Verify the foundation builds/imports; record resolved dependency versions.
2. Match Torch's Triton revision `9844da955a9db14ec69c9aac828ee9803085e288`
   and force CUDA 12.4 ptxas for SM80. A newer bundled ptxas/PTX can reintroduce
   the driver problem even when torch.version.cuda says 12.4.
3. Rebuild ABI-dependent CUDA extensions against this exact Torch: SGLang
   kernel, FlashAttention, Transformer Engine, Apex and torch_memory_saver.
   Audit FlashInfer/FLA and optional CUDA-13/FP8/FA4 imports as well. Existing
   cu129 extension wheels cannot simply be copied across.
4. Install the pinned Megatron/mbridge/Slime and SPADE, checking actual imports
   and package metadata. Preserve upstream patches and Qwen3.5 model support.
5. Only then register a separate training image/profile. On allocated GPUs,
   test CUDA/NCCL, Qwen conversion, inference, updates and weight sync.

EnvDuels environment adapters and GRPO semantics do not change in this track.
