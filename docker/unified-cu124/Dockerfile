# One CUDA 12.4 runtime for ms-swift LoRA GRPO and vLLM evaluation.
# PyTorch/vLLM are built for Ampere (A100) and Ada (RTX 4090) because upstream
# vLLM 0.20.2 requires Torch 2.11, for which no official cu124 wheel exists.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04@sha256:622e78a1d02c0f90ed900e3985d6c975d8e2dc9ee5e61643aed587dcf9129f42

ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_JOBS=64
ARG GIT_FETCH_JOBS=2
ARG TORCH_CUDA_ARCH_LIST="8.0;8.9"
ARG PYTORCH_COMMIT=70d99e998b4955e0049d13a98d77ae1b14db1f45
ARG VLLM_COMMIT=v0.20.2
ARG MS_SWIFT_COMMIT=8f07293409e7b24aa8bdc02e6fe43a90652121f1

LABEL org.envduels.runtime="unified-cu124" \
      org.envduels.pytorch.commit="${PYTORCH_COMMIT}" \
      org.envduels.vllm.commit="${VLLM_COMMIT}" \
      org.envduels.ms-swift.commit="${MS_SWIFT_COMMIT}"

ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    CMAKE_CUDA_ARCHITECTURES="80;89" \
    VLLM_TARGET_DEVICE=cuda \
    TOKENIZERS_PARALLELISM=false \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git build-essential python3 python3-dev python3-pip \
      python3-socks \
      libopenblas-dev libnuma-dev libaio-dev pkg-config ninja-build cmake \
      curl rsync tini && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --upgrade \
      'pip<26' 'setuptools>=77,<81' wheel 'cmake>=3.27,<4' ninja \
      'numpy==1.26.4' packaging pyyaml requests six typing-extensions \
      filelock sympy networkx jinja2 fsspec optree setuptools-scm

# Torch 2.11 compiled against the image's CUDA 12.4 toolkit.
RUN --mount=type=cache,id=envduels-pytorch-source,target=/opt/src \
    test "${BUILD_JOBS}" -ge 1 && test "${BUILD_JOBS}" -le 64 && \
    test "${GIT_FETCH_JOBS}" -ge 1 && test "${GIT_FETCH_JOBS}" -le 8 && \
    mkdir -p /opt/src/pytorch && cd /opt/src/pytorch && \
    if [ ! -d .git ]; then git init && \
      git remote add origin https://github.com/pytorch/pytorch.git; fi && \
    git config http.version HTTP/1.1 && \
    if ! git cat-file -e "${PYTORCH_COMMIT}^{commit}" 2>/dev/null; then \
      timeout --signal=TERM 300 git fetch --depth 1 origin "${PYTORCH_COMMIT}"; \
    fi && \
    git checkout --detach "${PYTORCH_COMMIT}" && \
    git config -f .gitmodules --get-regexp path | awk '{print $2}' | \
      while IFS= read -r path; do \
        if { [ -e "$path" ] && ! git -C "$path" rev-parse --verify HEAD >/dev/null 2>&1; } || \
           { [ -d ".git/modules/$path" ] && ! git --git-dir=".git/modules/$path" \
             rev-parse --verify HEAD >/dev/null 2>&1; }; then \
          echo "Removing incomplete submodule: $path"; \
          git submodule deinit -f -- "$path" >/dev/null 2>&1 || true; \
          rm -rf "$path" ".git/modules/$path"; \
        fi; \
      done && \
    find .git/modules -type f -name '*.lock' -delete 2>/dev/null || true && \
    git submodule sync --recursive && \
    attempt=1 && while ! timeout --signal=TERM 900 \
      git -c http.version=HTTP/1.1 submodule update \
      --init --recursive --depth 1 --jobs "${GIT_FETCH_JOBS}"; do \
      test "$attempt" -lt 5 || exit 1; \
      echo "Retrying incomplete PyTorch submodules (attempt $((attempt + 1))/5)"; \
      attempt=$((attempt + 1)); \
    done && \
    git submodule foreach --recursive 'git reset --hard HEAD' && \
    USE_CUDA=1 USE_CUDNN=1 USE_DISTRIBUTED=1 USE_NCCL=1 \
    BUILD_TEST=0 USE_CUSPARSELT=0 USE_CUFILE=0 \
    PYTORCH_BUILD_VERSION=2.11.0 PYTORCH_BUILD_NUMBER=1 \
    MAX_JOBS="${BUILD_JOBS}" CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}" \
      python3 setup.py bdist_wheel && \
    python3 -m pip install --no-deps dist/torch-*.whl && \
    cd /tmp && \
    python3 -c 'import torch; assert torch.__version__.split("+")[0] == "2.11.0"; assert torch.version.cuda == "12.4"'

# Build vLLM against that exact Torch/CUDA pair. FlashInfer's published wheels
# target newer CUDA toolkits; vLLM falls back to its CUDA/Triton backends here.
RUN git clone --filter=blob:none --branch "${VLLM_COMMIT}" --depth 1 \
      https://github.com/vllm-project/vllm.git /opt/vllm && \
    python3 -m pip install -r /opt/vllm/requirements/common.txt \
      'numba==0.65.0' 'fastsafetensors>=0.2.2' && \
    MAX_JOBS="${BUILD_JOBS}" CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}" \
      python3 -m pip install --no-build-isolation --no-deps -e /opt/vllm

# Qwen3.5's vLLM implementation imports torchvision transforms through the
# shared Qwen3-VL module.  Build it against our exact Torch/CUDA ABI; the PyPI
# wheel is built against a different CUDA Torch and cannot register its ops.
RUN git clone --filter=blob:none --branch v0.26.0 --depth 1 \
      https://github.com/pytorch/vision.git /opt/vision && \
    FORCE_CUDA=1 MAX_JOBS="${BUILD_JOBS}" \
      python3 -m pip install --no-build-isolation --no-deps -e /opt/vision && \
    cd /tmp && python3 -c 'import torchvision; assert torchvision.__version__.split("+")[0] == "0.26.0"'

# Pin ms-swift source plus the direct RL dependencies used by this project.
# The final freeze is retained in the image for cross-node provenance.
RUN git init /opt/ms-swift && cd /opt/ms-swift && \
    git remote add origin https://github.com/modelscope/ms-swift.git && \
    git fetch --depth 1 origin "${MS_SWIFT_COMMIT}" && \
    git checkout --detach FETCH_HEAD && \
    python3 -m pip install \
      'transformers==5.12.1' 'peft==0.20.0' 'accelerate==1.14.0' \
      'deepspeed==0.18.9' 'trl==0.27.0' 'datasets==4.8.4' \
      'openai==2.6.1' 'PyYAML==6.0.3' 'tqdm==4.70.1' \
      'math-verify==0.9.0' 'numpy==1.26.4' \
      'opencv-python-headless==4.11.0.86' tensorboard && \
    python3 -m pip install -e /opt/ms-swift

WORKDIR /opt/envduels-spade
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY LICENSES ./LICENSES
COPY spade ./spade
COPY eval_offline ./eval_offline
COPY scripts ./scripts
COPY cmd ./cmd
COPY configs ./configs
RUN python3 -m pip install --no-deps -e . && \
    python3 -c 'import numpy, torch, torchvision, transformers, peft, vllm, swift, spade; assert torch.ones(1).numpy().item() == 1; print(torch.__version__, torch.version.cuda, torchvision.__version__, numpy.__version__, transformers.__version__, vllm.__version__)' && \
    python3 -m pip freeze > /opt/envduels-unified-freeze.txt

ENV PYTHONPATH=/workspace/envduels/spade:/opt/envduels-spade:/opt/ms-swift
WORKDIR /workspace/envduels/spade
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
