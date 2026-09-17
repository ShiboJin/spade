FROM slimerl/slime@sha256:39be6cbb00f9b6770e664ace0c7b9f5ecff2977a1a205e7926a720f906fbc62c
WORKDIR /opt/envduels-spade
COPY slime /opt/envduels-spade/slime
# Always install the repository's pinned Slime, not the image's moving checkout.
RUN python -m pip install --no-deps -e /opt/envduels-spade/slime
RUN python -m pip install --no-deps \
    'git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c' \
    'ring-flash-attn==0.1.8'
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY LICENSES ./LICENSES
COPY spade ./spade
RUN python -m pip install --no-deps -e . && \
    python -m pip install 'transformers==5.12.1' 'openai==2.6.1' \
      'PyYAML==6.0.3' 'tqdm==4.70.1' 'math-verify==0.9.0' \
      'weave==0.53.9' 'pytest==8.4.2' && \
    chmod 755 /root && python -m pip freeze > /opt/envduels-runtime-freeze.txt
ENV PYTHONPATH=/opt/envduels-spade:/opt/envduels-spade/slime:/root/Megatron-LM
ENV WANDB_MODE=disabled WEAVE_DISABLED=true TOKENIZERS_PARALLELISM=false
WORKDIR /workspace/envduels/spade
