# Qwen3.8-27B: static compatibility audit

Model repository: https://huggingface.co/Qwen/Qwen3.8-27B

Pinned model revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`

Local download directory (relative to SPADE): `checkpoints/Qwen3.8-27B`.
This directory is Git-ignored. Keep this document in Git, not the weights.
The repository snapshot contains 18 safetensors shards and 55,586,114,863
bytes in total across its listed files.

Download completed: all 32 remote files were checked against the pinned
revision's filenames and byte sizes; all 18 shards referenced by the weight
index exist. This is a presence/size check, not an independent full-file hash
audit. Offline CPU loading of AutoConfig and AutoTokenizer succeeded, as did
rendering/tokenizing the chat template with thinking both enabled and disabled.
Tokenizer EOS is 248046 (different from the text config's EOS of 248044);
generation must use the tokenizer/generation configuration consistently.

## Conclusion

Basic architecture dimensions match the pinned Slime Qwen3.5-27B config.
The earlier suspected swish/sigmoid mismatch was a comparison of two different
gates, not an established incompatibility. Upstream SGLang consumes
`output_gate_type` in linear-attention RMSNorm gating, while full-attention
output gating remains sigmoid. In the pinned SGLang, RMSNormGated already
defaults to swish; Slime's GatedDeltaNet uses `hidden_act=silu` for that norm.
Swish/SiLU therefore agrees at that location. No full-attention gate patch is
needed or applied. See the [upstream implementation](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_5.py)
and the [pinned norm implementation](https://github.com/sgl-project/sglang/blob/5a15cde858ea09b77116212a39356f2fc51b8584/python/sglang/srt/layers/attention/fla/layernorm_gated.py).

No GPU model load, weight conversion, training, or backend changes were performed
for this audit. Static inspection cannot establish numerical equivalence.

## Matching configuration

| Field | Model / existing Slime Qwen3.5-27B configuration |
|---|---|
| Architecture / model type | Qwen3_5ForConditionalGeneration / qwen3_5 |
| Decoder layers | 64 |
| Hidden / FFN dimensions | 5120 / 17408 |
| Attention query / KV heads | 24 / 4 |
| Attention head dimension | 256 |
| Vocabulary | 248320 |
| RoPE base / partial factor | 10000000 / 0.25 |
| RMSNorm epsilon | 1e-6 |
| Word embedding tying | false |

The model declares three linear-attention layers followed by one full-attention
layer, repeated 16 times. Its linear attention uses 16 key heads, 48 value heads,
128-dimensional heads and convolution width 4. Slime's Qwen3.5 plugin reads the
nested HF text configuration for these parameters and the layer layout.

## Inspected implementation references

- Slime submodule: `bf14dc21f9500746447f2572d0692e981c4d2a7e`.
- `slime/scripts/models/qwen3.5-27B.sh`: architecture flags above.
- `slime/slime_plugins/models/qwen3_5.py`: nested text config, Gated DeltaNet.
- `slime/slime_plugins/mbridge/qwen3_5.py`: registered for `qwen3_5` and
  `qwen3_5_moe`, text weight mappings and MTP handling. Mapping support alone
  does not validate Qwen3.8 forward semantics or successful conversion.
- Slime-pinned Megatron commit `1dcf0dafa884ad52ffb243625717a3471643e087`,
  `megatron/core/transformer/attention.py`, `_apply_output_gate`:
  `x * torch.sigmoid(gate.float())`.
- Slime-pinned SGLang commit `5a15cde858ea09b77116212a39356f2fc51b8584`,
  `python/sglang/srt/models/qwen3_5.py`: sigmoid gate in full attention.
- Development venv Transformers 5.17.0 also uses sigmoid at the inspected
  `models/qwen3_5/modeling_qwen3_5.py` attention output. The venv has no PyTorch;
  this is source inspection, not execution.

## Other boundaries

The prepared runtime uses SGLang 0.5.15.post1 at commit
`0b3bb0cbe31873994c9f989fddfe2f87ca839fdd`, not the older source inspected above.
Its Qwen3.5 implementation explicitly forwards `output_gate_type` into the GDN
norm. Runtime versions include PyTorch 2.11.0+cu129, Transformers 5.12.1,
Megatron 0.16.0rc0 (the commit above), FLA 0.4.2 and mbridge 0.15.1
(`89eb10887887bc74853f89a4de258c0702932a1c`). CPU model/bridge imports pass.
The development venv remains separate. See ENVDUELS_RUN.md for the current
555.42.02 host-driver blocker; this runtime is not yet GPU-certified.

The checkpoint includes vision components and one MTP layer. Our intended
workload is text-only. Conversion and rollout must explicitly agree about
which weights participate; neither vision training nor speculative decoding
has been validated here.

Use the checkpoint's own tokenizer and `chat_template.jinja`. The template has
`enable_thinking`, `reasoning_effort` and `preserve_thinking` behavior. The
EnvDuels loop now forwards template kwargs, separates Qwen3.8 reasoning_content,
and appends observations against the actual decoded sampled prefix. It refuses
to rewrite sampled tokens if a template changes history. CPU preflight tests
both thinking modes. A working tokenizer alone is insufficient to establish
GPU numerical equivalence.

The A100 runtime/kernels and train-versus-rollout log probabilities remain
untested. The download does not change that status.

## Reproduce the download

From the SPADE checkout with its venv active:

```bash
hf download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-dir checkpoints/Qwen3.8-27B --max-workers 4
```
