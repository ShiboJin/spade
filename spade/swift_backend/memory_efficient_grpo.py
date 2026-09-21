"""ms-swift plugin: bound Qwen3.5 GRPO output-projection/log-softmax memory.

Uses ms-swift's existing FSDP-aware forward redirection (also used by its Liger
path). The root FSDP forward/backward hooks must run around the backbone and
the output projection; calling the unwrapped backbone alone is not sufficient.
"""
from functools import wraps
import os

from accelerate.utils import is_peft_model
import torch
from torch.utils.checkpoint import checkpoint
from swift.rlhf_trainers.grpo_trainer import GRPOTrainer
from swift.rlhf_trainers.utils import _ForwardRedirection
from swift.utils import get_logger
from torch import nn

from spade.swift_backend.chunked_logps import chunked_linear_logps
from spade.swift_backend.chunked_delta_rule import checkpointed_delta_rule


def install_delta_checkpointing():
    from functools import wraps
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    original = qwen.torch_chunk_gated_delta_rule
    if getattr(original, "_spade_checkpointed_delta_rule", False):
        return

    @wraps(original)
    def delta_rule(*args, **kwargs):
        return checkpointed_delta_rule(original, *args, **kwargs)

    delta_rule._spade_checkpointed_delta_rule = True
    qwen.torch_chunk_gated_delta_rule = delta_rule


def checkpoint_decoder_layers(model, *, offload_inputs=True):
    """Recompute complete decoder blocks inside their FSDP forward hooks.

    Keeping the modules themselves intact preserves FSDP's wrap selection and
    parameter names. Save each block's input on CPU; masks and position tensors
    remain shared on GPU instead of being copied once per layer. Preserve RNG
    state so LoRA dropout is identical on replay.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    for layer in model.modules():
        if not isinstance(layer, Qwen3_5DecoderLayer) or getattr(layer, "_spade_checkpointed", False):
            continue
        # During block replay, do not retain attention intermediates alongside
        # the expanded MLP activations. Recompute those submodules in turn.
        for name in ("linear_attn", "self_attn", "mlp"):
            child = getattr(layer, name, None)
            if child is None:
                continue
            child_forward = child.forward

            def submodule_forward(*args, _forward=child_forward, _module=child, **kwargs):
                # PyTorch discovers CUDA RNG devices from positional inputs.
                # Qwen attention passes hidden_states by keyword.
                if not args and "hidden_states" in kwargs:
                    args = (kwargs.pop("hidden_states"),)
                if _module.training and torch.is_grad_enabled():
                    return checkpoint(_forward, *args, use_reentrant=False, preserve_rng_state=True, **kwargs)
                return _forward(*args, **kwargs)

            child.forward = submodule_forward
        original_forward = layer.forward

        def forward(*args, _forward=original_forward, _layer=layer, **kwargs):
            if not args and "hidden_states" in kwargs:
                args = (kwargs.pop("hidden_states"),)
            if _layer.training and torch.is_grad_enabled():
                if not offload_inputs:
                    return checkpoint(_forward, *args, use_reentrant=False, preserve_rng_state=True, **kwargs)
                hidden_id = id(args[0] if args else kwargs["hidden_states"])

                def pack(tensor):
                    # Saved values must not retain their original autograd graph.
                    if id(tensor) == hidden_id:
                        return tensor.device, tensor.detach().cpu()
                    return None, tensor.detach()

                def unpack(saved):
                    device, tensor = saved
                    return tensor.to(device) if device is not None else tensor

                with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                    return checkpoint(_forward, *args, use_reentrant=False, preserve_rng_state=True, **kwargs)
            return _forward(*args, **kwargs)

        layer.forward = forward
        layer._spade_checkpointed = True
    # Swift/PEFT can enable HF reentrant checkpointing after FSDP preparation.
    # Its outer no-grad forward bypasses our checkpoint/offload code, while
    # retaining every decoder input on GPU. These layers already own their
    # checkpoint policy; restore that ownership before each root forward.
    if not getattr(model, "_spade_checkpoint_policy_guard", False):
        def disable_outer_checkpointing(root, _args):
            disabled = 0
            for module in root.modules():
                if (getattr(module, "_spade_checkpointed", False)
                        and getattr(module, "gradient_checkpointing", False)):
                    module.gradient_checkpointing = False
                    disabled += 1
            if disabled:
                get_logger().info("Disabled outer HF checkpointing on %s SPADE-managed decoder layers", disabled)

        model.register_forward_pre_hook(disable_outer_checkpointing)
        model._spade_checkpoint_policy_guard = True
    return model


def install_decoder_checkpointing(*, offload_inputs=True):
    import accelerate.accelerator as accelerator_module
    import accelerate.utils.fsdp_utils as fsdp_utils

    original = accelerator_module.fsdp2_apply_ac
    if getattr(original, "_spade_decoder_checkpointing", False):
        return

    def apply_ac(accelerator, model):
        if getattr(getattr(model, "config", None), "model_type", None) == "qwen3_5":
            get_logger().info("Enabled complete Qwen decoder recomputation; CPU activation offload=%s", offload_inputs)
            return checkpoint_decoder_layers(model, offload_inputs=offload_inputs)
        return original(accelerator, model)

    apply_ac._spade_decoder_checkpointing = True
    accelerator_module.fsdp2_apply_ac = apply_ac
    fsdp_utils.fsdp2_apply_ac = apply_ac


def install_chunked_grpo_logps(*, chunk_size=128):
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("GRPO logps chunk size must be a positive integer")
    original = GRPOTrainer._get_logps_via_local_forward
    if getattr(original, "_spade_chunked_logps", False):
        return
    # Swift initializes trainer._forward_redirection only for Liger loss.
    # The chunked path needs the same FSDP hooks independently of that option.
    forward_redirection = _ForwardRedirection()

    def get_logps(self, model, model_inputs, logits_to_keep, input_ids, compute_entropy=False):
        unwrapped = self.accelerator.unwrap_model(model)
        base = unwrapped.base_model.model if is_peft_model(unwrapped) else unwrapped
        if getattr(base.config, "model_type", None) != "qwen3_5":
            return original(self, model, model_inputs, logits_to_keep, input_ids, compute_entropy)
        head = base.get_output_embeddings()
        if not isinstance(head, nn.Linear):
            raise TypeError("SPADE chunked Qwen logps requires a plain linear output head")
        inputs = dict(model_inputs)
        if "use_cache" in self.model_kwarg_keys:
            inputs["use_cache"] = False

        def forward(*_args, **_kwargs):
            hidden = self._get_last_hidden_state(unwrapped, inputs, logits_to_keep)

            def project(hidden_states):
                return chunked_linear_logps(
                    hidden_states, head.weight, input_ids[:, -logits_to_keep:], bias=head.bias,
                    temperature=self.temperature, chunk_size=chunk_size, compute_entropy=compute_entropy)

            # Accelerate may shard the frozen head separately from the root.
            # Enter its forward hooks before reading its (unsharded) weights.
            return forward_redirection(head, head, project, hidden)

        return forward_redirection(model, unwrapped, forward, **inputs)

    get_logps._spade_chunked_logps = True
    GRPOTrainer._get_logps_via_local_forward = get_logps
    get_logger().info("Enabled Qwen GRPO projection/log-softmax in %s-token chunks", chunk_size)


def synchronize_qwen_fsdp(model, accelerator):
    """Finish asynchronous FSDP work before the next accumulated forward."""
    if not torch.cuda.is_available() or getattr(accelerator.state, "fsdp_plugin", None) is None:
        return False
    unwrapped = accelerator.unwrap_model(model)
    base = unwrapped.base_model.model if is_peft_model(unwrapped) else unwrapped
    if getattr(getattr(base, "config", None), "model_type", None) != "qwen3_5":
        return False
    torch.cuda.synchronize()
    return True


def install_fsdp_accumulation_sync(trainer_class=GRPOTrainer):
    original = trainer_class.training_step
    if getattr(original, "_spade_fsdp_accumulation_sync", False):
        return

    @wraps(original)
    def training_step(self, model, *args, **kwargs):
        result = original(self, model, *args, **kwargs)
        if not self.accelerator.sync_gradients:
            synchronize_qwen_fsdp(model, self.accelerator)
        return result

    training_step._spade_fsdp_accumulation_sync = True
    trainer_class.training_step = training_step
    get_logger().info("Enabled Qwen FSDP synchronization between accumulated backwards")


def _enabled(name):
    value = os.environ.get(name, "true").lower()
    if value not in ("true", "false"):
        raise ValueError(f"{name} must be true or false")
    return value == "true"


def install_from_environment():
    if _enabled("SPADE_GRPO_FSDP_SYNC"):
        install_fsdp_accumulation_sync()
    decoder_checkpointing = _enabled("SPADE_GRPO_DECODER_CHECKPOINTING")
    offload_inputs = _enabled("SPADE_GRPO_CPU_ACTIVATION_OFFLOAD")
    if offload_inputs and not decoder_checkpointing:
        raise ValueError("CPU activation offload requires SPADE_GRPO_DECODER_CHECKPOINTING=true")
    if _enabled("SPADE_GRPO_CHUNKED_LOGPS"):
        install_chunked_grpo_logps(chunk_size=int(os.environ.get("SPADE_GRPO_LOGPS_CHUNK_SIZE", "128")))
    if decoder_checkpointing:
        install_decoder_checkpointing(offload_inputs=offload_inputs)
    if _enabled("SPADE_GRPO_CHECKPOINT_DELTA_RULE"):
        install_delta_checkpointing()


install_from_environment()
