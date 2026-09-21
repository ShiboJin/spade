"""Include Qwen's nested final norm in Accelerate's FSDP2 output-tail group.

Accelerate 1.14's shallow final-norm lookup misses the multimodal language-model
nesting (and PEFT wrappers). That leaves the late-used frozen norm in the root
group with the unused vision parameters instead of the norm/head tail. In our
accumulation reproducer the first differing activation after backward was this
norm, while all sampled decoder activations still matched.

Only fix discovery: Accelerate still owns sharding, hooks, and parameter loading.
"""
from functools import wraps
import logging

from accelerate.utils import is_peft_model
from torch import nn


def qwen_final_norm(model):
    base = model.base_model.model if is_peft_model(model) else model
    if getattr(getattr(base, "config", None), "model_type", None) != "qwen3_5":
        return None
    backbone = getattr(base, "model", None)
    language_model = getattr(backbone, "language_model", None)
    norm = getattr(language_model, "norm", None)
    if not isinstance(norm, nn.Module):
        raise RuntimeError("Qwen3.5 FSDP2 requires model.language_model.norm")
    return norm


def install_qwen_fsdp_tail_norm():
    import accelerate.utils.fsdp_utils as fsdp_utils

    original = fsdp_utils._find_final_norm
    if getattr(original, "_spade_qwen_tail_norm", False):
        return

    @wraps(original)
    def find_final_norm(model):
        norm = qwen_final_norm(model)
        if norm is None:
            return original(model)
        logging.getLogger(__name__).info("Grouping Qwen final language norm with the FSDP2 output head")
        return norm

    find_final_norm._spade_qwen_tail_norm = True
    fsdp_utils._find_final_norm = find_final_norm
