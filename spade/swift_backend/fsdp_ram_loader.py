"""Keep FSDP2 worker ranks from materializing a full HF checkpoint on CPU.

Transformers 5.12.1 builds models on the meta device, but its new
``core_model_loading`` path still materializes every safetensors value on every
distributed rank.  Accelerate's FSDP2 RAM-efficient preparation only needs the
real state dict on global rank 0; it broadcasts those values while replacing
the meta model with sharded parameters on all other ranks.

This ms-swift external plugin preserves the normal loader on rank 0 and uses
uninitialized, shape-compatible CPU tensors on the other ranks.  The tensors
reserve virtual address space but do not read or fault in the checkpoint
payload. It also replaces the subsequent FSDP worker parameter zero-fill with
empty allocations: Transformers otherwise touches a complete model on every
worker even after checkpoint materialization has been skipped. Buffers keep
their upstream initialization behavior.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import os
from typing import Callable

import torch


_SUPPORTED_TRANSFORMERS_VERSION = "5.12.1"
_PATCH_MARKER = "_spade_fsdp_rank0_checkpoint_loader"
logger = logging.getLogger(__name__)


def _true_environment(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _non_main_fsdp_rank() -> bool:
    """Match the global-rank-0 broadcast used by Accelerate's FSDP2 loader."""
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_rank() != 0
        and _true_environment("ACCELERATE_USE_FSDP")
        and _true_environment("FSDP_CPU_RAM_EFFICIENT_LOADING")
    )


def _empty_checkpoint_tensor(tensor, device=None, dtype=None) -> torch.Tensor:
    if hasattr(tensor, "get_shape"):
        shape = tuple(tensor.get_shape())
    else:
        shape = tuple(tensor.shape)

    if dtype is None:
        if isinstance(tensor, torch.Tensor):
            dtype = tensor.dtype
        elif hasattr(tensor, "get_dtype"):
            # This private helper is deliberately guarded by the exact
            # Transformers version in install_transformers_fsdp_ram_loader().
            from safetensors.torch import _getdtype

            dtype = _getdtype(tensor.get_dtype())
        else:
            raise TypeError(f"Cannot determine checkpoint tensor dtype from {type(tensor)!r}")

    return torch.empty(shape, device=device or "cpu", dtype=dtype)


def install_transformers_fsdp_ram_loader() -> None:
    import transformers
    from transformers import core_model_loading, modeling_utils

    current = core_model_loading.spawn_materialize
    if getattr(current, _PATCH_MARKER, False):
        return
    if transformers.__version__ != _SUPPORTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "SPADE's FSDP RAM loader patch supports transformers=="
            f"{_SUPPORTED_TRANSFORMERS_VERSION}, found {transformers.__version__}. "
            "Review the upstream loader before changing this version pin."
        )

    original = current
    original_move = modeling_utils.PreTrainedModel._move_missing_keys_from_meta_to_device

    def move_missing_keys_rank0_only(self, missing_keys, device_map, device_mesh, hf_quantizer):
        if not _non_main_fsdp_rank() or hf_quantizer is not None:
            return original_move(self, missing_keys, device_map, device_mesh, hf_quantizer)
        # These parameters are overwritten by Accelerate's rank-0 broadcast.
        # zeros_like faults in ~54 GB/rank for the 27B model; empty_like does not.
        for key, param in self.named_parameters():
            value = torch.empty_like(param, device="cpu")
            modeling_utils._load_parameter_into_model(self, key, value)
        # Non-persistent buffers are not broadcast by FSDP; preserve HF's
        # zero-fill + _initialize_missing_keys behavior for all buffers.
        for key, buffer in self.named_buffers():
            value = torch.zeros_like(buffer, device="cpu")
            modeling_utils._load_parameter_into_model(self, key, value)

    def spawn_materialize_rank0_only(
        thread_pool: ThreadPoolExecutor | None,
        tensor,
        device=None,
        dtype=None,
    ) -> Callable:
        if not _non_main_fsdp_rank():
            return original(thread_pool, tensor, device, dtype)

        def _job():
            return _empty_checkpoint_tensor(tensor, device, dtype)

        if thread_pool is not None:
            return thread_pool.submit(_job)
        return _job

    setattr(spawn_materialize_rank0_only, _PATCH_MARKER, True)
    core_model_loading.spawn_materialize = spawn_materialize_rank0_only
    modeling_utils.PreTrainedModel._move_missing_keys_from_meta_to_device = move_missing_keys_rank0_only
    logger.info(
        "Installed SPADE FSDP rank-0-only checkpoint materialization for transformers %s",
        transformers.__version__,
    )


install_transformers_fsdp_ram_loader()
