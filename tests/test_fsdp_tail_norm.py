"""Discover the nested Qwen norm without changing other FSDP architectures."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from spade.swift_backend.fsdp_tail_norm import qwen_final_norm, install_qwen_fsdp_tail_norm


def qwen_model():
    model = torch.nn.Module()
    model.config = SimpleNamespace(model_type="qwen3_5")
    model.base_model_prefix = "model"
    model.model = torch.nn.Module()
    model.model.language_model = torch.nn.Module()
    model.model.language_model.norm = torch.nn.LayerNorm(4)
    return model


class TailNormTests(unittest.TestCase):
    def test_nested_qwen_norm_is_found(self):
        model = qwen_model()
        self.assertIs(qwen_final_norm(model), model.model.language_model.norm)

    def test_peft_wrapper_is_unwrapped(self):
        base = qwen_model()
        wrapper = SimpleNamespace(base_model=SimpleNamespace(model=base))
        with patch("spade.swift_backend.fsdp_tail_norm.is_peft_model", return_value=True):
            self.assertIs(qwen_final_norm(wrapper), base.model.language_model.norm)

    def test_unknown_architecture_is_not_assumed_to_be_qwen(self):
        model = torch.nn.Module()
        model.config = SimpleNamespace(model_type="llama")
        self.assertIsNone(qwen_final_norm(model))

    def test_unexpected_qwen_layout_fails_clearly(self):
        model = qwen_model()
        del model.model.language_model.norm
        with self.assertRaisesRegex(RuntimeError, "language_model.norm"):
            qwen_final_norm(model)

    def test_install_is_idempotent_and_delegates_other_models(self):
        import accelerate.utils.fsdp_utils as fsdp_utils
        fallback = torch.nn.LayerNorm(4)
        original = Mock(return_value=fallback)
        # Mock attributes are truthy unless explicitly set.
        original._spade_qwen_tail_norm = False
        with patch.object(fsdp_utils, "_find_final_norm", original):
            install_qwen_fsdp_tail_norm()
            installed = fsdp_utils._find_final_norm
            install_qwen_fsdp_tail_norm()
            self.assertIs(installed, fsdp_utils._find_final_norm)
            model = qwen_model()
            self.assertIs(installed(model), model.model.language_model.norm)
            original.assert_not_called()
            other = torch.nn.Linear(4, 4)
            self.assertIs(installed(other), fallback)
            original.assert_called_once_with(other)


if __name__ == "__main__":
    unittest.main()
