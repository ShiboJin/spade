"""Qwen FSDP accumulation synchronization is narrow and idempotent."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from spade.swift_backend.memory_efficient_grpo import (
    install_fsdp_accumulation_sync,
    synchronize_qwen_fsdp,
)


class FakeTrainer:
    def training_step(self, model, marker=None):
        return marker


def accelerator(*, fsdp=True, sync_gradients=False):
    return SimpleNamespace(
        state=SimpleNamespace(fsdp_plugin=object() if fsdp else None),
        sync_gradients=sync_gradients,
        unwrap_model=lambda model: model,
    )


def model(model_type):
    value = torch.nn.Module()
    value.config = SimpleNamespace(model_type=model_type)
    return value


class FSDPAccumulationSyncTests(unittest.TestCase):
    def test_qwen_fsdp_synchronizes(self):
        qwen = model("qwen3_5")
        with (patch("spade.swift_backend.memory_efficient_grpo.torch.cuda.is_available", return_value=True),
              patch("spade.swift_backend.memory_efficient_grpo.torch.cuda.synchronize") as sync):
            self.assertTrue(synchronize_qwen_fsdp(qwen, accelerator()))
            sync.assert_called_once_with()

    def test_other_paths_do_not_synchronize(self):
        llama = model("llama")
        qwen = model("qwen3_5")
        with (patch("spade.swift_backend.memory_efficient_grpo.torch.cuda.is_available", return_value=True),
              patch("spade.swift_backend.memory_efficient_grpo.torch.cuda.synchronize") as sync):
            self.assertFalse(synchronize_qwen_fsdp(llama, accelerator()))
            self.assertFalse(synchronize_qwen_fsdp(qwen, accelerator(fsdp=False)))
            sync.assert_not_called()

    def test_training_step_syncs_only_between_accumulated_backwards(self):
        class Trainer(FakeTrainer):
            pass

        install_fsdp_accumulation_sync(Trainer)
        installed = Trainer.training_step
        install_fsdp_accumulation_sync(Trainer)
        self.assertIs(installed, Trainer.training_step)
        trainer = Trainer()
        trainer.accelerator = accelerator(sync_gradients=False)
        qwen = model("qwen3_5")
        with patch("spade.swift_backend.memory_efficient_grpo.synchronize_qwen_fsdp") as sync:
            self.assertEqual(trainer.training_step(qwen, marker="loss"), "loss")
            sync.assert_called_once_with(qwen, trainer.accelerator)
            trainer.accelerator.sync_gradients = True
            trainer.training_step(qwen, marker="final")
            sync.assert_called_once()


if __name__ == "__main__":
    unittest.main()
