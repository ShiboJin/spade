import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

from spade.swift_backend.epoch_checkpoints import EpochCheckpointCallback


class EpochCheckpointTests(unittest.TestCase):
    def test_swift_registry_constructs_launcher_callbacks(self):
        from swift.trainers.mixin import SwiftMixin
        from spade.swift_backend import wandb_config  # Register the other launcher callback.

        args = SimpleNamespace(callbacks=["envduels_wandb_config", "envduels_epoch_checkpoints"])
        trainer = object()
        callbacks = SwiftMixin._get_callbacks(trainer, args)
        self.assertEqual(len(callbacks), 2)
        self.assertIsInstance(callbacks[0], wandb_config.EnvDuelsWandbConfigCallback)
        self.assertIsInstance(callbacks[1], EpochCheckpointCallback)
        for callback in callbacks:
            self.assertIs(callback.args, args)
            self.assertIs(callback.trainer, trainer)
        self.assertIsNone(callbacks[1].last_saved_step)

    def test_real_trainer_retains_latest_epoch_and_can_resume(self):
        import torch
        from transformers import Trainer, TrainingArguments

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)

            def forward(self, input_ids, labels):
                logits = self.linear(input_ids)
                return {"loss": (logits - labels).square().mean(), "logits": logits}

        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(model=TinyModel(), args=TrainingArguments(
                output_dir=directory, use_cpu=True, num_train_epochs=3,
                per_device_train_batch_size=2, save_strategy="steps", save_steps=3,
                save_total_limit=1, report_to=[], disable_tqdm=True),
                train_dataset=[{"input_ids": torch.tensor([float(i)]),
                                "labels": torch.tensor([float(i + 1)])} for i in range(4)],
                callbacks=[EpochCheckpointCallback()])
            trainer.train()
            root = Path(directory)
            self.assertEqual([p.name for p in root.glob("checkpoint-*")], ["checkpoint-6"])
            self.assertEqual([p.name for p in (root / "epoch_checkpoints").iterdir()], ["epoch-003-step-6"])
            for epoch, step in ((3, 6),):
                archive = root / "epoch_checkpoints" / f"epoch-{epoch:03d}-step-{step}"
                for filename in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
                    self.assertTrue((archive / filename).is_file(), filename)
                self.assertEqual(json.loads((archive / "trainer_state.json").read_text())["global_step"], step)
            trainer.args.output_dir = str(root / "resumed")
            trainer.args.num_train_epochs = 4
            resumed = Trainer(model=TinyModel(), args=trainer.args,
                              train_dataset=trainer.train_dataset,
                              callbacks=[EpochCheckpointCallback()])
            resumed.train(resume_from_checkpoint=str(root / "epoch_checkpoints/epoch-003-step-6"))
            self.assertEqual(resumed.state.global_step, 8)
            self.assertEqual(resumed.state.epoch, 4.0)

    def test_epoch_save_is_requested_off_step_interval(self):
        callback = EpochCheckpointCallback()
        control = SimpleNamespace(should_save=False)
        callback.on_epoch_end(None, SimpleNamespace(epoch=1.0, global_step=13), control)
        self.assertTrue(control.should_save)

    def test_partial_epoch_does_not_request_epoch_save(self):
        for epoch in (None, 0.0, 0.4, 1.2):
            control = SimpleNamespace(should_save=False)
            EpochCheckpointCallback().on_epoch_end(None, SimpleNamespace(epoch=epoch, global_step=7), control)
            self.assertFalse(control.should_save)

    def test_epoch_archive_survives_rotation_and_is_not_saved_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "checkpoint-15"
            source.mkdir()
            files = {"adapter_model.safetensors": "weights", "optimizer.pt": "optimizer",
                     "scheduler.pt": "scheduler", "rng_state_0.pth": "rng0",
                     "rng_state_1.pth": "rng1", "trainer_state.json": json.dumps({"global_step": 15})}
            for name, value in files.items():
                (source / name).write_text(value)
            callback = EpochCheckpointCallback()
            state = SimpleNamespace(epoch=1.0, global_step=15, is_world_process_zero=True)
            args = SimpleNamespace(output_dir=directory)
            control = SimpleNamespace(should_save=False)
            callback.on_save(args, state, control)
            callback.on_epoch_end(args, state, control)
            self.assertFalse(control.should_save)
            shutil.rmtree(source)  # Simulate deletion of the rolling checkpoint.
            archive = root / "epoch_checkpoints/epoch-001-step-15"
            self.assertEqual({p.name: p.read_text() for p in archive.iterdir()}, files)
            callback.on_save(args, state, control)  # Existing archive stays intact.
            self.assertEqual(len(list(archive.parent.iterdir())), 1)

    def test_regular_step_does_not_create_epoch_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            EpochCheckpointCallback().on_save(SimpleNamespace(output_dir=directory),
                SimpleNamespace(epoch=0.3, global_step=5, is_world_process_zero=True), None)
            self.assertFalse((Path(directory) / "epoch_checkpoints").exists())

    def test_missing_source_reports_failure_without_publishing_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path(directory) / "epoch_checkpoints/epoch-001-step-15"
            previous.mkdir(parents=True)
            (previous / "optimizer.pt").write_text("previous optimizer")
            with self.assertRaisesRegex(RuntimeError, "Could not archive epoch"):
                EpochCheckpointCallback().on_save(SimpleNamespace(output_dir=directory),
                    SimpleNamespace(epoch=2.0, global_step=30, is_world_process_zero=True), None)
            self.assertEqual(list(previous.parent.iterdir()), [previous])
            self.assertEqual((previous / "optimizer.pt").read_text(), "previous optimizer")
