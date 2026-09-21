"""Real CPU Trainer: constant windows still run backward, AdamW and the LR scheduler."""
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from transformers import Trainer, TrainingArguments

from spade.swift_backend.envduels_sage import install_update_metrics
from spade.swift_backend.hint_resampling import refill_constant_groups, resample_with_hints
from test_hint_resampling import Generator, samples, hints_for


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.constant = False

    def forward(self, input_ids, labels=None):
        loss = (self.weight * input_ids.float() - labels.float()).square().mean()
        return {"loss": loss * (0.0 if self.constant else 1.0)}


class FullBatchTrainer(Trainer):
    def __init__(self, *, constant_windows, **kwargs):
        super().__init__(**kwargs)
        self.constant_windows = constant_windows
        self.loss_calls = 0
        self._step = 0
        self._metrics = {"train": defaultdict(list)}

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.constant = self.state.global_step in self.constant_windows
        if model.constant and self._step % self.args.gradient_accumulation_steps == 0:
            generator = Generator({("hard", 0): [0]*4, ("hard", 1): [0]*4})
            retained = refill_constant_groups(
                samples(), sage_generate=lambda rows, attempt: resample_with_hints(
                    rows, generate=generator, gather=lambda x: x, group_size=4, hints_for=hints_for),
                gather=lambda x: x, group_size=4, max_attempts=1,
                min_valid_groups=2, refill=None, keep_constant_groups=True)
            assert len(retained) == 4
        result = super().training_step(model, inputs, num_items_in_batch)
        self.loss_calls += 1
        self._step += 1
        return result


def main():
    install_update_metrics(FullBatchTrainer)
    for constant in ({0}, {1}, {0, 1, 2}):
        with TemporaryDirectory() as directory:
            trainer = FullBatchTrainer(
                constant_windows=constant, model=Model(),
                args=TrainingArguments(output_dir=directory, use_cpu=True, report_to="none",
                                       per_device_train_batch_size=1, gradient_accumulation_steps=2,
                                       num_train_epochs=1, learning_rate=0.01, weight_decay=0.1,
                                       lr_scheduler_type="linear", save_strategy="steps", save_steps=1, disable_tqdm=True),
                train_dataset=[dict(input_ids=[1.0], labels=[0.0]) for _ in range(6)])
            trainer.train()
            assert trainer.state.global_step == 3
            assert trainer._step == trainer.loss_calls == 6
            assert trainer.lr_scheduler.last_epoch == 3
            assert trainer.optimizer.state
            assert all(int(s["step"]) == 3 for s in trainer.optimizer.state.values())
            # Even all-zero policy gradients retain normal AdamW weight decay.
            assert trainer.model.weight.item() < 1.0
            for window in range(3):
                checkpoint = Path(directory) / f"checkpoint-{window + 1}"
                saved_lr = torch.load(checkpoint / "scheduler.pt", weights_only=True)
                assert saved_lr["last_epoch"] == window + 1
            assert trainer._metrics["train"]["sage/skipped_update"] == [0.0]*3
            assert trainer._metrics["train"]["sage/applied_update"] == [1.0]*3
    print("PASS: constant first/later/all windows run backward, AdamW and scheduler; checkpoints record every update")


if __name__ == "__main__":
    main()
