"""Real CPU Trainer: skipped windows leave parameters, AdamW and LR state intact."""
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from spade.swift_backend.envduels_sage import install_skip_updates
from spade.swift_backend.hint_resampling import refill_constant_groups, resample_with_hints
from test_hint_resampling import Generator, samples, hints_for


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, labels=None):
        return {"loss": (self.weight * input_ids.float() - labels.float()).square().mean()}


class SkipTrainer(Trainer):
    def __init__(self, *, skip_windows, **kwargs):
        super().__init__(**kwargs)
        self.skip_windows = skip_windows
        self.loss_calls = 0
        self._step = 0  # Swift/TRL increments after each successful training_step.
        self._metrics = {"train": defaultdict(list)}

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.state.global_step in self.skip_windows:
            generator = Generator({("hard", 0): [0]*4, ("hard", 1): [0]*4,
                                   ("other", 0): [1]*4})
            refill_constant_groups(
                samples(),
                sage_generate=lambda rows, attempt: resample_with_hints(
                    rows, generate=generator, gather=lambda x: x, group_size=4, hints_for=hints_for),
                gather=lambda x: x, group_size=4, max_attempts=2,
                refill=lambda pending, current, attempt: {g: samples("other")[0] for g in pending})
            raise AssertionError("Expected bounded skip")
        result = super().training_step(model, inputs, num_items_in_batch)
        self.loss_calls += 1
        self._step += 1
        return result


def assert_equal(before, after):
    if isinstance(before, torch.Tensor):
        assert torch.equal(before, after), (before, after)
    elif isinstance(before, dict):
        assert before.keys() == after.keys()
        for key in before:
            assert_equal(before[key], after[key])
    elif isinstance(before, (tuple, list)):
        assert len(before) == len(after)
        for left, right in zip(before, after):
            assert_equal(left, right)
    else:
        assert before == after, (before, after)


class CheckSkippedState(TrainerCallback):
    def snapshot(self, model, optimizer, lr_scheduler):
        return deepcopy((model.state_dict(), optimizer.state_dict(), lr_scheduler.state_dict()))

    def on_step_begin(self, args, state, control, model, optimizer, lr_scheduler, **kwargs):
        self.before = self.snapshot(model, optimizer, lr_scheduler)

    def on_step_end(self, args, state, control, model, optimizer, lr_scheduler, **kwargs):
        if state.global_step - 1 in self.skip_windows:
            assert_equal(self.before, self.snapshot(model, optimizer, lr_scheduler))
            assert all(p.grad is None for p in model.parameters())


def main():
    install_skip_updates(SkipTrainer)
    # First-window failure, failure after AdamW has momentum, and all windows fail.
    for skipped in ({0}, {1}, {0, 1, 2}):
        with TemporaryDirectory() as directory:
            checker = CheckSkippedState()
            checker.skip_windows = skipped
            trainer = SkipTrainer(
                skip_windows=skipped, model=Model(), callbacks=[checker],
                args=TrainingArguments(output_dir=directory, use_cpu=True, report_to="none",
                                       per_device_train_batch_size=1, gradient_accumulation_steps=2,
                                       num_train_epochs=1, learning_rate=0.01, weight_decay=0.1,
                                       lr_scheduler_type="linear", save_strategy="steps", save_steps=1, disable_tqdm=True),
                train_dataset=[dict(input_ids=[1.0], labels=[0.0]) for _ in range(6)])
            trainer.train()
            applied = 3 - len(skipped)
            assert trainer.state.global_step == 3  # Consumed windows, including skips.
            assert trainer._step == 6
            assert trainer.loss_calls == applied * 2
            assert trainer.lr_scheduler.last_epoch == applied
            if applied:
                assert trainer.optimizer.state
                assert all(int(s["step"]) == applied for s in trainer.optimizer.state.values())
            else:
                assert not trainer.optimizer.state
                assert trainer.model.weight.item() == 1.0
            for window in range(3):
                checkpoint = Path(directory) / f"checkpoint-{window + 1}"
                saved_lr = torch.load(checkpoint / "scheduler.pt", weights_only=True)
                assert saved_lr["last_epoch"] == sum(i not in skipped for i in range(window + 1))
            assert trainer._metrics["train"]["sage/skipped_update"] == [float(i in skipped) for i in range(3)]
            assert sum(trainer._metrics["train"]["sage/applied_update"]) == applied
            events = [json.loads(line) for line in (Path(directory) / "hint_resampling.jsonl").read_text().splitlines()]
            assert {e["global_step"] for e in events} == skipped
            assert all(e["event"] == "skipped_update" for e in events)
    print("PASS: skip before/after AdamW state exists preserves parameters, momentum and LR; later windows train")
    print("PASS: all-skipped run consumes windows without backward, optimizer or scheduler updates")


if __name__ == "__main__":
    main()
