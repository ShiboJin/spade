"""Known-ratio and masking regressions for first-update loss diagnostics."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from spade.swift_backend.logps_diagnostics import (
    input_fingerprint, install_logps_diagnostics, summarize_logps,
)


class LogpsDiagnosticsTests(unittest.TestCase):
    def summarize(self, current, old, advantage, mask):
        return summarize_logps(current, old, advantage, mask, epsilon_low=0.2, epsilon_high=0.28)

    def test_same_policy_group_is_zero_even_with_masked_extreme_tokens(self):
        old = torch.tensor([[-1., -2.]] * 5)
        current = old.clone()
        current[4] = 1000
        mask = torch.tensor([[True, True]] * 4 + [[False, False]])
        advantage = torch.tensor([[.75], [-.25], [-.25], [-.25], [0.]]).expand_as(old)
        stats = self.summarize(current, old, advantage, mask)
        self.assertEqual(stats["policy_loss_before_sage_scale"], 0)
        self.assertEqual(stats["ratio_max"], 1)
        self.assertEqual(stats["ratio_nonfinite"], 0)
        self.assertEqual(stats["valid_tokens"], 8)
        json.dumps(stats, allow_nan=False)

    def test_negative_advantage_large_ratio_is_not_clipped_away(self):
        old = torch.tensor([[-12.]])
        current = old + torch.tensor(10000.).log()
        stats = self.summarize(current, old, torch.tensor([[-.25]]), torch.tensor([[True]]))
        self.assertAlmostEqual(stats["policy_loss_before_sage_scale"], 2500., places=2)
        self.assertAlmostEqual(stats["worst_token"]["ratio"], 10000., places=1)

    def test_nonfinite_and_empty_mask_produce_valid_json(self):
        for mask in (torch.tensor([[True]]), torch.tensor([[False]])):
            stats = self.summarize(torch.tensor([[1000.]]), torch.tensor([[-1.]]),
                                   torch.tensor([[-.25]]), mask)
            json.dumps(stats, allow_nan=False)
            self.assertEqual(stats["ratio_nonfinite"], int(mask.item()))

    def test_context_change_is_detected_and_hook_keeps_gradients(self):
        class Trainer:
            def _get_per_token_logps_and_entropies(self, model, inputs, batch):
                return model(inputs["input_ids"].float()).log_softmax(-1), None

        install_logps_diagnostics(Trainer)
        with TemporaryDirectory() as directory:
            trainer = Trainer()
            trainer.model = torch.nn.Linear(2, 2)
            trainer.state = SimpleNamespace(global_step=0)
            trainer.args = SimpleNamespace(output_dir=directory)
            trainer.accelerator = SimpleNamespace(process_index=0)
            trainer.overlong_filter = False
            trainer.epsilon_low, trainer.epsilon_high = .2, .28
            trainer.beta = 0.
            trainer.importance_sampling_level = "token"
            trainer.rollout_importance_sampling_mode = None
            batch = SimpleNamespace(old_per_token_logps=None, completion_mask=torch.ones(1, 2, dtype=torch.bool),
                                    advantages=torch.tensor([[.75, .75]]))
            inputs = {"input_ids": torch.tensor([[1, 2]])}
            with torch.no_grad():
                old, _ = trainer._get_per_token_logps_and_entropies(trainer.model, inputs, batch)
            batch.old_per_token_logps = old
            current, _ = trainer._get_per_token_logps_and_entropies(trainer.model, inputs, batch)
            current.sum().backward()
            self.assertTrue(torch.isfinite(trainer.model.weight.grad).all())
            fingerprint = input_fingerprint(inputs)
            inputs["input_ids"][0, 0] = 3
            self.assertNotEqual(fingerprint, input_fingerprint(inputs))
            trainer._get_per_token_logps_and_entropies(trainer.model, inputs, batch)
            records = [json.loads(line) for line in (Path(directory) / "logps_diagnostics.rank0.jsonl").read_text().splitlines()]
            self.assertEqual([r["input_matches_old"] for r in records], [True, False])
            self.assertEqual(records[0]["ratio_max"], 1)


if __name__ == "__main__":
    unittest.main()
