"""Guard and dispatch checks for redundant old-policy scoring."""
from types import SimpleNamespace
import unittest

import torch

from spade.swift_backend.memory_efficient_grpo import (
    install_single_iteration_fastpath,
    single_iteration_fastpath_eligible,
)


def trainer(**overrides):
    values = dict(
        model=torch.nn.Linear(2, 2),
        num_iterations=1,
        beta=0.0,
        loss_type="grpo",
        importance_sampling_level="token",
        compute_entropy=False,
        kl_in_reward=False,
        async_generate=False,
        rollout_importance_sampling_mode=None,
        log_rollout_offpolicy_metrics=False,
        off_policy_sequence_mask_delta=None,
        chord_sft_iterator=None,
        use_liger_loss=False,
        sdar_loss_coef=0.0,
        advantage_reweight=None,
        use_teacher_api=False,
        _teacher_use_disable_adapter=False,
        _has_teacher_explicit=lambda: False,
        old_policy=lambda: False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class SingleIterationFastpathTests(unittest.TestCase):
    def test_exact_on_policy_case_is_eligible(self):
        samples = [SimpleNamespace(teacher_prompt=None, teacher_images=None, teacher_messages=None)]
        self.assertTrue(single_iteration_fastpath_eligible(trainer(), samples))
        # Dynamic teacher capability alone is not an active teacher.
        self.assertTrue(single_iteration_fastpath_eligible(
            trainer(_is_dynamic_self_distillation=True, _has_teacher=True), samples))

    def test_training_features_that_consume_old_logps_fall_back(self):
        samples = [SimpleNamespace()]
        settings = (
            dict(num_iterations=2), dict(beta=0.1), dict(loss_type="dapo"),
            dict(importance_sampling_level="sequence"), dict(compute_entropy=True),
            dict(kl_in_reward=True), dict(async_generate=True),
            dict(rollout_importance_sampling_mode="token"),
            dict(log_rollout_offpolicy_metrics=True), dict(off_policy_sequence_mask_delta=1.0),
            dict(chord_sft_iterator=object()), dict(use_liger_loss=True),
            dict(sdar_loss_coef=0.1), dict(advantage_reweight="rlsd"),
            dict(use_teacher_api=True), dict(_teacher_model=object()),
            dict(teacher_model_server="http://teacher"),
            dict(_teacher_use_disable_adapter=True), dict(_has_teacher_explicit=lambda: True),
            dict(old_policy=lambda: True),
        )
        for setting in settings:
            with self.subTest(setting=setting):
                self.assertFalse(single_iteration_fastpath_eligible(trainer(**setting), samples))
        eval_trainer = trainer()
        eval_trainer.model.eval()
        self.assertFalse(single_iteration_fastpath_eligible(eval_trainer, samples))

    def test_teacher_inputs_fall_back(self):
        for sample in (SimpleNamespace(teacher_prompt="answer"),
                       SimpleNamespace(teacher_images=[]),
                       SimpleNamespace(teacher_messages=[])):
            with self.subTest(sample=sample):
                self.assertFalse(single_iteration_fastpath_eligible(trainer(), [sample]))

    def test_prepare_skips_only_the_guarded_no_grad_call(self):
        class Batch:
            old_per_token_logps = "unset"

        class Harness:
            calls = 0

            def _get_per_token_logps_and_entropies(self, model, inputs, batch):
                self.calls += 1
                return torch.ones(1, 2), None

            def _prepare_batch_inputs(self, samples):
                batch = Batch()
                with torch.no_grad():
                    batch.old_per_token_logps = self._get_per_token_logps_and_entropies(
                        self.model, {}, batch)[0]
                return [batch]

        install_single_iteration_fastpath(Harness)
        eligible = Harness()
        eligible.__dict__.update(trainer().__dict__)
        result = eligible._prepare_batch_inputs([SimpleNamespace()])
        self.assertEqual(eligible.calls, 0)
        self.assertIsNone(result[0].old_per_token_logps)
        self.assertTrue(result[0]._spade_old_policy_fastpath)
        # Calls outside preparation, including the differentiable policy pass,
        # still execute the original implementation.
        value, _ = eligible._get_per_token_logps_and_entropies(
            eligible.model, {}, result[0])
        self.assertEqual(eligible.calls, 1)
        self.assertTrue(torch.equal(value, torch.ones(1, 2)))

        fallback = Harness()
        fallback.__dict__.update(trainer(beta=0.1).__dict__)
        result = fallback._prepare_batch_inputs([SimpleNamespace()])
        self.assertEqual(fallback.calls, 1)
        self.assertTrue(torch.equal(result[0].old_per_token_logps, torch.ones(1, 2)))
        self.assertFalse(hasattr(result[0], "_spade_old_policy_fastpath"))


if __name__ == "__main__":
    unittest.main()
