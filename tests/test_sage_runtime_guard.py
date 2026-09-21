"""SAGE validates active training features, not Swift's dynamic OPSD capability."""
from types import SimpleNamespace
import unittest

from spade.swift_backend.envduels_sage import has_teacher_input, runtime_conflicts


def trainer(**overrides):
    values = dict(args=SimpleNamespace(steps_per_generation=6, gradient_accumulation_steps=6),
                  template=SimpleNamespace(sequence_parallel_size=1), num_iterations=1,
                  loss_type="grpo", scale_rewards="none", use_liger_loss=False, kl_in_reward=False,
                  chord_sft_iterator=None, use_gym_env=True, vllm_mode="colocate",
                  _has_teacher=True, _is_dynamic_self_distillation=True)
    values.update(overrides)
    return SimpleNamespace(**values)


class SageRuntimeGuardTests(unittest.TestCase):
    def test_dynamic_teacher_capability_is_allowed_without_active_teacher(self):
        self.assertEqual(runtime_conflicts(trainer()), [])
        self.assertEqual(runtime_conflicts(trainer(_has_teacher_explicit=lambda: False)), [])

    def test_explicit_teacher_is_still_rejected(self):
        for setting in (dict(_teacher_model=object()), dict(teacher_model_server="http://teacher"),
                        dict(_teacher_use_disable_adapter=True), dict(use_teacher_api=True),
                        dict(_has_teacher_explicit=lambda: True)):
            with self.subTest(setting=setting):
                self.assertIn("explicit_teacher=True (required False)", runtime_conflicts(trainer(**setting)))

    def test_teacher_input_detection_matches_swift(self):
        for setting in (dict(), dict(teacher_prompt=""), dict(teacher_prompt=None, teacher_images=None)):
            self.assertFalse(has_teacher_input(SimpleNamespace(**setting)))
        for setting in (dict(teacher_prompt="answer"), dict(teacher_images=[]), dict(teacher_messages=[])):
            self.assertTrue(has_teacher_input(SimpleNamespace(**setting)))

    def test_error_identifies_actual_conflicting_values(self):
        conflicts = runtime_conflicts(trainer(dynamic_sample=True,
            args=SimpleNamespace(steps_per_generation=3, gradient_accumulation_steps=6)))
        self.assertEqual(conflicts, ["dynamic_sample=True (required False)",
                                    "steps_per_generation=3 (required 6)"])

    def test_other_unsupported_features_remain_rejected(self):
        for key, value in (("async_generate", True), ("dynamic_num_samples", True),
                           ("num_iterations", 2), ("loss_type", "dapo"), ("scale_rewards", "batch"),
                           ("use_liger_loss", True), ("kl_in_reward", True),
                           ("chord_sft_iterator", object()), ("use_gym_env", False), ("vllm_mode", "server")):
            with self.subTest(key=key):
                self.assertTrue(any(key in error for error in runtime_conflicts(trainer(**{key: value}))))
        self.assertTrue(runtime_conflicts(trainer(), is_global_inputs=True))
        self.assertTrue(runtime_conflicts(trainer(template=SimpleNamespace(sequence_parallel_size=2))))


if __name__ == "__main__":
    unittest.main()
