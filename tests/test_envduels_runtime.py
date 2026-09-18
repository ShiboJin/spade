"""CPU regression tests for Qwen history and GPU-free command preparation."""
import os
from pathlib import Path
import subprocess
import unittest

from spade.core.utils.token_utils import get_observation_delta


class RuntimeTests(unittest.TestCase):
    def test_spruce_foundation_defaults_to_plan(self):
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env.pop("BUILD_JOBS", None)
        result = subprocess.run(["bash", "scripts/build_spruce_foundation.sh"],
                                cwd=root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("foundation only", result.stdout)
        self.assertIn("BUILD_JOBS=4", result.stdout)
        self.assertNotIn("--gpus", result.stdout)

    def test_spruce_build_jobs_are_bounded(self):
        root = Path(__file__).resolve().parents[1]
        for value in ("0", "17", "64", "bad", "01"):
            with self.subTest(value=value):
                result = subprocess.run(["bash", "scripts/build_spruce_foundation.sh"],
                    cwd=root, env={**os.environ, "BUILD_JOBS": value},
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)

    def test_spruce_larger_parallelism_plan(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(["bash", "scripts/build_spruce_foundation.sh", "--plan"],
            cwd=root, env={**os.environ, "BUILD_JOBS": "16"},
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("BUILD_JOBS=16", result.stdout)

    def test_foundation_cannot_be_used_as_training_image(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(["bash", "scripts/envduels_runtime.sh", "smoke"],
            cwd=root, env={**os.environ, "ENVDUELS_IMAGE": "envduels-spade:spruce-cu124-foundation"},
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("not a training image", result.stderr)

    def test_rewritten_history_is_rejected(self):
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return "changed history"
            def decode(self, *args, **kwargs):
                return "original history"
        with self.assertRaisesRegex(ValueError, "rewrites"):
            get_observation_delta(Tokenizer(), [], [1])

    def test_append_preserves_eos_separator(self):
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                assert kwargs["enable_thinking"] is False
                return "answer<EOS>\nUSER observation\nASSISTANT"
            def decode(self, *args, **kwargs):
                return "answer<EOS>"
            def encode(self, text, **kwargs):
                return list(text.encode())
        delta, mask = get_observation_delta(Tokenizer(), [], [1], {"enable_thinking": False})
        self.assertEqual(bytes(delta).decode(), "\nUSER observation\nASSISTANT")
        self.assertFalse(any(mask))

    def test_conversion_dry_run_needs_no_gpu_runtime(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(["bash", "scripts/convert_qwen38.sh"], cwd=root,
                                capture_output=True, text=True, env=os.environ.copy())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("convert_hf_to_torch_dist.py", result.stdout)
        self.assertIn("Qwen3.8-27B", result.stdout)


if __name__ == "__main__":
    unittest.main()
