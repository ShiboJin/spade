"""Launcher preparation tests: no GPU/runtime or model downloads."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="envduels launcher ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = Path(__file__).resolve().parents[1]
        (self.root / "manifest.json").write_text(json.dumps({
            "format_version": 1,
            "environments": [{"id": "example", "source": "NOT_EXECUTED.py", "split": "test"}],
        }))
        self.env = dict(os.environ, PYTHON_BIN=sys.executable,
                        MODEL_CONFIG=str(self.project / "cmd/models/qwen3-8B.sh"),
                        HF_CHECKPOINT=str(self.root / "missing model"),
                        REF_CHECKPOINT=str(self.root / "missing reference"),
                        ENVDUELS_EXPORT_DIR=str(self.root),
                        OUTPUT_DIR=str(self.root / "not created"),
                        MEGATRON_DIR=str(self.root / "missing megatron"),
                        NUM_GPUS="8", TP="2", PP="1", CP="1", ROLLOUT_TP="2",
                        GROUP_SIZE="8", GLOBAL_BATCH_SIZE="32", NUM_ROLLOUT="2",
                        MAX_TURNS="24", MAX_CONTEXT_LENGTH="8192", ACTOR_MAX_TOKENS="1024",
                        MAX_TOKENS_PER_GPU="2048", THINKING="false")
        for key in ("ENVDUELS_IDS_FILE", "RAY_ADDRESS", "LOAD_DIR"):
            self.env.pop(key, None)

    def run_launcher(self, *args):
        return subprocess.run(["bash", "cmd/games/train_envduels.sh", *args],
                              cwd=self.project, env=self.env, capture_output=True, text=True)

    def test_default_is_read_only_dry_run(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Selected environments: 1", result.stdout)
        self.assertIn("--disable-rewards-normalization", result.stdout)
        self.assertIn("4 problems x 8 episodes", result.stdout)
        self.assertIn("missing\\ model", result.stdout)
        self.assertFalse(Path(self.env["OUTPUT_DIR"]).exists())

    def test_invalid_parallelism(self):
        self.env["NUM_GPUS"] = "10"
        self.env["TP"] = "4"
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GPU count must be divisible", result.stderr)

    def test_invalid_group_size(self):
        self.env["GROUP_SIZE"] = "3"
        self.assertNotEqual(self.run_launcher().returncode, 0)

    def test_run_requires_explicit_cluster(self):
        result = self.run_launcher("--run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("dedicated Ray cluster", result.stderr)

    def test_model_is_required(self):
        del self.env["MODEL_CONFIG"]
        self.assertNotEqual(self.run_launcher().returncode, 0)


if __name__ == "__main__":
    unittest.main()
