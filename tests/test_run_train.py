"""CPU-only checks for the config-driven LoRA GRPO launcher."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_train import ROOT, docker_command, load_config


class TrainingLauncherTests(unittest.TestCase):
    def config(self, **updates):
        document = json.loads((ROOT / "configs/train_qwen38_envduels_lora.json").read_text())
        document["training"].update(updates)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.json"
            path.write_text(json.dumps(document))
            return load_config(path)

    def test_default_config_and_docker_command(self):
        cfg = self.config()
        self.assertEqual(cfg["dataset_rows"], 90)
        self.assertEqual(cfg["gpu_ids"], list(range(8)))
        run_dir = ROOT / "outputs/training/test-run"
        command, checkpoint = docker_command(cfg, run_dir, "test-training")
        self.assertEqual(checkpoint, run_dir / "checkpoint")
        self.assertIn("envduels-unified:cu124", command)
        self.assertIn("NUM_GPUS=8", command)
        self.assertNotIn("MAX_STEPS=100", command)
        self.assertIn("NUM_TRAIN_EPOCHS=1", command)
        self.assertIn("GENERATION_BATCH_SIZE=8", command)
        self.assertIn("NUM_GENERATIONS=8", command)
        self.assertIn("NUM_ITERATIONS=1", command)
        self.assertIn("LEARNING_RATE=1e-06", command)
        self.assertIn("LORA_RANK=32", command)
        self.assertIn("SCALE_REWARDS=none", command)
        self.assertIn("PPO_CLIP_LOW=0.2", command)
        self.assertIn("PPO_CLIP_HIGH=0.28", command)
        self.assertIn("BETA=0.0", command)
        self.assertFalse(any("RESUME_FROM_CHECKPOINT=" in item for item in command))
        self.assertEqual(cfg["derived"]["dataset_environments"], 90)
        self.assertEqual(cfg["derived"]["fixed_instances_per_environment"], 1)
        self.assertEqual(cfg["derived"]["rollouts_per_environment_per_pool_epoch"], 8)
        self.assertEqual(cfg["derived"]["rollouts_per_environment_total"], 8)
        self.assertEqual(cfg["derived"]["environments_per_optimizer_step"], 1)
        self.assertEqual(cfg["derived"]["optimizer_steps_per_dataset_pass"], 90)
        self.assertEqual(cfg["derived"]["rollouts_per_dataset_pass"], 720)
        self.assertEqual(cfg["derived"]["configured_dataset_passes"], 1)
        self.assertIn(
            "ACCELERATE_CONFIG=/workspace/envduels/spade/outputs/training/test-run/accelerate_config.json",
            command,
        )
        self.assertIn(
            f"type=bind,src={cfg['export_dir']},dst=/workspace/envduels/exports/duel_harness_004_rl,readonly",
            command,
        )
        self.assertNotIn("--publish", command)

    def test_step_override(self):
        path = ROOT / "configs/train_qwen38_envduels_lora.json"
        self.assertEqual(load_config(path, max_steps=1)["max_steps"], 1)
        epoch_cfg = load_config(path, epochs=2)
        self.assertIsNone(epoch_cfg["max_steps"])
        self.assertEqual(epoch_cfg["fixed_pool_epochs"], 2)
        self.assertEqual(epoch_cfg["derived"]["rollouts_per_environment_total"], 16)

    def test_invalid_settings(self):
        for update in (
            {"gpu_ids": [0, 0]},
            {"vllm_tensor_parallel": 3},
            {"actor_max_tokens": 8192},
            {"learning_rate": float("nan")},
            {"max_steps": True},
            {"trajectories_per_game": 1, "batch_size": 1},
            {"batch_size": 16},
            {"num_games_per_rollout": 8, "batch_size": 64},
            {"dataset_shuffle": "true"},
            {"reward_normalization": "typo"},
            {"ppo_clip_low": 0.3, "ppo_clip_high": 0.2},
            {"fixed_pool_seed": -1},
            {"resume_from_checkpoint": ""},
        ):
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.config(**update)

    def test_accelerate_is_embedded_and_validated(self):
        document = json.loads((ROOT / "configs/train_qwen38_envduels_lora.json").read_text())
        document["accelerate"]["num_processes"] = 4
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.json"
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "num_processes"):
                load_config(path)

    def test_json_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.yaml"
            path.write_text("training: {}")
            with self.assertRaisesRegex(ValueError, "json"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
