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
        self.assertIn("envduels-unified:cu124-wandb", command)
        self.assertEqual(command[command.index("--network") + 1], "bridge")
        self.assertIn("NUM_GPUS=8", command)
        self.assertIn("VLLM_TP=8", command)
        self.assertIn("MOVE_MODEL_BATCHES=64", command)
        self.assertNotIn("MAX_STEPS=100", command)
        self.assertIn("NUM_TRAIN_EPOCHS=3", command)
        self.assertIn("GENERATION_BATCH_SIZE=24", command)
        self.assertIn("NUM_GENERATIONS=4", command)
        self.assertIn("GRADIENT_ACCUMULATION_STEPS=3", command)
        self.assertIn("NUM_ITERATIONS=1", command)
        self.assertIn("LEARNING_RATE=1e-06", command)
        self.assertIn("LORA_RANK=32", command)
        self.assertIn("SCALE_REWARDS=none", command)
        self.assertIn("PPO_CLIP_LOW=0.2", command)
        self.assertIn("PPO_CLIP_HIGH=0.28", command)
        self.assertIn("BETA=0.0", command)
        self.assertIn("MODELSCOPE_CACHE=/tmp/modelscope", command)
        self.assertIn("REPORT_TO=tensorboard,wandb", command)
        self.assertIn(f"RUN_NAME={cfg['wandb_run_name'] or 'test-training'}", command)
        self.assertIn("WANDB_MODE=online", command)
        self.assertIn(f"WANDB_PROJECT={cfg['wandb_project']}", command)
        self.assertIn(
            "WANDB_DIR=/workspace/envduels/spade/outputs/training/test-run/wandb",
            command,
        )
        self.assertIn(
            "SPADE_RESOLVED_CONFIG=/workspace/envduels/spade/outputs/training/test-run/resolved_config.json",
            command,
        )
        self.assertIn("WANDB_API_KEY", command)
        self.assertFalse(any(item.startswith("WANDB_API_KEY=") for item in command))
        self.assertFalse(any("RESUME_FROM_CHECKPOINT=" in item for item in command))
        self.assertEqual(cfg["derived"]["dataset_environments"], 90)
        self.assertEqual(cfg["derived"]["fixed_instances_per_environment"], 1)
        self.assertEqual(cfg["derived"]["rollouts_per_environment_per_pool_epoch"], 4)
        self.assertEqual(cfg["derived"]["rollouts_per_environment_total"], 12)
        self.assertEqual(cfg["derived"]["environments_per_generation_batch"], 6)
        self.assertEqual(cfg["derived"]["environments_per_optimizer_step"], 6)
        self.assertEqual(cfg["derived"]["gradient_accumulation_steps"], 3)
        self.assertEqual(cfg["derived"]["optimizer_steps_per_dataset_pass"], 15)
        self.assertEqual(cfg["derived"]["rollouts_per_dataset_pass"], 360)
        self.assertEqual(cfg["derived"]["configured_dataset_passes"], 3)
        self.assertIn(
            "ACCELERATE_CONFIG=/workspace/envduels/spade/outputs/training/test-run/accelerate_config.json",
            command,
        )
        self.assertIn(
            f"type=bind,src={cfg['export_dir']},dst=/workspace/envduels/exports/duel_harness_004_rl,readonly",
            command,
        )
        self.assertNotIn("--publish", command)
        self.assertEqual(command[command.index("--memory") + 1], "160g")
        self.assertEqual(command[command.index("--memory-swap") + 1], "160g")
        self.assertEqual(command[command.index("--memory-swappiness") + 1], "0")
        self.assertIn("SPADE_MEMORY_LIMIT_GIB=160", command)
        self.assertIn("USER=envduels", command)
        self.assertIn("XDG_CONFIG_HOME=/tmp/config", command)
        self.assertIn("VLLM_CONFIG_ROOT=/tmp/vllm-config", command)
        self.assertIn("VLLM_NO_USAGE_STATS=1", command)

    def test_step_override(self):
        path = ROOT / "configs/train_qwen38_envduels_lora.json"
        self.assertEqual(load_config(path, max_steps=1)["max_steps"], 1)
        epoch_cfg = load_config(path, epochs=2)
        self.assertIsNone(epoch_cfg["max_steps"])
        self.assertEqual(epoch_cfg["fixed_pool_epochs"], 2)
        self.assertEqual(epoch_cfg["derived"]["rollouts_per_environment_total"], 8)

    def test_invalid_settings(self):
        for update in (
            {"gpu_ids": [0, 0]},
            {"vllm_tensor_parallel": 3},
            {"move_model_batches": 0},
            {"actor_max_tokens": 8192},
            {"learning_rate": float("nan")},
            {"max_steps": True},
            {"trajectories_per_game": 1, "batch_size": 6},
            {"batch_size": 16},
            {"num_games_per_rollout": 8, "batch_size": 32},
            {"dataset_shuffle": "true"},
            {"reward_normalization": "typo"},
            {"ppo_clip_low": 0.3, "ppo_clip_high": 0.2},
            {"fixed_pool_seed": -1},
            {"resume_from_checkpoint": ""},
            {"wandb_enabled": "true"},
            {"wandb_mode": "disabled"},
            {"wandb_project": ""},
            {"wandb_entity": ""},
            {"wandb_run_name": ""},
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

    def test_ms_swift_fsdp_config_matches_accelerate(self):
        document = json.loads((ROOT / "configs/train_qwen38_envduels_lora.json").read_text())
        swift = json.loads((ROOT / "configs/ms_swift_fsdp2.json").read_text())["fsdp_config"]
        accelerate = document["accelerate"]["fsdp_config"]
        self.assertEqual(swift["fsdp_version"], accelerate["fsdp_version"])
        self.assertEqual(swift["auto_wrap_policy"], accelerate["fsdp_auto_wrap_policy"])
        self.assertEqual(
            swift["transformer_layer_cls_to_wrap"],
            accelerate["fsdp_transformer_layer_cls_to_wrap"],
        )
        self.assertEqual(swift["cpu_ram_efficient_loading"], accelerate["fsdp_cpu_ram_efficient_loading"])
        self.assertEqual(swift["sync_module_states"], accelerate["fsdp_sync_module_states"])
        self.assertEqual(swift["reshard_after_forward"], accelerate["fsdp_reshard_after_forward"])
        self.assertEqual(swift["state_dict_type"], accelerate["fsdp_state_dict_type"])

    def test_json_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.yaml"
            path.write_text("training: {}")
            with self.assertRaisesRegex(ValueError, "json"):
                load_config(path)

    def test_training_script_saves_resumable_checkpoints(self):
        script = (ROOT / "cmd/games/train_envduels_lora_swift.sh").read_text()
        self.assertNotIn("--save_only_model", script)
        self.assertIn('--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"', script)
        self.assertIn('--fsdp "$FSDP_CONFIG"', script)
        self.assertIn('--move_model_batches "$MOVE_MODEL_BATCHES"', script)
        self.assertIn("spade/swift_backend/fsdp_ram_loader.py", script)
        self.assertNotIn("--gradient_checkpointing true", script)

    def test_offline_wandb_keeps_training_network_disabled(self):
        cfg = self.config(wandb_mode="offline")
        command, _ = docker_command(cfg, ROOT / "outputs/training/offline", "offline")
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertIn("WANDB_MODE=offline", command)
        self.assertNotIn("WANDB_API_KEY", command)

    def test_wandb_can_be_disabled(self):
        cfg = self.config(wandb_enabled=False)
        command, _ = docker_command(cfg, ROOT / "outputs/training/no-wandb", "no-wandb")
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertIn("REPORT_TO=tensorboard", command)
        self.assertFalse(any(item.startswith("WANDB_") for item in command))


if __name__ == "__main__":
    unittest.main()
