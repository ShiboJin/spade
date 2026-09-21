"""CPU-only checks for the config-driven LoRA GRPO launcher."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_train import ROOT, ROLLOUT_DEFAULTS, docker_command, load_config


class TrainingLauncherTests(unittest.TestCase):
    def test_sage_defaults_and_exclusive_refill_control(self):
        cfg = self.config(remove_constant_reward_groups=True)
        self.assertTrue(cfg["sage_hint_resampling"])
        command, _ = docker_command(cfg, ROOT / "outputs/training/test-sage", "test-sage")
        self.assertIn("SPADE_SAGE_HINT_RESAMPLING=true", command)
        self.assertIn("SPADE_MAX_ROLLOUT_ATTEMPTS=2", command)
        self.assertIn("SPADE_MIN_VALID_GROUPS=4", command)
        self.assertIn("DYNAMIC_SAMPLE=false", command)
        cfg = self.config(sage_hint_resampling=False, remove_constant_reward_groups=True)
        command, _ = docker_command(cfg, ROOT / "outputs/training/test-plain", "test-plain")
        self.assertIn("SPADE_SAGE_HINT_RESAMPLING=false", command)
        self.assertIn("DYNAMIC_SAMPLE=true", command)

    def test_min_valid_groups_config(self):
        for value in (0, -1, True, 2.5, 7):
            with self.assertRaisesRegex(ValueError, "min_valid_groups"):
                self.config(min_valid_groups=value)
        cfg = self.config(min_valid_groups=5)
        command, _ = docker_command(cfg, ROOT / "outputs/training/test-sage", "test-sage")
        self.assertIn("SPADE_MIN_VALID_GROUPS=5", command)

    def test_sage_rejects_multiple_substeps(self):
        with self.assertRaisesRegex(ValueError, "num_substeps=1"):
            self.config(num_substeps=2)
        self.config(num_substeps=2, sage_hint_resampling=False)

    def test_disjoint_training_and_evaluation_use_shared_gpu_reservations(self):
        from scripts.run_train import task_resources as training_resources
        from scripts.run_eval import task_resources as eval_resources
        from scripts.run_eval import load_config as eval_config
        accelerate = {**self.config()["accelerate"], "num_processes": 4}
        first = self.config(gpu_ids=[0, 1, 2, 3], vllm_tensor_parallel=4, memory_limit_gib=80,
                            accelerate=accelerate)
        second = self.config(gpu_ids=[4, 5, 6, 7], vllm_tensor_parallel=4, memory_limit_gib=80,
                             accelerate=accelerate)
        for cfg in (first, second):
            command, _ = docker_command(cfg, ROOT / "outputs/training/test", "test")
            self.assertIn('"device=' + ','.join(map(str, cfg["gpu_ids"])) + '"', command)
            self.assertIn("spade.gpu-concurrent=true", command)
            self.assertIn("NUM_GPUS=4", command)
            self.assertEqual(cfg["accelerate"]["num_processes"], 4)
        with tempfile.TemporaryDirectory() as directory, patch("scripts.memory_guard.concurrent_preflight"):
            root = Path(directory)
            with training_resources(first, root):
                with training_resources(second, root):
                    with self.assertRaisesRegex(RuntimeError, "GPU"):
                        with eval_resources(eval_config(ROOT / "configs/evaluation.json"), root):
                            self.fail("eval overlapped training")
                with eval_resources(eval_config(ROOT / "configs/evaluation.json",
                                                dict(gpu_ids=[4, 5, 6, 7])), root):
                    pass

    def config(self, accelerate=None, **updates):
        document = json.loads((ROOT / "configs/train_qwen38_envduels_lora.json").read_text())
        document["training"].update(updates)
        if accelerate is not None:
            document["accelerate"] = accelerate
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.json"
            path.write_text(json.dumps(document))
            return load_config(path)

    def test_online_config_and_docker_command(self):
        cfg = self.config()
        self.assertEqual(cfg["wandb_mode"], "online")
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
        for value in ("VLLM_ENFORCE_EAGER=true", "SLEEP_LEVEL=2",
                      "OFFLOAD_MODEL=true", "OFFLOAD_OPTIMIZER=true"):
            self.assertIn(value, command)
        self.assertNotIn("MAX_STEPS=100", command)
        self.assertIn("NUM_TRAIN_EPOCHS=3", command)
        self.assertIn("GENERATION_BATCH_SIZE=24", command)
        self.assertIn("NUM_GENERATIONS=4", command)
        self.assertIn("GRADIENT_ACCUMULATION_STEPS=3", command)
        self.assertIn("NUM_ITERATIONS=1", command)
        self.assertIn("LEARNING_RATE=1e-06", command)
        self.assertIn("LORA_RANK=32", command)
        self.assertIn("SPADE_GRPO_CHUNKED_LOGPS=true", command)
        self.assertIn("SPADE_GRPO_LOGPS_CHUNK_SIZE=128", command)
        self.assertIn("SPADE_GRPO_DECODER_CHECKPOINTING=true", command)
        self.assertIn("SPADE_GRPO_CPU_ACTIVATION_OFFLOAD=true", command)
        self.assertIn("SPADE_GRPO_CHECKPOINT_DELTA_RULE=true", command)
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
        self.assertIn("WANDB_DATA_DIR=/workspace/envduels/spade/outputs/training/test-run/wandb/data", command)
        self.assertIn("WANDB_CACHE_DIR=/workspace/envduels/spade/outputs/training/test-run/wandb/cache", command)
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
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", command)
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
            {"vllm_enforce_eager": "false"},
            {"offload_model": 0},
            {"offload_optimizer": "false"},
            {"sleep_level": True},
            {"sleep_level": -1},
            {"sleep_level": 3},
            {"sleep_level": 1.0},
            {"max_context_length": 8192, "actor_max_tokens": 8192},
            {"grpo_logps_chunk_size": 0},
            {"grpo_chunked_logps": "true"},
            {"grpo_decoder_checkpointing": False, "grpo_cpu_activation_offload": True},
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

    def test_a100_profile_disables_extra_activation_offload_and_recomputation(self):
        cfg = load_config(ROOT / "configs/train_qwen38_envduels_lora_a100_80gb.json")
        command, _ = docker_command(cfg, ROOT / "outputs/training/test-a100", "test-a100")
        self.assertIn("SPADE_GRPO_CHUNKED_LOGPS=true", command)
        self.assertIn("SPADE_GRPO_DECODER_CHECKPOINTING=false", command)
        self.assertIn("SPADE_GRPO_CPU_ACTIVATION_OFFLOAD=false", command)
        self.assertIn("SPADE_GRPO_CHECKPOINT_DELTA_RULE=false", command)
        for value in ("SPADE_GRPO_LOGPS_CHUNK_SIZE=512", "MOVE_MODEL_BATCHES=32",
                      "VLLM_ENFORCE_EAGER=false", "SLEEP_LEVEL=1",
                      "OFFLOAD_MODEL=false", "OFFLOAD_OPTIMIZER=false"):
            self.assertIn(value, command)

    def test_legacy_rollout_defaults_and_zero_sleep(self):
        document = json.loads((ROOT / "configs/train_qwen38_envduels_lora.json").read_text())
        for key in ROLLOUT_DEFAULTS:
            document["training"].pop(key, None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.json"
            path.write_text(json.dumps(document))
            cfg = load_config(path)
        for key, value in ROLLOUT_DEFAULTS.items():
            self.assertEqual(cfg[key], value)
        command, _ = docker_command(self.config(sleep_level=0), ROOT / "outputs/training/test-zero", "test-zero")
        self.assertIn("SLEEP_LEVEL=0", command)

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
        self.assertIn("spade/swift_backend/rlhf_entry.py", script)
        self.assertIn("spade/swift_backend/epoch_checkpoints.py", script)
        self.assertIn("--callbacks envduels_wandb_config envduels_epoch_checkpoints", script)
        self.assertNotIn("--save_only_model", script)
        self.assertIn('--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"', script)
        self.assertIn('--fsdp "$FSDP_CONFIG"', script)
        self.assertIn('--move_model_batches "$MOVE_MODEL_BATCHES"', script)
        for flag, env in (("vllm_enforce_eager", "VLLM_ENFORCE_EAGER"),
                          ("sleep_level", "SLEEP_LEVEL"), ("offload_model", "OFFLOAD_MODEL"),
                          ("offload_optimizer", "OFFLOAD_OPTIMIZER")):
            self.assertIn(f'--{flag} "${env}"', script)
        self.assertIn("spade/swift_backend/fsdp_ram_loader.py", script)
        self.assertNotIn("--gradient_checkpointing true", script)

    def test_offline_wandb_keeps_training_network_disabled(self):
        cfg = self.config(wandb_mode="offline")
        self.assertEqual(cfg["wandb_mode"], "offline")
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
