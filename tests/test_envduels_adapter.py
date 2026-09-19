"""CPU-only contracts; no Slime runtime, model download or GPU required."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from spade.core.envs.envduels_adapter import (
    EnvDuelsAdapter, extract_action, select_complete_problem_groups,
)
from spade.core.fixed_env_orchestrator import FixedEnvOrchestrator
from spade.core.types import SpadeConfig, TrajectoryStatus


SOURCE = '''
class Env:
    def __init__(self, max_turns):
        self.turns = 0
    def reset(self, seed=None):
        self.turns = 0
        return str(seed), {}
    def step(self, action):
        self.turns += 1
        if action == "CRASH":
            raise RuntimeError("environment crash")
        if action == "READ":
            return "observation", 0.2, False, False, {}
        return "done", 10.0 if action == "WIN" else -1.0, True, False, {}
'''


class FakeModel:
    tokenizer = SimpleNamespace(eos_token_id=99)

    def __init__(self, responses):
        self.responses = iter(responses)

    def apply_template(self, messages):
        return [10, 11]

    async def generate_async(self, **kwargs):
        return [{"text": next(self.responses), "token_ids": [20, 99],
                 "logprobs": [-0.2, -0.3]}]


class EnvDuelsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "env.py").write_text(SOURCE)
        self.row = {"id": "author/env_001", "source": "env.py", "class_name": "Env",
                    "max_turns": 3, "domain": "reasoning", "split": "test",
                    "source_sha256": hashlib.sha256(SOURCE.encode()).hexdigest()}
        self.write_manifest()

    def write_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps(
            {"format_version": 1, "environments": [self.row]}))

    def adapter(self):
        return EnvDuelsAdapter(self.root, seed=123)

    def test_seed_binding_and_independent_state(self):
        adapter = self.adapter()
        first, second = adapter.create_instances_same_problem(self.row["id"], n=2)
        self.assertEqual(first.reset(), second.reset())
        self.assertEqual(first.metadata, second.metadata)
        first.step(r"\boxed{READ}")
        self.assertEqual(second.env.env.turns, 0)
        third = adapter.create_instance(self.row["id"])
        self.assertNotEqual(first.metadata["problem_id"], third.metadata["problem_id"])
        with self.assertRaises(ValueError):
            first.reset(seed=0)
        repeat = self.adapter().create_instance(self.row["id"])
        self.assertEqual(first.metadata["seed"], repeat.metadata["seed"])

    def test_explicit_seed_for_external_grpo_group(self):
        adapter = self.adapter()
        first = adapter.create_instance_with_seed(self.row["id"], 987)
        second = adapter.create_instance_with_seed(self.row["id"], 987)
        self.assertEqual(first.reset(), second.reset())
        self.assertEqual(first.metadata["problem_id"], "author/env_001@seed=987")
        first.step(r"\boxed{READ}")
        self.assertEqual(second.env.env.turns, 0)
        for seed in (-1, 2**63, True, "987"):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                adapter.create_instance_with_seed(self.row["id"], seed)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            adapter.create_instance_with_seed("missing", 987)

    def test_parsing(self):
        self.assertEqual(extract_action(r'\boxed{READ} then \boxed{{"a": 1}}'), '{"a": 1}')
        self.assertIsNone(extract_action(r'\boxed{READ'))

    def test_batch_keeps_whole_groups_without_padding(self):
        trajectories = [SimpleNamespace(metadata={"problem_id": key})
                        for key in ("a", "b", "a", "c", "b")]
        selected = select_complete_problem_groups(trajectories, 4, 2)
        self.assertEqual([t.metadata["problem_id"] for t in selected], ["a", "a", "b", "b"])
        with self.assertRaises(RuntimeError):
            select_complete_problem_groups(trajectories, 6, 2)
        with self.assertRaises(ValueError):
            select_complete_problem_groups(trajectories, 3, 2)

    def test_crash_filtered_but_normal_loss_kept(self):
        adapter = self.adapter()
        orchestrator = FixedEnvOrchestrator(
            FakeModel([r"\boxed{CRASH}", r"\boxed{LOSE}"]),
            SpadeConfig(max_turns=1), [adapter],
        )
        _, trajectories, info = asyncio.run(orchestrator.collect_trajectories_async(2, 2))
        self.assertEqual(len(trajectories), 1)
        self.assertEqual(info["num_failed"], 1)
        self.assertEqual(trajectories[0].metadata["original_reward"], 0)

    def test_rewards_format_errors_and_crashes(self):
        instance = self.adapter().create_instance(self.row["id"])
        instance.reset()
        result = instance.step(r"reasoning \boxed{READ}")
        self.assertEqual(result[1], 0)
        self.assertEqual(result[4]["raw_reward"], 0.2)
        result = instance.step(r"\boxed{WIN}")
        self.assertEqual(result[1], 1)
        self.assertEqual(result[4]["raw_reward"], 10)
        instance.reset()
        self.assertEqual(instance.step(r"\boxed{LOSE}")[1], 0)
        instance.reset()
        for _ in range(3):
            result = instance.step("unboxed action")
        self.assertEqual(result[1:4], (0.0, False, True))
        instance.reset()
        with self.assertRaisesRegex(RuntimeError, "environment crash"):
            instance.step(r"\boxed{CRASH}")

    def test_hash_path_selection(self):
        adapter = self.adapter()
        self.assertEqual(adapter.list_environments(), [self.row["id"]])
        (self.root / "env.py").write_text(SOURCE + "# changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            adapter.create_instance(self.row["id"])
        self.row["source"] = "../outside.py"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.adapter().create_instance(self.row["id"])
        selection = self.root / "ids.txt"
        selection.write_text("unknown\n")
        with self.assertRaises(ValueError):
            EnvDuelsAdapter(self.root, env_ids_file=selection)

    def test_rollout_masks_metadata_and_failure(self):
        adapter = self.adapter()
        orchestrator = FixedEnvOrchestrator(
            FakeModel([r"\boxed{READ}", r"\boxed{WIN}"]),
            SpadeConfig(max_turns=3), [adapter],
        )
        with patch("spade.core.utils.token_utils.get_observation_delta", return_value=([30, 31], [0, 0])):
            trajectory = asyncio.run(orchestrator.play_env_async(adapter.create_instance(self.row["id"])))
        self.assertEqual(trajectory.reward, 1)
        self.assertEqual(trajectory.loss_mask, [1, 1, 0, 0, 1, 1])
        self.assertEqual(trajectory.metadata["game_file"], trajectory.metadata["problem_id"])
        self.assertEqual(trajectory.metadata["reward_diagnostics"][-1]["raw_reward"], 10)
        orchestrator.model = FakeModel([r"\boxed{CRASH}"])
        failed = asyncio.run(orchestrator.play_env_async(adapter.create_instance(self.row["id"])))
        self.assertEqual(failed.status, TrajectoryStatus.FAILED)

    def test_two_seeds_are_normalized_separately_in_both_paths(self):
        for fixed in (False, True):
            with self.subTest(fixed_pool=fixed):
                adapter = self.adapter()
                orchestrator = FixedEnvOrchestrator(
                    FakeModel([r"\boxed{WIN}", r"\boxed{WIN}", r"\boxed{LOSE}", r"\boxed{LOSE}"]),
                    SpadeConfig(max_turns=1, reward_normalization="grpo"), [adapter],
                    fixed_pool=[(self.row["id"], 0)] if fixed else None,
                )
                # Async scheduling order is not group order. Tie the fake
                # outcome to the task's observation, not call arrival order.
                winning_seed = str(self.adapter().create_instance(self.row["id"]).metadata["seed"])

                async def generate(**kwargs):
                    text = r"\boxed{WIN}" if winning_seed in str(kwargs["messages"]) else r"\boxed{LOSE}"
                    return [{"text": text, "token_ids": [20, 99], "logprobs": [-0.2, -0.3]}]

                orchestrator.model.generate_async = generate
                with patch.object(orchestrator, "_select_env_ids", return_value=[self.row["id"]] * 2), \
                     patch.object(orchestrator, "_select_pool_entries", return_value=[(self.row["id"], 0)] * 2):
                    _, trajectories, _ = asyncio.run(orchestrator.collect_trajectories_async(4, 2))
                groups = {}
                for trajectory in trajectories:
                    groups.setdefault(trajectory.metadata["problem_id"], []).append(trajectory)
                self.assertEqual(sorted(map(len, groups.values())), [2, 2])
                self.assertTrue(all(abs(t.reward) < 1e-6 for t in trajectories))

    @unittest.skipUnless(os.environ.get("ENVDUELS_TEST_EXPORT"), "optional real export")
    def test_real_rotation_lock(self):
        adapter = EnvDuelsAdapter(os.environ["ENVDUELS_TEST_EXPORT"])
        env_id = next(e for e in adapter.list_environments() if "qwen3.8-27b/env_001/" in e)
        instances = adapter.create_instances_same_problem(env_id, n=2)
        self.assertEqual(instances[0].reset(), instances[1].reset())
        for instance in instances:
            orchestrator = FixedEnvOrchestrator(
                FakeModel([r"\boxed{READ}", r"\boxed{COMMIT}"]),
                SpadeConfig(max_turns=2), [adapter],
            )
            with patch("spade.core.utils.token_utils.get_observation_delta", return_value=([30], [0])):
                trajectory = asyncio.run(orchestrator.play_env_async(instance))
            self.assertNotEqual(trajectory.status, TrajectoryStatus.FAILED)
            self.assertIn(trajectory.reward, (0, 1))
            self.assertEqual(trajectory.metadata["problem_id"], instances[0].metadata["problem_id"])


if __name__ == "__main__":
    unittest.main()
