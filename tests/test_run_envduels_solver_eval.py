import json
import asyncio
from pathlib import Path
import tempfile
import unittest

from scripts.run_envduels_solver_eval import (
    DEFAULTS,
    LocalOpenAIClient,
    build_ranking,
    docker_command,
    episode_key,
    load_cases,
    play_case,
    validate_export_matches_baseline,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]


class EnvDuelsSolverEvalTests(unittest.TestCase):
    def test_http_requests_can_use_more_than_default_32_workers(self):
        cfg = self.config()
        cfg["max_concurrent_episodes"] = 64

        def request(_messages, _seed):
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

        client = LocalOpenAIClient(cfg, "http://localhost")
        try:
            self.assertEqual(client.executor._max_workers, 64)
            client._request = request
            result = asyncio.run(client.chat([], 1))
            self.assertEqual(result["text"], "ok")
        finally:
            client.close()

    def config(self):
        return {
            **DEFAULTS,
            "checkpoint": str(ROOT / "checkpoints/Qwen3.8-27B-spade-merged-ckpt24"),
            "export_dir": str(ROOT.parent / "exports/duel_harness_004_rl"),
            "baseline_run": str(ROOT.parent / "exports/duel_harness_004_push"),
            "output_dir": "/tmp/result",
        }

    def test_full_export_selects_four_seeds_for_ninety_environments(self):
        cfg = self.config()
        validate_export_matches_baseline(Path(cfg["export_dir"]), Path(cfg["baseline_run"]))
        cases, digest = load_cases(cfg)
        self.assertEqual(len(cases), 720)
        self.assertEqual(len({case["env_id"] for case in cases}), 90)
        self.assertEqual(len(digest), 64)
        self.assertEqual(len({episode_key(case) for case in cases}), 720)
        self.assertEqual({case["condition"] for case in cases}, {"without_hint", "with_hint"})

    def test_docker_command_uses_exactly_selected_gpus_and_read_only_inputs(self):
        cfg = self.config()
        cfg["gpu_ids"] = [2, 3]
        with tempfile.TemporaryDirectory() as directory:
            command = docker_command(cfg, Path(directory), "test-eval", False)
        self.assertIn('"device=2,3"', command)
        self.assertIn("tensor_parallel", cfg)
        joined = " ".join(command)
        self.assertIn("dst=/model,readonly", joined)
        self.assertIn("dst=/envduels-export,readonly", joined)
        self.assertIn("dst=/baseline-run,readonly", joined)
        self.assertIn("--network none", joined)

    def test_tensor_parallel_must_match_gpu_count(self):
        cfg = self.config()
        cfg["gpu_ids"] = [0]
        with self.assertRaisesRegex(ValueError, "tensor_parallel"):
            validate_config(cfg)

    def test_ranking_uses_terminal_results_and_adds_solver(self):
        cfg = self.config()
        cfg["solver_name"] = "new-solver"
        cfg["max_environments"] = 1
        cfg["seeds_per_environment"] = 1
        case = load_cases(cfg)[0][0]
        hinted = {**case, "condition": "with_hint"}
        rows = {
            episode_key(case): {**case, "key": episode_key(case), "status": "success"},
            episode_key(hinted): {**hinted, "key": episode_key(hinted), "status": "failure"},
        }
        with tempfile.TemporaryDirectory() as directory:
            result = build_ranking(cfg, rows, Path(directory))
            saved = json.loads((Path(directory) / "ranking_solver.json").read_text())
        self.assertEqual(result, saved)
        new = next(row for row in result["ranking"] if row["model"] == "new-solver")
        self.assertEqual(new["canonical_autonomous_solve"], 1.0)
        self.assertEqual(new["hint_gain"], -1.0)
        self.assertEqual(new["overall"]["without_hint"]["accuracy"], 1.0)
        self.assertEqual(new["overall"]["with_hint"]["accuracy"], 0.0)
        self.assertIsNone(new["design_rank"])
        self.assertEqual(new["solve_rank"], 1)
        self.assertEqual(len(result["ranking"]), 10)
        self.assertEqual(next(row for row in result["ranking"] if row["model"] == "qwen3.8-27b")["hint_gain"], 0.12179487179487179)

    def test_hint_is_only_in_hinted_first_prompt(self):
        class Instance:
            def reset(self):
                return "initial observation", {}

            def step(self, response):
                return "done", 1.0, True, False, {}

            def close(self):
                pass

        class Adapter:
            def create_instance_with_seed(self, env_id, seed):
                return Instance()

            def get_hint_levels(self, env_id):
                return ("secret strategy",)

        class Client:
            def __init__(self):
                self.messages = []

            async def chat(self, messages, request_seed):
                self.messages.append(messages)
                return {"text": "\\boxed{WIN}", "usage": {}, "finish_reason": "stop"}

        cfg = self.config()
        case = load_cases(cfg)[0][0]
        client = Client()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "trajectories").mkdir()
            for condition in ("without_hint", "with_hint"):
                row = asyncio.run(play_case(cfg, Adapter(), client, {**case, "condition": condition}, out))
                self.assertEqual(row["status"], "success")
            self.assertNotIn("secret strategy", client.messages[0][1]["content"])
            self.assertIn("secret strategy", client.messages[1][1]["content"])


if __name__ == "__main__":
    unittest.main()
