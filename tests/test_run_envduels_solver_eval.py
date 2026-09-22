import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_envduels_solver_eval import (
    DEFAULTS,
    build_ranking,
    docker_command,
    episode_key,
    load_cases,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]


class EnvDuelsSolverEvalTests(unittest.TestCase):
    def config(self):
        return {
            **DEFAULTS,
            "checkpoint": str(ROOT / "checkpoints/Qwen3.8-27B-spade-merged-ckpt19"),
            "export_dir": str(ROOT.parent / "exports/duel_harness_004_rl"),
            "baseline_run": str(ROOT.parent / "exports/duel_harness_004_push"),
            "output_dir": "/tmp/result",
        }

    def test_full_export_selects_four_seeds_for_ninety_environments(self):
        cfg = self.config()
        cases, digest = load_cases(cfg)
        self.assertEqual(len(cases), 360)
        self.assertEqual(len({case["env_id"] for case in cases}), 90)
        self.assertEqual(len(digest), 64)
        self.assertEqual(len({episode_key(case) for case in cases}), 360)

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

    def test_ranking_adds_solver_and_uses_terminal_results_only(self):
        cfg = self.config()
        cfg["solver_name"] = "new-solver"
        cfg["max_environments"] = 1
        cfg["seeds_per_environment"] = 1
        case = load_cases(cfg)[0][0]
        rows = {episode_key(case): {**case, "key": episode_key(case), "status": "success"}}
        with tempfile.TemporaryDirectory() as directory:
            result = build_ranking(cfg, rows, Path(directory))
            saved = json.loads((Path(directory) / "ranking_solver.json").read_text())
        self.assertEqual(result, saved)
        new = next(row for row in result["ranking"] if row["model"] == "new-solver")
        self.assertEqual(new["canonical_autonomous_solve"], 1.0)


if __name__ == "__main__":
    unittest.main()
