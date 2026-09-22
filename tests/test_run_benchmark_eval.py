"""CPU-only checks for the Docker multi-benchmark launcher."""

from pathlib import Path
import tempfile
import unittest

import yaml

from eval_offline.suites._acebench_patches import _patch_request_timeout
from scripts.run_benchmark_eval import (
    ROOT, checkpoint_mount, docker_command, load_config,
)


class BenchmarkEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "configs/qwen38_benchmarks.json")

    def test_protocol_and_concurrency(self):
        self.assertEqual(self.cfg["max_concurrent"], 64)
        self.assertEqual(self.cfg["max_num_seqs"], 64)
        self.assertEqual(self.cfg["request_timeout_seconds"], 3600)
        selected = [
            suite
            for entry in self.cfg["evaluations"]
            for suite in entry["suites"]
        ]
        self.assertEqual(selected, ["gem"])

        games = yaml.safe_load(
            (ROOT / "eval_offline/configs/games.yaml").read_text()
        )
        ace = yaml.safe_load(
            (ROOT / "eval_offline/configs/acebench_agent.yaml").read_text()
        )
        self.assertEqual(games["suites"]["gem"]["defaults"]["max_concurrent"], 64)
        self.assertEqual(ace["suites"]["acebench"]["num_threads"], 16)

    def test_docker_command_uses_renewed_source_and_vllm_image(self):
        output = Path("/tmp/qwen38 benchmark output")
        command = docker_command(
            self.cfg, output, "benchmark-test", ["HF_TOKEN", "OPENROUTER_API_KEY"]
        )
        self.assertIn("envduels-unified:cu124", command)
        self.assertIn('"device=8,9"', command)
        self.assertIn("HF_TOKEN", command)
        self.assertIn("OPENROUTER_API_KEY", command)
        self.assertNotIn("--network", command)
        self.assertNotIn("--user", command)
        self.assertIn("spade.gpu-concurrent=true", command)
        self.assertNotIn("spade.eval-concurrent=true", command)
        renewed_mount = (
            f"type=bind,src={ROOT},dst=/workspace/spade,readonly"
        )
        self.assertIn(renewed_mount, command)
        self.assertNotIn("/home/yzo/home/spade-latest", " ".join(command))
        source, target, checkpoint = checkpoint_mount(Path(self.cfg["checkpoint"]))
        self.assertEqual(source.name, "hub")
        self.assertEqual(target, "/model-hub")
        self.assertIn("/models--Qwen--Qwen3.8-27B/snapshots/", checkpoint)
        self.assertIn(
            f"type=bind,src={source},dst={target},readonly", command
        )
        self.assertEqual(command[-4:], [
            "scripts/run_eval.py", "--inside-container",
            "--config", "/results/container_config.json",
        ])

    def test_acebench_uses_the_configured_request_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targets = [
                root / "model_inference/apimodel_inference.py",
                root / "model_inference/multi_step/APIModel_agent.py",
                root / "model_inference/multi_turn/APIModel_agent.py",
            ]
            for target in targets:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    "self.client = OpenAI(base_url=base_url, api_key=api_key)\n"
                )
            _patch_request_timeout(root)
            _patch_request_timeout(root)
            for target in targets:
                patched = target.read_text()
                self.assertEqual(
                    patched.count("SPADE_EVAL_REQUEST_TIMEOUT_SECONDS"), 1
                )


if __name__ == "__main__":
    unittest.main()
