"""CPU-only regression checks for the configurable evaluation protocol."""
import asyncio
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.run_eval import ROOT, build_server_command, docker_command, load_config, load_dataset, score_dataset
from scripts.benchmark_data import grade


class EvaluationTests(unittest.TestCase):
    def config(self, **overrides):
        return load_config(ROOT / "configs/evaluation.json", overrides)

    def dataset(self, records, **overrides):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
            return load_dataset(self.config(data=str(path), **overrides))

    def test_arbitrary_size_and_subset(self):
        records = [dict(id=f"q{i}", problem="Compute something", answer=i) for i in range(3)]
        rows, digest, total = self.dataset(records, max_problems=2)
        self.assertEqual([r["id"] for r in rows], ["q0", "q1"])
        self.assertEqual(total, 3)
        self.assertEqual(len(digest), 64)
        self.assertEqual(rows[0]["answer"], 0)
        self.assertNotIn("answer", rows[0]["messages"][0])

    def test_chat_fields_and_no_duplicate_suffix(self):
        suffix = self.config()["prompt_suffix"]
        records = [dict(prompt=[dict(role="system", content="Be precise."),
                               dict(role="user", content="Question" + suffix)], label="7")]
        rows, _, _ = self.dataset(records, prompt_key="prompt", answer_key="label", id_key=None)
        self.assertEqual(rows[0]["id"], 1)
        self.assertEqual(rows[0]["messages"][-1]["content"], "Question" + suffix)
        self.assertTrue(suffix.startswith("\n"))
        self.assertIn(r"\boxed{}", suffix)

    def test_invalid_datasets_fail_before_inference(self):
        good = dict(id="a", problem="Question", answer=1)
        for rows in ([], [good, good], [dict(good, answer=True)], [dict(good, answer="1.5")],
                     [dict(good, problem=[])], [dict(good, problem=[dict(role="assistant", content="leaked answer")])]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.dataset(rows)

    def test_invalid_config(self):
        for override in (dict(gpu_ids=[0, 0]), dict(samples_per_problem=0), dict(tensor_parallel=2),
                         dict(temperature=0), dict(top_p=float("nan")), dict(max_problems=0),
                         dict(max_tokens=12288), dict(enforce_eager="false")):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.config(**override)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"evaluation": {"checkpiont": "bad"}}))
            with self.assertRaisesRegex(ValueError, "Unknown"):
                load_config(path)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("evaluation: {}")
            with self.assertRaisesRegex(ValueError, "json"):
                load_config(path)

    def test_external_paths_mounted_and_relative_paths_stable(self):
        cfg = self.config(checkpoint="/tmp/external model", data="/tmp/tests.jsonl")
        command = docker_command(cfg, Path("/tmp/eval output"), "test-evaluation")
        self.assertEqual(command[command.index("--memory") + 1], "96g")
        self.assertEqual(command[command.index("--memory-swap") + 1], "96g")
        self.assertEqual(command[command.index("--memory-swappiness") + 1], "0")
        self.assertIn("type=bind,src=/tmp/external model,dst=/model,readonly", command)
        self.assertIn("type=bind,src=/tmp/tests.jsonl,dst=/dataset/test.jsonl,readonly", command)
        self.assertEqual(self.config()["checkpoint"], str(ROOT / "checkpoints/Qwen3.8-27B"))
        self.assertNotIn("--publish", command)

    def test_lora_mount_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "adapter_config.json").write_text(json.dumps({"r": 16}))
            (root / "adapter_model.safetensors").touch()
            cfg = self.config(lora=str(root))
            command = docker_command(cfg, Path("/tmp/eval-output"), "test-lora")
            self.assertIn(f"type=bind,src={root},dst=/lora,readonly", command)
            self.assertEqual(cfg["lora"], str(root))
            server = build_server_command(cfg)
            self.assertIn("--enable-lora", server)
            self.assertEqual(server[server.index("--max-lora-rank") + 1], "16")
            self.assertEqual(server[server.index("--served-model-name") + 1], "evaluation-base")
            self.assertIn(f"evaluation-model={root}", server)

    def test_boxed_grading(self):
        self.assertEqual(grade(r"\boxed{5} then \boxed{007}", 7, "boxed_integer"), ("007", True, True))
        self.assertEqual(grade(r"\boxed{\frac{1}{2}}", r"\frac{1}{2}", "boxed_exact_match")[1:], (True, True))
        self.assertFalse(grade(r"\boxed{7", 7, "boxed_integer")[2])
        self.assertFalse(grade("The answer is 7", 7, "boxed_integer")[2])

    def test_variable_k_metrics_and_request(self):
        requests = []

        class Client:
            async def chat(self, **kwargs):
                requests.append(kwargs)
                answers = [r"\boxed{7}", "no box", r"\boxed{0}"] if kwargs["messages"][0]["content"] == "first" else [r"\boxed{0}"] * 3
                return [SimpleNamespace(text=a, finish_reason="length" if i == 1 else "stop", raw={}) for i, a in enumerate(answers)]

        rows = [dict(id=i, answer=7, messages=[dict(role="user", content=p)]) for i, p in enumerate(("first", "second"))]
        stream = io.StringIO()
        cfg = self.config(samples_per_problem=3)
        metrics = asyncio.run(score_dataset(Client(), rows, cfg, stream))
        self.assertEqual(metrics["pass_at_3"], 0.5)
        self.assertEqual(metrics["avg_at_3"], 1 / 6)
        self.assertEqual(metrics["n_completions"], 6)
        self.assertEqual(metrics["invalid_answer_fraction"], 1 / 6)
        self.assertEqual(metrics["length_stop_fraction"], 2 / 6)
        self.assertEqual(len(stream.getvalue().splitlines()), 6)
        self.assertEqual(requests[0]["extra_body"]["seed"], 43)
        self.assertEqual(requests[1]["extra_body"]["seed"], 44)
        self.assertEqual(requests[0]["n"], 3)

    def test_incomplete_sampling_fails(self):
        class Client:
            async def chat(self, **kwargs):
                return []
        rows = [dict(id="x", answer=7, messages=[dict(role="user", content="question")])]
        with self.assertRaisesRegex(RuntimeError, "expected 8 samples"):
            asyncio.run(score_dataset(Client(), rows, self.config(), io.StringIO()))

    def test_dry_run_never_starts_docker_or_creates_output(self):
        from scripts.run_eval import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            (root / "model.safetensors").touch()
            (root / "data.jsonl").write_text(json.dumps(dict(id=1, problem="Q", answer=1)))
            args = ["run_eval.py", "--dry-run", "--checkpoint", str(root), "--data", str(root / "data.jsonl"), "--output-dir", str(root / "output")]
            with patch("sys.argv", args), patch("subprocess.Popen", side_effect=AssertionError("unexpected process")), patch("sys.stdout", new=io.StringIO()):
                main()
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
