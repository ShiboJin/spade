import asyncio
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.eval_aime26 import configuration, inside_checkout, score_answer, score_dataset
from scripts.prepare_aime26 import validate_rows
from scripts.benchmark_data import grade, load_benchmark


class ExperimentTests(unittest.TestCase):
    def test_generic_protocols(self):
        self.assertEqual(grade(r"\boxed{-007}", "-7", "boxed_integer")[1:], (True, True))
        self.assertTrue(grade(r"\boxed{Paris}", "Paris", "boxed_exact_match")[1])
        self.assertFalse(grade(r"\boxed{paris}", "Paris", "boxed_exact_match")[1])
        self.assertFalse(grade(r"\boxed{0.5}", "1/2", "boxed_exact_match")[1])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.jsonl"
            path.write_text(json.dumps(dict(id="test", problem="Capital?", answer="Paris")))
            self.assertEqual(len(load_benchmark(path, "boxed_exact_match")[0]), 1)
            with self.assertRaises(ValueError):
                load_benchmark(path, "boxed_integer")
            with self.assertRaises(ValueError):
                load_benchmark(path, "unknown")
            path.write_text("")
            with self.assertRaises(ValueError):
                load_benchmark(path, "boxed_exact_match")

    def test_data_count_duplicates_labels(self):
        rows = [dict(id=i + 1, problem=f"Problem {i}", answer=i) for i in range(30)]
        self.assertEqual(len(validate_rows(rows)), 30)
        with self.assertRaises(ValueError):
            validate_rows(rows[:15])
        with self.assertRaises(ValueError):
            validate_rows(rows[:-1] + [rows[0]])
        with self.assertRaises(ValueError):
            validate_rows(rows[:-1] + [dict(id=30, problem="Other", answer=1000)])

    def test_exact_integer_scoring(self):
        self.assertTrue(score_answer(r"Reasoning \boxed{007}", 7)[1])
        self.assertFalse(score_answer(r"\boxed{7} then \boxed{8}", 7)[1])
        for answer in ("7", r"\boxed{7.0}", r"\boxed{1000}", r"\boxed{seven}"):
            self.assertFalse(score_answer(answer, 7)[1])

    def test_eval_config_validation(self):
        with patch.dict(os.environ, {"EVAL_CONTEXT_LENGTH": "100", "EVAL_MAX_TOKENS": "100"}):
            with self.assertRaises(ValueError):
                configuration()
        with self.assertRaises(ValueError):
            inside_checkout("/tmp/not-mounted")

    def test_metrics_and_thinking_forwarding(self):
        class Client:
            async def chat(inner, **kwargs):
                self.assertFalse(kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"])
                return [SimpleNamespace(text=r"\boxed{7}", raw={}, finish_reason="stop", completion_tokens=3),
                        SimpleNamespace(text=r"\boxed{8}", raw={}, finish_reason="length", completion_tokens=4)]
        cfg = dict(n_samples=2, temperature=0.7, top_p=0.8, top_k=20, max_tokens=128, thinking=False, seed=42)
        stream = io.StringIO()
        metrics = asyncio.run(score_dataset(Client(), [dict(id=1, problem="x", answer=7)], cfg, stream))
        self.assertEqual(metrics["sample_accuracy"], 0.5)
        self.assertEqual(metrics["fraction_problems_any_correct"], 1)
        self.assertEqual(metrics["length_stop_fraction"], 0.5)
        self.assertEqual(len(stream.getvalue().splitlines()), 2)

    def test_missing_completions_fail(self):
        class Client:
            async def chat(self, **kwargs):
                return []
        cfg = dict(n_samples=2, temperature=0.7, top_p=0.8, top_k=20, max_tokens=128, thinking=False, seed=42)
        with self.assertRaises(RuntimeError):
            asyncio.run(score_dataset(Client(), [dict(id=1, problem="x", answer=7)], cfg, io.StringIO()))

    def test_generic_string_id_and_metrics(self):
        class Client:
            async def chat(self, **kwargs):
                return [SimpleNamespace(text=r"\boxed{Paris}", raw={}, finish_reason="stop", completion_tokens=3)]
        cfg = dict(benchmark="boxed_exact_match", n_samples=1, temperature=0.7,
                   top_p=0.8, top_k=20, max_tokens=128, thinking=False, seed=42)
        metrics = asyncio.run(score_dataset(Client(),
            [dict(id="geography-1", problem="Capital of France?", answer="Paris")], cfg, io.StringIO()))
        self.assertEqual(metrics["sample_accuracy"], 1)
        self.assertEqual(metrics["invalid_answer_fraction"], 0)


if __name__ == "__main__":
    unittest.main()
