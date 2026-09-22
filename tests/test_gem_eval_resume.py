"""Regression checks for GEM answer normalization and failed-episode resume."""

import unittest

from eval_offline.suites.gem import _merge_retry_result
from spade.core.eval.gem_evaluator import (
    GemEvalResult,
    GemTaskResult,
    _parse_action,
)
from spade.core.eval.gem_tasks import GemTaskSpec


def _result(items):
    return GemEvalResult(0, 0, 0, sum(x.num_episodes for x in items), len(items), items)


class GemEvalResumeTests(unittest.TestCase):
    def test_prime_factorization_normalizes_nested_latex_box(self):
        response = r"<answer>\boxed{2 \times 2 \cdot 3}</answer>"
        self.assertEqual(
            _parse_action(response, "rg:prime_factorization-hard"),
            "<answer>2 × 2 × 3</answer>",
        )
        self.assertEqual(_parse_action(response, "rg:gcd-hard"), response)

    def test_retry_fills_only_prior_error_slots(self):
        old = GemTaskResult("rg:prime_factorization-hard", "rg", 0, 0, 0, 0, 0, 8)
        retry = GemTaskResult("rg:prime_factorization-hard", "rg", 8, 6, .75, .8, 1, 0)
        spec = GemTaskSpec("rg", "rg:prime_factorization-hard", episodes=8)
        merged = _merge_retry_result(_result([old]), _result([retry]), [spec])
        self.assertEqual(merged.total_episodes, 8)
        self.assertEqual(merged.errors, 0)
        self.assertEqual(merged.per_task_results[0].num_wins, 6)
        self.assertEqual(merged.per_category_metrics["rg_math"]["num_episodes"], 8.0)


if __name__ == "__main__":
    unittest.main()
