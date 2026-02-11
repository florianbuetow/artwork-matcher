"""Unit tests for evaluation metric helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

import sys

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import evaluate  # noqa: E402


class EvaluateMetricsTest(unittest.TestCase):
    """Validate pure metric functions in tools/evaluate.py."""

    def setUp(self) -> None:
        self.labels = {
            "p1": evaluate.LabelEntry(picture_id="p1", valid_object_ids=frozenset({"o1"})),
            "p2": evaluate.LabelEntry(picture_id="p2", valid_object_ids=frozenset({"o2"})),
            "p3": evaluate.LabelEntry(picture_id="p3", valid_object_ids=frozenset({"o3"})),
        }

        self.results = [
            evaluate.MatchResult(
                picture_id="p1",
                mode="embedding_only",
                matched_object_id="o1",
                ranked_results=[{"object_id": "o1"}, {"object_id": "x1"}],
            ),
            evaluate.MatchResult(
                picture_id="p2",
                mode="embedding_only",
                matched_object_id="x2",
                ranked_results=[{"object_id": "x2"}, {"object_id": "o2"}],
            ),
            evaluate.MatchResult(
                picture_id="p3",
                mode="embedding_only",
                error="request_error",
                error_message="timeout",
                ranked_results=[],
            ),
        ]

    def test_calculate_classification_metrics(self) -> None:
        metrics = evaluate.calculate_classification_metrics(self.results, self.labels)

        self.assertEqual(metrics.true_positives, 1)
        self.assertEqual(metrics.false_positives, 1)
        self.assertEqual(metrics.false_negatives, 1)
        self.assertAlmostEqual(metrics.precision, 0.5, places=6)
        self.assertAlmostEqual(metrics.recall, 0.5, places=6)
        self.assertAlmostEqual(metrics.f1_score, 0.5, places=6)

    def test_calculate_mrr(self) -> None:
        mrr = evaluate.calculate_mrr(self.results, self.labels)
        self.assertAlmostEqual(mrr, 0.5, places=6)

    def test_calculate_hit_at_k(self) -> None:
        hit_at_1 = evaluate.calculate_hit_at_k(self.results, self.labels, 1)
        hit_at_2 = evaluate.calculate_hit_at_k(self.results, self.labels, 2)

        self.assertAlmostEqual(hit_at_1, 1.0 / 3.0, places=6)
        self.assertAlmostEqual(hit_at_2, 2.0 / 3.0, places=6)

    def test_calculate_ndcg_at_k(self) -> None:
        ndcg_at_2 = evaluate.calculate_ndcg_at_k(self.results, self.labels, 2)
        expected = (1.0 + (1.0 / 1.5849625007211563) + 0.0) / 3.0
        self.assertAlmostEqual(ndcg_at_2, expected, places=6)


if __name__ == "__main__":
    unittest.main()
