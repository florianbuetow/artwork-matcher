"""Unit tests for evaluation metric helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

import sys

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from evaluation.metrics import (  # noqa: E402
    calculate_classification_metrics,
    calculate_hit_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
)
from evaluation.models import LabelEntry, MatchResult, RankedResultItem  # noqa: E402


class EvaluateMetricsTest(unittest.TestCase):
    """Validate pure metric functions in tools/evaluation/metrics.py."""

    def setUp(self) -> None:
        self.labels = {
            "p1": LabelEntry(picture_id="p1", valid_object_ids=frozenset({"o1"})),
            "p2": LabelEntry(picture_id="p2", valid_object_ids=frozenset({"o2"})),
            "p3": LabelEntry(picture_id="p3", valid_object_ids=frozenset({"o3"})),
        }

        self.results = [
            MatchResult(
                picture_id="p1",
                mode="embedding_only",
                matched_object_id="o1",
                similarity_score=0.95,
                geometric_score=None,
                confidence=0.95,
                ranked_results=[
                    RankedResultItem("o1", 0.95, None, 0.95),
                    RankedResultItem("x1", 0.80, None, 0.80),
                ],
                embedding_ms=10.0,
                search_ms=8.0,
                geometric_ms=0.0,
                total_ms=20.0,
                error=None,
                error_message=None,
            ),
            MatchResult(
                picture_id="p2",
                mode="embedding_only",
                matched_object_id="x2",
                similarity_score=0.87,
                geometric_score=None,
                confidence=0.87,
                ranked_results=[
                    RankedResultItem("x2", 0.87, None, 0.87),
                    RankedResultItem("o2", 0.70, None, 0.70),
                ],
                embedding_ms=11.0,
                search_ms=8.0,
                geometric_ms=0.0,
                total_ms=21.0,
                error=None,
                error_message=None,
            ),
            MatchResult(
                picture_id="p3",
                mode="embedding_only",
                matched_object_id=None,
                similarity_score=None,
                geometric_score=None,
                confidence=None,
                error="request_error",
                error_message="timeout",
                ranked_results=[],
                embedding_ms=0.0,
                search_ms=0.0,
                geometric_ms=0.0,
                total_ms=0.0,
            ),
        ]

    def test_calculate_classification_metrics(self) -> None:
        metrics = calculate_classification_metrics(self.results, self.labels)

        self.assertEqual(metrics.true_positives, 1)
        self.assertEqual(metrics.false_positives, 1)
        self.assertEqual(metrics.false_negatives, 1)
        self.assertAlmostEqual(metrics.precision, 0.5, places=6)
        self.assertAlmostEqual(metrics.recall, 0.5, places=6)
        self.assertAlmostEqual(metrics.f1_score, 0.5, places=6)

    def test_calculate_mrr(self) -> None:
        mrr = calculate_mrr(self.results, self.labels)
        self.assertAlmostEqual(mrr, 0.5, places=6)

    def test_calculate_hit_at_k(self) -> None:
        hit_at_1 = calculate_hit_at_k(self.results, self.labels, 1)
        hit_at_2 = calculate_hit_at_k(self.results, self.labels, 2)

        self.assertAlmostEqual(hit_at_1, 1.0 / 3.0, places=6)
        self.assertAlmostEqual(hit_at_2, 2.0 / 3.0, places=6)

    def test_calculate_ndcg_at_k(self) -> None:
        ndcg_at_2 = calculate_ndcg_at_k(self.results, self.labels, 2)
        expected = (1.0 + (1.0 / 1.5849625007211563) + 0.0) / 3.0
        self.assertAlmostEqual(ndcg_at_2, expected, places=6)


if __name__ == "__main__":
    unittest.main()
