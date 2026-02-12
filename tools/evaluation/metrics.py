"""Metric calculations for evaluation tooling."""

from __future__ import annotations

import math

import numpy as np

from evaluation.models import (
    ClassificationMetrics,
    LabelEntry,
    MatchResult,
    RankingMetrics,
    TimingStats,
)


def calculate_classification_metrics(
    results: list[MatchResult],
    labels: dict[str, LabelEntry],
) -> ClassificationMetrics:
    """Calculate classification metrics for top-1 predictions."""
    tp = fp = fn = 0

    for result in results:
        if result.error:
            fn += 1
            continue

        label = labels.get(result.picture_id)
        if not label or not label.valid_object_ids:
            continue

        predicted = result.matched_object_id
        valid_set = label.valid_object_ids

        if predicted is None:
            fn += 1
        elif predicted in valid_set:
            tp += 1
        else:
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return ClassificationMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )


def calculate_mrr(
    results: list[MatchResult],
    labels: dict[str, LabelEntry],
) -> float:
    """Calculate Mean Reciprocal Rank."""
    reciprocal_ranks = []

    for result in results:
        label = labels.get(result.picture_id)
        if not label or not label.valid_object_ids:
            continue

        if result.error or not result.ranked_results:
            reciprocal_ranks.append(0.0)
            continue

        found = False
        for rank, item in enumerate(result.ranked_results, start=1):
            if item.object_id in label.valid_object_ids:
                reciprocal_ranks.append(1.0 / rank)
                found = True
                break

        if not found:
            reciprocal_ranks.append(0.0)

    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def calculate_hit_at_k(
    results: list[MatchResult],
    labels: dict[str, LabelEntry],
    k: int,
) -> float:
    """Calculate Hit@K."""
    hits = 0
    total = 0

    for result in results:
        label = labels.get(result.picture_id)
        if not label or not label.valid_object_ids:
            continue

        total += 1

        if result.error:
            continue

        top_k = result.ranked_results[:k]
        for item in top_k:
            if item.object_id in label.valid_object_ids:
                hits += 1
                break

    return hits / total if total > 0 else 0.0


def calculate_ndcg_at_k(
    results: list[MatchResult],
    labels: dict[str, LabelEntry],
    k: int,
) -> float:
    """Calculate NDCG@K."""
    ndcg_scores = []

    for result in results:
        label = labels.get(result.picture_id)
        if not label or not label.valid_object_ids:
            continue

        if result.error or not result.ranked_results:
            ndcg_scores.append(0.0)
            continue

        # DCG@K
        dcg = 0.0
        for i, item in enumerate(result.ranked_results[:k]):
            rel = 1.0 if item.object_id in label.valid_object_ids else 0.0
            dcg += rel / math.log2(i + 2)

        # Ideal DCG
        num_relevant = min(len(label.valid_object_ids), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(num_relevant))

        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def calculate_ranking_metrics(
    results: list[MatchResult],
    labels: dict[str, LabelEntry],
    hit_at_k_values: list[int],
    ndcg_at_k_values: list[int],
) -> RankingMetrics:
    """Calculate all ranking metrics."""
    return RankingMetrics(
        mrr=calculate_mrr(results, labels),
        hit_at_k={k: calculate_hit_at_k(results, labels, k) for k in hit_at_k_values},
        ndcg_at_k={k: calculate_ndcg_at_k(results, labels, k) for k in ndcg_at_k_values},
    )


def calculate_timing_stats(results: list[MatchResult]) -> TimingStats:
    """Calculate latency statistics."""
    times = [r.total_ms for r in results if r.is_successful and r.total_ms > 0]

    if not times:
        return TimingStats(0, 0, 0, 0, 0, 0)

    return TimingStats(
        mean_ms=float(np.mean(times)),
        median_ms=float(np.median(times)),
        p95_ms=float(np.percentile(times, 95)),
        p99_ms=float(np.percentile(times, 99)),
        min_ms=float(np.min(times)),
        max_ms=float(np.max(times)),
    )
