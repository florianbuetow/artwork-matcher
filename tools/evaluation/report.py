"""Report analysis and rendering for evaluation tooling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from evaluation.models import (
    ClassificationMetrics,
    LabelEntry,
    MatchResult,
    RankingMetrics,
    TimingStats,
)


@dataclass(frozen=True)
class PerImageResult:
    """Structured per-image comparison across evaluation modes."""

    picture_id: str
    expected_ids: tuple[str, ...]
    embed_object_id: str | None
    embed_similarity: float | None
    geo_object_id: str | None
    geo_confidence: float | None
    embed_correct: bool
    geo_correct: bool
    corrected: bool
    embed_error: str | None
    geo_error: str | None


@dataclass(frozen=True)
class ReportAnalysis:
    """Precomputed analysis used by report rendering."""

    per_image_results: list[PerImageResult]
    embed_failures: list[PerImageResult]
    geo_failures: list[PerImageResult]
    corrections: list[PerImageResult]
    multi_match_count: int
    error_count: int


def analyze_per_image_results(
    embed_results: list[MatchResult],
    geo_results: list[MatchResult],
    labels: dict[str, LabelEntry],
) -> ReportAnalysis:
    """Build reusable per-image analysis independent of markdown rendering."""
    per_image_results: list[PerImageResult] = []
    embed_failures: list[PerImageResult] = []
    geo_failures: list[PerImageResult] = []
    corrections: list[PerImageResult] = []

    for embed_r, geo_r in zip(embed_results, geo_results):
        label = labels.get(embed_r.picture_id)
        if not label:
            continue

        expected_ids = tuple(sorted(label.valid_object_ids))
        valid = label.valid_object_ids

        embed_correct = (
            embed_r.matched_object_id in valid
            if embed_r.matched_object_id and valid
            else False
        )
        geo_correct = (
            geo_r.matched_object_id in valid
            if geo_r.matched_object_id and valid
            else False
        )

        result = PerImageResult(
            picture_id=embed_r.picture_id,
            expected_ids=expected_ids,
            embed_object_id=embed_r.matched_object_id,
            embed_similarity=embed_r.similarity_score,
            geo_object_id=geo_r.matched_object_id,
            geo_confidence=geo_r.confidence,
            embed_correct=embed_correct,
            geo_correct=geo_correct,
            corrected=(not embed_correct and geo_correct),
            embed_error=embed_r.error,
            geo_error=geo_r.error,
        )
        per_image_results.append(result)

        if not result.embed_correct and result.embed_error is None:
            embed_failures.append(result)
        if not result.geo_correct and result.geo_error is None:
            geo_failures.append(result)
        if result.corrected:
            corrections.append(result)

    multi_match_count = sum(1 for l in labels.values() if len(l.valid_object_ids) > 1)
    error_count = sum(1 for r in embed_results + geo_results if r.error)

    return ReportAnalysis(
        per_image_results=per_image_results,
        embed_failures=embed_failures,
        geo_failures=geo_failures,
        corrections=corrections,
        multi_match_count=multi_match_count,
        error_count=error_count,
    )


def _format_embed_result(item: PerImageResult) -> str:
    if item.embed_object_id is not None and item.embed_similarity is not None:
        return f"{item.embed_object_id} ({item.embed_similarity:.2f})"
    return item.embed_object_id or "None"


def _format_geo_result(item: PerImageResult) -> str:
    if item.geo_object_id is not None and item.geo_confidence is not None:
        return f"{item.geo_object_id} ({item.geo_confidence:.2f})"
    return item.geo_object_id or "None"


def generate_report(
    embed_class: ClassificationMetrics,
    geo_class: ClassificationMetrics,
    embed_rank: RankingMetrics,
    geo_rank: RankingMetrics,
    embed_timing: TimingStats,
    geo_timing: TimingStats,
    report_analysis: ReportAnalysis,
    testdata_path: str,
    num_objects: int,
    hit_at_k_values: list[int],
    ndcg_at_k_values: list[int],
) -> str:
    """Generate markdown evaluation report."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    num_pictures = len(report_analysis.per_image_results)

    # Build report
    lines = [
        "# Artwork Matcher Evaluation Report",
        "",
        f"**Generated**: {timestamp}",
        f"**Test Dataset**: {testdata_path}",
        f"**Pictures Evaluated**: {num_pictures}",
        f"**Objects in Index**: {num_objects}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "| Mode | Precision | Recall | F1 | MRR | Mean Latency |",
        "|------|-----------|--------|-----|-----|--------------|",
        f"| Embedding Only | {embed_class.precision*100:.1f}% | {embed_class.recall*100:.1f}% | {embed_class.f1_score:.3f} | {embed_rank.mrr:.3f} | {embed_timing.mean_ms:.0f}ms |",
        f"| With Geometric | {geo_class.precision*100:.1f}% | {geo_class.recall*100:.1f}% | {geo_class.f1_score:.3f} | {geo_rank.mrr:.3f} | {geo_timing.mean_ms:.0f}ms |",
        f"| **Delta** | **{(geo_class.precision - embed_class.precision)*100:+.1f}%** | **{(geo_class.recall - embed_class.recall)*100:+.1f}%** | **{geo_class.f1_score - embed_class.f1_score:+.3f}** | **{geo_rank.mrr - embed_rank.mrr:+.3f}** | **+{geo_timing.mean_ms - embed_timing.mean_ms:.0f}ms** |",
        "",
        "---",
        "",
        "## Classification Metrics (Top-1)",
        "",
        "| Metric | Embedding Only | With Geometric | Delta |",
        "|--------|----------------|----------------|-------|",
        f"| Precision | {embed_class.precision:.3f} | {geo_class.precision:.3f} | {geo_class.precision - embed_class.precision:+.3f} |",
        f"| Recall | {embed_class.recall:.3f} | {geo_class.recall:.3f} | {geo_class.recall - embed_class.recall:+.3f} |",
        f"| F1-Score | {embed_class.f1_score:.3f} | {geo_class.f1_score:.3f} | {geo_class.f1_score - embed_class.f1_score:+.3f} |",
        f"| True Positives | {embed_class.true_positives} | {geo_class.true_positives} | {geo_class.true_positives - embed_class.true_positives:+d} |",
        f"| False Positives | {embed_class.false_positives} | {geo_class.false_positives} | {geo_class.false_positives - embed_class.false_positives:+d} |",
        f"| False Negatives | {embed_class.false_negatives} | {geo_class.false_negatives} | {geo_class.false_negatives - embed_class.false_negatives:+d} |",
        "",
        "---",
        "",
        "## Ranking Metrics",
        "",
        "| Metric | Embedding Only | With Geometric | Delta |",
        "|--------|----------------|----------------|-------|",
        f"| MRR | {embed_rank.mrr:.3f} | {geo_rank.mrr:.3f} | {geo_rank.mrr - embed_rank.mrr:+.3f} |",
    ]

    for k in hit_at_k_values:
        embed_hit = embed_rank.hit_at_k[k]
        geo_hit = geo_rank.hit_at_k[k]
        lines.append(f"| Hit@{k} | {embed_hit:.3f} | {geo_hit:.3f} | {geo_hit - embed_hit:+.3f} |")

    for k in ndcg_at_k_values:
        embed_ndcg = embed_rank.ndcg_at_k[k]
        geo_ndcg = geo_rank.ndcg_at_k[k]
        lines.append(f"| NDCG@{k} | {embed_ndcg:.3f} | {geo_ndcg:.3f} | {geo_ndcg - embed_ndcg:+.3f} |")

    lines.extend([
        "",
        "---",
        "",
        "## Latency Analysis",
        "",
        "| Metric | Embedding Only | With Geometric | Overhead |",
        "|--------|----------------|----------------|----------|",
        f"| Mean | {embed_timing.mean_ms:.0f}ms | {geo_timing.mean_ms:.0f}ms | +{geo_timing.mean_ms - embed_timing.mean_ms:.0f}ms |",
        f"| Median | {embed_timing.median_ms:.0f}ms | {geo_timing.median_ms:.0f}ms | +{geo_timing.median_ms - embed_timing.median_ms:.0f}ms |",
        f"| P95 | {embed_timing.p95_ms:.0f}ms | {geo_timing.p95_ms:.0f}ms | +{geo_timing.p95_ms - embed_timing.p95_ms:.0f}ms |",
        f"| P99 | {embed_timing.p99_ms:.0f}ms | {geo_timing.p99_ms:.0f}ms | +{geo_timing.p99_ms - embed_timing.p99_ms:.0f}ms |",
        "",
        "---",
        "",
        "## Per-Image Results",
        "",
        "| Picture | Expected | Embedding Result | Geo Result | Corrected? |",
        "|---------|----------|------------------|------------|------------|",
    ])

    for item in report_analysis.per_image_results:
        expected = ";".join(item.expected_ids)
        corrected = "✓" if item.corrected else "-"
        lines.append(
            f"| {item.picture_id} | {expected} | {_format_embed_result(item)} | {_format_geo_result(item)} | {corrected} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Failures Analysis",
        "",
        "### Embedding-Only Mode Failures",
        "",
    ])

    if report_analysis.embed_failures:
        lines.append("| Picture | Expected | Got | Score |")
        lines.append("|---------|----------|-----|-------|")
        for item in report_analysis.embed_failures:
            expected = ";".join(item.expected_ids)
            got = item.embed_object_id or "None"
            score = item.embed_similarity or 0.0
            lines.append(f"| {item.picture_id} | {expected} | {got} | {score:.3f} |")
    else:
        lines.append("*No failures in embedding-only mode.*")

    lines.extend([
        "",
        "### Geometric Mode Failures",
        "",
    ])

    if report_analysis.geo_failures:
        lines.append("| Picture | Expected | Got | Confidence |")
        lines.append("|---------|----------|-----|------------|")
        for item in report_analysis.geo_failures:
            expected = ";".join(item.expected_ids)
            got = item.geo_object_id or "None"
            confidence = item.geo_confidence or 0.0
            lines.append(f"| {item.picture_id} | {expected} | {got} | {confidence:.2f} |")
    else:
        lines.append("*No failures in geometric mode.*")

    lines.extend([
        "",
        "### Corrected by Geometric Verification",
        "",
    ])

    if report_analysis.corrections:
        lines.append("| Picture | Wrong (Embedding) | Correct (Geometric) | Expected |")
        lines.append("|---------|-------------------|---------------------|----------|")
        for item in report_analysis.corrections:
            expected = ";".join(item.expected_ids)
            lines.append(
                f"| {item.picture_id} | {item.embed_object_id or 'None'} | {item.geo_object_id} | {expected} |"
            )
    else:
        lines.append("*No corrections made by geometric verification.*")

    lines.extend([
        "",
        "---",
        "",
        "## Summary Statistics",
        "",
        "| Category | Count |",
        "|----------|-------|",
        f"| Total pictures evaluated | {num_pictures} |",
        f"| Objects in index | {num_objects} |",
        f"| Pictures with multiple valid matches | {report_analysis.multi_match_count} |",
        f"| Errors during evaluation | {report_analysis.error_count} |",
        "",
        "---",
        "",
        "*Report generated by tools/evaluate.py*",
        "",
    ])

    return "\n".join(lines)

