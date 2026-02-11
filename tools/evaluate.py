#!/usr/bin/env python3
"""
Evaluate artwork-matcher accuracy against ground truth labels.

Runs identification on all test pictures in both modes (embedding-only and
with geometric verification) and computes accuracy metrics.

Usage:
    uv run python evaluate.py --testdata ../data/evaluation
    uv run python evaluate.py --testdata ../data/evaluation --output ../reports/evaluation
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

console = Console()


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class LabelEntry:
    """Single ground truth label entry."""

    picture_id: str
    valid_object_ids: frozenset[str]

    @classmethod
    def from_csv_row(cls, picture_path: str, painting_paths: str) -> LabelEntry:
        """Create from CSV row values."""
        picture_id = Path(picture_path.strip()).stem
        object_ids = frozenset(
            Path(p.strip()).stem for p in painting_paths.split(";") if p.strip()
        )
        return cls(picture_id=picture_id, valid_object_ids=object_ids)


@dataclass
class MatchResult:
    """Result from a single identification request."""

    picture_id: str
    mode: str  # "embedding_only" or "geometric"

    # Match info
    matched_object_id: str | None = None
    similarity_score: float | None = None
    geometric_score: float | None = None
    confidence: float | None = None

    # Full ranked results
    ranked_results: list[dict] = field(default_factory=list)

    # Timing
    embedding_ms: float = 0.0
    search_ms: float = 0.0
    geometric_ms: float = 0.0
    total_ms: float = 0.0

    # Error
    error: str | None = None
    error_message: str | None = None

    @property
    def is_successful(self) -> bool:
        """Returns True if the request succeeded."""
        return self.error is None


@dataclass
class ClassificationMetrics:
    """Classification metrics for top-1 match."""

    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass
class RankingMetrics:
    """Ranking quality metrics."""

    mrr: float
    hit_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]


@dataclass
class TimingStats:
    """Latency statistics."""

    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


# =============================================================================
# Data Loading
# =============================================================================


def load_labels(labels_path: Path) -> dict[str, LabelEntry]:
    """Load labels.csv into a dictionary keyed by picture_id."""
    df = pd.read_csv(labels_path, skipinitialspace=True)
    picture_col = df.columns[0]
    painting_col = df.columns[1]

    labels = {}
    for _, row in df.iterrows():
        entry = LabelEntry.from_csv_row(
            str(row[picture_col]), str(row[painting_col])
        )
        labels[entry.picture_id] = entry

    return labels


# =============================================================================
# API Client
# =============================================================================


def check_gateway_health(client: httpx.Client, gateway_url: str) -> dict:
    """Check gateway health status."""
    response = client.get(
        f"{gateway_url}/health", params={"check_backends": "true"}
    )
    response.raise_for_status()
    data = response.json()
    status = data.get("status")
    if status is None:
        msg = "Gateway health response missing 'status'"
        raise ValueError(msg)
    return data


def identify_image(
    client: httpx.Client,
    gateway_url: str,
    image_path: Path,
    geometric_verification: bool,
    k: int,
    threshold: float,
) -> MatchResult:
    """Call the /identify endpoint."""
    picture_id = image_path.stem
    mode = "geometric" if geometric_verification else "embedding_only"

    try:
        image_bytes = image_path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
    except OSError as e:
        return MatchResult(
            picture_id=picture_id,
            mode=mode,
            error="file_error",
            error_message=str(e),
        )

    try:
        response = client.post(
            f"{gateway_url}/identify",
            json={
                "image": image_b64,
                "options": {
                    "k": k,
                    "threshold": threshold,
                    "geometric_verification": geometric_verification,
                    "include_alternatives": True,
                },
            },
        )
        response.raise_for_status()
        data = response.json()

        timing = data.get("timing", {})
        match = data.get("match")
        alternatives = data.get("alternatives", [])

        ranked = []
        if match:
            ranked.append(match)
        ranked.extend(alternatives)

        return MatchResult(
            picture_id=picture_id,
            mode=mode,
            matched_object_id=match["object_id"] if match else None,
            similarity_score=match.get("similarity_score") if match else None,
            geometric_score=match.get("geometric_score") if match else None,
            confidence=match.get("confidence") if match else None,
            ranked_results=ranked,
            embedding_ms=timing.get("embedding_ms", 0),
            search_ms=timing.get("search_ms", 0),
            geometric_ms=timing.get("geometric_ms", 0),
            total_ms=timing.get("total_ms", 0),
        )

    except httpx.HTTPStatusError as e:
        error_data = {}
        try:
            error_data = e.response.json()
        except (JSONDecodeError, ValueError, TypeError):
            error_data = {}

        return MatchResult(
            picture_id=picture_id,
            mode=mode,
            error=error_data.get("error", "http_error"),
            error_message=error_data.get("message", str(e)),
        )

    except httpx.RequestError as e:
        return MatchResult(
            picture_id=picture_id,
            mode=mode,
            error="request_error",
            error_message=str(e),
        )
    except (JSONDecodeError, ValueError, TypeError) as e:
        return MatchResult(
            picture_id=picture_id,
            mode=mode,
            error="invalid_response",
            error_message=str(e),
        )


# =============================================================================
# Metrics Calculation
# =============================================================================


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
            if item.get("object_id") in label.valid_object_ids:
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
            if item.get("object_id") in label.valid_object_ids:
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
            rel = 1.0 if item.get("object_id") in label.valid_object_ids else 0.0
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


# =============================================================================
# Report Generation
# =============================================================================


def generate_report(
    embed_class: ClassificationMetrics,
    geo_class: ClassificationMetrics,
    embed_rank: RankingMetrics,
    geo_rank: RankingMetrics,
    embed_timing: TimingStats,
    geo_timing: TimingStats,
    embed_results: list[MatchResult],
    geo_results: list[MatchResult],
    labels: dict[str, LabelEntry],
    testdata_path: str,
    num_objects: int,
    hit_at_k_values: list[int],
    ndcg_at_k_values: list[int],
) -> str:
    """Generate markdown evaluation report."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    num_pictures = len(embed_results)

    # Analyze per-image results
    per_image_rows = []
    embed_failures = []
    geo_failures = []
    corrections = []

    for embed_r, geo_r in zip(embed_results, geo_results):
        label = labels.get(embed_r.picture_id)
        if not label:
            continue

        valid = label.valid_object_ids
        expected = ";".join(sorted(valid))

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

        embed_str = (
            f"{embed_r.matched_object_id} ({embed_r.similarity_score:.2f})"
            if embed_r.matched_object_id and embed_r.similarity_score
            else embed_r.matched_object_id or "None"
        )
        geo_str = (
            f"{geo_r.matched_object_id} ({geo_r.confidence:.2f})"
            if geo_r.matched_object_id and geo_r.confidence
            else geo_r.matched_object_id or "None"
        )

        corrected = "✓" if not embed_correct and geo_correct else "-"

        per_image_rows.append(
            f"| {embed_r.picture_id} | {expected} | {embed_str} | {geo_str} | {corrected} |"
        )

        if not embed_correct and not embed_r.error:
            embed_failures.append(
                f"| {embed_r.picture_id} | {expected} | {embed_r.matched_object_id or 'None'} | {embed_r.similarity_score or 0:.3f} |"
            )

        if not geo_correct and not geo_r.error:
            geo_failures.append(
                f"| {geo_r.picture_id} | {expected} | {geo_r.matched_object_id or 'None'} | {geo_r.confidence or 0:.2f} |"
            )

        if not embed_correct and geo_correct:
            corrections.append(
                f"| {embed_r.picture_id} | {embed_r.matched_object_id or 'None'} | {geo_r.matched_object_id} | {expected} |"
            )

    # Count multi-match labels
    multi_match_count = sum(1 for l in labels.values() if len(l.valid_object_ids) > 1)
    error_count = sum(1 for r in embed_results + geo_results if r.error)

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
    lines.extend(per_image_rows)

    lines.extend([
        "",
        "---",
        "",
        "## Failures Analysis",
        "",
        "### Embedding-Only Mode Failures",
        "",
    ])

    if embed_failures:
        lines.append("| Picture | Expected | Got | Score |")
        lines.append("|---------|----------|-----|-------|")
        lines.extend(embed_failures)
    else:
        lines.append("*No failures in embedding-only mode.*")

    lines.extend([
        "",
        "### Geometric Mode Failures",
        "",
    ])

    if geo_failures:
        lines.append("| Picture | Expected | Got | Confidence |")
        lines.append("|---------|----------|-----|------------|")
        lines.extend(geo_failures)
    else:
        lines.append("*No failures in geometric mode.*")

    lines.extend([
        "",
        "### Corrected by Geometric Verification",
        "",
    ])

    if corrections:
        lines.append("| Picture | Wrong (Embedding) | Correct (Geometric) | Expected |")
        lines.append("|---------|-------------------|---------------------|----------|")
        lines.extend(corrections)
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
        f"| Pictures with multiple valid matches | {multi_match_count} |",
        f"| Errors during evaluation | {error_count} |",
        "",
        "---",
        "",
        "*Report generated by tools/evaluate.py*",
        "",
    ])

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate artwork-matcher accuracy"
    )
    parser.add_argument(
        "--testdata",
        type=Path,
        required=True,
        help="Path to test data directory containing objects/, pictures/, labels.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output directory for reports",
    )
    parser.add_argument(
        "--gateway-url",
        type=str,
        required=True,
        help="Gateway service URL",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of candidates to retrieve",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Similarity threshold passed to gateway /identify options",
    )
    return parser.parse_args()


def main() -> int:
    """Main evaluation entry point."""
    args = parse_args()

    console.print()
    console.print("[bold blue]" + "=" * 60)
    console.print("[bold blue]       Artwork Matcher Evaluation")
    console.print("[bold blue]" + "=" * 60)
    console.print()

    # Validate paths
    labels_path = args.testdata / "labels.csv"
    pictures_path = args.testdata / "pictures"
    objects_path = args.testdata / "objects"

    # Step 1: Load labels
    console.print("[dim]Step 1: Loading labels...[/dim]")

    if not labels_path.exists():
        console.print(f"[red]Error: Labels file not found: {labels_path}[/red]")
        return 1

    labels = load_labels(labels_path)
    console.print(f"[green]Loaded {len(labels)} labels[/green]")

    # Step 2: Find pictures
    console.print()
    console.print("[dim]Step 2: Finding pictures...[/dim]")

    if not pictures_path.exists():
        console.print(f"[red]Error: Pictures directory not found: {pictures_path}[/red]")
        return 1

    picture_files = sorted(
        list(pictures_path.glob("*.jpg"))
        + list(pictures_path.glob("*.jpeg"))
        + list(pictures_path.glob("*.png"))
    )

    if not picture_files:
        console.print(f"[red]Error: No pictures found in {pictures_path}[/red]")
        return 1

    console.print(f"[green]Found {len(picture_files)} pictures to evaluate[/green]")

    # Count objects
    num_objects = (
        len(list(objects_path.glob("*.jpg")))
        + len(list(objects_path.glob("*.jpeg")))
        + len(list(objects_path.glob("*.png")))
        if objects_path.exists()
        else 0
    )
    console.print(f"[green]Objects in index: {num_objects}[/green]")

    # Create HTTP client
    client = httpx.Client(timeout=120.0)

    try:
        # Step 3: Check gateway health
        console.print()
        console.print("[dim]Step 3: Checking gateway health...[/dim]")

        try:
            health = check_gateway_health(client, args.gateway_url)
        except (httpx.RequestError, httpx.HTTPStatusError, JSONDecodeError, ValueError, TypeError) as e:
            console.print(f"[red]Error: Gateway health check failed: {e}[/red]")
            return 1

        if health.get("status") not in ("healthy", "degraded"):
            console.print(f"[red]Error: Gateway unhealthy: {health}[/red]")
            return 1

        console.print(f"[green]✓ Gateway healthy[/green]")

        # Step 4: Phase 1 - Embedding only
        console.print()
        console.print("[bold cyan]Step 4: Phase 1 - Embedding Only Evaluation[/bold cyan]")

        embed_results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Evaluating (embedding only)", total=len(picture_files))

            for pic_path in picture_files:
                result = identify_image(
                    client, args.gateway_url, pic_path,
                    geometric_verification=False, k=args.k, threshold=args.threshold
                )
                embed_results.append(result)
                progress.advance(task)

        # Step 5: Phase 2 - With geometric
        console.print()
        console.print("[bold cyan]Step 5: Phase 2 - Geometric Verification Evaluation[/bold cyan]")

        geo_results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Evaluating (with geometric)", total=len(picture_files))

            for pic_path in picture_files:
                result = identify_image(
                    client, args.gateway_url, pic_path,
                    geometric_verification=True, k=args.k, threshold=args.threshold
                )
                geo_results.append(result)
                progress.advance(task)

    finally:
        client.close()

    # Step 6: Compute metrics
    console.print()
    console.print("[dim]Step 6: Computing metrics...[/dim]")

    hit_at_k_values = [1, 3, 5, 10]
    ndcg_at_k_values = [5, 10]

    embed_class = calculate_classification_metrics(embed_results, labels)
    geo_class = calculate_classification_metrics(geo_results, labels)

    embed_rank = calculate_ranking_metrics(embed_results, labels, hit_at_k_values, ndcg_at_k_values)
    geo_rank = calculate_ranking_metrics(geo_results, labels, hit_at_k_values, ndcg_at_k_values)

    embed_timing = calculate_timing_stats(embed_results)
    geo_timing = calculate_timing_stats(geo_results)

    console.print("[green]✓ Metrics computed[/green]")

    # Display summary table
    console.print()
    table = Table(title="Evaluation Results Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Embedding Only", style="yellow")
    table.add_column("With Geometric", style="green")
    table.add_column("Delta", style="magenta")

    table.add_row(
        "Precision@1",
        f"{embed_class.precision:.3f}",
        f"{geo_class.precision:.3f}",
        f"{geo_class.precision - embed_class.precision:+.3f}",
    )
    table.add_row(
        "Recall@1",
        f"{embed_class.recall:.3f}",
        f"{geo_class.recall:.3f}",
        f"{geo_class.recall - embed_class.recall:+.3f}",
    )
    table.add_row(
        "F1-Score",
        f"{embed_class.f1_score:.3f}",
        f"{geo_class.f1_score:.3f}",
        f"{geo_class.f1_score - embed_class.f1_score:+.3f}",
    )
    table.add_row(
        "MRR",
        f"{embed_rank.mrr:.3f}",
        f"{geo_rank.mrr:.3f}",
        f"{geo_rank.mrr - embed_rank.mrr:+.3f}",
    )
    table.add_row(
        "Mean Latency",
        f"{embed_timing.mean_ms:.0f}ms",
        f"{geo_timing.mean_ms:.0f}ms",
        f"+{geo_timing.mean_ms - embed_timing.mean_ms:.0f}ms",
    )

    console.print(table)

    # Step 7: Generate report
    console.print()
    console.print("[dim]Step 7: Generating report...[/dim]")

    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "evaluation_report.md"

    report = generate_report(
        embed_class=embed_class,
        geo_class=geo_class,
        embed_rank=embed_rank,
        geo_rank=geo_rank,
        embed_timing=embed_timing,
        geo_timing=geo_timing,
        embed_results=embed_results,
        geo_results=geo_results,
        labels=labels,
        testdata_path=str(args.testdata),
        num_objects=num_objects,
        hit_at_k_values=hit_at_k_values,
        ndcg_at_k_values=ndcg_at_k_values,
    )

    report_path.write_text(report)
    console.print(f"[green]✓ Report saved to: {report_path}[/green]")

    # Save JSON results
    results_path = args.output / "evaluation_results.json"
    results_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "testdata_path": str(args.testdata),
        "num_pictures": len(picture_files),
        "num_objects": num_objects,
        "config": {
            "search_k": args.k,
            "threshold": args.threshold,
            "gateway_url": args.gateway_url,
        },
        "metrics": {
            "embedding_only": {
                "classification": {
                    "precision": embed_class.precision,
                    "recall": embed_class.recall,
                    "f1_score": embed_class.f1_score,
                    "true_positives": embed_class.true_positives,
                    "false_positives": embed_class.false_positives,
                    "false_negatives": embed_class.false_negatives,
                },
                "ranking": {
                    "mrr": embed_rank.mrr,
                    "hit_at_k": embed_rank.hit_at_k,
                    "ndcg_at_k": embed_rank.ndcg_at_k,
                },
                "timing": {
                    "mean_ms": embed_timing.mean_ms,
                    "median_ms": embed_timing.median_ms,
                    "p95_ms": embed_timing.p95_ms,
                    "p99_ms": embed_timing.p99_ms,
                },
            },
            "geometric": {
                "classification": {
                    "precision": geo_class.precision,
                    "recall": geo_class.recall,
                    "f1_score": geo_class.f1_score,
                    "true_positives": geo_class.true_positives,
                    "false_positives": geo_class.false_positives,
                    "false_negatives": geo_class.false_negatives,
                },
                "ranking": {
                    "mrr": geo_rank.mrr,
                    "hit_at_k": geo_rank.hit_at_k,
                    "ndcg_at_k": geo_rank.ndcg_at_k,
                },
                "timing": {
                    "mean_ms": geo_timing.mean_ms,
                    "median_ms": geo_timing.median_ms,
                    "p95_ms": geo_timing.p95_ms,
                    "p99_ms": geo_timing.p99_ms,
                },
            },
        },
    }

    results_path.write_text(json.dumps(results_data, indent=2))
    console.print(f"[green]✓ Results JSON saved to: {results_path}[/green]")

    embed_errors = sum(1 for r in embed_results if r.error is not None)
    geo_errors = sum(1 for r in geo_results if r.error is not None)
    total_errors = embed_errors + geo_errors
    if total_errors > 0:
        console.print(
            f"[yellow]Evaluation completed with request errors: "
            f"embedding={embed_errors}, geometric={geo_errors}[/yellow]"
        )

    # Step 8: Summary
    console.print()
    console.print("[bold blue]" + "=" * 60)
    console.print("[bold green]       Evaluation Complete!")
    console.print("[bold blue]" + "=" * 60)
    console.print()

    return 1 if total_errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
