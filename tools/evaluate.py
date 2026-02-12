#!/usr/bin/env python3
"""
Evaluate artwork-matcher accuracy against ground truth labels.

Runs identification on all test pictures in both modes (embedding-only and
with geometric verification) and computes accuracy metrics.

Usage:
    uv run python evaluate.py --testdata ../data/evaluation --output ../reports/evaluation --gateway-url http://localhost:8000 --k 10 --threshold 0.0
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from evaluation.client import check_gateway_health, identify_image
from evaluation.data import count_images, discover_images, load_labels
from evaluation.metrics import (
    calculate_classification_metrics,
    calculate_ranking_metrics,
    calculate_timing_stats,
)
from evaluation.report import analyze_per_image_results, generate_report

if TYPE_CHECKING:
    from evaluation.models import ClassificationMetrics, MatchResult, RankingMetrics, TimingStats

console = Console()


@dataclass(frozen=True)
class EvaluationModeConfig:
    """Configuration for one evaluation mode."""

    name: str
    geometric_verification: bool
    description: str


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


def validate_input_paths(testdata_path: Path) -> tuple[Path, Path, Path]:
    """Validate test data layout and return resolved key paths."""
    labels_path = testdata_path / "labels.csv"
    pictures_path = testdata_path / "pictures"
    objects_path = testdata_path / "objects"

    if not labels_path.exists():
        msg = f"Labels file not found: {labels_path}"
        raise FileNotFoundError(msg)

    if not pictures_path.exists():
        msg = f"Pictures directory not found: {pictures_path}"
        raise FileNotFoundError(msg)

    if not objects_path.exists():
        msg = f"Objects directory not found: {objects_path}"
        raise FileNotFoundError(msg)

    return labels_path, pictures_path, objects_path


def run_evaluation_phase(
    client: httpx.Client,
    gateway_url: str,
    picture_files: list[Path],
    mode: EvaluationModeConfig,
    k: int,
    threshold: float,
) -> list[MatchResult]:
    """Run one evaluation phase for a configured mode."""
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task(mode.description, total=len(picture_files))

        for pic_path in picture_files:
            result = identify_image(
                client=client,
                gateway_url=gateway_url,
                image_path=pic_path,
                geometric_verification=mode.geometric_verification,
                k=k,
                threshold=threshold,
            )
            results.append(result)
            progress.advance(task)

    return results


def render_summary_table(
    embed_class: ClassificationMetrics,
    geo_class: ClassificationMetrics,
    embed_rank: RankingMetrics,
    geo_rank: RankingMetrics,
    embed_timing: TimingStats,
    geo_timing: TimingStats,
) -> None:
    """Render a concise evaluation summary table."""
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


def main() -> int:
    """Main evaluation entry point."""
    args = parse_args()

    console.print()
    console.print("[bold blue]" + "=" * 60)
    console.print("[bold blue]       Artwork Matcher Evaluation")
    console.print("[bold blue]" + "=" * 60)
    console.print()

    # Step 1: Validate paths and load labels
    console.print("[dim]Step 1: Loading labels...[/dim]")
    try:
        labels_path, pictures_path, objects_path = validate_input_paths(args.testdata)
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1

    labels = load_labels(labels_path)
    console.print(f"[green]Loaded {len(labels)} labels[/green]")

    # Step 2: Discover pictures/objects
    console.print()
    console.print("[dim]Step 2: Finding pictures...[/dim]")

    picture_files = discover_images(pictures_path)
    if not picture_files:
        console.print(f"[red]Error: No pictures found in {pictures_path}[/red]")
        return 1

    num_objects = count_images(objects_path)
    console.print(f"[green]Found {len(picture_files)} pictures to evaluate[/green]")
    console.print(f"[green]Objects in index: {num_objects}[/green]")

    # Step 3: Gateway health check
    console.print()
    console.print("[dim]Step 3: Checking gateway health...[/dim]")

    client = httpx.Client(timeout=120.0)
    try:
        try:
            health = check_gateway_health(client, args.gateway_url)
        except (
            httpx.RequestError,
            httpx.HTTPStatusError,
            JSONDecodeError,
            ValueError,
            TypeError,
        ) as e:
            console.print(f"[red]Error: Gateway health check failed: {e}[/red]")
            return 1

        if health.get("status") not in ("healthy", "degraded"):
            console.print(f"[red]Error: Gateway unhealthy: {health}[/red]")
            return 1

        console.print("[green]✓ Gateway healthy[/green]")

        # Step 4/5: Run data-driven evaluation modes
        modes = [
            EvaluationModeConfig(
                name="embedding_only",
                geometric_verification=False,
                description="Evaluating (embedding only)",
            ),
            EvaluationModeConfig(
                name="geometric",
                geometric_verification=True,
                description="Evaluating (with geometric)",
            ),
        ]

        results_by_mode: dict[str, list] = {}
        for idx, mode in enumerate(modes, start=4):
            console.print()
            console.print(
                f"[bold cyan]Step {idx}: Phase {idx-3} - {mode.name.replace('_', ' ').title()} Evaluation[/bold cyan]"
            )
            results_by_mode[mode.name] = run_evaluation_phase(
                client=client,
                gateway_url=args.gateway_url,
                picture_files=picture_files,
                mode=mode,
                k=args.k,
                threshold=args.threshold,
            )
    finally:
        client.close()

    embed_results = results_by_mode["embedding_only"]
    geo_results = results_by_mode["geometric"]

    # Step 6: Compute metrics
    console.print()
    console.print("[dim]Step 6: Computing metrics...[/dim]")

    hit_at_k_values = [1, 3, 5, 10]
    ndcg_at_k_values = [5, 10]

    embed_class = calculate_classification_metrics(embed_results, labels)
    geo_class = calculate_classification_metrics(geo_results, labels)

    embed_rank = calculate_ranking_metrics(
        embed_results,
        labels,
        hit_at_k_values,
        ndcg_at_k_values,
    )
    geo_rank = calculate_ranking_metrics(
        geo_results,
        labels,
        hit_at_k_values,
        ndcg_at_k_values,
    )

    embed_timing = calculate_timing_stats(embed_results)
    geo_timing = calculate_timing_stats(geo_results)

    console.print("[green]✓ Metrics computed[/green]")
    console.print()
    render_summary_table(
        embed_class,
        geo_class,
        embed_rank,
        geo_rank,
        embed_timing,
        geo_timing,
    )

    # Step 7: Analyze and generate report
    console.print()
    console.print("[dim]Step 7: Generating report...[/dim]")

    report_analysis = analyze_per_image_results(embed_results, geo_results, labels)

    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "evaluation_report.md"

    report = generate_report(
        embed_class=embed_class,
        geo_class=geo_class,
        embed_rank=embed_rank,
        geo_rank=geo_rank,
        embed_timing=embed_timing,
        geo_timing=geo_timing,
        report_analysis=report_analysis,
        testdata_path=str(args.testdata),
        num_objects=num_objects,
        hit_at_k_values=hit_at_k_values,
        ndcg_at_k_values=ndcg_at_k_values,
    )

    report_path.write_text(report)
    console.print(f"[green]✓ Report saved to: {report_path}[/green]")

    # Step 8: Save JSON summary
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

    total_errors = report_analysis.error_count
    if total_errors > 0:
        embed_errors = sum(1 for r in embed_results if r.error is not None)
        geo_errors = sum(1 for r in geo_results if r.error is not None)
        console.print(
            "[yellow]Evaluation completed with request errors: "
            f"embedding={embed_errors}, geometric={geo_errors}[/yellow]"
        )

    console.print()
    console.print("[bold blue]" + "=" * 60)
    console.print("[bold green]       Evaluation Complete!")
    console.print("[bold blue]" + "=" * 60)
    console.print()

    return 1 if total_errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
