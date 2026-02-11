#!/usr/bin/env python3
"""
Verify that the E2E evaluation pipeline is correctly wired up.

Runs a series of checks against running services and evaluation output
to confirm the pipeline produces valid results.

Usage:
    uv run python verify_evaluation.py --testdata ../data/evaluation --reports ../reports/evaluation --gateway-url http://localhost:8000 --embeddings-url http://localhost:8001 --search-url http://localhost:8002 --geometric-url http://localhost:8003 --k 10 --threshold 0.0
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import httpx
import pandas as pd
from rich.console import Console

console = Console()


# =============================================================================
# Check Helpers
# =============================================================================


class CheckRunner:
    """Tracks pass/fail results across all checks."""

    def __init__(self) -> None:
        self.passed: int = 0
        self.failed: int = 0
        self.errors: list[str] = []

    def pass_check(self, name: str, detail: str) -> None:
        self.passed += 1
        console.print(f"  [green]PASS[/green]  {name}: {detail}")

    def fail_check(self, name: str, detail: str) -> None:
        self.failed += 1
        self.errors.append(f"{name}: {detail}")
        console.print(f"  [red]FAIL[/red]  {name}: {detail}")

    @property
    def all_passed(self) -> bool:
        return self.failed == 0


# =============================================================================
# Checks
# =============================================================================


def check_service_health(
    runner: CheckRunner,
    client: httpx.Client,
    gateway_url: str,
    embeddings_url: str,
    search_url: str,
    geometric_url: str,
) -> None:
    """Check 1: All services healthy, gateway sees all backends."""
    console.print("\n[bold]Check 1: Service Health[/bold]")

    services = {
        "embeddings": embeddings_url,
        "search": search_url,
        "geometric": geometric_url,
        "gateway": gateway_url,
    }
    for name, url in services.items():
        try:
            resp = client.get(f"{url}/health")
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "unknown")
            if status == "healthy":
                runner.pass_check(name, f"status={status}")
            else:
                runner.fail_check(name, f"status={status}")
        except (httpx.RequestError, httpx.HTTPStatusError, ValueError, TypeError) as e:
            runner.fail_check(name, f"unreachable ({e})")

    try:
        resp = client.get(f"{gateway_url}/health", params={"check_backends": "true"})
        resp.raise_for_status()
        data = resp.json()
        backends = data.get("backends", {})
        all_healthy = all(v == "healthy" for v in backends.values())
        if all_healthy and len(backends) == 3:
            runner.pass_check("gateway-backends", f"all 3 backends healthy")
        else:
            runner.fail_check("gateway-backends", f"backends={backends}")
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError, TypeError) as e:
        runner.fail_check("gateway-backends", f"check failed ({e})")


def check_index_state(
    runner: CheckRunner,
    client: httpx.Client,
    objects_dir: Path,
    search_url: str,
) -> None:
    """Check 2: Index item count matches objects directory."""
    console.print("\n[bold]Check 2: Index State[/bold]")

    object_files = list(objects_dir.glob("*.jpg")) + list(objects_dir.glob("*.jpeg")) + list(objects_dir.glob("*.png"))
    expected_count = len(object_files)

    try:
        resp = client.get(f"{search_url}/info")
        resp.raise_for_status()
        data = resp.json()
        index_count = data.get("index", {}).get("count", 0)
        if index_count == expected_count:
            runner.pass_check("index-count", f"{index_count} items (matches {expected_count} objects)")
        else:
            runner.fail_check("index-count", f"index has {index_count} items, expected {expected_count}")
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError, TypeError) as e:
        runner.fail_check("index-count", f"could not query search service ({e})")


def check_label_consistency(
    runner: CheckRunner, labels_path: Path, pictures_dir: Path, objects_dir: Path
) -> None:
    """Check 3: Labels reference existing files."""
    console.print("\n[bold]Check 3: Label Consistency[/bold]")

    df = pd.read_csv(labels_path, skipinitialspace=True)
    picture_col = df.columns[0]
    painting_col = df.columns[1]

    picture_files = {p.stem for p in pictures_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    object_files = {o.stem for o in objects_dir.iterdir() if o.suffix.lower() in (".jpg", ".jpeg", ".png")}

    missing_pictures: list[str] = []
    missing_objects: list[str] = []

    for _, row in df.iterrows():
        pic_id = Path(str(row[picture_col]).strip()).stem
        if pic_id not in picture_files:
            missing_pictures.append(pic_id)

        for painting_path in str(row[painting_col]).split(";"):
            obj_id = Path(painting_path.strip()).stem
            if obj_id and obj_id not in object_files:
                missing_objects.append(obj_id)

    if not missing_pictures:
        runner.pass_check("pictures-exist", f"all {len(df)} pictures found in {pictures_dir.name}/")
    else:
        runner.fail_check("pictures-exist", f"missing: {missing_pictures}")

    if not missing_objects:
        runner.pass_check("objects-exist", f"all referenced objects found in {objects_dir.name}/")
    else:
        runner.fail_check("objects-exist", f"missing: {missing_objects}")


def check_api_smoke_test(
    runner: CheckRunner,
    client: httpx.Client,
    labels_path: Path,
    pictures_dir: Path,
    gateway_url: str,
    k: int,
    threshold: float,
) -> None:
    """Check 4: API returns expected object for a known picture."""
    console.print("\n[bold]Check 4: API Smoke Test[/bold]")

    df = pd.read_csv(labels_path, skipinitialspace=True)
    picture_col = df.columns[0]
    painting_col = df.columns[1]

    # Pick the first single-match label (simplest case)
    test_row = None
    for _, row in df.iterrows():
        paintings = str(row[painting_col])
        if ";" not in paintings:
            test_row = row
            break

    if test_row is None:
        runner.fail_check("smoke-test", "no single-match label found for testing")
        return

    pic_id = Path(str(test_row[picture_col]).strip()).stem
    expected_obj = Path(str(test_row[painting_col]).strip()).stem

    # Find the picture file
    pic_file = None
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = pictures_dir / f"{pic_id}{ext}"
        if candidate.exists():
            pic_file = candidate
            break

    if pic_file is None:
        runner.fail_check("smoke-test", f"picture file not found for {pic_id}")
        return

    try:
        image_b64 = base64.b64encode(pic_file.read_bytes()).decode("ascii")
        resp = client.post(
            f"{gateway_url}/identify",
            json={
                "image": image_b64,
                "options": {
                    "k": k,
                    "threshold": threshold,
                    "geometric_verification": False,
                    "include_alternatives": True,
                },
            },
        )
        resp.raise_for_status()
        data = resp.json()

        match = data.get("match")
        alternatives = data.get("alternatives", [])
        timing = data.get("timing", {})

        # Check: got a result at all
        if match is None:
            runner.fail_check("smoke-test-result", f"no match returned for {pic_id}")
            return

        runner.pass_check("smoke-test-result", f"got match: {match['object_id']} (confidence={match.get('confidence', 'N/A')})")

        # Check: expected object in ranked results
        ranked_ids = [match["object_id"]] + [a["object_id"] for a in alternatives]
        if expected_obj in ranked_ids:
            rank = ranked_ids.index(expected_obj) + 1
            runner.pass_check("smoke-test-ranking", f"expected '{expected_obj}' found at rank {rank}")
        else:
            runner.fail_check("smoke-test-ranking", f"expected '{expected_obj}' not in top {len(ranked_ids)} results: {ranked_ids}")

        # Check: timing is non-zero
        total_ms = timing.get("total_ms", 0)
        if total_ms > 0:
            runner.pass_check("smoke-test-timing", f"total_ms={total_ms:.0f}ms")
        else:
            runner.fail_check("smoke-test-timing", f"total_ms={total_ms} (expected > 0)")

    except (OSError, httpx.RequestError, httpx.HTTPStatusError, ValueError, TypeError) as e:
        runner.fail_check("smoke-test", f"API call failed: {e}")


def check_metric_sanity(runner: CheckRunner, reports_dir: Path) -> None:
    """Check 5: Evaluation metrics are non-zero and consistent."""
    console.print("\n[bold]Check 5: Metric Sanity[/bold]")

    results_path = reports_dir / "evaluation_results.json"
    if not results_path.exists():
        runner.fail_check("results-file", f"{results_path} not found")
        return

    data = json.loads(results_path.read_text())
    metrics = data.get("metrics", {})

    for mode in ("embedding_only", "geometric"):
        mode_metrics = metrics.get(mode, {})
        classification = mode_metrics.get("classification", {})
        timing = mode_metrics.get("timing", {})

        # Precision/recall/F1 should be > 0
        f1 = classification.get("f1_score", 0)
        if f1 > 0:
            runner.pass_check(f"{mode}-f1", f"F1={f1:.3f}")
        else:
            runner.fail_check(f"{mode}-f1", f"F1={f1} (expected > 0)")

        # Latency should be > 0
        mean_ms = timing.get("mean_ms", 0)
        if mean_ms > 0:
            runner.pass_check(f"{mode}-latency", f"mean={mean_ms:.0f}ms")
        else:
            runner.fail_check(f"{mode}-latency", f"mean_ms={mean_ms} (expected > 0)")

    # Check no errors
    num_pictures = data.get("num_pictures", 0)
    em_fn = metrics.get("embedding_only", {}).get("classification", {}).get("false_negatives", -1)
    em_tp = metrics.get("embedding_only", {}).get("classification", {}).get("true_positives", 0)
    em_fp = metrics.get("embedding_only", {}).get("classification", {}).get("false_positives", 0)
    total_classified = em_tp + em_fp + em_fn
    if total_classified == num_pictures:
        runner.pass_check("classification-count", f"TP+FP+FN={total_classified} matches {num_pictures} pictures")
    else:
        runner.fail_check("classification-count", f"TP+FP+FN={total_classified}, expected {num_pictures}")


def check_report_existence(runner: CheckRunner, reports_dir: Path) -> None:
    """Check 6: Report files exist."""
    console.print("\n[bold]Check 6: Report Files[/bold]")

    for filename in ("evaluation_report.md", "evaluation_results.json"):
        filepath = reports_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            runner.pass_check(filename, f"exists ({size:,} bytes)")
        else:
            runner.fail_check(filename, "file not found")


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify E2E evaluation pipeline")
    parser.add_argument(
        "--testdata",
        type=Path,
        required=True,
        help="Path to test data directory (contains labels.csv, objects/, pictures/)",
    )
    parser.add_argument(
        "--reports",
        type=Path,
        required=True,
        help="Path to evaluation reports directory",
    )
    parser.add_argument(
        "--gateway-url",
        type=str,
        required=True,
        help="Gateway URL",
    )
    parser.add_argument(
        "--embeddings-url",
        type=str,
        required=True,
        help="Embeddings URL",
    )
    parser.add_argument(
        "--search-url",
        type=str,
        required=True,
        help="Search URL",
    )
    parser.add_argument(
        "--geometric-url",
        type=str,
        required=True,
        help="Geometric URL",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Top-K candidates for smoke-test identify call",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Similarity threshold for smoke-test identify call",
    )
    args = parser.parse_args()

    testdata_dir = args.testdata
    reports_dir = args.reports
    gateway_url = args.gateway_url
    embeddings_url = args.embeddings_url
    search_url = args.search_url
    geometric_url = args.geometric_url

    labels_path = testdata_dir / "labels.csv"
    pictures_dir = testdata_dir / "pictures"
    objects_dir = testdata_dir / "objects"

    # Validate directories exist
    for path, desc in [
        (testdata_dir, "testdata directory"),
        (labels_path, "labels.csv"),
        (pictures_dir, "pictures directory"),
        (objects_dir, "objects directory"),
    ]:
        if not path.exists():
            console.print(f"[red]ERROR[/red]: {desc} not found: {path}")
            return 1

    console.print("=" * 60)
    console.print("       [bold]E2E Evaluation Verification[/bold]")
    console.print("=" * 60)

    runner = CheckRunner()
    client = httpx.Client(timeout=30.0)

    try:
        check_service_health(
            runner,
            client,
            gateway_url=gateway_url,
            embeddings_url=embeddings_url,
            search_url=search_url,
            geometric_url=geometric_url,
        )
        check_index_state(runner, client, objects_dir, search_url=search_url)
        check_label_consistency(runner, labels_path, pictures_dir, objects_dir)
        check_api_smoke_test(
            runner,
            client,
            labels_path,
            pictures_dir,
            gateway_url=gateway_url,
            k=args.k,
            threshold=args.threshold,
        )
        check_metric_sanity(runner, reports_dir)
        check_report_existence(runner, reports_dir)
    finally:
        client.close()

    # Summary
    console.print("\n" + "=" * 60)
    total = runner.passed + runner.failed
    if runner.all_passed:
        console.print(f"[bold green]ALL {total} CHECKS PASSED[/bold green]")
    else:
        console.print(f"[bold red]{runner.failed}/{total} CHECKS FAILED[/bold red]")
        for err in runner.errors:
            console.print(f"  [red]- {err}[/red]")
    console.print("=" * 60)

    return 0 if runner.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
