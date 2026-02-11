#!/usr/bin/env python3
"""
Run full end-to-end evaluation pipeline.

This script orchestrates:
1. Starting Docker services (unless --skip-docker)
2. Building the FAISS index from objects
3. Running evaluation against labels

Usage:
    uv run python run_evaluation.py --testdata ../data/evaluation --output ../reports/evaluation --gateway-url http://localhost:8000 --embeddings-url http://localhost:8001 --search-url http://localhost:8002 --geometric-url http://localhost:8003 --k 10 --threshold 0.0
    uv run python run_evaluation.py --testdata ../data/evaluation --output ../reports/evaluation --gateway-url http://localhost:8000 --embeddings-url http://localhost:8001 --search-url http://localhost:8002 --geometric-url http://localhost:8003 --k 10 --threshold 0.0 --skip-docker
    uv run python run_evaluation.py --testdata ../data/evaluation --output ../reports/evaluation --gateway-url http://localhost:8000 --embeddings-url http://localhost:8001 --search-url http://localhost:8002 --geometric-url http://localhost:8003 --k 10 --threshold 0.0 --skip-index
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import httpx
from rich.console import Console

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run full end-to-end evaluation pipeline"
    )
    parser.add_argument(
        "--testdata",
        type=Path,
        required=True,
        help="Path to test data directory (contains objects/, pictures/, labels.csv)",
    )
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Skip Docker startup (assume services are already running)",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip index building (assume index is already built)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for reports",
    )
    parser.add_argument(
        "--gateway-url",
        type=str,
        required=True,
        help="Gateway base URL",
    )
    parser.add_argument(
        "--embeddings-url",
        type=str,
        required=True,
        help="Embeddings base URL",
    )
    parser.add_argument(
        "--search-url",
        type=str,
        required=True,
        help="Search base URL",
    )
    parser.add_argument(
        "--geometric-url",
        type=str,
        required=True,
        help="Geometric base URL",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of retrieval candidates for evaluation",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Similarity threshold passed to gateway during evaluation",
    )
    return parser.parse_args()


def check_service_health(url: str, timeout: float = 2.0) -> tuple[bool, str]:
    """Check if a service is healthy."""
    try:
        response = httpx.get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        status = data.get("status")
        healthy = status in ("healthy", "degraded")
        detail = f"status={status}" if status is not None else "status missing in response"
        return healthy, detail
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError, TypeError) as e:
        return False, f"{type(e).__name__}: {e}"


def wait_for_services(services: list[tuple[str, str]], max_wait: int = 120) -> bool:
    """Wait for all services to become healthy."""
    console.print("[dim]Waiting for services to become healthy...[/dim]")

    start_time = time.time()
    while time.time() - start_time < max_wait:
        all_healthy = True
        for name, url in services:
            healthy, detail = check_service_health(url)
            if not healthy:
                all_healthy = False
                console.print(f"[dim]  {name} not ready: {detail}[/dim]")
                break

        if all_healthy:
            return True

        time.sleep(2)
        elapsed = int(time.time() - start_time)
        console.print(f"[dim]  Still waiting... ({elapsed}s elapsed)[/dim]")

    return False


def start_docker_services() -> bool:
    """Start Docker services using docker compose."""
    console.print("[dim]Starting Docker services...[/dim]")

    # Get the project root (parent of tools/)
    project_root = Path(__file__).parent.parent

    try:
        result = subprocess.run(
            ["docker", "compose", "up", "-d"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            console.print(f"[red]Error starting Docker:[/red]")
            console.print(result.stderr)
            return False

        console.print("[green]✓ Docker compose started[/green]")
        return True

    except FileNotFoundError:
        console.print("[red]Error: docker command not found[/red]")
        return False


def run_build_index(testdata: Path, embeddings_url: str, search_url: str) -> bool:
    """Run the build_index.py script."""
    console.print()
    console.print("[bold cyan]" + "=" * 60)
    console.print("[bold cyan]       Building Index")
    console.print("[bold cyan]" + "=" * 60)

    objects_path = testdata / "objects"

    result = subprocess.run(
        [
            "uv", "run", "python", "build_index.py",
            "--objects", str(objects_path),
            "--embeddings-url", embeddings_url,
            "--search-url", search_url,
        ],
        cwd=Path(__file__).parent,
    )

    return result.returncode == 0


def run_evaluation(
    testdata: Path,
    output: Path,
    gateway_url: str,
    k: int,
    threshold: float,
) -> bool:
    """Run the evaluate.py script."""
    console.print()
    console.print("[bold cyan]" + "=" * 60)
    console.print("[bold cyan]       Running Evaluation")
    console.print("[bold cyan]" + "=" * 60)

    result = subprocess.run(
        [
            "uv", "run", "python", "evaluate.py",
            "--testdata", str(testdata),
            "--output", str(output),
            "--gateway-url", gateway_url,
            "--k", str(k),
            "--threshold", str(threshold),
        ],
        cwd=Path(__file__).parent,
    )

    return result.returncode == 0


def main() -> int:
    """Main entry point."""
    args = parse_args()
    services = [
        ("Gateway", f"{args.gateway_url}/health"),
        ("Embeddings", f"{args.embeddings_url}/health"),
        ("Search", f"{args.search_url}/health"),
        ("Geometric", f"{args.geometric_url}/health"),
    ]

    console.print()
    console.print("[bold blue]" + "=" * 60)
    console.print("[bold blue]       Artwork Matcher E2E Evaluation")
    console.print("[bold blue]" + "=" * 60)
    console.print()

    # Validate testdata path
    if not args.testdata.exists():
        console.print(f"[red]Error: Test data directory not found: {args.testdata}[/red]")
        return 1

    labels_path = args.testdata / "labels.csv"
    if not labels_path.exists():
        console.print(f"[red]Error: Labels file not found: {labels_path}[/red]")
        return 1

    objects_path = args.testdata / "objects"
    if not objects_path.exists():
        console.print(f"[red]Error: Objects directory not found: {objects_path}[/red]")
        return 1

    pictures_path = args.testdata / "pictures"
    if not pictures_path.exists():
        console.print(f"[red]Error: Pictures directory not found: {pictures_path}[/red]")
        return 1

    console.print(f"[green]✓ Test data validated: {args.testdata}[/green]")

    # Step 1: Start Docker (unless --skip-docker)
    if not args.skip_docker:
        console.print()
        console.print("[bold cyan]Step 1: Starting Docker Services[/bold cyan]")

        if not start_docker_services():
            return 1

        if not wait_for_services(services=services):
            console.print("[red]Error: Services did not become healthy within timeout[/red]")
            console.print("[yellow]Check Docker logs with: just docker-logs[/yellow]")
            return 1

        console.print("[green]✓ All services healthy[/green]")
    else:
        console.print()
        console.print("[bold cyan]Step 1: Checking Services (--skip-docker)[/bold cyan]")

        # Verify services are running
        all_healthy = True
        for name, url in services:
            healthy, detail = check_service_health(url)
            if healthy:
                console.print(f"[green]✓ {name} healthy ({detail})[/green]")
            else:
                console.print(f"[red]✗ {name} not healthy ({detail})[/red]")
                all_healthy = False

        if not all_healthy:
            console.print()
            console.print("[red]Error: Some services are not healthy[/red]")
            console.print("[yellow]Start services with: just docker-up[/yellow]")
            return 1

    # Step 2: Build index (unless --skip-index)
    if not args.skip_index:
        console.print()
        console.print("[bold cyan]Step 2: Building Index[/bold cyan]")

        if not run_build_index(args.testdata, args.embeddings_url, args.search_url):
            console.print("[red]Error: Index building failed[/red]")
            return 1
    else:
        console.print()
        console.print("[bold cyan]Step 2: Skipping Index Build (--skip-index)[/bold cyan]")

    # Step 3: Run evaluation
    console.print()
    console.print("[bold cyan]Step 3: Running Evaluation[/bold cyan]")

    if not run_evaluation(
        args.testdata,
        args.output,
        args.gateway_url,
        args.k,
        args.threshold,
    ):
        console.print("[red]Error: Evaluation failed[/red]")
        return 1

    # Summary
    console.print()
    console.print("[bold blue]" + "=" * 60)
    console.print("[bold green]       E2E Evaluation Complete!")
    console.print("[bold blue]" + "=" * 60)
    console.print()
    console.print(f"[green]Report: {args.output / 'evaluation_report.md'}[/green]")
    console.print(f"[green]Results: {args.output / 'evaluation_results.json'}[/green]")
    console.print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
