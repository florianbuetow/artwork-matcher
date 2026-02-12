#!/usr/bin/env python3
"""
Build FAISS index from object images.

Reads images from the objects directory, extracts embeddings via the
Embeddings service, and adds them to the Search service index.

Usage:
    uv run python build_index.py --objects ../data/evaluation/objects --embeddings-url http://localhost:8001 --search-url http://localhost:8002
"""
from __future__ import annotations

import argparse
import base64
import sys
from json import JSONDecodeError
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build FAISS index from object images"
    )
    parser.add_argument(
        "--objects",
        type=Path,
        required=True,
        help="Path to objects directory containing images to index",
    )
    parser.add_argument(
        "--embeddings-url",
        type=str,
        required=True,
        help="Embeddings service URL",
    )
    parser.add_argument(
        "--search-url",
        type=str,
        required=True,
        help="Search service URL",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if index exists",
    )
    return parser.parse_args()


def main() -> int:
    """Main index building entry point."""
    args = parse_args()

    console.print()
    console.print("[bold blue]" + "=" * 60)
    console.print("[bold blue]       Build FAISS Index")
    console.print("[bold blue]" + "=" * 60)
    console.print()

    # Step 1: Validate objects directory
    console.print("[dim]Step 1: Validating objects directory...[/dim]")

    if not args.objects.exists():
        console.print(
            f"[red]Error: Objects directory not found: {args.objects}[/red]"
        )
        return 1

    if not args.objects.is_dir():
        console.print(
            f"[red]Error: Path is not a directory: {args.objects}[/red]"
        )
        return 1

    # Find object images
    object_files = sorted(
        list(args.objects.glob("*.jpg"))
        + list(args.objects.glob("*.jpeg"))
        + list(args.objects.glob("*.png"))
    )

    if not object_files:
        console.print(
            f"[red]Error: No image files found in {args.objects}[/red]"
        )
        return 1

    console.print(f"[green]Found {len(object_files)} objects to index[/green]")

    # Create HTTP client with timeout
    client = httpx.Client(timeout=60.0)

    try:
        # Step 2: Check service health
        console.print()
        console.print("[dim]Step 2: Checking service health...[/dim]")

        try:
            embed_response = client.get(f"{args.embeddings_url}/health")
            embed_response.raise_for_status()
            embed_health = embed_response.json()

            if embed_health.get("status") != "healthy":
                console.print(
                    f"[yellow]Warning: Embeddings service status: "
                    f"{embed_health.get('status', 'unknown')}[/yellow]"
                )
            else:
                console.print(
                    f"[green]✓ Embeddings service healthy[/green]"
                )

        except (
            httpx.RequestError,
            httpx.HTTPStatusError,
            JSONDecodeError,
            ValueError,
            TypeError,
        ) as e:
            console.print(
                f"[red]Error: Cannot connect to embeddings service at "
                f"{args.embeddings_url} ({e})[/red]"
            )
            console.print(
                "[yellow]Make sure services are running: just docker-up[/yellow]"
            )
            return 1

        try:
            search_response = client.get(f"{args.search_url}/health")
            search_response.raise_for_status()
            search_health = search_response.json()

            if search_health.get("status") != "healthy":
                console.print(
                    f"[yellow]Warning: Search service status: "
                    f"{search_health.get('status', 'unknown')}[/yellow]"
                )
            else:
                console.print(f"[green]✓ Search service healthy[/green]")

        except (
            httpx.RequestError,
            httpx.HTTPStatusError,
            JSONDecodeError,
            ValueError,
            TypeError,
        ) as e:
            console.print(
                f"[red]Error: Cannot connect to search service at "
                f"{args.search_url} ({e})[/red]"
            )
            console.print(
                "[yellow]Make sure services are running: just docker-up[/yellow]"
            )
            return 1

        # Step 3: Decide whether to rebuild
        console.print()
        console.print("[dim]Step 3: Checking current index state...[/dim]")

        try:
            info_response = client.get(f"{args.search_url}/info")
            info_response.raise_for_status()
            info_data = info_response.json()
            index_data = info_data.get("index", {})
            index_count = int(index_data.get("count", 0))
        except (
            httpx.RequestError,
            httpx.HTTPStatusError,
            JSONDecodeError,
            ValueError,
            TypeError,
        ) as e:
            console.print(f"[red]Error: Failed to query current index state: {e}[/red]")
            return 1

        if index_count > 0 and not args.force:
            console.print(
                f"[green]✓ Existing index detected with {index_count} items. "
                "[/green]"
                "[green]Skipping rebuild (use --force to rebuild).[/green]"
            )
            return 0

        if index_count > 0 or args.force:
            console.print("[dim]Clearing existing index...[/dim]")
            try:
                response = client.delete(f"{args.search_url}/index")
                response.raise_for_status()
                result = response.json()
                previous_count = result.get("previous_count", 0)
                console.print(
                    f"[green]✓ Cleared index (had {previous_count} items)[/green]"
                )
            except (
                httpx.RequestError,
                httpx.HTTPStatusError,
                JSONDecodeError,
                ValueError,
                TypeError,
            ) as e:
                console.print(f"[red]Error: Failed to clear existing index: {e}[/red]")
                return 1

        # Step 4: Index objects
        console.print()
        console.print("[dim]Step 4: Indexing objects...[/dim]")

        success_count = 0
        error_count = 0
        errors: list[tuple[str, str]] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Indexing objects", total=len(object_files)
            )

            for image_path in object_files:
                object_id = image_path.stem

                try:
                    # Read and encode image
                    image_bytes = image_path.read_bytes()
                    image_b64 = base64.b64encode(image_bytes).decode("ascii")

                    # Get embedding from embeddings service
                    embed_response = client.post(
                        f"{args.embeddings_url}/embed",
                        json={
                            "image": image_b64,
                            "image_id": object_id,
                        },
                    )
                    embed_response.raise_for_status()
                    embedding_data = embed_response.json()
                    embedding = embedding_data["embedding"]

                    # Add to search index
                    add_response = client.post(
                        f"{args.search_url}/add",
                        json={
                            "object_id": object_id,
                            "embedding": embedding,
                            "metadata": {
                                "name": object_id,
                                "image_path": str(image_path),
                            },
                        },
                    )
                    add_response.raise_for_status()

                    success_count += 1

                except httpx.HTTPStatusError as e:
                    error_msg = str(e)
                    try:
                        error_data = e.response.json()
                        error_msg = error_data.get("message", str(e))
                    except (JSONDecodeError, ValueError, TypeError):
                        error_data = {}
                    errors.append((object_id, error_msg))
                    error_count += 1

                except (OSError, KeyError, ValueError, TypeError) as e:
                    errors.append((object_id, str(e)))
                    error_count += 1

                progress.advance(task)

        # Step 5: Save index
        console.print()
        console.print("[dim]Step 5: Saving index to disk...[/dim]")

        try:
            save_response = client.post(
                f"{args.search_url}/index/save",
                json={},
            )
            save_response.raise_for_status()
            save_result = save_response.json()
            index_count = save_result.get("count", 0)
            index_bytes = save_result.get("size_bytes", 0)
            console.print(
                f"[green]✓ Index saved: {index_count} items, "
                f"{index_bytes:,} bytes[/green]"
            )
        except (
            httpx.RequestError,
            httpx.HTTPStatusError,
            JSONDecodeError,
            ValueError,
            TypeError,
        ) as e:
            console.print(f"[red]Error saving index: {e}[/red]")
            return 1

        # Step 6: Summary
        console.print()
        console.print("[bold blue]" + "=" * 60)

        if error_count == 0:
            console.print(
                f"[bold green]✓ Index built successfully: "
                f"{success_count} objects indexed[/bold green]"
            )
        else:
            console.print(
                f"[bold yellow]Index built with errors: "
                f"{success_count} succeeded, {error_count} failed[/bold yellow]"
            )
            console.print()
            console.print("[red]Errors:[/red]")
            for obj_id, err_msg in errors[:10]:
                console.print(f"  - {obj_id}: {err_msg}")
            if len(errors) > 10:
                console.print(f"  ... and {len(errors) - 10} more errors")

        console.print("[bold blue]" + "=" * 60)
        console.print()

        return 0 if error_count == 0 else 1

    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
