"""
Rijksmuseum Data Downloader.

Downloads collection metadata and images from the Rijksmuseum APIs.

Uses the new Linked Art APIs (no API key required):
    - Search API: https://data.rijksmuseum.nl/search/collection
    - Object Resolver: https://data.rijksmuseum.nl/{id}
    - Images via IIIF: https://iiif.micr.io/{id}/full/max/0/default.jpg

Features:
    - Downloads both metadata (JSON) and images (JPEG)
    - Resume capability (saves pageToken between runs)
    - Progress tracking with rich output
    - Rate limiting to respect API limits
    - Skips objects without downloadable images

Usage:
    # Download 10 objects with images
    uv run python download_data.py --limit 10

    # Force re-download (ignore existing files)
    uv run python download_data.py --force --limit 10

API Documentation:
    - Search API: https://data.rijksmuseum.nl/docs/search
    - Resolver: https://data.rijksmuseum.nl/docs/http
    - IIIF Image: https://data.rijksmuseum.nl/docs/iiif/image
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# API endpoints
SEARCH_API = "https://data.rijksmuseum.nl/search/collection"
DATA_API = "https://data.rijksmuseum.nl"

# Getty AAT term for accession/object numbers
OBJECT_NUMBER_CLASSIFIER = "300312355"

console = Console()


@dataclass
class DownloadConfig:
    """Configuration for the data downloader."""

    download_dir: Path
    rate_limit_delay: float = 0.5
    chunk_size: int = 8192
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 5.0


def load_config(args: argparse.Namespace) -> DownloadConfig:
    """Load configuration from CLI args."""
    if args.download_dir:
        download_dir = Path(args.download_dir)
    else:
        # Default to project root data/downloads (two levels up from this script)
        project_root = Path(__file__).parent.parent.parent
        download_dir = project_root / "data" / "downloads"

    return DownloadConfig(
        download_dir=download_dir,
        rate_limit_delay=0.5,
        chunk_size=8192,
        timeout=30.0,
        max_retries=3,
        retry_delay=5.0,
    )


@dataclass
class DownloadStats:
    """Statistics for the download session."""

    total_requested: int = 0
    downloaded_objects: int = 0
    skipped_existing: int = 0
    skipped_not_downloadable: int = 0
    skipped_no_image: int = 0
    failed: int = 0
    total_bytes: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def duration(self) -> float:
        return time.time() - self.start_time


class RijksmuseumDownloader:
    """Downloads collection data from Rijksmuseum APIs."""

    def __init__(self, config: DownloadConfig) -> None:
        self.config = config
        self._client: httpx.Client | None = None
        self.stats = DownloadStats()

    def __enter__(self) -> RijksmuseumDownloader:
        self._client = httpx.Client(
            timeout=httpx.Timeout(self.config.timeout),
            follow_redirects=True,
            headers={
                "Accept": "application/ld+json",
                "User-Agent": "RijksmuseumDownloader/2.0 (artwork-matcher)",
            },
        )
        return self

    def __exit__(self, *_args: Any) -> None:
        if self._client:
            self._client.close()
            self._client = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            raise RuntimeError("Downloader must be used as context manager")
        return self._client

    def ensure_dirs(self) -> None:
        """Create necessary directories."""
        (self.config.download_dir / "metadata").mkdir(parents=True, exist_ok=True)
        (self.config.download_dir / "images").mkdir(parents=True, exist_ok=True)
        console.print(f"[dim]Download directory: {self.config.download_dir.resolve()}[/dim]")

    def get_state_path(self) -> Path:
        return self.config.download_dir / ".download_state.json"

    def load_state(self) -> dict[str, Any]:
        state_path = self.get_state_path()
        if state_path.exists():
            with state_path.open() as f:
                return json.load(f)
        return {"downloaded_ids": [], "next_page_token": None}

    def save_state(self, state: dict[str, Any]) -> None:
        state_path = self.get_state_path()
        state["updated_at"] = datetime.now().isoformat()
        with state_path.open("w") as f:
            json.dump(state, f, indent=2)

    def _make_request(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Make HTTP request with retries."""
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self.client.get(url, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                console.print(f"[red]Attempt {attempt}/{self.config.max_retries}: HTTP {e.response.status_code}[/red]")
            except httpx.HTTPError as e:
                console.print(f"[red]Attempt {attempt}/{self.config.max_retries}: {e}[/red]")

            if attempt < self.config.max_retries:
                time.sleep(self.config.retry_delay)
        return None

    # =========================================================================
    # Search API
    # =========================================================================

    def search(self, page_token: str | None = None) -> tuple[list[str], str | None, int]:
        """
        Search collection using the new Search API.

        Returns (list of object IDs, next_page_token, total_count).
        """
        params: dict[str, Any] = {"imageAvailable": "true"}
        if page_token:
            params["pageToken"] = page_token

        data = self._make_request(SEARCH_API, params)
        if data is None:
            return [], None, 0

        # Extract object IDs from orderedItems
        object_ids = []
        for item in data.get("orderedItems", []):
            if isinstance(item, dict):
                obj_id = item.get("id", "")
                # Extract numeric ID from URL: https://id.rijksmuseum.nl/20010000
                if "/" in obj_id:
                    numeric_id = obj_id.split("/")[-1]
                    object_ids.append(numeric_id)

        # Get next page token from next.id URL
        next_token = None
        next_info = data.get("next", {})
        if isinstance(next_info, dict):
            next_url = next_info.get("id", "")
            if "pageToken=" in next_url:
                next_token = next_url.split("pageToken=")[-1]

        # Get total count
        total = data.get("partOf", {}).get("totalItems", 0)

        return object_ids, next_token, total

    # =========================================================================
    # Object Chain Fetching
    # =========================================================================

    def get_object(self, object_id: str) -> dict[str, Any] | None:
        """Fetch object metadata."""
        url = f"{DATA_API}/{object_id}"
        return self._make_request(url)

    def get_linked_resource(self, resource_id: str) -> dict[str, Any] | None:
        """Fetch a linked resource (VisualItem or DigitalObject)."""
        # Resource ID might be full URL or just the ID
        if resource_id.startswith("https://"):
            numeric_id = resource_id.split("/")[-1]
        else:
            numeric_id = resource_id
        url = f"{DATA_API}/{numeric_id}"
        return self._make_request(url)

    def extract_object_number(self, obj_data: dict[str, Any]) -> str | None:
        """Extract the object number (e.g., RP-P-1906-2550) from metadata."""
        for identifier in obj_data.get("identified_by", []):
            if identifier.get("type") != "Identifier":
                continue
            # Check if classified as object/accession number
            for classifier in identifier.get("classified_as", []):
                classifier_id = classifier.get("id", "") if isinstance(classifier, dict) else str(classifier)
                if OBJECT_NUMBER_CLASSIFIER in classifier_id:
                    return identifier.get("content")
        return None

    def extract_visual_item_id(self, obj_data: dict[str, Any]) -> str | None:
        """Extract the VisualItem ID from object metadata."""
        shows = obj_data.get("shows", [])
        if shows and isinstance(shows[0], dict):
            return shows[0].get("id")
        return None

    def extract_digital_object_id(self, visual_item: dict[str, Any]) -> str | None:
        """Extract the DigitalObject ID from VisualItem."""
        digitally_shown_by = visual_item.get("digitally_shown_by", [])
        if digitally_shown_by and isinstance(digitally_shown_by[0], dict):
            return digitally_shown_by[0].get("id")
        return None

    def extract_image_url(self, digital_object: dict[str, Any]) -> str | None:
        """Extract the IIIF image URL from DigitalObject."""
        access_points = digital_object.get("access_point", [])
        if access_points and isinstance(access_points[0], dict):
            return access_points[0].get("id")
        return None

    def is_downloadable(self, digital_object: dict[str, Any]) -> bool:
        """Check if the image is downloadable."""
        for ref in digital_object.get("referred_to_by", []):
            content = ref.get("content", "") if isinstance(ref, dict) else ""
            if "downloadbaar" in content.lower() and "niet" not in content.lower():
                return True
            if "niet downloadbaar" in content.lower():
                return False
        # Default to not downloadable if status unclear
        return False

    def resolve_image_url(self, object_id: str) -> tuple[str | None, dict[str, Any] | None, str | None]:
        """
        Resolve the full chain to get image URL.

        Returns (image_url, object_metadata, object_number) or (None, None, None) on failure.
        """
        # Step 1: Get object metadata
        obj_data = self.get_object(object_id)
        if not obj_data:
            return None, None, None
        time.sleep(self.config.rate_limit_delay)

        object_number = self.extract_object_number(obj_data)
        visual_item_id = self.extract_visual_item_id(obj_data)
        if not visual_item_id:
            return None, obj_data, object_number

        # Step 2: Get VisualItem
        visual_item = self.get_linked_resource(visual_item_id)
        if not visual_item:
            return None, obj_data, object_number
        time.sleep(self.config.rate_limit_delay)

        digital_object_id = self.extract_digital_object_id(visual_item)
        if not digital_object_id:
            return None, obj_data, object_number

        # Step 3: Get DigitalObject
        digital_object = self.get_linked_resource(digital_object_id)
        if not digital_object:
            return None, obj_data, object_number
        time.sleep(self.config.rate_limit_delay)

        # Check if downloadable
        if not self.is_downloadable(digital_object):
            return "NOT_DOWNLOADABLE", obj_data, object_number

        image_url = self.extract_image_url(digital_object)
        return image_url, obj_data, object_number

    # =========================================================================
    # Image Download
    # =========================================================================

    def download_image(self, image_url: str, output_path: Path) -> bool:
        """Download an image file."""
        if output_path.exists():
            return True

        try:
            with self.client.stream("GET", image_url) as response:
                response.raise_for_status()
                with output_path.open("wb") as f:
                    for chunk in response.iter_bytes(self.config.chunk_size):
                        f.write(chunk)
                        self.stats.total_bytes += len(chunk)
            return True
        except httpx.HTTPError as e:
            console.print(f"[red]Image download failed: {e}[/red]")
            if output_path.exists():
                output_path.unlink()
            return False

    # =========================================================================
    # Main Download Logic
    # =========================================================================

    def download_all(self, limit: int | None = None, force: bool = False) -> DownloadStats:
        """Download collection data."""
        self.ensure_dirs()
        self.stats = DownloadStats()

        console.print("[blue]Using Rijksmuseum Linked Art API[/blue]")

        state = {} if force else self.load_state()
        downloaded_ids = set(state.get("downloaded_ids", []))
        page_token = None if force else state.get("next_page_token")

        # Get initial count
        console.print("[blue]Querying collection...[/blue]")
        _, _, total_available = self.search()

        if total_available == 0:
            console.print("[yellow]No objects found or API unavailable[/yellow]")
            return self.stats

        total_to_download = min(limit, total_available) if limit else total_available
        self.stats.total_requested = total_to_download

        console.print(f"[green]Found {total_available:,} objects with images[/green]")
        console.print(f"[green]Downloading up to {total_to_download:,} objects[/green]")

        if downloaded_ids:
            console.print(f"[yellow]Resuming: {len(downloaded_ids)} already downloaded[/yellow]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Downloading...",
                total=total_to_download,
                completed=self.stats.downloaded_objects,
            )

            # Track new downloads (not skipped) to honor the limit
            new_downloads = 0

            while new_downloads < total_to_download:
                # Search for objects
                object_ids, next_token, _ = self.search(page_token)

                if not object_ids:
                    console.print("[yellow]No more objects to fetch[/yellow]")
                    break

                for obj_id in object_ids:
                    if new_downloads >= total_to_download:
                        break

                    # Resolve image URL through the chain
                    image_url, obj_data, object_number = self.resolve_image_url(obj_id)

                    # Use object_number for filename, fallback to numeric ID
                    file_id = object_number or obj_id

                    # Skip already downloaded (don't count toward limit)
                    if file_id in downloaded_ids:
                        self.stats.skipped_existing += 1
                        continue

                    # Check various skip conditions (count toward limit)
                    if image_url is None:
                        self.stats.skipped_no_image += 1
                        new_downloads += 1
                        progress.update(task, completed=new_downloads)
                        continue

                    if image_url == "NOT_DOWNLOADABLE":
                        self.stats.skipped_not_downloadable += 1
                        new_downloads += 1
                        progress.update(task, completed=new_downloads)
                        continue

                    # Download image
                    image_path = self.config.download_dir / "images" / f"{file_id}.jpg"
                    if not self.download_image(image_url, image_path):
                        self.stats.failed += 1
                        new_downloads += 1
                        progress.update(task, completed=new_downloads)
                        continue

                    # Save metadata only if image downloaded successfully
                    metadata_path = self.config.download_dir / "metadata" / f"{file_id}.json"
                    with metadata_path.open("w") as f:
                        json.dump(obj_data, f, indent=2, ensure_ascii=False)

                    self.stats.downloaded_objects += 1
                    downloaded_ids.add(file_id)
                    new_downloads += 1
                    progress.update(task, completed=new_downloads)

                    time.sleep(self.config.rate_limit_delay)

                # Save state after each page
                page_token = next_token
                state["downloaded_ids"] = list(downloaded_ids)
                state["next_page_token"] = page_token
                self.save_state(state)

                if not next_token:
                    break

        return self.stats


def print_summary(stats: DownloadStats, config: DownloadConfig) -> None:
    """Print download summary."""
    console.print()

    table = Table(title="Download Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Requested", f"{stats.total_requested:,}")
    table.add_row("Downloaded", f"[green]{stats.downloaded_objects:,}[/green]")
    table.add_row("Skipped (existing)", f"[dim]{stats.skipped_existing:,}[/dim]")
    table.add_row("Skipped (not downloadable)", f"[yellow]{stats.skipped_not_downloadable:,}[/yellow]")
    table.add_row("Skipped (no image)", f"[yellow]{stats.skipped_no_image:,}[/yellow]")
    table.add_row("Failed", f"[red]{stats.failed:,}[/red]")
    table.add_row("Data downloaded", f"{stats.total_bytes / 1024 / 1024:.1f} MB")
    table.add_row("Duration", f"{stats.duration:.1f}s")

    console.print(table)
    console.print()
    console.print(f"[dim]Metadata: {config.download_dir / 'metadata'}[/dim]")
    console.print(f"[dim]Images: {config.download_dir / 'images'}[/dim]")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download Rijksmuseum collection data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Uses the Rijksmuseum Linked Art APIs (no API key required).
Downloads metadata (JSON) and max-resolution images (JPEG) for objects
that have downloadable images.

Examples:
  # Download 10 objects
  uv run python download_data.py --limit 10

  # Force re-download
  uv run python download_data.py --force --limit 10

  # Custom download directory
  uv run python download_data.py --download-dir /path/to/data --limit 100
        """,
    )
    parser.add_argument("--download-dir", type=Path, help="Download directory")
    parser.add_argument("--limit", type=int, help="Max objects to download")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--info", action="store_true", help="Show API info")
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.info:
        console.print("\n[bold]Rijksmuseum Data Downloader v2.0[/bold]\n")

        table = Table()
        table.add_column("API", style="cyan")
        table.add_column("URL")
        table.add_column("Key Required")

        table.add_row("Search", "data.rijksmuseum.nl/search/collection", "[green]No[/green]")
        table.add_row("Object Resolver", "data.rijksmuseum.nl/{id}", "[green]No[/green]")
        table.add_row("IIIF Images", "iiif.micr.io/{id}/full/max/0/default.jpg", "[green]No[/green]")

        console.print(table)
        console.print()
        console.print("[dim]All data is CC0 (public domain)[/dim]")
        console.print()
        return 0

    console.print()
    console.print("[bold blue]" + "=" * 60 + "[/bold blue]")
    console.print("[bold blue]Rijksmuseum Data Downloader[/bold blue]")
    console.print("[bold blue]" + "=" * 60 + "[/bold blue]")
    console.print()

    config = load_config(args)

    with RijksmuseumDownloader(config) as downloader:
        stats = downloader.download_all(
            limit=args.limit,
            force=args.force,
        )

    print_summary(stats, config)
    return 1 if stats.failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
