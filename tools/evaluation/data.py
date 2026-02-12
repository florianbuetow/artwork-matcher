"""Data-loading helpers for evaluation tooling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluation.models import LabelEntry

IMAGE_PATTERNS: tuple[str, ...] = ("*.jpg", "*.jpeg", "*.png")


def load_labels(labels_path: Path) -> dict[str, LabelEntry]:
    """Load labels.csv into a dictionary keyed by picture_id."""
    df = pd.read_csv(labels_path, skipinitialspace=True)
    picture_col = df.columns[0]
    painting_col = df.columns[1]

    labels: dict[str, LabelEntry] = {}
    for _, row in df.iterrows():
        entry = LabelEntry.from_csv_row(
            str(row[picture_col]), str(row[painting_col])
        )
        labels[entry.picture_id] = entry

    return labels


def discover_images(directory: Path) -> list[Path]:
    """Discover supported image files in a directory."""
    files: list[Path] = []
    for pattern in IMAGE_PATTERNS:
        files.extend(directory.glob(pattern))
    files.sort()
    return files


def count_images(directory: Path) -> int:
    """Count supported image files in a directory."""
    return len(discover_images(directory))

