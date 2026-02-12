"""Unit tests for evaluation data loading helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from evaluation.data import count_images, discover_images, load_labels  # noqa: E402


class EvaluationDataTest(unittest.TestCase):
    """Validate labels parsing and image discovery behavior."""

    def test_load_labels_parses_picture_and_object_stems(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            labels_path = Path(tmpdir) / "labels.csv"
            labels_path.write_text(
                "picture_path,painting_paths\n"
                "pictures/p1.jpg,objects/o1.jpg; objects/o2.jpg\n"
                "pictures/p2.png,objects/o3.jpeg\n"
            )

            labels = load_labels(labels_path)

            self.assertEqual(set(labels.keys()), {"p1", "p2"})
            self.assertEqual(labels["p1"].valid_object_ids, frozenset({"o1", "o2"}))
            self.assertEqual(labels["p2"].valid_object_ids, frozenset({"o3"}))

    def test_discover_and_count_images_only_include_supported_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "b.jpeg").write_bytes(b"jpeg")
            (data_dir / "a.jpg").write_bytes(b"jpg")
            (data_dir / "c.png").write_bytes(b"png")
            (data_dir / "ignore.gif").write_bytes(b"gif")

            files = discover_images(data_dir)

            self.assertEqual([p.name for p in files], ["a.jpg", "b.jpeg", "c.png"])
            self.assertEqual(count_images(data_dir), 3)


if __name__ == "__main__":
    unittest.main()
