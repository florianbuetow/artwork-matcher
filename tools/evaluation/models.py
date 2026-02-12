"""Data models used by evaluation tooling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class LabelEntry:
    """Single ground truth label entry."""

    picture_id: str
    valid_object_ids: frozenset[str]

    @classmethod
    def from_csv_row(cls, picture_path: str, painting_paths: str) -> LabelEntry:
        """Create a label entry from CSV row values."""
        picture_id = Path(picture_path.strip()).stem
        object_ids = frozenset(
            Path(p.strip()).stem for p in painting_paths.split(";") if p.strip()
        )
        return cls(picture_id=picture_id, valid_object_ids=object_ids)


@dataclass
class RankedResultItem:
    """One ranked candidate from gateway /identify response."""

    object_id: str
    similarity_score: float | None
    geometric_score: float | None
    confidence: float | None


@dataclass
class MatchResult:
    """Result from a single identification request."""

    picture_id: str
    mode: Literal["embedding_only", "geometric"]
    matched_object_id: str | None
    similarity_score: float | None
    geometric_score: float | None
    confidence: float | None
    ranked_results: list[RankedResultItem]
    embedding_ms: float
    search_ms: float
    geometric_ms: float
    total_ms: float
    error: str | None
    error_message: str | None

    @property
    def is_successful(self) -> bool:
        """Return True when the request succeeded."""
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
