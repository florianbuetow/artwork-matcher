# Artwork Matcher Evaluation Guide

This guide details how to evaluate the accuracy of the artwork matching system, comparing performance **with** and **without** the geometric reranker (Geometric Service).

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Test Data Format](#test-data-format)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Implementation](#implementation)
6. [Running the Evaluation](#running-the-evaluation)
7. [Report Format](#report-format)
8. [Interpreting Results](#interpreting-results)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The evaluation system measures how well the artwork matcher identifies artworks from visitor photos by comparing system predictions against ground truth labels.

### Goals

1. **Measure accuracy** — How often does the system correctly identify the artwork?
2. **Compare modes** — Quantify the improvement from geometric verification
3. **Rank quality** — Assess the quality of the ranked candidate list
4. **Latency impact** — Measure the latency cost of geometric verification

### Two Evaluation Modes

| Mode | Description | API Call |
|------|-------------|----------|
| **Embedding Only** | DINOv2 embeddings + FAISS search | Gateway `/identify` with `geometric_verification: false` |
| **With Geometric** | Embedding search + ORB/RANSAC reranking | Gateway `/identify` with `geometric_verification: true` |

---

## Architecture

### Evaluation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Evaluation Pipeline                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. SETUP                                                                │
│     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐            │
│     │  Check      │────▶│  Build      │────▶│  Load       │            │
│     │  Services   │     │  Index      │     │  Labels     │            │
│     └─────────────┘     └─────────────┘     └─────────────┘            │
│                                                                          │
│  2. EVALUATION (for each picture)                                        │
│     ┌─────────────────────────────────────────────────────────────┐     │
│     │                                                              │     │
│     │  ┌─────────────────────┐    ┌─────────────────────┐        │     │
│     │  │  Call Gateway       │    │  Call Gateway       │        │     │
│     │  │  geometric=false    │    │  geometric=true     │        │     │
│     │  │  (embedding only)   │    │  (with reranker)    │        │     │
│     │  └──────────┬──────────┘    └──────────┬──────────┘        │     │
│     │             │                           │                   │     │
│     │             ▼                           ▼                   │     │
│     │  ┌─────────────────────────────────────────────────────┐   │     │
│     │  │           Compare to Ground Truth                    │   │     │
│     │  └─────────────────────────────────────────────────────┘   │     │
│     │                                                              │     │
│     └──────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  3. METRICS                                                              │
│     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐            │
│     │ Classification│    │  Ranking    │    │   Timing    │            │
│     │  P, R, F1   │    │ MRR, Hit@K  │    │  Latency    │            │
│     └─────────────┘     └─────────────┘     └─────────────┘            │
│                                                                          │
│  4. REPORT                                                               │
│     ┌─────────────────────────────────────────────────────────────┐     │
│     │  Generate Markdown Report with comparisons and analysis     │     │
│     └─────────────────────────────────────────────────────────────┘     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Service Dependencies

| Service | Port | Role in Evaluation |
|---------|------|-------------------|
| Gateway | 8000 | Main API endpoint for `/identify` |
| Embeddings | 8001 | DINOv2 embedding extraction |
| Search | 8002 | FAISS vector search |
| Geometric | 8003 | ORB + RANSAC verification (optional) |

---

## Test Data Format

### Directory Structure

Test data must be placed in `./data/testdata/`:

```
data/testdata/
├── objects/           # Reference artwork images (to be indexed)
│   ├── nl-SK-A-4999.jpg
│   ├── nl-SK-A-5000.jpg
│   ├── nl-SK-A-4821.jpg
│   └── ...
├── pictures/          # Visitor test photos (to be evaluated)
│   ├── sk_2.jpg
│   ├── sk_3.jpg
│   ├── sk_4.jpg
│   └── ...
└── labels.csv         # Ground truth mappings
```

### Labels File Format

The `labels.csv` file maps visitor photos to their expected artwork matches:

```csv
picture_image_path, painting_image_paths
sk_2.jpg, nl-SK-A-4999.jpg
sk_3.jpg, nl-SK-A-5000.jpg
sk_4.jpg, nl-SK-A-4821.jpg
sk_6.jpg, nl-SK-A-359.jpg;nl-SK-A-851.jpg
```

**Fields:**
- `picture_image_path`: Filename of the visitor photo in `pictures/`
- `painting_image_paths`: One or more valid artwork matches (semicolon-separated for multiple valid matches)

### Object ID Derivation

Object IDs are derived from filenames by removing the extension:

| Filename | Object ID |
|----------|-----------|
| `nl-SK-A-4999.jpg` | `nl-SK-A-4999` |
| `sk_2.jpg` | `sk_2` |

**Important:** The evaluation compares the `object_id` returned by the Gateway API against the expected object IDs from `labels.csv`.

### Multiple Valid Matches

Some pictures may legitimately match multiple artworks (e.g., different angles or editions). These are encoded with semicolons:

```csv
sk_6.jpg, nl-SK-A-359.jpg;nl-SK-A-851.jpg
```

For metrics calculation:
- **Classification (Top-1)**: Correct if the predicted match is ANY of the valid options
- **Ranking**: Correct at rank K if ANY valid match appears at that rank or better

---

## Evaluation Metrics

### Classification Metrics (Top-1 Match)

These measure the accuracy of the **primary** (top-1) identification result:

| Metric | Formula | Description |
|--------|---------|-------------|
| **Precision** | `TP / (TP + FP)` | Of all predictions, how many were correct? |
| **Recall** | `TP / (TP + FN)` | Of all actual matches, how many did we find? |
| **F1-Score** | `2 * (P * R) / (P + R)` | Harmonic mean of precision and recall |

**Definition of TP/FP/FN:**
- **True Positive (TP)**: Predicted match is in the set of valid matches
- **False Positive (FP)**: Predicted a match, but it's not in the valid set
- **False Negative (FN)**: Failed to predict a match when valid matches exist

### Ranking Metrics

These measure the quality of the **full ranked list** of candidates:

| Metric | Formula | Description |
|--------|---------|-------------|
| **MRR** | `(1/n) * Σ(1/rank_i)` | Mean Reciprocal Rank — average of 1/rank for first correct result |
| **Hit@K** | `Σ(hit_i) / n` | Fraction of queries with correct result in top-K |
| **NDCG@K** | `DCG@K / IDCG@K` | Normalized Discounted Cumulative Gain at K |
| **MAP** | `(1/n) * Σ(AP_i)` | Mean Average Precision across all queries |

**Example MRR Calculation:**

| Query | First Correct Rank | Reciprocal Rank |
|-------|-------------------|-----------------|
| pic_1 | 1 | 1/1 = 1.0 |
| pic_2 | 3 | 1/3 = 0.33 |
| pic_3 | 2 | 1/2 = 0.5 |
| pic_4 | Not found | 0 |

**MRR = (1.0 + 0.33 + 0.5 + 0) / 4 = 0.458**

### Timing Metrics

| Metric | Description |
|--------|-------------|
| **Mean Latency** | Average request time |
| **Median (P50)** | 50th percentile latency |
| **P95** | 95th percentile latency |
| **P99** | 99th percentile latency |
| **Geometric Overhead** | Additional latency from geometric verification |

---

## Implementation

### Dependencies

Add to `tools/pyproject.toml`:

```toml
[project]
name = "artwork-matcher-tools"
version = "0.1.0"
description = "CLI tools for artwork matcher"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.28.0",
    "pillow>=11.0.0",
    "pandas>=2.2.0",
    "rich>=13.9.0",
    "numpy>=2.0.0",
    "jinja2>=3.1.0",
    "pydantic>=2.0.0",
]
```

### Package Structure

Create the evaluation package at `tools/evaluation/`:

```
tools/
├── evaluation/
│   ├── __init__.py
│   ├── config.py       # Configuration models
│   ├── data.py         # Data loading and result types
│   ├── client.py       # HTTP client for Gateway API
│   ├── metrics.py      # Metric calculations
│   └── report.py       # Markdown report generation
├── build_index.py      # Index building script
├── evaluate.py         # Main evaluation script
├── justfile
└── pyproject.toml
```

### Configuration Module (`evaluation/config.py`)

```python
"""
Evaluation configuration with Pydantic validation.
"""
from pathlib import Path
from pydantic import BaseModel, Field


class ServiceConfig(BaseModel):
    """Service connection configuration."""
    gateway_url: str = "http://localhost:8000"
    embeddings_url: str = "http://localhost:8001"
    search_url: str = "http://localhost:8002"
    geometric_url: str = "http://localhost:8003"
    timeout_seconds: float = 60.0


class DataConfig(BaseModel):
    """Test data paths configuration."""
    testdata_path: Path = Path("data/testdata")
    objects_subdir: str = "objects"
    pictures_subdir: str = "pictures"
    labels_file: str = "labels.csv"

    @property
    def objects_path(self) -> Path:
        return self.testdata_path / self.objects_subdir

    @property
    def pictures_path(self) -> Path:
        return self.testdata_path / self.pictures_subdir

    @property
    def labels_path(self) -> Path:
        return self.testdata_path / self.labels_file


class EvaluationConfig(BaseModel):
    """Evaluation parameters."""
    # Search parameters
    search_k: int = 10

    # Thresholds (use 0.0 for evaluation to get full ranking)
    similarity_threshold: float = 0.0

    # Hit@K values to calculate
    hit_at_k_values: list[int] = Field(default=[1, 3, 5, 10])

    # NDCG@K values to calculate
    ndcg_at_k_values: list[int] = Field(default=[5, 10])


class OutputConfig(BaseModel):
    """Output configuration."""
    report_dir: Path = Path("reports/evaluation")
    report_filename: str = "evaluation_report.md"
    results_json_filename: str = "evaluation_results.json"
```

### Data Module (`evaluation/data.py`)

```python
"""
Data loading and result dataclasses.
"""
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd


@dataclass(frozen=True)
class LabelEntry:
    """Single ground truth label entry."""
    picture_id: str
    valid_object_ids: frozenset[str]

    @classmethod
    def from_csv_row(cls, picture_path: str, painting_paths: str) -> "LabelEntry":
        """Create from CSV row values."""
        # Extract picture_id from filename (stem without extension)
        picture_id = Path(picture_path.strip()).stem

        # Parse semicolon-separated painting paths and extract object IDs
        object_ids = frozenset(
            Path(p.strip()).stem
            for p in painting_paths.split(";")
            if p.strip()
        )

        return cls(picture_id=picture_id, valid_object_ids=object_ids)


@dataclass
class MatchResult:
    """Result from a single identification request."""
    picture_id: str
    mode: str  # "embedding_only" or "geometric"

    # Match info (from top result)
    matched_object_id: str | None = None
    similarity_score: float | None = None
    geometric_score: float | None = None
    geometric_inliers: int | None = None
    confidence: float | None = None
    verification_method: str | None = None

    # Full ranked results (top-K)
    ranked_results: list[dict] = field(default_factory=list)

    # Timing breakdown
    embedding_ms: float = 0.0
    search_ms: float = 0.0
    geometric_ms: float = 0.0
    total_ms: float = 0.0

    # Error info
    error: str | None = None
    error_message: str | None = None

    @property
    def is_successful(self) -> bool:
        """Returns True if the request succeeded without errors."""
        return self.error is None


def load_labels(labels_path: Path) -> dict[str, LabelEntry]:
    """
    Load labels.csv into a dictionary keyed by picture_id.

    Args:
        labels_path: Path to the labels.csv file

    Returns:
        Dictionary mapping picture_id -> LabelEntry
    """
    df = pd.read_csv(labels_path, skipinitialspace=True)

    # Handle different possible column names
    picture_col = df.columns[0]
    painting_col = df.columns[1]

    labels = {}
    for _, row in df.iterrows():
        entry = LabelEntry.from_csv_row(
            str(row[picture_col]),
            str(row[painting_col])
        )
        labels[entry.picture_id] = entry

    return labels
```

### Client Module (`evaluation/client.py`)

```python
"""
HTTP client for calling the Gateway API.
"""
import base64
import time
from pathlib import Path
import httpx

from .data import MatchResult


class ArtworkMatcherClient:
    """Client for the Artwork Matcher Gateway API."""

    def __init__(
        self,
        gateway_url: str = "http://localhost:8000",
        timeout: float = 60.0,
    ):
        self.gateway_url = gateway_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def check_health(self) -> dict:
        """
        Check health of all services via Gateway.

        Returns:
            Health status dictionary
        """
        try:
            response = self.client.get(
                f"{self.gateway_url}/health",
                params={"check_backends": "true"}
            )
            return response.json()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def identify(
        self,
        image_path: Path,
        geometric_verification: bool,
        k: int = 10,
        threshold: float = 0.0,
    ) -> MatchResult:
        """
        Call the /identify endpoint.

        Args:
            image_path: Path to the visitor photo
            geometric_verification: Whether to enable geometric reranking
            k: Number of candidates to retrieve
            threshold: Minimum similarity threshold

        Returns:
            MatchResult with identification results
        """
        picture_id = image_path.stem
        mode = "geometric" if geometric_verification else "embedding_only"

        start_time = time.perf_counter()

        try:
            # Read and encode image
            image_bytes = image_path.read_bytes()
            image_b64 = base64.b64encode(image_bytes).decode("ascii")

            # Call Gateway
            response = self.client.post(
                f"{self.gateway_url}/identify",
                json={
                    "image": image_b64,
                    "options": {
                        "k": k,
                        "threshold": threshold,
                        "geometric_verification": geometric_verification,
                        "include_alternatives": True,
                    }
                }
            )
            response.raise_for_status()
            data = response.json()

            # Extract timing
            timing = data.get("timing", {})

            # Extract match info
            match = data.get("match")
            alternatives = data.get("alternatives", [])

            # Build ranked results list
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
                geometric_inliers=match.get("inliers") if match else None,
                confidence=match.get("confidence") if match else None,
                verification_method=match.get("verification_method") if match else None,
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
            except Exception:
                pass

            return MatchResult(
                picture_id=picture_id,
                mode=mode,
                error=error_data.get("error", "http_error"),
                error_message=error_data.get("message", str(e)),
                total_ms=(time.perf_counter() - start_time) * 1000,
            )

        except Exception as e:
            return MatchResult(
                picture_id=picture_id,
                mode=mode,
                error="client_error",
                error_message=str(e),
                total_ms=(time.perf_counter() - start_time) * 1000,
            )

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
```

### Metrics Module (`evaluation/metrics.py`)

```python
"""
Evaluation metric calculations.
"""
from dataclasses import dataclass
import numpy as np

from .data import MatchResult, LabelEntry


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
    map_score: float


@dataclass
class TimingStats:
    """Latency statistics."""
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


def calculate_classification_metrics(
    results: list[MatchResult],
    labels: dict[str, LabelEntry],
) -> ClassificationMetrics:
    """
    Calculate classification metrics for top-1 predictions.

    A prediction is correct if the matched_object_id is in the
    set of valid object IDs for that picture.
    """
    tp = fp = fn = 0

    for result in results:
        if result.error:
            # Errors count as false negatives
            fn += 1
            continue

        label = labels.get(result.picture_id)
        if not label or not label.valid_object_ids:
            # No ground truth - skip
            continue

        predicted = result.matched_object_id
        valid_set = label.valid_object_ids

        if predicted is None:
            # No prediction made
            fn += 1
        elif predicted in valid_set:
            # Correct prediction
            tp += 1
        else:
            # Wrong prediction
            fp += 1

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

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
    """
    Calculate Mean Reciprocal Rank.

    For each query, finds the rank of the first correct result
    and computes the reciprocal. Returns the mean across all queries.
    """
    reciprocal_ranks = []

    for result in results:
        label = labels.get(result.picture_id)
        if not label or not label.valid_object_ids:
            continue

        if result.error or not result.ranked_results:
            reciprocal_ranks.append(0.0)
            continue

        # Find first correct result in ranking
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
    """
    Calculate Hit@K.

    Returns the fraction of queries where at least one correct
    result appears in the top-K positions.
    """
    hits = 0
    total = 0

    for result in results:
        label = labels.get(result.picture_id)
        if not label or not label.valid_object_ids:
            continue

        total += 1

        if result.error:
            continue

        # Check if any valid match in top-K
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
    """
    Calculate NDCG@K (Normalized Discounted Cumulative Gain).

    Uses binary relevance: 1 if object_id is in valid set, 0 otherwise.
    """
    ndcg_scores = []

    for result in results:
        label = labels.get(result.picture_id)
        if not label or not label.valid_object_ids:
            continue

        if result.error or not result.ranked_results:
            ndcg_scores.append(0.0)
            continue

        # Calculate DCG@K
        dcg = 0.0
        for i, item in enumerate(result.ranked_results[:k]):
            rel = 1.0 if item.get("object_id") in label.valid_object_ids else 0.0
            dcg += rel / np.log2(i + 2)  # i+2 because ranks start at 1

        # Calculate Ideal DCG (all relevant items at top)
        num_relevant = min(len(label.valid_object_ids), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))

        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def calculate_map(
    results: list[MatchResult],
    labels: dict[str, LabelEntry],
) -> float:
    """
    Calculate Mean Average Precision (MAP).

    Computes average precision for each query, then takes the mean.
    """
    average_precisions = []

    for result in results:
        label = labels.get(result.picture_id)
        if not label or not label.valid_object_ids:
            continue

        if result.error or not result.ranked_results:
            average_precisions.append(0.0)
            continue

        # Calculate Average Precision
        relevant_found = 0
        precision_sum = 0.0

        for i, item in enumerate(result.ranked_results):
            if item.get("object_id") in label.valid_object_ids:
                relevant_found += 1
                precision_at_i = relevant_found / (i + 1)
                precision_sum += precision_at_i

        num_relevant = len(label.valid_object_ids)
        ap = precision_sum / num_relevant if num_relevant > 0 else 0.0
        average_precisions.append(ap)

    return float(np.mean(average_precisions)) if average_precisions else 0.0


def calculate_timing_stats(results: list[MatchResult]) -> TimingStats:
    """Calculate latency statistics from successful results."""
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
        map_score=calculate_map(results, labels),
    )
```

### Report Module (`evaluation/report.py`)

```python
"""
Markdown report generation using Jinja2.
"""
from datetime import datetime
from pathlib import Path
from jinja2 import Environment, BaseLoader

from .metrics import ClassificationMetrics, RankingMetrics, TimingStats
from .data import MatchResult, LabelEntry


REPORT_TEMPLATE = '''# Artwork Matcher Evaluation Report

**Generated**: {{ timestamp }}
**Test Dataset**: {{ testdata_path }}
**Pictures Evaluated**: {{ num_pictures }}
**Objects in Index**: {{ num_objects }}

---

## Executive Summary

| Mode | Precision@1 | Recall@1 | F1-Score | MRR | Mean Latency |
|------|-------------|----------|----------|-----|--------------|
| Embedding Only | {{ "%.1f"|format(embed_class.precision * 100) }}% | {{ "%.1f"|format(embed_class.recall * 100) }}% | {{ "%.3f"|format(embed_class.f1_score) }} | {{ "%.3f"|format(embed_rank.mrr) }} | {{ "%.0f"|format(embed_timing.mean_ms) }}ms |
| With Geometric | {{ "%.1f"|format(geo_class.precision * 100) }}% | {{ "%.1f"|format(geo_class.recall * 100) }}% | {{ "%.3f"|format(geo_class.f1_score) }} | {{ "%.3f"|format(geo_rank.mrr) }} | {{ "%.0f"|format(geo_timing.mean_ms) }}ms |
| **Improvement** | **{{ "%+.1f"|format((geo_class.precision - embed_class.precision) * 100) }}%** | **{{ "%+.1f"|format((geo_class.recall - embed_class.recall) * 100) }}%** | **{{ "%+.3f"|format(geo_class.f1_score - embed_class.f1_score) }}** | **{{ "%+.3f"|format(geo_rank.mrr - embed_rank.mrr) }}** | **+{{ "%.0f"|format(geo_timing.mean_ms - embed_timing.mean_ms) }}ms** |

{% if geo_class.precision > embed_class.precision %}
**Recommendation**: Geometric verification improves precision by {{ "%.1f"|format((geo_class.precision - embed_class.precision) * 100) }}% at a cost of {{ "%.0f"|format(geo_timing.mean_ms - embed_timing.mean_ms) }}ms additional latency.
{% elif geo_class.precision == embed_class.precision %}
**Observation**: Geometric verification shows no precision improvement for this dataset. Consider disabling it for faster responses.
{% else %}
**Warning**: Geometric verification decreased precision. This may indicate issues with the geometric service or test data quality.
{% endif %}

---

## Classification Metrics (Top-1 Match)

| Metric | Embedding Only | With Geometric | Delta |
|--------|----------------|----------------|-------|
| Precision | {{ "%.3f"|format(embed_class.precision) }} | {{ "%.3f"|format(geo_class.precision) }} | {{ "%+.3f"|format(geo_class.precision - embed_class.precision) }} |
| Recall | {{ "%.3f"|format(embed_class.recall) }} | {{ "%.3f"|format(geo_class.recall) }} | {{ "%+.3f"|format(geo_class.recall - embed_class.recall) }} |
| F1-Score | {{ "%.3f"|format(embed_class.f1_score) }} | {{ "%.3f"|format(geo_class.f1_score) }} | {{ "%+.3f"|format(geo_class.f1_score - embed_class.f1_score) }} |
| True Positives | {{ embed_class.true_positives }} | {{ geo_class.true_positives }} | {{ "%+d"|format(geo_class.true_positives - embed_class.true_positives) }} |
| False Positives | {{ embed_class.false_positives }} | {{ geo_class.false_positives }} | {{ "%+d"|format(geo_class.false_positives - embed_class.false_positives) }} |
| False Negatives | {{ embed_class.false_negatives }} | {{ geo_class.false_negatives }} | {{ "%+d"|format(geo_class.false_negatives - embed_class.false_negatives) }} |

---

## Ranking Metrics

| Metric | Embedding Only | With Geometric | Delta |
|--------|----------------|----------------|-------|
| MRR | {{ "%.3f"|format(embed_rank.mrr) }} | {{ "%.3f"|format(geo_rank.mrr) }} | {{ "%+.3f"|format(geo_rank.mrr - embed_rank.mrr) }} |
{% for k in hit_at_k_values %}
| Hit@{{ k }} | {{ "%.3f"|format(embed_rank.hit_at_k[k]) }} | {{ "%.3f"|format(geo_rank.hit_at_k[k]) }} | {{ "%+.3f"|format(geo_rank.hit_at_k[k] - embed_rank.hit_at_k[k]) }} |
{% endfor %}
{% for k in ndcg_at_k_values %}
| NDCG@{{ k }} | {{ "%.3f"|format(embed_rank.ndcg_at_k[k]) }} | {{ "%.3f"|format(geo_rank.ndcg_at_k[k]) }} | {{ "%+.3f"|format(geo_rank.ndcg_at_k[k] - embed_rank.ndcg_at_k[k]) }} |
{% endfor %}
| MAP | {{ "%.3f"|format(embed_rank.map_score) }} | {{ "%.3f"|format(geo_rank.map_score) }} | {{ "%+.3f"|format(geo_rank.map_score - embed_rank.map_score) }} |

---

## Latency Analysis

| Metric | Embedding Only | With Geometric | Geometric Overhead |
|--------|----------------|----------------|-------------------|
| Mean | {{ "%.0f"|format(embed_timing.mean_ms) }}ms | {{ "%.0f"|format(geo_timing.mean_ms) }}ms | +{{ "%.0f"|format(geo_timing.mean_ms - embed_timing.mean_ms) }}ms |
| Median (P50) | {{ "%.0f"|format(embed_timing.median_ms) }}ms | {{ "%.0f"|format(geo_timing.median_ms) }}ms | +{{ "%.0f"|format(geo_timing.median_ms - embed_timing.median_ms) }}ms |
| P95 | {{ "%.0f"|format(embed_timing.p95_ms) }}ms | {{ "%.0f"|format(geo_timing.p95_ms) }}ms | +{{ "%.0f"|format(geo_timing.p95_ms - embed_timing.p95_ms) }}ms |
| P99 | {{ "%.0f"|format(embed_timing.p99_ms) }}ms | {{ "%.0f"|format(geo_timing.p99_ms) }}ms | +{{ "%.0f"|format(geo_timing.p99_ms - embed_timing.p99_ms) }}ms |

---

## Per-Image Failure Analysis

### Embedding Only Mode Failures

{% if embed_failures %}
| Picture | Expected | Got | Similarity |
|---------|----------|-----|------------|
{% for f in embed_failures %}
| {{ f.picture_id }} | {{ f.expected }} | {{ f.got or "None" }} | {{ "%.3f"|format(f.similarity or 0) }} |
{% endfor %}
{% else %}
*No failures in embedding-only mode.*
{% endif %}

### Geometric Mode Failures

{% if geo_failures %}
| Picture | Expected | Got | Confidence | Inliers |
|---------|----------|-----|------------|---------|
{% for f in geo_failures %}
| {{ f.picture_id }} | {{ f.expected }} | {{ f.got or "None" }} | {{ "%.2f"|format(f.confidence or 0) }} | {{ f.inliers or "-" }} |
{% endfor %}
{% else %}
*No failures in geometric mode.*
{% endif %}

### Corrections by Geometric Verification

{% if corrected %}
These cases were **incorrect** with embedding-only but **corrected** by geometric verification:

| Picture | Embedding Result | Geometric Result | Expected |
|---------|------------------|------------------|----------|
{% for c in corrected %}
| {{ c.picture_id }} | {{ c.embed_result }} ({{ "%.3f"|format(c.embed_sim) }}) | {{ c.geo_result }} ({{ "%.2f"|format(c.geo_conf) }}) | {{ c.expected }} |
{% endfor %}
{% else %}
*No corrections made by geometric verification.*
{% endif %}

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Total pictures evaluated | {{ num_pictures }} |
| Objects in index | {{ num_objects }} |
| Pictures with multiple valid matches | {{ multi_match_count }} |
| Errors during evaluation | {{ error_count }} |

---

*Report generated by `tools/evaluate.py`*
'''


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
    """
    Generate the markdown evaluation report.

    Args:
        embed_class: Classification metrics for embedding-only mode
        geo_class: Classification metrics for geometric mode
        embed_rank: Ranking metrics for embedding-only mode
        geo_rank: Ranking metrics for geometric mode
        embed_timing: Timing stats for embedding-only mode
        geo_timing: Timing stats for geometric mode
        embed_results: Raw results for embedding-only mode
        geo_results: Raw results for geometric mode
        labels: Ground truth labels
        testdata_path: Path to test data (for display)
        num_objects: Number of objects in the index
        hit_at_k_values: K values for Hit@K metrics
        ndcg_at_k_values: K values for NDCG@K metrics

    Returns:
        Markdown report as a string
    """
    env = Environment(loader=BaseLoader())
    template = env.from_string(REPORT_TEMPLATE)

    # Analyze failures
    embed_failures = []
    geo_failures = []
    corrected = []

    for embed_r, geo_r in zip(embed_results, geo_results):
        label = labels.get(embed_r.picture_id)
        if not label:
            continue

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

        # Track failures
        if not embed_correct and not embed_r.error and valid:
            embed_failures.append({
                "picture_id": embed_r.picture_id,
                "expected": ";".join(valid),
                "got": embed_r.matched_object_id,
                "similarity": embed_r.similarity_score,
            })

        if not geo_correct and not geo_r.error and valid:
            geo_failures.append({
                "picture_id": geo_r.picture_id,
                "expected": ";".join(valid),
                "got": geo_r.matched_object_id,
                "confidence": geo_r.confidence,
                "inliers": geo_r.geometric_inliers,
            })

        # Track corrections
        if not embed_correct and geo_correct:
            corrected.append({
                "picture_id": embed_r.picture_id,
                "embed_result": embed_r.matched_object_id or "None",
                "embed_sim": embed_r.similarity_score or 0,
                "geo_result": geo_r.matched_object_id,
                "geo_conf": geo_r.confidence or 0,
                "expected": ";".join(valid),
            })

    # Count edge cases
    multi_match_count = sum(
        1 for label in labels.values()
        if len(label.valid_object_ids) > 1
    )
    error_count = sum(
        1 for r in embed_results + geo_results
        if r.error
    )

    return template.render(
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        testdata_path=testdata_path,
        num_pictures=len(embed_results),
        num_objects=num_objects,
        embed_class=embed_class,
        geo_class=geo_class,
        embed_rank=embed_rank,
        geo_rank=geo_rank,
        embed_timing=embed_timing,
        geo_timing=geo_timing,
        hit_at_k_values=hit_at_k_values,
        ndcg_at_k_values=ndcg_at_k_values,
        embed_failures=embed_failures[:20],
        geo_failures=geo_failures[:20],
        corrected=corrected[:20],
        multi_match_count=multi_match_count,
        error_count=error_count,
    )
```

### Main Evaluation Script (`evaluate.py`)

```python
#!/usr/bin/env python3
"""
Artwork Matcher Evaluation Script.

Evaluates identification accuracy WITH and WITHOUT geometric verification,
comparing results against ground truth labels.

Usage:
    uv run python evaluate.py
    uv run python evaluate.py --testdata ./data/testdata
    uv run python evaluate.py --gateway-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from evaluation.config import DataConfig, EvaluationConfig, OutputConfig
from evaluation.client import ArtworkMatcherClient
from evaluation.data import load_labels
from evaluation.metrics import (
    calculate_classification_metrics,
    calculate_ranking_metrics,
    calculate_timing_stats,
)
from evaluation.report import generate_report

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate artwork-matcher accuracy"
    )
    parser.add_argument(
        "--testdata",
        type=Path,
        default=Path("data/testdata"),
        help="Path to test data directory (default: data/testdata)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/evaluation"),
        help="Path to output directory for reports (default: reports/evaluation)",
    )
    parser.add_argument(
        "--gateway-url",
        type=str,
        default="http://localhost:8000",
        help="Gateway service URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of candidates to retrieve (default: 10)",
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

    # Initialize configuration
    data_config = DataConfig(testdata_path=args.testdata)
    eval_config = EvaluationConfig(search_k=args.k)
    output_config = OutputConfig(report_dir=args.output)

    # Validate paths
    if not data_config.labels_path.exists():
        console.print(
            f"[red]Error: Labels file not found: {data_config.labels_path}[/red]"
        )
        return 1

    if not data_config.pictures_path.exists():
        console.print(
            f"[red]Error: Pictures directory not found: {data_config.pictures_path}[/red]"
        )
        return 1

    # Load labels
    console.print(f"[dim]Loading labels from {data_config.labels_path}...[/dim]")
    labels = load_labels(data_config.labels_path)
    console.print(f"[green]Loaded {len(labels)} label entries[/green]")

    # Find picture files
    picture_files = sorted(
        list(data_config.pictures_path.glob("*.jpg")) +
        list(data_config.pictures_path.glob("*.png")) +
        list(data_config.pictures_path.glob("*.jpeg"))
    )

    if not picture_files:
        console.print(
            f"[red]Error: No picture files found in {data_config.pictures_path}[/red]"
        )
        return 1

    console.print(f"[green]Found {len(picture_files)} pictures to evaluate[/green]")

    # Count objects in index (from objects directory)
    objects_path = data_config.objects_path
    num_objects = len(list(objects_path.glob("*.jpg"))) if objects_path.exists() else 0
    console.print(f"[green]Found {num_objects} objects in index[/green]")

    # Initialize client
    with ArtworkMatcherClient(
        gateway_url=args.gateway_url,
        timeout=60.0,
    ) as client:

        # Check service health
        console.print()
        console.print("[dim]Checking service health...[/dim]")
        health = client.check_health()

        if health.get("status") not in ("healthy", "degraded"):
            console.print(
                f"[red]Warning: Gateway health check failed: {health}[/red]"
            )
            console.print("[yellow]Proceeding anyway...[/yellow]")
        else:
            console.print(f"[green]Services healthy: {health.get('status')}[/green]")

        # Phase 1: Embedding-only evaluation
        console.print()
        console.print("[bold cyan]Phase 1: Embedding-Only Evaluation[/bold cyan]")
        embed_results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Evaluating (embedding only)",
                total=len(picture_files)
            )

            for pic_path in picture_files:
                result = client.identify(
                    image_path=pic_path,
                    geometric_verification=False,
                    k=eval_config.search_k,
                    threshold=eval_config.similarity_threshold,
                )
                embed_results.append(result)
                progress.advance(task)

        # Phase 2: Geometric verification evaluation
        console.print()
        console.print("[bold cyan]Phase 2: Geometric Verification Evaluation[/bold cyan]")
        geo_results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Evaluating (with geometric)",
                total=len(picture_files)
            )

            for pic_path in picture_files:
                result = client.identify(
                    image_path=pic_path,
                    geometric_verification=True,
                    k=eval_config.search_k,
                    threshold=eval_config.similarity_threshold,
                )
                geo_results.append(result)
                progress.advance(task)

    # Calculate metrics
    console.print()
    console.print("[dim]Calculating metrics...[/dim]")

    embed_class = calculate_classification_metrics(embed_results, labels)
    geo_class = calculate_classification_metrics(geo_results, labels)

    embed_rank = calculate_ranking_metrics(
        embed_results,
        labels,
        eval_config.hit_at_k_values,
        eval_config.ndcg_at_k_values,
    )
    geo_rank = calculate_ranking_metrics(
        geo_results,
        labels,
        eval_config.hit_at_k_values,
        eval_config.ndcg_at_k_values,
    )

    embed_timing = calculate_timing_stats(embed_results)
    geo_timing = calculate_timing_stats(geo_results)

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

    # Generate and save report
    output_config.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_config.report_dir / output_config.report_filename

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
        testdata_path=str(data_config.testdata_path),
        num_objects=num_objects,
        hit_at_k_values=eval_config.hit_at_k_values,
        ndcg_at_k_values=eval_config.ndcg_at_k_values,
    )

    report_path.write_text(report)
    console.print()
    console.print(f"[green]Report saved to: {report_path}[/green]")

    # Save raw results as JSON
    results_path = output_config.report_dir / output_config.results_json_filename
    results_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "testdata_path": str(data_config.testdata_path),
        "num_pictures": len(picture_files),
        "num_objects": num_objects,
        "config": {
            "search_k": eval_config.search_k,
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
                    "map": embed_rank.map_score,
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
                    "map": geo_rank.map_score,
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
    console.print(f"[green]Results JSON saved to: {results_path}[/green]")

    console.print()
    console.print("[bold blue]" + "=" * 60)
    console.print("[bold green]       Evaluation Complete!")
    console.print("[bold blue]" + "=" * 60)
    console.print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Index Building Script (`build_index.py`)

```python
#!/usr/bin/env python3
"""
Build FAISS index from object images.

Reads images from the objects directory, extracts embeddings via the
Embeddings service, and adds them to the Search service index.

Usage:
    uv run python build_index.py
    uv run python build_index.py --objects ./data/testdata/objects
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build FAISS index from object images")
    parser.add_argument(
        "--objects",
        type=Path,
        default=Path("data/objects"),
        help="Path to objects directory (default: data/objects)",
    )
    parser.add_argument(
        "--embeddings-url",
        type=str,
        default="http://localhost:8001",
        help="Embeddings service URL (default: http://localhost:8001)",
    )
    parser.add_argument(
        "--search-url",
        type=str,
        default="http://localhost:8002",
        help="Search service URL (default: http://localhost:8002)",
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

    # Validate objects directory
    if not args.objects.exists():
        console.print(f"[red]Error: Objects directory not found: {args.objects}[/red]")
        return 1

    # Find object images
    object_files = sorted(
        list(args.objects.glob("*.jpg")) +
        list(args.objects.glob("*.png")) +
        list(args.objects.glob("*.jpeg"))
    )

    if not object_files:
        console.print(f"[red]Error: No image files found in {args.objects}[/red]")
        return 1

    console.print(f"[green]Found {len(object_files)} objects to index[/green]")

    # Create HTTP client
    client = httpx.Client(timeout=60.0)

    try:
        # Check service health
        console.print("[dim]Checking service health...[/dim]")

        try:
            embed_health = client.get(f"{args.embeddings_url}/health").json()
            search_health = client.get(f"{args.search_url}/health").json()

            if embed_health.get("status") != "healthy":
                console.print(
                    f"[yellow]Warning: Embeddings service status: {embed_health}[/yellow]"
                )
            if search_health.get("status") != "healthy":
                console.print(
                    f"[yellow]Warning: Search service status: {search_health}[/yellow]"
                )

            console.print("[green]Services healthy[/green]")

        except Exception as e:
            console.print(f"[red]Error checking services: {e}[/red]")
            return 1

        # Clear existing index
        console.print()
        console.print("[dim]Clearing existing index...[/dim]")
        try:
            response = client.delete(f"{args.search_url}/index")
            result = response.json()
            console.print(
                f"[green]Cleared index (had {result.get('previous_count', 0)} items)[/green]"
            )
        except Exception as e:
            console.print(f"[yellow]Warning: Could not clear index: {e}[/yellow]")

        # Build index
        console.print()
        console.print("[bold cyan]Building index...[/bold cyan]")

        success_count = 0
        error_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Indexing objects", total=len(object_files))

            for image_path in object_files:
                object_id = image_path.stem

                try:
                    # Read and encode image
                    image_b64 = base64.b64encode(image_path.read_bytes()).decode()

                    # Get embedding
                    embed_response = client.post(
                        f"{args.embeddings_url}/embed",
                        json={
                            "image": image_b64,
                            "image_id": object_id,
                        }
                    )
                    embed_response.raise_for_status()
                    embedding = embed_response.json()["embedding"]

                    # Add to index
                    add_response = client.post(
                        f"{args.search_url}/add",
                        json={
                            "object_id": object_id,
                            "embedding": embedding,
                            "metadata": {
                                "name": object_id,
                                "image_path": str(image_path),
                            }
                        }
                    )
                    add_response.raise_for_status()

                    success_count += 1

                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Failed to index {object_id}: {e}[/yellow]"
                    )
                    error_count += 1

                progress.advance(task)

        # Save index
        console.print()
        console.print("[dim]Saving index to disk...[/dim]")

        try:
            save_response = client.post(
                f"{args.search_url}/index/save",
                json={}
            )
            save_response.raise_for_status()
            save_result = save_response.json()
            console.print(
                f"[green]Index saved: {save_result.get('count', 0)} items, "
                f"{save_result.get('size_bytes', 0)} bytes[/green]"
            )
        except Exception as e:
            console.print(f"[red]Error saving index: {e}[/red]")
            return 1

        # Summary
        console.print()
        console.print("[bold blue]" + "=" * 60)
        console.print(
            f"[bold green]Index built successfully: "
            f"{success_count} objects indexed"
        )
        if error_count > 0:
            console.print(f"[yellow]Errors: {error_count}[/yellow]")
        console.print("[bold blue]" + "=" * 60)
        console.print()

        return 0 if error_count == 0 else 1

    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
```

### Updated Justfile (`tools/justfile`)

```just
# Default recipe: show available commands
_default:
    @just --list

# Initialize the tools environment
init:
    @echo ""
    @printf "\033[0;34m=== Initializing Tools Environment ===\033[0m\n"
    @uv sync
    @printf "\033[0;32m✓ Tools environment ready\033[0m\n"
    @echo ""

# Destroy the virtual environment
destroy:
    @echo ""
    @printf "\033[0;34m=== Destroying Virtual Environment ===\033[0m\n"
    @rm -rf .venv
    @printf "\033[0;32m✓ Virtual environment removed\033[0m\n"
    @echo ""

# Build the FAISS index from object images
build-index objects="data/objects":
    @echo ""
    @printf "\033[0;34m=== Building Index ===\033[0m\n"
    @uv run python build_index.py --objects "{{objects}}"
    @echo ""

# Evaluate accuracy against labels.csv (full pipeline)
evaluate testdata="data/testdata":
    #!/usr/bin/env bash
    set -e
    echo ""
    printf "\033[0;34m=== Checking Service Health ===\033[0m\n"

    # Check services are running
    services=("8000:Gateway" "8001:Embeddings" "8002:Search" "8003:Geometric")
    all_healthy=true

    for svc in "${services[@]}"; do
        port="${svc%%:*}"
        name="${svc##*:}"
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            printf "\033[0;32m✓ %s (:%s) healthy\033[0m\n" "$name" "$port"
        else
            printf "\033[0;31m✗ %s (:%s) not responding\033[0m\n" "$name" "$port"
            all_healthy=false
        fi
    done

    if [ "$all_healthy" = false ]; then
        printf "\n\033[0;31mError: Some services not healthy. Start them first:\033[0m\n"
        printf "  just run-embeddings &\n"
        printf "  just run-search &\n"
        printf "  just run-geometric &\n"
        printf "  just run-gateway &\n"
        exit 1
    fi

    echo ""
    printf "\033[0;34m=== Building Index from {{testdata}}/objects ===\033[0m\n"
    uv run python build_index.py --objects "{{testdata}}/objects"

    echo ""
    printf "\033[0;34m=== Running Evaluation ===\033[0m\n"
    uv run python evaluate.py --testdata "{{testdata}}"

    echo ""
    printf "\033[0;32m✓ Evaluation complete. Report saved to reports/evaluation/\033[0m\n"
    echo ""

# Run evaluation only (assumes index is already built)
evaluate-only testdata="data/testdata":
    @echo ""
    @printf "\033[0;34m=== Running Evaluation ===\033[0m\n"
    @uv run python evaluate.py --testdata "{{testdata}}"
    @echo ""
```

---

## Running the Evaluation

### Prerequisites

1. **Start all services** (in separate terminals or background):

   ```bash
   # Option 1: Local development
   just run-embeddings &
   just run-search &
   just run-geometric &
   just run-gateway &

   # Option 2: Docker
   just docker-up
   ```

2. **Prepare test data** in `./data/testdata/`:

   ```
   data/testdata/
   ├── objects/       # Reference artwork images
   ├── pictures/      # Visitor test photos
   └── labels.csv     # Ground truth
   ```

### Run Evaluation

```bash
# Full pipeline (builds index + evaluates)
just evaluate

# With custom test data location
just evaluate testdata=./my-test-data

# Evaluation only (assumes index already built)
just evaluate-only
```

### Output

Reports are saved to `./reports/evaluation/`:

- `evaluation_report.md` — Human-readable markdown report
- `evaluation_results.json` — Machine-readable metrics for CI/automation

---

## Report Format

### Example Output

```markdown
# Artwork Matcher Evaluation Report

**Generated**: 2026-02-02 14:30:00 UTC
**Test Dataset**: data/testdata
**Pictures Evaluated**: 17
**Objects in Index**: 28

---

## Executive Summary

| Mode | Precision@1 | Recall@1 | F1-Score | MRR | Mean Latency |
|------|-------------|----------|----------|-----|--------------|
| Embedding Only | 85.0% | 85.0% | 0.850 | 0.892 | 52ms |
| With Geometric | 94.0% | 94.0% | 0.940 | 0.956 | 198ms |
| **Improvement** | **+9.0%** | **+9.0%** | **+0.090** | **+0.064** | **+146ms** |

**Recommendation**: Geometric verification improves precision by 9.0% at a cost of 146ms additional latency.
```

---

## Interpreting Results

### When Geometric Verification Helps

Geometric verification typically improves results when:

| Scenario | Why it helps |
|----------|--------------|
| Similar artworks by same artist | Embedding similarity high, but local features differ |
| Different editions/prints | Global appearance similar, details different |
| Reproductions vs originals | Color/texture differences detected by local features |
| Unusual photo angles | Geometric transform validates spatial consistency |

### When to Disable Geometric Verification

Consider disabling if:

| Condition | Reason |
|-----------|--------|
| Precision improvement < 3% | Cost doesn't justify benefit |
| Latency critical (< 100ms target) | Geometric adds 100-200ms |
| High-quality, well-framed photos | Embedding alone is sufficient |
| Small reference database | Few similar artworks to confuse |

### Metric Interpretation Guide

| Metric | Good | Excellent |
|--------|------|-----------|
| Precision@1 | > 0.85 | > 0.95 |
| Recall@1 | > 0.80 | > 0.90 |
| MRR | > 0.85 | > 0.95 |
| Hit@3 | > 0.90 | > 0.98 |

---

## Troubleshooting

### "Service not responding" Errors

Check that all services are running:

```bash
just status  # From project root

# Or manually:
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
```

### Low Recall Scores

Possible causes:
- Test pictures don't match any indexed objects
- Index was built from different objects
- Similarity threshold too high

Solutions:
- Verify test data matches expectations
- Rebuild index from correct objects directory
- Lower threshold in evaluation config

### Slow Evaluation

- Use `--k 5` for fewer candidates (faster)
- Check if GPU is available for embeddings service
- Reduce number of test pictures for quick testing

### Index Build Failures

- Check embeddings service is responding
- Verify image files are valid (JPEG/PNG)
- Check disk space for index storage
