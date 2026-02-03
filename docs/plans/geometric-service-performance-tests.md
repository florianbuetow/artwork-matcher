# Geometric Service Performance Tests Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add comprehensive performance tests to the geometric service that match the patterns and metrics of existing services (embeddings, search, gateway).

**Architecture:** Create 4 files in `tests/performance/` following the established pattern: metrics collection classes, test data generators, pytest fixtures, and parametrized test classes measuring latency and throughput.

**Tech Stack:** pytest, numpy, PIL, FastAPI TestClient, time.perf_counter()

---

## Task 1: Create Performance Test Directory Structure

**Files:**
- Create: `services/geometric/tests/performance/__init__.py`

**Step 1: Create the directory and init file**

```python
"""Performance tests for the geometric service."""
```

**Step 2: Verify directory exists**

Run: `ls -la services/geometric/tests/performance/`
Expected: Directory exists with `__init__.py`

**Step 3: Commit**

```bash
git add services/geometric/tests/performance/__init__.py
git commit -m "feat(geometric): add performance tests directory"
```

---

## Task 2: Create Metrics Module

**Files:**
- Create: `services/geometric/tests/performance/metrics.py`

**Step 1: Write the LatencyMetrics class**

```python
"""
Metrics collection utilities for performance testing.

Provides the LatencyMetrics, ThroughputMetrics, and PerformanceReport classes
for collecting, analyzing, and reporting performance data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class LatencyMetrics:
    """
    Collects and analyzes latency samples.

    Provides statistical summaries including mean, min, max,
    percentiles, and histogram distribution.
    """

    samples: list[float] = field(default_factory=list)

    def add(self, sample: float) -> None:
        """Add a latency sample in milliseconds."""
        self.samples.append(sample)

    @property
    def count(self) -> int:
        """Number of samples collected."""
        return len(self.samples)

    @property
    def mean(self) -> float:
        """Average latency in milliseconds."""
        if not self.samples:
            return 0.0
        return float(np.mean(self.samples))

    @property
    def min(self) -> float:
        """Minimum latency in milliseconds."""
        if not self.samples:
            return 0.0
        return float(np.min(self.samples))

    @property
    def max(self) -> float:
        """Maximum latency in milliseconds."""
        if not self.samples:
            return 0.0
        return float(np.max(self.samples))

    @property
    def std(self) -> float:
        """Standard deviation in milliseconds."""
        if not self.samples:
            return 0.0
        return float(np.std(self.samples))

    def percentile(self, p: int) -> float:
        """
        Calculate percentile value.

        Args:
            p: Percentile (0-100), e.g., 50 for median, 95 for p95

        Returns:
            Latency value at the given percentile in milliseconds
        """
        if not self.samples:
            return 0.0
        return float(np.percentile(self.samples, p))

    @property
    def p50(self) -> float:
        """Median (50th percentile) in milliseconds."""
        return self.percentile(50)

    @property
    def p95(self) -> float:
        """95th percentile in milliseconds."""
        return self.percentile(95)

    @property
    def p99(self) -> float:
        """99th percentile in milliseconds."""
        return self.percentile(99)

    def histogram(self, bins: int = 10) -> dict[str, int]:
        """
        Generate histogram of latency distribution.

        Args:
            bins: Number of bins for the histogram

        Returns:
            Dictionary mapping bin ranges (as strings) to counts
        """
        if not self.samples:
            return {}

        counts: NDArray[np.intp]
        bin_edges: NDArray[np.floating[object]]
        counts, bin_edges = np.histogram(self.samples, bins=bins)

        result: dict[str, int] = {}
        for i, count in enumerate(counts):
            low = bin_edges[i]
            high = bin_edges[i + 1]
            key = f"{low:.1f}-{high:.1f}ms"
            result[key] = int(count)

        return result

    def summary(self) -> str:
        """
        Generate formatted summary string for console output.

        Returns:
            Multi-line string with all metrics
        """
        if not self.samples:
            return "No samples collected"

        lines = [
            f"Samples: {self.count}",
            f"Mean:    {self.mean:.2f} ms",
            f"Min:     {self.min:.2f} ms",
            f"Max:     {self.max:.2f} ms",
            f"Std:     {self.std:.2f} ms",
            f"P50:     {self.p50:.2f} ms",
            f"P95:     {self.p95:.2f} ms",
            f"P99:     {self.p99:.2f} ms",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, float | int]:
        """
        Convert metrics to dictionary format.

        Returns:
            Dictionary with all metric values
        """
        return {
            "count": self.count,
            "mean": self.mean,
            "min": self.min,
            "max": self.max,
            "std": self.std,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
        }
```

**Step 2: Add ThroughputMetrics class**

Append to the same file:

```python
@dataclass
class ThroughputMetrics:
    """
    Collects and analyzes throughput measurements.

    Tracks total requests, duration, and calculates requests per second.
    """

    total_requests: int = 0
    total_duration_seconds: float = 0.0

    @property
    def requests_per_second(self) -> float:
        """Calculate throughput in requests per second."""
        if self.total_duration_seconds <= 0:
            return 0.0
        return self.total_requests / self.total_duration_seconds

    def summary(self) -> str:
        """Generate formatted summary string."""
        return (
            f"Requests: {self.total_requests}\n"
            f"Duration: {self.total_duration_seconds:.2f} s\n"
            f"Throughput: {self.requests_per_second:.2f} req/s"
        )

    def to_dict(self) -> dict[str, float | int]:
        """Convert metrics to dictionary format."""
        return {
            "total_requests": self.total_requests,
            "total_duration_seconds": self.total_duration_seconds,
            "requests_per_second": self.requests_per_second,
        }
```

**Step 3: Add PerformanceReport class**

Append to the same file:

```python
@dataclass
class PerformanceReport:
    """Collects geometric service performance test results and generates markdown report."""

    dimension_results: dict[str, dict[str, float]] = field(default_factory=dict)
    feature_count_results: dict[str, dict[str, float]] = field(default_factory=dict)
    match_results: dict[str, dict[str, float]] = field(default_factory=dict)
    throughput_results: dict[str, dict[str, float]] = field(default_factory=dict)
    image_sizes: dict[str, float] = field(default_factory=dict)

    def add_dimension_result(
        self, size: int, latency: LatencyMetrics, image_size_kb: float
    ) -> None:
        """Record dimension test result."""
        key = f"{size}x{size}"
        self.dimension_results[key] = latency.to_dict()
        self.image_sizes[f"dim_{key}"] = image_size_kb

    def add_feature_count_result(
        self, max_features: int, latency: LatencyMetrics, actual_features: float
    ) -> None:
        """Record feature count test result."""
        key = str(max_features)
        self.feature_count_results[key] = {
            **latency.to_dict(),
            "max_features": max_features,
            "avg_actual_features": actual_features,
        }

    def add_match_result(self, name: str, latency: LatencyMetrics) -> None:
        """Record match test result."""
        self.match_results[name] = latency.to_dict()

    def add_throughput_result(
        self,
        name: str,
        latency: LatencyMetrics,
        throughput: ThroughputMetrics,
    ) -> None:
        """Record throughput test result."""
        self.throughput_results[name] = {
            **latency.to_dict(),
            **throughput.to_dict(),
        }

    def generate_markdown(self) -> str:
        """Generate markdown report content with insights."""
        lines = [
            "# Geometric Service Performance Test Report",
            "",
        ]

        lines.extend(self._generate_key_findings())
        lines.extend(self._generate_test_configuration())

        if self.dimension_results:
            lines.extend(self._generate_dimension_section())

        if self.feature_count_results:
            lines.extend(self._generate_feature_count_section())

        if self.match_results:
            lines.extend(self._generate_match_section())

        if self.throughput_results:
            lines.extend(self._generate_throughput_section())

        lines.extend(
            [
                "---",
                "",
                "*Report generated by geometric service performance tests.*",
            ]
        )

        return "\n".join(lines)

    def _generate_key_findings(self) -> list[str]:
        """Generate key findings summary."""
        lines = ["## Key Findings", ""]
        findings = []

        if self.dimension_results:
            dims = list(self.dimension_results.keys())
            if len(dims) >= 2:
                smallest = self.dimension_results[dims[0]]
                largest = self.dimension_results[dims[-1]]
                ratio = largest["mean"] / smallest["mean"] if smallest["mean"] > 0 else 0
                findings.append(
                    f"- **Image dimension impact**: {dims[-1]} images take "
                    f"{ratio:.1f}x longer than {dims[0]} images "
                    f"({largest['mean']:.1f} ms vs {smallest['mean']:.1f} ms)"
                )

        if self.feature_count_results:
            counts = list(self.feature_count_results.values())
            if len(counts) >= 2:
                first = counts[0]
                last = counts[-1]
                ratio = last["mean"] / first["mean"] if first["mean"] > 0 else 0
                findings.append(
                    f"- **Feature count impact**: {last['max_features']:.0f} max features takes "
                    f"{ratio:.1f}x longer than {first['max_features']:.0f} max features"
                )

        if self.throughput_results:
            if "sequential" in self.throughput_results:
                seq = self.throughput_results["sequential"]
                findings.append(
                    f"- **Sequential throughput**: {seq['requests_per_second']:.1f} req/s"
                )

            concurrent = {k: v for k, v in self.throughput_results.items() if k != "sequential"}
            if concurrent:
                best = max(concurrent.items(), key=lambda x: x[1]["requests_per_second"])
                findings.append(
                    f"- **Best concurrent throughput**: "
                    f"{best[1]['requests_per_second']:.1f} req/s ({best[0]})"
                )

        if findings:
            lines.extend(findings)
        else:
            lines.append("*No test results available.*")

        lines.extend(["", ""])
        return lines

    def _generate_test_configuration(self) -> list[str]:
        """Generate test configuration section."""
        return [
            "## Test Configuration",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            "| Iterations per test | 30 |",
            "| Feature detector | ORB |",
            "| Matcher | BFMatcher (HAMMING) |",
            "| Verification | RANSAC homography |",
            "",
        ]

    def _generate_dimension_section(self) -> list[str]:
        """Generate dimension tests section with analysis."""
        lines = [
            "## Image Dimension Tests",
            "",
            "Measures ORB feature extraction time for different input image sizes.",
            "",
            "| Dimensions | File Size | Mean | Std | P50 | P95 | P99 |",
            "|------------|-----------|------|-----|-----|-----|-----|",
        ]

        for dim, metrics in self.dimension_results.items():
            img_key = f"dim_{dim}"
            img_size = self.image_sizes.get(img_key, 0)
            lines.append(
                f"| {dim} | {img_size:.0f} KB | "
                f"{metrics['mean']:.1f} ms | {metrics['std']:.1f} ms | "
                f"{metrics['p50']:.1f} ms | {metrics['p95']:.1f} ms | "
                f"{metrics['p99']:.1f} ms |"
            )

        lines.extend(["", "**Analysis:**", ""])
        dims = list(self.dimension_results.keys())
        if len(dims) >= 2:
            first = self.dimension_results[dims[0]]
            last = self.dimension_results[dims[-1]]
            overhead = last["mean"] - first["mean"]
            lines.append(f"- Processing larger images adds ~{overhead:.1f} ms overhead")

            avg_std = sum(m["std"] for m in self.dimension_results.values()) / len(dims)
            if avg_std < 5:
                lines.append("- Latency is **consistent** (low standard deviation)")
            else:
                lines.append(f"- Latency variance is notable (avg std: {avg_std:.1f} ms)")

        lines.extend(["", ""])
        return lines

    def _generate_feature_count_section(self) -> list[str]:
        """Generate feature count tests section with analysis."""
        lines = [
            "## Feature Count Tests",
            "",
            "Measures impact of max_features parameter on extraction time.",
            "",
            "| Max Features | Avg Actual | Mean | Std | P50 | P95 | P99 |",
            "|--------------|------------|------|-----|-----|-----|-----|",
        ]

        for _, metrics in self.feature_count_results.items():
            lines.append(
                f"| {metrics['max_features']:.0f} | {metrics['avg_actual_features']:.0f} | "
                f"{metrics['mean']:.1f} ms | {metrics['std']:.1f} ms | "
                f"{metrics['p50']:.1f} ms | {metrics['p95']:.1f} ms | "
                f"{metrics['p99']:.1f} ms |"
            )

        lines.extend(["", "**Analysis:**", ""])
        counts = list(self.feature_count_results.values())
        if len(counts) >= 2:
            ratio = counts[-1]["mean"] / counts[0]["mean"] if counts[0]["mean"] > 0 else 0
            if ratio < 1.5:
                lines.append(
                    "- Feature count has **minimal impact** on extraction time "
                    "(ORB is efficient at any scale)"
                )
            else:
                lines.append(
                    f"- Feature count has **moderate impact**: "
                    f"{ratio:.1f}x slowdown with {counts[-1]['max_features']:.0f} vs "
                    f"{counts[0]['max_features']:.0f} features"
                )

        lines.extend(["", ""])
        return lines

    def _generate_match_section(self) -> list[str]:
        """Generate match tests section with analysis."""
        lines = [
            "## Match Tests",
            "",
            "Measures geometric verification latency for different scenarios.",
            "",
            "| Scenario | Mean | Std | P50 | P95 | P99 |",
            "|----------|------|-----|-----|-----|-----|",
        ]

        for name, metrics in self.match_results.items():
            lines.append(
                f"| {name} | {metrics['mean']:.1f} ms | {metrics['std']:.1f} ms | "
                f"{metrics['p50']:.1f} ms | {metrics['p95']:.1f} ms | "
                f"{metrics['p99']:.1f} ms |"
            )

        lines.extend(["", "**Analysis:**", ""])

        if "extract_only" in self.match_results and "full_match" in self.match_results:
            extract = self.match_results["extract_only"]["mean"]
            full = self.match_results["full_match"]["mean"]
            matching_overhead = full - extract
            lines.append(
                f"- Feature matching + RANSAC verification adds ~{matching_overhead:.1f} ms "
                f"to extraction time"
            )

        lines.extend(["", ""])
        return lines

    def _generate_throughput_section(self) -> list[str]:
        """Generate throughput tests section with analysis."""
        lines = [
            "## Throughput Tests",
            "",
            "Measures sustained request handling capacity.",
            "",
            "| Test | Requests | Duration | Throughput | Mean | P99 |",
            "|------|----------|----------|------------|------|-----|",
        ]

        for name, metrics in self.throughput_results.items():
            lines.append(
                f"| {name} | {metrics['total_requests']:.0f} | "
                f"{metrics['total_duration_seconds']:.2f} s | "
                f"**{metrics['requests_per_second']:.1f} req/s** | "
                f"{metrics['mean']:.1f} ms | {metrics['p99']:.1f} ms |"
            )

        lines.extend(["", "**Analysis:**", ""])

        if "sequential" in self.throughput_results:
            seq = self.throughput_results["sequential"]
            lines.append(f"- Sequential baseline: {seq['requests_per_second']:.1f} req/s")

        concurrent = {k: v for k, v in self.throughput_results.items() if k != "sequential"}
        if concurrent and "sequential" in self.throughput_results:
            seq_rps = self.throughput_results["sequential"]["requests_per_second"]
            for name, metrics in concurrent.items():
                speedup = metrics["requests_per_second"] / seq_rps if seq_rps > 0 else 0
                if speedup > 1.1:
                    lines.append(f"- {name}: {speedup:.2f}x speedup over sequential")
                elif speedup < 0.9:
                    lines.append(
                        f"- {name}: **contention detected** ({speedup:.2f}x of sequential)"
                    )
                else:
                    lines.append(f"- {name}: no significant speedup (likely CPU-bound)")

        lines.extend(["", ""])
        return lines
```

**Step 4: Run linting to verify syntax**

Run: `cd services/geometric && uv run ruff check tests/performance/metrics.py`
Expected: No errors

**Step 5: Commit**

```bash
git add services/geometric/tests/performance/metrics.py
git commit -m "feat(geometric): add performance metrics module"
```

---

## Task 3: Create Test Data Generators

**Files:**
- Create: `services/geometric/tests/performance/generators.py`

**Step 1: Write the generators module**

```python
"""
Image generators for performance testing.

Provides functions to generate test images with controlled properties
for measuring ORB feature extraction and geometric matching performance.
"""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from PIL import Image


def create_noise_image_base64(
    width: int,
    height: int,
    quality: int = 85,
) -> str:
    """
    Generate a JPEG image with random noise as base64.

    Random noise creates images with many detectable features,
    ideal for testing ORB extraction performance.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        quality: JPEG quality (1-100), higher = larger file

    Returns:
        Base64-encoded JPEG image string
    """
    noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    image = Image.fromarray(noise, mode="RGB")

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def create_checkerboard_base64(
    width: int,
    height: int,
    block_size: int = 20,
) -> str:
    """
    Generate a checkerboard image with detectable corner features.

    Checkerboard patterns provide consistent corner features
    that are reliably detected by ORB.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        block_size: Size of checkerboard squares

    Returns:
        Base64-encoded PNG image string
    """
    img = np.zeros((height, width), dtype=np.uint8)
    for i in range(0, height, block_size * 2):
        for j in range(0, width, block_size * 2):
            img[i : i + block_size, j : j + block_size] = 255
            img[i + block_size : i + block_size * 2, j + block_size : j + block_size * 2] = 255

    pil_image = Image.fromarray(img)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def create_transformed_image_base64(
    base_image_b64: str,
    rotation_deg: float = 0,
    scale: float = 1.0,
) -> str:
    """
    Create a transformed version of an image for matching tests.

    Applies rotation and/or scale to simulate real-world image variations.

    Args:
        base_image_b64: Base64-encoded source image
        rotation_deg: Rotation angle in degrees
        scale: Scale factor (1.0 = no change)

    Returns:
        Base64-encoded transformed PNG image
    """
    image_bytes = base64.b64decode(base_image_b64)
    image = Image.open(BytesIO(image_bytes))

    if rotation_deg != 0:
        image = image.rotate(rotation_deg, expand=True, fillcolor=(128, 128, 128))

    if scale != 1.0:
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def get_image_size_kb(image_base64: str) -> float:
    """
    Calculate the size of a base64-encoded image in kilobytes.

    Args:
        image_base64: Base64-encoded image string

    Returns:
        Size in kilobytes
    """
    image_bytes = base64.b64decode(image_base64)
    return len(image_bytes) / 1024
```

**Step 2: Run linting to verify**

Run: `cd services/geometric && uv run ruff check tests/performance/generators.py`
Expected: No errors

**Step 3: Commit**

```bash
git add services/geometric/tests/performance/generators.py
git commit -m "feat(geometric): add performance test generators"
```

---

## Task 4: Create Fixtures Module

**Files:**
- Create: `services/geometric/tests/performance/conftest.py`

**Step 1: Write the conftest module**

```python
"""
Shared fixtures for performance tests.

Performance tests use the real application with actual ORB extraction
and RANSAC verification. Images are pre-generated to avoid contaminating
latency measurements.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from geometric_service.app import create_app
from geometric_service.config import clear_settings_cache
from geometric_service.core.state import reset_app_state

from .generators import (
    create_checkerboard_base64,
    create_noise_image_base64,
    create_transformed_image_base64,
    get_image_size_kb,
)
from .metrics import PerformanceReport

if TYPE_CHECKING:
    from collections.abc import Iterator


# Service root directory (services/geometric/)
SERVICE_ROOT = Path(__file__).parent.parent.parent

# Test configuration constants
ITERATIONS_PER_SCENARIO = 30
DIMENSION_SIZES = [100, 250, 500, 750, 1000, 1500, 2000]
FEATURE_COUNT_LEVELS = [100, 250, 500, 1000, 2000]
THROUGHPUT_REQUESTS = 250
CONCURRENCY_LEVELS = [2, 4, 8, 16]

# Report output path
REPORTS_DIR = SERVICE_ROOT / "reports" / "performance"
REPORT_PATH = REPORTS_DIR / "geometric_service_performance.md"


@pytest.fixture(scope="session")
def performance_report() -> Iterator[PerformanceReport]:
    """
    Session-scoped fixture that collects test results and writes report.

    The report is written to reports/performance/geometric_service_performance.md
    when all tests complete.
    """
    report = PerformanceReport()
    yield report

    # Write report after all tests complete
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report.generate_markdown())
    print(f"\n\nPerformance report written to: {REPORT_PATH}")


@pytest.fixture(scope="module")
def performance_client() -> Iterator[TestClient]:
    """
    Create test client with the real application.

    Uses context manager to trigger lifespan events.
    This client is shared across all tests in the module.
    """
    clear_settings_cache()
    reset_app_state()
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    # Clean up after all tests in this module
    clear_settings_cache()
    reset_app_state()


@pytest.fixture
def client(performance_client: TestClient) -> TestClient:
    """Alias for performance_client for test compatibility."""
    return performance_client


@pytest.fixture(scope="module")
def pregenerated_images() -> dict[str, tuple[str, float]]:
    """
    Pre-generate all test images before any tests run.

    This ensures image generation time is NOT included in latency
    measurements. All tests use identical pre-generated images.

    Returns:
        Dictionary mapping image key to (base64 string, size in KB) tuple.
    """
    images: dict[str, tuple[str, float]] = {}

    print("\n" + "=" * 60)
    print("Pre-generating test images...")
    print("=" * 60)

    # Dimension test images (noise for feature-rich content)
    print("\nDimension images:")
    for size in DIMENSION_SIZES:
        key = f"dim_{size}"
        image_b64 = create_noise_image_base64(size, size)
        image_size_kb = get_image_size_kb(image_b64)
        images[key] = (image_b64, image_size_kb)
        print(f"  {key}: {size}x{size} ({image_size_kb:.1f} KB)")

    # Feature count test image (large, consistent size)
    print("\nFeature count image:")
    feature_b64 = create_noise_image_base64(800, 800)
    feature_size = get_image_size_kb(feature_b64)
    images["feature_count"] = (feature_b64, feature_size)
    print(f"  feature_count: 800x800 ({feature_size:.1f} KB)")

    # Match test images (checkerboard for reliable matching)
    print("\nMatch test images:")
    query_b64 = create_checkerboard_base64(500, 500, block_size=25)
    query_size = get_image_size_kb(query_b64)
    images["match_query"] = (query_b64, query_size)
    print(f"  match_query: 500x500 checkerboard ({query_size:.1f} KB)")

    # Create transformed version for matching tests
    ref_b64 = create_transformed_image_base64(query_b64, rotation_deg=5, scale=0.95)
    ref_size = get_image_size_kb(ref_b64)
    images["match_reference"] = (ref_b64, ref_size)
    print(f"  match_reference: transformed ({ref_size:.1f} KB)")

    # Throughput test image (moderate size)
    print("\nThroughput image:")
    throughput_b64 = create_checkerboard_base64(400, 400, block_size=20)
    throughput_size = get_image_size_kb(throughput_b64)
    images["throughput"] = (throughput_b64, throughput_size)
    print(f"  throughput: 400x400 ({throughput_size:.1f} KB)")

    print(f"\nTotal pre-generated images: {len(images)}")
    print("=" * 60)
    return images
```

**Step 2: Run linting to verify**

Run: `cd services/geometric && uv run ruff check tests/performance/conftest.py`
Expected: No errors

**Step 3: Commit**

```bash
git add services/geometric/tests/performance/conftest.py
git commit -m "feat(geometric): add performance test fixtures"
```

---

## Task 5: Create Dimension Latency Tests

**Files:**
- Create: `services/geometric/tests/performance/test_performance.py`

**Step 1: Write the test file header and dimension tests**

```python
"""
Performance tests for the geometric service.

Tests measure latency and throughput under various conditions:
- Different image dimensions (ORB extraction scaling)
- Different max_features settings
- Match scenarios (extract only vs full match)
- Sequential and concurrent request patterns
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import pytest

from .conftest import (
    CONCURRENCY_LEVELS,
    DIMENSION_SIZES,
    FEATURE_COUNT_LEVELS,
    ITERATIONS_PER_SCENARIO,
    THROUGHPUT_REQUESTS,
)
from .metrics import LatencyMetrics, PerformanceReport, ThroughputMetrics

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


@pytest.mark.slow
@pytest.mark.performance
class TestDimensionLatency:
    """Test latency across different image dimensions."""

    @pytest.mark.parametrize(
        "size",
        DIMENSION_SIZES,
        ids=[f"{s}x{s}" for s in DIMENSION_SIZES],
    )
    def test_dimension_latency(
        self,
        client: TestClient,
        pregenerated_images: dict[str, tuple[str, float]],
        performance_report: PerformanceReport,
        size: int,
    ) -> None:
        """
        Measure extraction latency for different image dimensions.

        Args:
            client: Test client with real application
            pregenerated_images: Pre-generated test images
            performance_report: Report collector
            size: Image dimension (width and height)
        """
        image_key = f"dim_{size}"
        image_base64, image_size_kb = pregenerated_images[image_key]

        metrics = LatencyMetrics()

        for i in range(ITERATIONS_PER_SCENARIO):
            start = time.perf_counter()
            response = client.post(
                "/extract",
                json={"image": image_base64, "image_id": f"{image_key}_iter{i}"},
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response.status_code == 200, f"Request failed: {response.text}"
            metrics.add(elapsed_ms)

        # Record to report
        performance_report.add_dimension_result(size, metrics, image_size_kb)

        # Print results
        print(f"\n{'=' * 60}")
        print(f"Dimension Test: {size}x{size} ({image_size_kb:.0f} KB)")
        print(f"{'=' * 60}")
        print(metrics.summary())
```

**Step 2: Run the dimension test to verify it works**

Run: `cd services/geometric && uv run pytest tests/performance/test_performance.py::TestDimensionLatency::test_dimension_latency[100x100] -v -s`
Expected: PASS with latency output

**Step 3: Commit**

```bash
git add services/geometric/tests/performance/test_performance.py
git commit -m "feat(geometric): add dimension latency tests"
```

---

## Task 6: Add Feature Count Tests

**Files:**
- Modify: `services/geometric/tests/performance/test_performance.py`

**Step 1: Add feature count test class**

Append to test_performance.py:

```python
@pytest.mark.slow
@pytest.mark.performance
class TestFeatureCountLatency:
    """Test latency across different max_features settings."""

    @pytest.mark.parametrize(
        "max_features",
        FEATURE_COUNT_LEVELS,
        ids=[f"{f}_features" for f in FEATURE_COUNT_LEVELS],
    )
    def test_feature_count_latency(
        self,
        client: TestClient,
        pregenerated_images: dict[str, tuple[str, float]],
        performance_report: PerformanceReport,
        max_features: int,
    ) -> None:
        """
        Measure extraction latency for different max_features values.

        Args:
            client: Test client with real application
            pregenerated_images: Pre-generated test images
            performance_report: Report collector
            max_features: Maximum features to extract
        """
        image_base64, _ = pregenerated_images["feature_count"]

        metrics = LatencyMetrics()
        actual_features_sum = 0

        for i in range(ITERATIONS_PER_SCENARIO):
            start = time.perf_counter()
            response = client.post(
                "/extract",
                json={
                    "image": image_base64,
                    "image_id": f"features_{max_features}_iter{i}",
                    "max_features": max_features,
                },
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response.status_code == 200, f"Request failed: {response.text}"
            metrics.add(elapsed_ms)
            actual_features_sum += response.json()["num_features"]

        avg_features = actual_features_sum / ITERATIONS_PER_SCENARIO

        # Record to report
        performance_report.add_feature_count_result(max_features, metrics, avg_features)

        # Print results
        print(f"\n{'=' * 60}")
        print(f"Feature Count Test: max={max_features}, avg_actual={avg_features:.0f}")
        print(f"{'=' * 60}")
        print(metrics.summary())
```

**Step 2: Run a feature count test to verify**

Run: `cd services/geometric && uv run pytest tests/performance/test_performance.py::TestFeatureCountLatency::test_feature_count_latency[500_features] -v -s`
Expected: PASS with latency output

**Step 3: Commit**

```bash
git add services/geometric/tests/performance/test_performance.py
git commit -m "feat(geometric): add feature count latency tests"
```

---

## Task 7: Add Match Scenario Tests

**Files:**
- Modify: `services/geometric/tests/performance/test_performance.py`

**Step 1: Add match scenario test class**

Append to test_performance.py:

```python
@pytest.mark.slow
@pytest.mark.performance
class TestMatchLatency:
    """Test latency for different matching scenarios."""

    def test_extract_only_latency(
        self,
        client: TestClient,
        pregenerated_images: dict[str, tuple[str, float]],
        performance_report: PerformanceReport,
    ) -> None:
        """
        Measure baseline extraction latency (no matching).

        Establishes baseline for comparison with full match.
        """
        image_base64, _ = pregenerated_images["match_query"]
        metrics = LatencyMetrics()

        for i in range(ITERATIONS_PER_SCENARIO):
            start = time.perf_counter()
            response = client.post(
                "/extract",
                json={"image": image_base64, "image_id": f"extract_only_{i}"},
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response.status_code == 200, f"Request failed: {response.text}"
            metrics.add(elapsed_ms)

        performance_report.add_match_result("extract_only", metrics)

        print(f"\n{'=' * 60}")
        print("Match Test: extract_only (baseline)")
        print(f"{'=' * 60}")
        print(metrics.summary())

    def test_full_match_latency(
        self,
        client: TestClient,
        pregenerated_images: dict[str, tuple[str, float]],
        performance_report: PerformanceReport,
    ) -> None:
        """
        Measure full match latency (extract + match + RANSAC).

        Tests the complete geometric verification pipeline.
        """
        query_b64, _ = pregenerated_images["match_query"]
        ref_b64, _ = pregenerated_images["match_reference"]

        metrics = LatencyMetrics()

        for i in range(ITERATIONS_PER_SCENARIO):
            start = time.perf_counter()
            response = client.post(
                "/match",
                json={
                    "query_image": query_b64,
                    "reference_image": ref_b64,
                    "query_id": f"match_{i}",
                    "reference_id": "ref",
                },
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response.status_code == 200, f"Request failed: {response.text}"
            metrics.add(elapsed_ms)

        performance_report.add_match_result("full_match", metrics)

        print(f"\n{'=' * 60}")
        print("Match Test: full_match (extract + match + RANSAC)")
        print(f"{'=' * 60}")
        print(metrics.summary())
```

**Step 2: Run a match test to verify**

Run: `cd services/geometric && uv run pytest tests/performance/test_performance.py::TestMatchLatency::test_full_match_latency -v -s`
Expected: PASS with latency output

**Step 3: Commit**

```bash
git add services/geometric/tests/performance/test_performance.py
git commit -m "feat(geometric): add match scenario latency tests"
```

---

## Task 8: Add Throughput Tests

**Files:**
- Modify: `services/geometric/tests/performance/test_performance.py`

**Step 1: Add throughput test class**

Append to test_performance.py:

```python
@pytest.mark.slow
@pytest.mark.performance
class TestThroughput:
    """Test request throughput (sequential and concurrent)."""

    def test_sequential_throughput(
        self,
        client: TestClient,
        pregenerated_images: dict[str, tuple[str, float]],
        performance_report: PerformanceReport,
    ) -> None:
        """
        Measure throughput for sequential requests.

        Sends requests one after another and measures total time.
        """
        image_base64, _ = pregenerated_images["throughput"]

        latency_metrics = LatencyMetrics()

        start_total = time.perf_counter()
        for i in range(THROUGHPUT_REQUESTS):
            start = time.perf_counter()
            response = client.post(
                "/extract",
                json={"image": image_base64, "image_id": f"throughput_seq_{i}"},
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response.status_code == 200, f"Request failed: {response.text}"
            latency_metrics.add(elapsed_ms)

        total_duration = time.perf_counter() - start_total

        throughput = ThroughputMetrics(
            total_requests=THROUGHPUT_REQUESTS,
            total_duration_seconds=total_duration,
        )

        # Record to report
        performance_report.add_throughput_result("sequential", latency_metrics, throughput)

        # Print results
        print(f"\n{'=' * 60}")
        print("Sequential Throughput Test")
        print(f"{'=' * 60}")
        print(throughput.summary())
        print("\nLatency Stats:")
        print(latency_metrics.summary())

    @pytest.mark.parametrize(
        "workers",
        CONCURRENCY_LEVELS,
        ids=[f"{w}_workers" for w in CONCURRENCY_LEVELS],
    )
    def test_concurrent_throughput(
        self,
        client: TestClient,
        pregenerated_images: dict[str, tuple[str, float]],
        performance_report: PerformanceReport,
        workers: int,
    ) -> None:
        """
        Measure throughput for concurrent requests.

        Sends requests in parallel using a thread pool.

        Args:
            client: Test client with real application
            pregenerated_images: Pre-generated test images
            performance_report: Report collector
            workers: Number of concurrent workers
        """
        image_base64, _ = pregenerated_images["throughput"]
        requests_per_worker = THROUGHPUT_REQUESTS // workers

        latency_metrics = LatencyMetrics()
        errors: list[str] = []

        def make_request(request_id: int) -> float:
            """Make a single request and return latency in ms."""
            start = time.perf_counter()
            response = client.post(
                "/extract",
                json={"image": image_base64, "image_id": f"throughput_c{workers}_{request_id}"},
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            if response.status_code != 200:
                errors.append(f"Request failed: {response.text}")

            return elapsed_ms

        start_total = time.perf_counter()
        total_requests = requests_per_worker * workers

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(make_request, i) for i in range(total_requests)]

            for future in as_completed(futures):
                latency_ms = future.result()
                latency_metrics.add(latency_ms)

        total_duration = time.perf_counter() - start_total

        # Check for errors
        assert not errors, f"Some requests failed: {errors[:5]}"

        throughput = ThroughputMetrics(
            total_requests=total_requests,
            total_duration_seconds=total_duration,
        )

        # Record to report
        performance_report.add_throughput_result(
            f"concurrent_{workers}", latency_metrics, throughput
        )

        # Print results
        print(f"\n{'=' * 60}")
        print(f"Concurrent Throughput Test ({workers} workers)")
        print(f"{'=' * 60}")
        print(throughput.summary())
        print("\nLatency Stats:")
        print(latency_metrics.summary())
```

**Step 2: Run a throughput test to verify**

Run: `cd services/geometric && uv run pytest tests/performance/test_performance.py::TestThroughput::test_sequential_throughput -v -s`
Expected: PASS with throughput output

**Step 3: Commit**

```bash
git add services/geometric/tests/performance/test_performance.py
git commit -m "feat(geometric): add throughput tests"
```

---

## Task 9: Run Full Performance Test Suite

**Files:**
- None (verification only)

**Step 1: Run all performance tests**

Run: `cd services/geometric && uv run pytest tests/performance -m performance -v -s`
Expected: All tests PASS, report generated at `reports/performance/geometric_service_performance.md`

**Step 2: Verify report was generated**

Run: `cat services/geometric/reports/performance/geometric_service_performance.md`
Expected: Markdown report with Key Findings, Test Configuration, and all test sections

**Step 3: Run CI checks to ensure code quality**

Run: `cd services/geometric && just ci`
Expected: All CI checks pass

**Step 4: Commit any formatting fixes**

```bash
git add services/geometric/tests/performance/
git commit -m "style(geometric): fix formatting in performance tests"
```

---

## Task 10: Final Commit and Push

**Files:**
- None (git operations only)

**Step 1: Verify all files are committed**

Run: `git status`
Expected: Clean working directory (nothing to commit)

**Step 2: Push to remote**

Run: `git push`
Expected: Successfully pushed to remote

---

## Summary

This plan creates 4 files matching the established performance test pattern:

1. `tests/performance/__init__.py` - Package marker
2. `tests/performance/metrics.py` - LatencyMetrics, ThroughputMetrics, PerformanceReport
3. `tests/performance/generators.py` - Image generators for test data
4. `tests/performance/conftest.py` - Fixtures with exact constants (30 iterations, 250 requests, [2,4,8,16] concurrency)
5. `tests/performance/test_performance.py` - Test classes with @pytest.mark.slow and @pytest.mark.performance

Test categories:
- **TestDimensionLatency**: 7 sizes (100-2000px)
- **TestFeatureCountLatency**: 5 max_features levels (100-2000)
- **TestMatchLatency**: extract_only and full_match scenarios
- **TestThroughput**: sequential + 4 concurrency levels

Report output: `reports/performance/geometric_service_performance.md`
