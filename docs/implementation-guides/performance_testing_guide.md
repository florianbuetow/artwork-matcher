# Performance Testing Guide

This guide documents the performance testing framework used across artwork-matcher microservices. The **embeddings service** serves as the reference implementation.

## Design Principles

1. **Measurement isolation** — Pre-generate test data to exclude I/O from latency measurements
2. **Statistical significance** — 30 iterations per scenario provides reliable statistics
3. **Reproducibility** — Same pre-generated data for all iterations ensures consistent results
4. **Real components** — Use actual service with dependencies loaded (no mocking)
5. **Automated reporting** — Generate markdown reports with analysis after each run

---

## Test Categories

| Category | Purpose | Metrics |
|----------|---------|---------|
| **Input Variation** | How input characteristics affect latency | Mean, Std, P50, P95, P99 |
| **Sequential Throughput** | Baseline single-threaded capacity | Requests/second |
| **Concurrent Throughput** | Scalability under parallel load | Requests/second, contention detection |

---

## Directory Structure

Every service should follow this layout:

```
services/<service_name>/
├── tests/
│   └── performance/
│       ├── __init__.py
│       ├── conftest.py          # Fixtures and configuration constants
│       ├── test_performance.py  # Test classes
│       ├── metrics.py           # Metrics collection (REUSABLE)
│       └── generators.py        # Service-specific test data generation
└── reports/
    └── performance/
        └── <service_name>_performance.md  # Generated report
```

**Reference:** [`services/embeddings/tests/performance/`](../../services/embeddings/tests/performance/)

---

## Configuration Constants

Define these constants in `conftest.py`:

```python
# Standard constants (same across all services)
ITERATIONS_PER_SCENARIO = 30        # Statistical significance
THROUGHPUT_REQUESTS = 250           # Sustained load testing
CONCURRENCY_LEVELS = [2, 4, 8, 16]  # Thread counts for concurrent tests

# Service-specific constants (CUSTOMIZE THESE)
# Examples:
# - Embeddings: DIMENSION_SIZES = [100, 512, 1024, 2048, 4096]
# - Search: VECTOR_COUNTS = [10, 100, 1000, 10000]
# - Gateway: PIPELINE_CONFIGS = ["embedding_only", "with_search", "full"]
INPUT_VARIATION_PARAMS = [...]

# Report output path
SERVICE_ROOT = Path(__file__).parent.parent.parent
REPORTS_DIR = SERVICE_ROOT / "reports" / "performance"
REPORT_PATH = REPORTS_DIR / "<service_name>_performance.md"
```

---

## Metrics Module (Reusable)

The `metrics.py` module is **reusable across all services**. Copy it from the embeddings service and customize only the `PerformanceReport` class for service-specific test categories.

### `LatencyMetrics`

Collects and analyzes latency samples with statistical summaries.

```python
from dataclasses import dataclass, field

import numpy as np


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

### `ThroughputMetrics`

Calculates sustained request handling capacity.

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

### `PerformanceReport`

Aggregates results and generates markdown report. **Customize this class** for service-specific test categories.

```python
from datetime import UTC, datetime


@dataclass
class PerformanceReport:
    """
    Collects performance test results and generates markdown report.

    Customize the add_*_result methods and _generate_*_section methods
    for your service's specific test categories.
    """

    # Results storage - customize field names for your service
    input_variation_results: dict[str, dict[str, float]] = field(default_factory=dict)
    throughput_results: dict[str, dict[str, float]] = field(default_factory=dict)

    def add_input_variation_result(
        self, key: str, latency: LatencyMetrics, **extra: float
    ) -> None:
        """Record input variation test result."""
        self.input_variation_results[key] = {
            **latency.to_dict(),
            **extra,
        }

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
            "# <Service Name> Performance Test Report",
            "",
            f"**Generated:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
        ]

        # Key findings summary
        lines.extend(self._generate_key_findings())

        # Test configuration
        lines.extend(self._generate_config_section())

        # Input variation tests
        if self.input_variation_results:
            lines.extend(self._generate_input_variation_section())

        # Throughput tests
        if self.throughput_results:
            lines.extend(self._generate_throughput_section())

        lines.extend([
            "---",
            "",
            "*Report generated by performance tests.*",
        ])

        return "\n".join(lines)

    def _generate_key_findings(self) -> list[str]:
        """Generate key findings summary. Customize for your service."""
        lines = ["## Key Findings", ""]
        findings = []

        # Add service-specific findings based on collected results
        # Example: Compare smallest vs largest input variation
        if self.input_variation_results:
            keys = list(self.input_variation_results.keys())
            if len(keys) >= 2:
                first = self.input_variation_results[keys[0]]
                last = self.input_variation_results[keys[-1]]
                ratio = last["mean"] / first["mean"] if first["mean"] > 0 else 0
                findings.append(
                    f"- **Input variation impact**: {keys[-1]} takes "
                    f"{ratio:.1f}x longer than {keys[0]}"
                )

        # Throughput findings
        if "sequential" in self.throughput_results:
            seq = self.throughput_results["sequential"]
            findings.append(
                f"- **Sequential throughput**: {seq['requests_per_second']:.1f} req/s"
            )

        if findings:
            lines.extend(findings)
        else:
            lines.append("*No test results available.*")

        lines.extend(["", ""])
        return lines

    def _generate_config_section(self) -> list[str]:
        """Generate test configuration section."""
        return [
            "## Test Configuration",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            "| Iterations per test | 30 |",
            "| Throughput requests | 250 |",
            "| Concurrency levels | 2, 4, 8, 16 |",
            "",
        ]

    def _generate_input_variation_section(self) -> list[str]:
        """Generate input variation results section. Customize for your service."""
        lines = [
            "## Input Variation Tests",
            "",
            "| Input | Mean | Std | P50 | P95 | P99 |",
            "|-------|------|-----|-----|-----|-----|",
        ]

        for key, metrics in self.input_variation_results.items():
            lines.append(
                f"| {key} | "
                f"{metrics['mean']:.1f} ms | {metrics['std']:.1f} ms | "
                f"{metrics['p50']:.1f} ms | {metrics['p95']:.1f} ms | "
                f"{metrics['p99']:.1f} ms |"
            )

        lines.extend(["", ""])
        return lines

    def _generate_throughput_section(self) -> list[str]:
        """Generate throughput results section."""
        lines = [
            "## Throughput Tests",
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

        # Analysis
        lines.extend(["", "**Analysis:**", ""])

        if "sequential" in self.throughput_results:
            seq_rps = self.throughput_results["sequential"]["requests_per_second"]
            concurrent = {k: v for k, v in self.throughput_results.items() if k != "sequential"}

            for name, metrics in concurrent.items():
                speedup = metrics["requests_per_second"] / seq_rps if seq_rps > 0 else 0
                if speedup > 1.1:
                    lines.append(f"- {name}: {speedup:.2f}x speedup over sequential")
                elif speedup < 0.9:
                    lines.append(f"- {name}: **contention detected** ({speedup:.2f}x of sequential)")
                else:
                    lines.append(f"- {name}: no significant speedup (likely CPU-bound)")

        lines.extend(["", ""])
        return lines
```

---

## Test Data Generators (Service-Specific)

Create `generators.py` with functions to generate test inputs. **This is service-specific** — each service has different input types.

### Pattern

```python
"""
Test data generators for <service_name> performance testing.

Provides functions to generate test inputs with controlled properties.
"""

def create_test_input(param: <ParamType>) -> <InputType>:
    """
    Generate test input with specific characteristics.

    Args:
        param: Parameter controlling input characteristics

    Returns:
        Test input ready for API request
    """
    ...

def get_input_size(input_data: <InputType>) -> float:
    """
    Calculate size/complexity of input for reporting.

    Args:
        input_data: The test input

    Returns:
        Size metric (KB, count, etc.)
    """
    ...
```

### Example: Embeddings Service (Images)

```python
import base64
from io import BytesIO

import numpy as np
from PIL import Image


def create_noise_image_base64(width: int, height: int, quality: int = 85) -> str:
    """
    Generate a JPEG image with random noise as base64.

    Random noise creates images that don't compress well,
    resulting in larger file sizes compared to solid colors.
    """
    noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    image = Image.fromarray(noise, mode="RGB")

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def get_image_size_kb(image_base64: str) -> float:
    """Calculate the size of a base64-encoded image in kilobytes."""
    image_bytes = base64.b64decode(image_base64)
    return len(image_bytes) / 1024
```

### Example: Search Service (Vectors)

```python
import numpy as np


def create_random_embedding(dimension: int) -> list[float]:
    """Generate a normalized random embedding vector."""
    vec = np.random.randn(dimension).astype(np.float32)
    vec = vec / np.linalg.norm(vec)  # Normalize
    return vec.tolist()


def create_index_with_vectors(count: int, dimension: int) -> list[dict]:
    """Generate test index data with random vectors."""
    return [
        {
            "object_id": f"object_{i:04d}",
            "embedding": create_random_embedding(dimension),
            "metadata": {"name": f"Test Object {i}"},
        }
        for i in range(count)
    ]
```

---

## Fixtures (conftest.py)

### Session-Scoped Report Fixture

```python
import pytest
from pathlib import Path
from typing import Iterator

from .metrics import PerformanceReport


SERVICE_ROOT = Path(__file__).parent.parent.parent
REPORTS_DIR = SERVICE_ROOT / "reports" / "performance"
REPORT_PATH = REPORTS_DIR / "<service_name>_performance.md"


@pytest.fixture(scope="session")
def performance_report() -> Iterator[PerformanceReport]:
    """
    Session-scoped fixture that collects test results and writes report.

    The report is written after all tests complete.
    """
    report = PerformanceReport()
    yield report

    # Write report after all tests complete
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report.generate_markdown())
    print(f"\n\nPerformance report written to: {REPORT_PATH}")
```

### Module-Scoped Client Fixture

```python
from fastapi.testclient import TestClient

from <service_name>.app import create_app
from <service_name>.config import clear_settings_cache
from <service_name>.core.state import reset_app_state


@pytest.fixture(scope="module")
def performance_client() -> Iterator[TestClient]:
    """
    Create test client with the real application.

    Uses context manager to trigger lifespan events (dependency loading).
    This client is shared across all tests in the module to avoid
    loading dependencies multiple times.
    """
    clear_settings_cache()
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
```

### Module-Scoped Pre-generated Data Fixture

```python
from .generators import create_test_input, get_input_size


# Service-specific parameters
INPUT_VARIATION_PARAMS = [...]  # Customize for your service


@pytest.fixture(scope="module")
def pregenerated_data() -> dict[str, <InputType>]:
    """
    Pre-generate all test data before any tests run.

    This ensures data generation time is NOT included in latency
    measurements. All tests use identical pre-generated data.

    Returns:
        Dictionary mapping data key to test input.
    """
    data: dict[str, <InputType>] = {}

    print("\nPre-generating test data...")
    for param in INPUT_VARIATION_PARAMS:
        key = f"variation_{param}"
        data[key] = create_test_input(param)
        size = get_input_size(data[key])
        print(f"  {key}: {size:.1f} <unit>")

    # Throughput test data (single standard input)
    data["throughput"] = create_test_input(DEFAULT_PARAM)

    print(f"\nTotal pre-generated inputs: {len(data)}")
    return data
```

---

## Test Classes

### Input Variation Latency Tests

```python
import time

import pytest

from .conftest import INPUT_VARIATION_PARAMS, ITERATIONS_PER_SCENARIO
from .generators import get_input_size
from .metrics import LatencyMetrics, PerformanceReport


@pytest.mark.slow
@pytest.mark.performance
class TestInputVariationLatency:
    """Test latency across different input characteristics."""

    @pytest.mark.parametrize(
        "param",
        INPUT_VARIATION_PARAMS,
        ids=[f"param_{p}" for p in INPUT_VARIATION_PARAMS],
    )
    def test_input_variation_latency(
        self,
        client: TestClient,
        pregenerated_data: dict[str, <InputType>],
        performance_report: PerformanceReport,
        param: <ParamType>,
    ) -> None:
        """
        Measure latency for different input characteristics.

        Args:
            client: Test client with real dependencies loaded
            pregenerated_data: Pre-generated test inputs
            performance_report: Report collector
            param: Input variation parameter
        """
        data_key = f"variation_{param}"
        input_data = pregenerated_data[data_key]
        input_size = get_input_size(input_data)

        metrics = LatencyMetrics()

        for i in range(ITERATIONS_PER_SCENARIO):
            request_id = f"{data_key}_iter{i}"
            start = time.perf_counter()
            response = client.post("/endpoint", json={"data": input_data, "id": request_id})
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response.status_code == 200, f"Request failed: {response.text}"
            metrics.add(elapsed_ms)

        # Record to report
        performance_report.add_input_variation_result(
            key=str(param),
            latency=metrics,
            input_size=input_size,
        )

        # Print results
        print(f"\n{'=' * 60}")
        print(f"Input Variation Test: {param} ({input_size:.1f} <unit>)")
        print(f"{'=' * 60}")
        print(metrics.summary())
```

### Throughput Tests

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

from .conftest import CONCURRENCY_LEVELS, THROUGHPUT_REQUESTS
from .metrics import ThroughputMetrics


@pytest.mark.slow
@pytest.mark.performance
class TestThroughput:
    """Test request throughput (sequential and concurrent)."""

    def test_sequential_throughput(
        self,
        client: TestClient,
        pregenerated_data: dict[str, <InputType>],
        performance_report: PerformanceReport,
    ) -> None:
        """
        Measure throughput for sequential requests.

        Sends requests one after another and measures total time.
        """
        input_data = pregenerated_data["throughput"]

        latency_metrics = LatencyMetrics()

        start_total = time.perf_counter()
        for i in range(THROUGHPUT_REQUESTS):
            request_id = f"throughput_seq_{i}"
            start = time.perf_counter()
            response = client.post("/endpoint", json={"data": input_data, "id": request_id})
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
        pregenerated_data: dict[str, <InputType>],
        performance_report: PerformanceReport,
        workers: int,
    ) -> None:
        """
        Measure throughput for concurrent requests.

        Sends requests in parallel using a thread pool.

        Args:
            client: Test client with real dependencies loaded
            pregenerated_data: Pre-generated test inputs
            performance_report: Report collector
            workers: Number of concurrent workers
        """
        input_data = pregenerated_data["throughput"]
        requests_per_worker = THROUGHPUT_REQUESTS // workers

        latency_metrics = LatencyMetrics()
        errors: list[str] = []

        def make_request(request_id: int) -> float:
            """Make a single request and return latency in ms."""
            req_id = f"throughput_c{workers}_{request_id}"
            start = time.perf_counter()
            response = client.post("/endpoint", json={"data": input_data, "id": req_id})
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

---

## Pytest Configuration

Register markers in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-v --tb=short --strict-markers"
markers = [
    "unit: Unit tests (fast, isolated, no external dependencies)",
    "integration: Integration tests (require running services)",
    "slow: Tests that take more than 1 second",
    "performance: Performance benchmark tests (require dependencies, slow)",
]
```

**Key flags:**
- `--strict-markers` — Enforces all markers must be declared (prevents typos)
- `-v` — Verbose output showing test names
- `--tb=short` — Minimal tracebacks on failures

---

## Justfile Recipes

Add to service `justfile`:

```makefile
# Run performance tests (requires dependencies, slow)
test-performance:
    @echo ""
    @printf "\033[0;34m=== Running Performance Tests ===\033[0m\n"
    @uv run pytest tests/performance -m performance -v -s
    @echo ""
```

**Key flags:**
- `-m performance` — Only run tests marked with `@pytest.mark.performance`
- `-v` — Verbose output
- `-s` — Show print statements (captures console output from tests)

### Running Tests

```bash
# From service directory
cd services/<service_name>

# Run all performance tests
just test-performance

# Run specific test class
just test-performance -- -k "TestInputVariationLatency"

# Run specific scenario
just test-performance -- -k "test_input_variation_latency[param_100]"
```

---

## Report Output

After tests complete, a markdown report is generated at:

```
services/<service_name>/reports/performance/<service_name>_performance.md
```

### Report Structure

1. **Header** — Service name and generation timestamp (UTC)
2. **Key Findings** — Summary bullet points of most important metrics
3. **Test Configuration** — Parameters table
4. **Input Variation Results** — Table with Mean/Std/P50/P95/P99
5. **Throughput Results** — Table with Requests/Duration/RPS/Mean/P99
6. **Analysis** — Dynamic insights (speedup calculations, contention detection)

### Interpreting Results

**Latency Analysis:**
- Low P99/P50 ratio (< 2x) indicates consistent performance
- P95 close to mean indicates few outliers
- Tight histogram distribution is good

**Warning signs:**
- High P99 with low mean indicates occasional slow requests
- Bimodal histogram may indicate GC pauses or resource contention

**Throughput Analysis:**
- Sequential throughput is the baseline
- Concurrent should scale with workers (up to hardware limits)
- Speedup < 1.0 indicates contention issues

---

## Implementation Checklist

When adding performance tests to a new service:

### Directory Structure
- [ ] Create `tests/performance/` directory
- [ ] Create `tests/performance/__init__.py`
- [ ] Create `reports/performance/` directory

### Metrics Module
- [ ] Copy `metrics.py` from embeddings service
- [ ] Customize `PerformanceReport` for service-specific test categories
- [ ] Update report title and section generators

### Generators
- [ ] Create `generators.py` with service-specific functions
- [ ] Implement `create_test_input()` for your input type
- [ ] Implement `get_input_size()` for reporting

### Fixtures
- [ ] Create `conftest.py` with configuration constants
- [ ] Define `INPUT_VARIATION_PARAMS` for your service
- [ ] Implement `performance_report` fixture (session-scoped)
- [ ] Implement `performance_client` fixture (module-scoped)
- [ ] Implement `pregenerated_data` fixture (module-scoped)
- [ ] Implement `client` fixture alias

### Test Classes
- [ ] Create `test_performance.py`
- [ ] Implement `TestInputVariationLatency` class
- [ ] Implement `TestThroughput` class with sequential and concurrent tests
- [ ] Add `@pytest.mark.slow` and `@pytest.mark.performance` markers

### Configuration
- [ ] Add pytest markers to `pyproject.toml`
- [ ] Add `test-performance` recipe to `justfile`

### Verification
- [ ] Run `just test-performance` and verify all tests pass
- [ ] Check generated report at `reports/performance/`
- [ ] Verify report contains all expected sections

---

## Summary

| Aspect | Choice | Why |
|--------|--------|-----|
| Iterations | 30 per scenario | Statistical significance without excessive runtime |
| Throughput | 250 requests | Sustained load measurement |
| Concurrency | 2, 4, 8, 16 workers | Covers light to heavy parallelism |
| Data | Pre-generated | Excludes I/O from latency measurements |
| Client | Module-scoped | Avoids reloading dependencies |
| Report | Session-scoped | Single report for entire test run |
| Timing | `time.perf_counter()` | High-precision wall-clock timing |
| Markers | `slow` + `performance` | Enables filtering in CI |

**Reference implementation:** [`services/embeddings/tests/performance/`](../../services/embeddings/tests/performance/)
