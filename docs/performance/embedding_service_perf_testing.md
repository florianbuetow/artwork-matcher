# Embeddings Service Performance Testing

This document describes the performance testing strategy for the embeddings service.

## Overview

The embeddings service extracts DINOv2 embeddings from images. Performance tests measure latency and throughput under various conditions to ensure the service meets requirements and to detect regressions.

## Metrics

### Latency Metrics (per-request)

| Metric | Description |
|--------|-------------|
| Mean | Average processing time across all samples |
| Min | Fastest request time |
| Max | Slowest request time |
| P50 | Median - 50% of requests are faster than this |
| P95 | 95th percentile - only 5% of requests are slower |
| P99 | 99th percentile - only 1% of requests are slower |
| Histogram | Distribution of request times across bins |

### Throughput Metrics

| Metric | Description |
|--------|-------------|
| Sequential RPS | Requests per second in a serial loop |
| Concurrent RPS | Requests per second with parallel workers |

## Test Scenarios

### 1. Image Dimension Tests

Tests how input image dimensions affect processing time. The DINOv2 model resizes all inputs to 518×518, so this measures preprocessing overhead for different input sizes.

| Test Case | Dimensions | Rationale |
|-----------|------------|-----------|
| tiny | 100×100 | Below model input size - minimal resize |
| small | 500×500 | Close to model input (518) - near-native |
| medium | 1000×1000 | ~2x model input - moderate downscale |
| large | 2000×2000 | ~4x model input - significant downscale |
| xlarge | 4000×4000 | ~8x model input - large image handling |

### 2. File Size Tests (Compressed JPEG)

Tests how compressed file size affects processing time. Uses random noise images to control JPEG file size (solid colors compress to tiny files, noise creates larger files).

| Test Case | Target Size | Description |
|-----------|-------------|-------------|
| tiny | ~10 KB | Highly compressed, simple content |
| small | ~100 KB | Typical web image |
| medium | ~500 KB | High quality photo |
| large | ~1 MB | Very high quality |
| xlarge | ~5 MB | Maximum quality/complexity |

**Why noise images?** JPEG compression is content-dependent. A 1000×1000 solid red image compresses to ~5KB, while the same dimensions with random noise can be 500KB+. Noise images let us independently test file size impact.

### 3. Throughput Tests

Measures sustained request handling capacity under different load patterns.

| Test Case | Type | Configuration | Purpose |
|-----------|------|---------------|---------|
| sequential | Loop | 50 requests back-to-back | Baseline throughput |
| concurrent_2 | Parallel | 2 workers, 25 requests each | Light concurrency |
| concurrent_4 | Parallel | 4 workers, 12 requests each | Moderate concurrency |
| concurrent_8 | Parallel | 8 workers, 6 requests each | Heavy concurrency |

## Test Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Iterations per scenario | 30 | Sufficient for statistical significance |
| Image format | JPEG only | Service production format |
| Output | Console | Real-time visibility during test runs |

## Image Pre-generation

**Critical**: All test images are generated ONCE before tests run to avoid contaminating latency measurements.

```python
# In conftest.py - module-scoped fixture
@pytest.fixture(scope="module")
def pregenerated_images() -> dict[str, str]:
    """
    Pre-generate all test images before any tests run.
    Returns dict mapping image key to base64 string.
    """
    images = {}

    # Dimension test images (5 sizes)
    for size in [100, 500, 1000, 2000, 4000]:
        images[f"dim_{size}x{size}"] = create_noise_image_base64(size, size)

    # File size test images (5 target sizes)
    for target_kb in [10, 100, 500, 1000, 5000]:
        images[f"size_{target_kb}kb"] = create_target_size_image_base64(target_kb)

    # Throughput test image (single standard image)
    images["throughput"] = create_noise_image_base64(518, 518)

    return images
```

This ensures:
- Image generation time is NOT included in latency measurements
- All tests use identical pre-generated images
- Large images (4000x4000, 5MB) are only generated once

## File Structure

```
services/embeddings/tests/
├── performance/                 # Dedicated performance test directory
│   ├── __init__.py              # Package marker
│   ├── conftest.py              # Performance test fixtures
│   ├── test_performance.py      # Main performance test cases
│   ├── metrics.py               # LatencyMetrics class for statistics
│   └── generators.py            # Noise image generators
└── ...
```

### Key Components

#### 1. Metrics Collector (`performance/metrics.py`)

```python
@dataclass
class LatencyMetrics:
    samples: list[float]

    @property
    def mean(self) -> float
    @property
    def min(self) -> float
    @property
    def max(self) -> float
    def percentile(self, p: int) -> float  # p50, p95, p99
    def histogram(self, bins: int) -> dict[str, int]
    def summary(self) -> str  # Formatted output for console
```

#### 2. Image Generators (`performance/generators.py`)

```python
def create_noise_image_base64(
    width: int,
    height: int,
    quality: int = 85,
) -> str:
    """Generate JPEG with random noise for controlled complexity."""

def create_target_size_image_base64(
    target_kb: int,
    max_dimension: int = 1000,
) -> str:
    """Generate JPEG that approximates target file size using noise."""
```

#### 3. Test Classes (`performance/test_performance.py`)

```python
@pytest.mark.slow
@pytest.mark.performance
class TestDimensionLatency:
    """Test latency across different image dimensions."""

    @pytest.mark.parametrize("width,height", [
        (100, 100), (500, 500), (1000, 1000), (2000, 2000), (4000, 4000)
    ])
    def test_dimension_latency(self, client, width, height):
        # Run N iterations, collect metrics, report


@pytest.mark.slow
@pytest.mark.performance
class TestFileSizeLatency:
    """Test latency across different compressed file sizes."""

    @pytest.mark.parametrize("target_kb", [10, 100, 500, 1000, 5000])
    def test_filesize_latency(self, client, target_kb):
        # Generate noise image of target size, measure latency


@pytest.mark.slow
@pytest.mark.performance
class TestThroughput:
    """Test request throughput."""

    def test_sequential_throughput(self, client):
        # 50 sequential requests, measure requests/second

    @pytest.mark.parametrize("workers", [2, 4, 8])
    def test_concurrent_throughput(self, client, workers):
        # Parallel requests using ThreadPoolExecutor
```

## Running Performance Tests

```bash
# Navigate to embeddings service
cd services/embeddings

# Run only performance tests
just test -- -m performance -v

# Run specific test class
just test -- -m performance -k "TestDimensionLatency" -v

# Run specific scenario
just test -- -m performance -k "test_dimension_latency[1000-1000]" -v

# Exclude performance tests from regular test runs
just test -- -m "not performance"
```

## Interpreting Results

### Latency Analysis

**Good performance indicators:**
- Low P99/P50 ratio (< 2x) indicates consistent performance
- P95 close to mean indicates few outliers
- Histogram shows tight distribution

**Warning signs:**
- High P99 with low mean indicates occasional slow requests
- Bimodal histogram may indicate GC pauses or resource contention
- Increasing latency with image size beyond expected scaling

### Throughput Analysis

**Expected behavior:**
- Sequential throughput limited by single-request latency
- Concurrent throughput should scale with workers (up to hardware limits)
- Diminishing returns beyond CPU/GPU core count

**Warning signs:**
- Concurrent throughput lower than sequential (contention issues)
- No scaling with additional workers (bottleneck)
- High variance in concurrent results (resource starvation)

## Dependencies

Performance tests use the same dependencies as integration tests:
- `pytest` - Test framework
- `fastapi.testclient` - HTTP client for testing
- `PIL` - Image generation
- `numpy` - Random noise generation, statistics
- `concurrent.futures` - Parallel request execution

No additional dependencies required (no pytest-benchmark).
