# Performance Testing Methodology

This document describes the performance testing strategy for the artwork-matcher system, covering individual service benchmarks, integration testing, and end-to-end validation.

## Table of Contents

1. [Philosophy](#philosophy)
2. [Testing Levels](#testing-levels)
3. [Common Framework](#common-framework)
4. [Per-Service Testing](#per-service-testing)
   - [Embeddings Service](#embeddings-service)
   - [Search Service](#search-service)
   - [Geometric Service](#geometric-service)
   - [Gateway Service](#gateway-service)
5. [End-to-End Testing](#end-to-end-testing)
6. [Running Tests](#running-tests)
7. [Interpreting Results](#interpreting-results)
8. [TODO: Missing Coverage](#todo-missing-coverage)

---

## Philosophy

### Why Performance Test?

Performance testing serves three critical purposes:

1. **Regression Detection** - Catch performance degradations before they reach production
2. **Capacity Planning** - Understand throughput limits and scaling characteristics
3. **Optimization Guidance** - Identify bottlenecks and measure improvement impact

### Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Measurement Isolation** | Pre-generate all test data before timing begins |
| **Statistical Significance** | 30 iterations per scenario for reliable percentiles |
| **Reproducibility** | Identical data across iterations, seeded randomness |
| **Real Components** | Test actual service code with real dependencies |
| **Automated Reporting** | Generate markdown reports with analysis |
| **Consistent Patterns** | Same framework structure across all services |

---

## Testing Levels

The system uses a layered testing approach:

```
┌─────────────────────────────────────────────────────────────┐
│                    END-TO-END TESTS                         │
│         Full pipeline with real services (TODO)             │
├─────────────────────────────────────────────────────────────┤
│                   GATEWAY ISOLATION                         │
│      Real gateway app, mocked backend services              │
├──────────────┬──────────────┬──────────────┬───────────────┤
│  EMBEDDINGS  │    SEARCH    │  GEOMETRIC   │   (Gateway)   │
│    (Real)    │    (Real)    │    (Real)    │   See above   │
│  DINOv2 GPU  │  FAISS index │  ORB+RANSAC  │               │
└──────────────┴──────────────┴──────────────┴───────────────┘
```

### What Each Level Measures

| Level | Real Components | Mocked Components | What's Measured |
|-------|-----------------|-------------------|-----------------|
| **Service Isolation** | Service code, ML models, algorithms | Nothing | Raw processing performance |
| **Gateway Isolation** | Gateway routing, orchestration | Backend services (respx) | Gateway overhead only |
| **End-to-End** | All services via HTTP | Nothing | Full pipeline latency |

---

## Common Framework

All services share identical testing infrastructure patterns.

### Standard Constants

```python
ITERATIONS_PER_SCENARIO = 30    # Statistical significance
THROUGHPUT_REQUESTS = 250       # Sustained load measurement
CONCURRENCY_LEVELS = [2, 4, 8, 16]  # Light to heavy parallelism
```

### File Structure (Per Service)

```
services/<name>/tests/performance/
├── __init__.py           # Package marker
├── conftest.py           # Fixtures and constants
├── generators.py         # Test data generation
├── metrics.py            # LatencyMetrics, ThroughputMetrics, PerformanceReport
└── test_performance.py   # Actual test classes
```

### Metrics Collected

**Latency Metrics (per request):**
- Mean, Min, Max, Std Dev
- P50 (median), P95 (tail), P99 (extreme tail)
- Histogram distribution

**Throughput Metrics:**
- Total requests
- Total duration (seconds)
- Requests per second (RPS)

### Test Markers

All performance tests use these pytest markers:

```python
@pytest.mark.slow         # Takes > 1 second
@pytest.mark.performance  # Performance benchmark
```

### Report Generation

Each service generates a markdown report at:
```
services/<name>/reports/performance/<name>_service_performance.md
```

Reports include:
- Key findings summary
- Test configuration
- Detailed results tables
- Scaling analysis
- Contention/bottleneck detection

---

## Per-Service Testing

### Embeddings Service

**Location:** `services/embeddings/tests/performance/`

**What's Real:**
- DINOv2 model (loaded once at startup)
- Image preprocessing pipeline
- GPU/CPU inference

**What's Mocked:** Nothing

**Test Scenarios:**

| Test Class | What's Varied | Values | Purpose |
|------------|---------------|--------|---------|
| `TestDimensionLatency` | Image dimensions | 100, 512, 1024, 2048, 4096 | Measure preprocessing overhead |
| `TestFileSizeLatency` | JPEG file size | 10KB to 5000KB | Measure decompression impact |
| `TestThroughput` | Request pattern | Sequential + concurrent | Measure capacity limits |

**Key Insights:**
- All images resized to 518x518 for model input
- File size tests isolate JPEG decompression from dimension scaling
- GPU memory limits concurrent request capacity

**Expected Performance:**
- Latency: 50-500ms depending on hardware (GPU vs CPU)
- Throughput: 0.5-5 RPS (highly hardware dependent)

---

### Search Service

**Location:** `services/search/tests/performance/`

**What's Real:**
- FAISS flat index (IndexFlatIP)
- Vector similarity computation
- Result ranking

**What's Mocked:** Nothing

**Test Scenarios:**

| Test Class | What's Varied | Values | Purpose |
|------------|---------------|--------|---------|
| `TestIndexSizeLatency` | Index size | 100 to 50,000 vectors | Measure O(n) scaling |
| `TestKValueLatency` | Results requested (k) | 1, 5, 10, 25, 50, 100 | Measure result count impact |
| `TestThroughput` | Request pattern | Sequential + concurrent | Measure capacity limits |

**Key Insights:**
- Flat index has O(n) search complexity
- k value has minimal impact (all distances computed anyway)
- CPU-bound, benefits from parallelism

**Expected Performance:**
- Latency: 0.1-10ms depending on index size
- Throughput: 100-1000+ RPS

---

### Geometric Service

**Location:** `services/geometric/tests/performance/`

**What's Real:**
- OpenCV ORB feature extractor
- Brute-force feature matcher
- RANSAC homography estimation

**What's Mocked:** Nothing

**Test Scenarios:**

| Test Class | What's Varied | Values | Purpose |
|------------|---------------|--------|---------|
| `TestDimensionLatency` | Image dimensions | 100 to 2000 | Measure ORB scaling |
| `TestFeatureCountLatency` | max_features param | 100 to 2000 | Measure feature limit impact |
| `TestMatchLatency` | Pipeline depth | extract_only, full_match | Isolate pipeline stages |
| `TestThroughput` | Request pattern | Sequential + concurrent | Measure capacity limits |

**Key Insights:**
- ORB extraction scales with image pixels
- Feature count affects both extraction and matching
- Full match includes: 2x extraction + matching + RANSAC
- CPU-bound, OpenCV uses internal parallelism

**Expected Performance:**
- Extract only: 5-50ms
- Full match: 15-150ms
- Throughput: 10-100 RPS

---

### Gateway Service

**Location:** `services/gateway/tests/performance/`

**What's Real:**
- FastAPI application
- Request routing and validation
- Response serialization
- HTTP client initialization

**What's Mocked:**
- Embeddings service (port 8001) via `respx`
- Search service (port 8002) via `respx`
- Geometric service (port 8003) via `respx`

All mocked responses return instantly (~1ms simulated processing).

**Test Scenarios:**

| Test Class | What's Varied | Values | Purpose |
|------------|---------------|--------|---------|
| `TestDimensionLatency` | Payload size | 100 to 2000 dimensions | Measure serialization overhead |
| `TestEndpointLatency` | Endpoint | /health, /info, /identify, /objects | Measure per-endpoint overhead |
| `TestGeometricService` | Service check | Port 8003 availability | Document test environment |
| `TestThroughput` | Request pattern | Sequential + concurrent | Measure gateway capacity |

**Key Insights:**
- Gateway tests measure **orchestration overhead only**
- Backend latency is deliberately excluded
- Useful for detecting gateway-specific regressions
- Does NOT represent real-world end-to-end latency

**Expected Performance:**
- Gateway overhead: 1-10ms per request
- Throughput: 100-500+ RPS (with mocked backends)

---

## End-to-End Testing

### Current Status: NOT IMPLEMENTED

The system currently lacks end-to-end performance tests that exercise the full pipeline with real services.

### What End-to-End Tests Would Measure

```
User Request
    │
    ▼
┌─────────┐     ┌────────────┐     ┌────────┐     ┌───────────┐
│ Gateway │────▶│ Embeddings │────▶│ Search │────▶│ Geometric │
└─────────┘     └────────────┘     └────────┘     └───────────┘
    │                                                   │
    ▼                                                   │
  Response ◀────────────────────────────────────────────┘
```

**Metrics to capture:**
- Full pipeline latency (user-perceived)
- Per-service breakdown within pipeline
- Network overhead between services
- Concurrent user capacity
- System behavior under load

### Why It Matters

| Scenario | Service Tests | E2E Tests |
|----------|---------------|-----------|
| Service regression | Detected | Detected |
| Network configuration issue | Not detected | Detected |
| Service interaction bug | Not detected | Detected |
| Realistic latency estimate | No | Yes |
| Capacity planning | Partial | Complete |

---

## Running Tests

### Single Service

```bash
cd services/<name>
uv run pytest tests/performance -m performance -v -s
```

### Specific Test Class

```bash
cd services/<name>
uv run pytest tests/performance/test_performance.py::TestDimensionLatency -v -s
```

### All Services (Sequential)

```bash
# From project root
just test-performance-all  # If defined in justfile
```

### With Docker (Future E2E)

```bash
# Start all services
just docker-up

# Run E2E tests (when implemented)
# TODO: Define E2E test runner
```

---

## Interpreting Results

### Latency Analysis

| Metric | What It Tells You |
|--------|-------------------|
| **Mean** | Average case performance |
| **P50** | Typical user experience |
| **P95** | Worst case for most users |
| **P99** | Tail latency (SLA compliance) |
| **Std Dev** | Consistency (lower = more predictable) |

### Throughput Analysis

| Pattern | Indication |
|---------|------------|
| Sequential > Concurrent | Lock contention or resource exhaustion |
| Linear scaling with workers | Good parallelism |
| Flat scaling with workers | CPU-bound (single-threaded bottleneck) |
| Decreasing with workers | Severe contention |

### Scaling Analysis

The reports calculate scaling ratios:

```
Scaling Factor = Latency(large) / Latency(small)
```

| Factor | Interpretation |
|--------|----------------|
| ~1.0 | O(1) - constant time |
| ~2.0 for 2x input | O(n) - linear scaling |
| ~4.0 for 2x input | O(n²) - quadratic scaling |

---
