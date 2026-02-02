# Developer Log: Search Service Implementation

**Branch:** `feature-search-service`
**Base:** `32fd177` (main)

---

## Overview

This branch implements the **Search Service**, a FAISS-based vector similarity search microservice for the artwork-matcher system. It manages a vector index for fast nearest-neighbor queries and returns ranked matches with metadata.

---

## Features Implemented

### Vector Search API

**POST /search** - Find similar vectors in the index

- Accepts query embedding, k (max results), and threshold (min score)
- Returns ranked results with object_id, score, rank, and metadata
- Inner product similarity for normalized vectors (equivalent to cosine similarity)
- Includes processing time in response

**POST /add** - Add embedding to the index

- Accepts object_id, embedding vector, and optional metadata
- Returns index position and total count
- In-memory until explicitly saved

**POST /index/save** - Persist index to disk

- Saves FAISS index and metadata JSON to configured paths
- Optional custom path parameter
- Returns file paths and size

**POST /index/load** - Load index from disk

- Loads previously saved index
- Automatic load on startup if configured

**DELETE /index** - Clear the index

- Removes all vectors and metadata
- Returns previous and current count

**GET /health** - Health check endpoint

- Returns service status with uptime

**GET /info** - Service configuration endpoint

- Exposes service name, version, index statistics
- Shows index type, metric, dimension, count, load status

---

## Architecture

### Application Structure

```
src/search_service/
├── app.py           # FastAPI factory function
├── config.py        # YAML configuration loading + Pydantic validation
├── schemas.py       # Request/response Pydantic models
├── logging.py       # Structured JSON logging
├── main.py          # Production entry point (uvicorn)
├── core/
│   ├── state.py     # Runtime state (index, uptime)
│   ├── lifespan.py  # Startup/shutdown lifecycle
│   └── exceptions.py# Error handling and HTTP responses
├── routers/
│   ├── health.py    # GET /health
│   ├── info.py      # GET /info
│   ├── search.py    # POST /search
│   └── index.py     # POST /add, /index/save, /index/load, DELETE /index
└── services/
    └── faiss_index.py  # FAISS index wrapper with metadata storage
```

### Configuration

All configuration is loaded from `config.yaml` with zero defaults in code:

```yaml
service:
  name: "search"
  version: "0.1.0"

faiss:
  embedding_dimension: 768
  index_type: "flat"         # flat, ivf, hnsw
  metric: "inner_product"    # inner_product, l2

index:
  path: "/data/index/faiss.index"
  metadata_path: "/data/index/metadata.json"
  auto_load: true

search:
  default_k: 5
  max_k: 100
  default_threshold: 0.0

server:
  host: "0.0.0.0"
  port: 8002

logging:
  level: "INFO"
  format: "json"
```

---

## Design Decisions

### 1. FAISS IndexFlatIP (Brute Force)

Using `faiss.IndexFlatIP` for exact nearest-neighbor search with inner product.

**Why:** 20 objects = 20 comparisons = <0.1ms. No training needed. 100% accuracy. Simplest configuration. See [API spec](../api/search_service_api_spec.md) for scaling recommendations.

### 2. Inner Product Metric

Using inner product instead of L2 distance.

**Why:** For L2-normalized vectors (from Embeddings Service), inner product equals cosine similarity. Higher score = more similar (intuitive). Scores in [-1, 1] range.

### 3. Separate Metadata Storage

FAISS index stores only vectors; metadata stored in JSON file alongside.

**Why:** FAISS optimized for vectors, not metadata. JSON is human-readable for debugging. Metadata updates don't require index rebuild. Follows FAISS best practices.

### 4. Explicit Save/Load Lifecycle

Index changes are in-memory until explicit `/index/save` call. Auto-load on startup if file exists.

**Why:** Fast batch operations (add many, save once). Explicit control over persistence. Good for index building workflows.

### 5. Zero-Default Configuration

All configuration values must be explicitly specified. No defaults in code.

**Why:** Fail-fast at startup. No hidden assumptions. Aligns with CLAUDE.md project philosophy.

### 6. FastAPI Factory Pattern

Using `create_app()` factory instead of module-level `app` variable.

**Why:** Better testability (fresh app per test). Avoids import-time side effects. Cleaner lifespan management.

**Implication:** Requires `uvicorn --factory` flag.

### 7. Singleton Application State

Module-level `AppState` singleton initialized during lifespan, accessed via `get_app_state()`.

**Why:** FAISS index initialization should happen once at startup. State needs to be accessible from request handlers. Explicit initialization makes testing easier.

### 8. Lifespan Context Manager

Using FastAPI's `lifespan` async context manager instead of deprecated `on_startup`/`on_shutdown`.

**Why:** Modern FastAPI pattern. Cleaner resource management. Index fully initialized before accepting requests. Cleanup guaranteed on shutdown.

### 9. Structured JSON Logging

All logs are JSON with consistent schema: timestamp, level, logger, message, extra fields.

**Why:** Machine-parseable for log aggregation. Easy to filter/search. Includes service context in every message.

### 10. Sensitive Data Redaction

The `/info` endpoint automatically redacts values for keys containing: key, secret, password, token, credential, etc.

**Why:** Safe to expose configuration for debugging without leaking secrets.

### 11. Test Data Factories

Dedicated `tests/factories.py` with deterministic vector generators.

**Why:** Reproducible tests with known similarities. Create vectors at exact angles for precise ranking tests. Shared across unit and integration tests.

### 12. Three-Tier Test Structure

Tests split into `tests/unit/`, `tests/integration/`, and `tests/performance/`.

**Why:** Unit tests are fast (mocked). Integration tests use real FAISS index. Performance tests measure latency/throughput. CI runs unit tests by default.

### 13. Separate Process Test Execution

Unit and integration tests run in separate pytest processes.

**Why:** Complete isolation prevents state leakage. FAISS global state doesn't persist between test tiers. Clean module imports per tier.

---

## Issues Encountered & Fixes

### Test Isolation with FAISS

**Problem:** Unit tests using mocks and integration tests using real FAISS index interfered when run together. State from one tier leaked to another.

**Fix:** Run `just test-unit` and `just test-integration` as separate pytest processes instead of combining markers in one run.

### Pytest Markers Required

**Problem:** `just test-unit` with `-m unit` marker found no tests because test classes weren't marked.

**Fix:** Added `@pytest.mark.unit` / `@pytest.mark.integration` / `@pytest.mark.performance` to all test classes. Documented in testing guide.

### ASGI App Not Found

**Problem:** `uvicorn search_service.app:app` failed because code uses factory pattern.

**Fix:** Use `--factory` flag: `uvicorn search_service.app:create_app --factory`

---

## Testing

### Unit Tests (51 tests)

Located in `tests/unit/`. Run with `just test-unit`.

- Router tests (health, info, search, index operations)
- Configuration loading and validation
- FAISS index wrapper logic
- Schema validation
- All external dependencies mocked

### Integration Tests (54 tests)

Located in `tests/integration/`. Run with `just test-integration`.

| Module | Tests | Purpose |
|--------|-------|---------|
| `test_search_behavior.py` | 10 | Basic search determinism, discrimination, ranking, lifecycle |
| `test_deterministic_search.py` | 21 | Exact match, orthogonal vectors, known similarity scores, precise ranking |
| `test_endpoints.py` | 14 | Health, info, save/load round-trip, metadata persistence |
| `test_error_responses.py` | 9 | Dimension mismatch, empty index, proper HTTP status codes |

Key integration test categories:

**Deterministic Search Tests:**
- Identical vectors return score=1.0
- Orthogonal basis vectors return score=0.0
- Vectors with computed target similarities (0.9, 0.7, 0.5) rank correctly
- Threshold filtering with precise cutoff validation

**Endpoint Tests:**
- Health returns proper format
- Info reflects index state (count, dimension, loaded status)
- Save creates files, load restores them
- Metadata survives save/load round-trip

**Error Response Tests:**
- Dimension mismatch returns 400 with expected/received details
- Empty index returns 422 with index_empty error
- Proper JSON error format

### Performance Tests (17 tests)

Located in `tests/performance/`. Run with `just test-performance`.

| Category | Tests | Purpose |
|----------|-------|---------|
| Index Size | 6 | Measure search latency as index grows (10 to 10K vectors) |
| Throughput | 6 | Sequential and concurrent (2/4/8/16 workers) search throughput |
| Add Operations | 3 | Bulk add performance |
| Save/Load | 2 | Persistence timing |

**Metrics Collected:**
- Latency: mean, min, max, std, p50, p95, p99
- Throughput: requests/second, total duration

**Report Generation:**

After tests complete, a markdown report is automatically generated at `reports/performance/search_service_performance.md` with:

- Key findings summary
- Detailed results tables for each test category
- Dynamic analysis (index size impact, scaling efficiency, bottleneck detection)

See [Performance Testing Guide](../implementation-guides/performance_testing_guide.md) for full documentation.

### Test Factories

The `tests/factories.py` module provides deterministic vector generators:

```python
# Basis vectors (orthogonal, similarity = 0)
create_basis_vector(dimension, index)

# Vector with exact target similarity to base
create_vector_with_known_similarity(base, target_similarity)

# Multiple vectors with known rankings
create_scaled_similarity_vectors(dimension, [0.9, 0.7, 0.5])

# Invalid vectors for error testing
create_invalid_embedding_nan(dimension)
create_wrong_dimension_embedding(target_dimension)
```

---

## Justfile Commands

### Root Level

| Command | Description |
|---------|-------------|
| `just docker-up` | Start all services (Docker) |
| `just docker-down` | Stop all services |
| `just docker-logs` | View Docker logs |
| `just docker-build` | Build all Docker images |
| `just status` | Check health of all services |

### Service Level (from `services/search/`)

| Command | Description |
|---------|-------------|
| `just run` | Run locally with hot reload |
| `just kill` | Stop local uvicorn process |
| `just status` | Check service health and info |
| `just test` | Run all tests (unit + integration) |
| `just test-unit` | Run unit tests only |
| `just test-integration` | Run integration tests only |
| `just test-performance` | Run performance tests (generates report) |
| `just ci` | Run all CI checks (uses unit tests) |
| `just ci-all` | Run CI including integration tests |
| `just ci-quiet` | Run CI silently (output on errors only) |
| `just docker-up` | Start this service in Docker |
| `just docker-down` | Stop this service |

---

## Documentation Added

| File | Description |
|------|-------------|
| `docs/implementation-guides/service_testing_guide.md` | Three-tier testing structure, markers, justfile recipes |
| `docs/implementation-guides/performance_testing_guide.md` | Metrics module, test patterns, report generation |
| `reports/performance/search_service_performance.md` | Performance baseline report |

---

## Commits Summary

| Commit | Description |
|--------|-------------|
| `a5e1421` | Initial implementation: FastAPI service, FAISS wrapper, 51 unit tests |
| `6a30fdc` | Fix lifecycle test to run from service directory |
| `2b8ec71` | Add service testing guide documentation |
| `3aa63da` | Align justfile with embeddings service patterns |
| `572860f` | Document pytest marker requirements |
| `0e9feb3` | Add integration tests (10 tests for search behavior) |
| `a92ced2` | Add performance testing implementation guide |
| `2a80561` | Add performance tests and track reports in git |
| `b8161cb` | Improve test isolation with separate processes |
| `1c51ca4` | Extract test factories into dedicated module |
| `472f18a` | Add comprehensive integration test suite (44 more tests) |

---

## Statistics

| Metric | Value |
|--------|-------|
| Total Tests | 122 |
| Unit Tests | 51 |
| Integration Tests | 54 |
| Performance Tests | 17 |
| Source Files | 17 |
| Test Files | 13 |
