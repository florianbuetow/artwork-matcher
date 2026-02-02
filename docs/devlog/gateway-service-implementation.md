# Developer Log: Gateway Service Implementation

**Branch:** `feature-gateway-service`
**Base:** `d3b0e3b` (main)

---

## Overview

This branch implements the **Gateway Service**, the public-facing API that orchestrates the artwork identification pipeline. It coordinates requests across internal services (Embeddings, Search, Geometric) and presents a unified interface to clients.

---

## Features Implemented

### Artwork Identification API

**POST /identify** - Identify artwork in a visitor photo

- Accepts base64-encoded image and optional pipeline parameters
- Orchestrates embedding extraction, vector search, and geometric verification
- Returns best match with confidence score, timing breakdown, and alternatives
- Graceful degradation when geometric service unavailable

**GET /health** - Health check with backend status

- Returns gateway status and individual backend health
- Status logic: healthy (all up), degraded (geometric down), unhealthy (embeddings/search down)
- Optional `check_backends=false` parameter for fast response

**GET /info** - Service configuration and backend info

- Exposes pipeline settings (search_k, thresholds, geometric enabled)
- Shows backend URLs and their current status
- Returns model info from embeddings service

**GET /objects** - List all artworks in database

- Returns object list with metadata (id, name, artist, year)
- Sorted by object_id

**GET /objects/{id}** - Get artwork details

- Returns full object metadata including description, location
- Includes image_url for reference image

**GET /objects/{id}/image** - Get artwork reference image

- Returns binary image data with correct content-type
- Supports JPEG and PNG formats

---

## Architecture

### Application Structure

```
src/gateway/
├── app.py           # FastAPI factory function with CORS
├── config.py        # YAML configuration loading + Pydantic validation
├── schemas.py       # Request/response Pydantic models
├── logging.py       # Structured JSON logging
├── main.py          # Production entry point (uvicorn)
├── core/
│   ├── state.py     # Runtime state (uptime, clients)
│   ├── lifespan.py  # Startup/shutdown lifecycle
│   └── exceptions.py# Error handling and HTTP responses
├── routers/
│   ├── health.py    # GET /health
│   ├── info.py      # GET /info
│   ├── identify.py  # POST /identify (pipeline orchestration)
│   └── objects.py   # GET /objects, /objects/{id}, /objects/{id}/image
└── clients/
    ├── base.py      # Base HTTP client with error handling
    ├── embeddings.py# Embeddings service client
    ├── search.py    # Search service client
    └── geometric.py # Geometric service client
```

### Configuration

All configuration is loaded from `config.yaml` with zero defaults in code:

```yaml
service:
  name: "gateway"
  version: "0.1.0"

backends:
  embeddings_url: "http://localhost:8001"
  search_url: "http://localhost:8002"
  geometric_url: "http://localhost:8003"
  timeout_seconds: 30.0

pipeline:
  search_k: 5
  similarity_threshold: 0.7
  geometric_verification: true
  confidence_threshold: 0.6

server:
  host: "0.0.0.0"
  port: 8000
  cors_origins:
    - "*"

logging:
  level: "INFO"
  format: "json"
```

---

## Design Decisions

### 1. Sequential Pipeline with Early Termination

Pipeline stages execute sequentially: Embed → Search → Geometric Verify.

**Why:** Each stage depends on the previous. Can't search without embedding. Can't verify without candidates. Early termination saves resources (if embedding fails, no point searching).

### 2. Graceful Degradation for Geometric Service

If geometric service is unavailable, pipeline returns embedding-only results with reduced confidence.

**Why:** Geometric verification is an enhancement, not a requirement. Better to return results than fail entirely. Confidence penalty signals reduced reliability.

### 3. Confidence Score Calculation

```python
if geometric_score is not None:
    if geometric_score > 0.5:
        confidence = 0.6 * similarity + 0.4 * geometric_score
    else:
        confidence = 0.3 * similarity + 0.2 * geometric_score
elif verification_enabled:
    confidence = similarity * 0.7  # Penalty for missing verification
else:
    confidence = similarity * 0.85  # Small penalty when intentionally skipped
```

**Why:** Geometric confirmation increases confidence. Geometric rejection decreases it despite high similarity. Missing verification gets a penalty.

### 4. Top-K with Threshold Filtering

Search returns top-K candidates, filtered by similarity threshold.

**Why:** Embedding similarity isn't perfect—closest match might be wrong. Geometric can re-rank. Multiple artworks might be in photo. Threshold prevents returning garbage matches.

### 5. Cascading Timeouts

| Service | Timeout | Rationale |
|---------|---------|-----------|
| Embeddings | 30s | Model inference can be slow on CPU |
| Search | 30s | Should be fast; if slow, something's wrong |
| Geometric | 30s | Multiple image comparisons |

**Why:** Individual timeouts prevent one slow service from blocking everything. Total pipeline timeout is the sum.

### 6. CORS on Gateway Only

CORS middleware configured only on Gateway service.

**Why:** Gateway is the only service that receives browser requests. Internal services communicate server-to-server. Single point of CORS configuration.

### 7. Backend Health Aggregation

Gateway health reflects backend health: healthy (all up), degraded (geometric down), unhealthy (critical backends down).

**Why:** Load balancers need to know if gateway can fulfill requests. Degraded status allows partial functionality. Unhealthy means pipeline is broken.

### 8. FastAPI Factory Pattern

Using `create_app()` factory instead of module-level `app` variable.

**Why:** Better testability (fresh app per test). Avoids import-time side effects. Cleaner lifespan management.

**Implication:** Requires `uvicorn --factory` flag.

### 9. Singleton Application State

Module-level `AppState` singleton initialized during lifespan, accessed via `get_app_state()`.

**Why:** Backend clients should be initialized once at startup. State needs to be accessible from request handlers. Explicit initialization makes testing easier.

### 10. Structured JSON Logging

All logs are JSON with consistent schema: timestamp, level, logger, message, extra fields.

**Why:** Machine-parseable for log aggregation. Easy to filter/search. Includes service context in every message.

### 11. Sensitive Data Redaction

The `/info` endpoint automatically redacts values for keys containing: key, secret, password, token, credential, etc.

**Why:** Safe to expose configuration for debugging without leaking secrets.

### 12. Test Data Factories

Dedicated `tests/factories.py` with mock backend response generators.

**Why:** Reproducible tests with controlled responses. Create specific scenarios (match found, no match, errors). Shared across unit and integration tests.

### 13. Three-Tier Test Structure

Tests split into `tests/unit/`, `tests/integration/`, and `tests/performance/`.

**Why:** Unit tests are fast (mocked backends). Integration tests use mocked HTTP responses. Performance tests measure actual throughput. CI runs unit tests by default.

---

## Issues Encountered & Fixes

### App Factory Not Found

**Problem:** `uvicorn gateway.app:app` failed because code uses factory pattern.

**Fix:** Use `--factory` flag: `uvicorn gateway.app:create_app --factory`

### Test Coverage Below Threshold

**Problem:** Coverage at 73%, threshold is 80%.

**Status:** Known issue. Low coverage in `main.py` (entry point), backend clients (error paths not exercised), and some router edge cases. Integration tests with real backends would improve this.

### Geometric Verification Warning

**Problem:** Performance tests log warnings about geometric verification being skipped.

**Status:** Expected behavior—geometric verification requires reference images which aren't available in test fixtures. Warning is informational.

---

## Testing

### Unit Tests (114 tests)

Located in `tests/unit/`. Run with `just test-unit`.

- Router tests (health, info, identify, objects)
- Configuration loading and validation
- Schema validation
- All backend dependencies mocked

| Module | Tests | Purpose |
|--------|-------|---------|
| `test_health.py` | 7 | Health endpoint with various backend states |
| `test_info.py` | 13 | Info endpoint response structure |
| `test_identify.py` | 11 | Identify pipeline and confidence calculation |
| `test_objects.py` | 19 | Object listing, details, and images |
| `test_config.py` | 35 | Config loading, validation, sensitive data |
| `test_schemas.py` | 29 | Schema validation and serialization |

### Integration Tests (27 tests)

Located in `tests/integration/`. Run with `just test-integration`.

| Module | Tests | Purpose |
|--------|-------|---------|
| `test_pipeline.py` | 15 | End-to-end pipeline with mocked backends |
| `test_consistency.py` | 12 | Response consistency and structure |

Key integration test categories:

**Pipeline Tests:**
- Health, info endpoints return correct structure
- Identify returns match with timing
- Identify returns no match when appropriate
- Error responses propagated correctly

**Consistency Tests:**
- Same image produces consistent result
- Response always has required fields
- Timing values are non-negative
- Total timing >= component sum

### Performance Tests (18 tests)

Located in `tests/performance/`. Run with `just test-performance`.

| Category | Tests | Purpose |
|----------|-------|---------|
| Dimension Latency | 7 | Measure latency vs image size (100x100 to 2000x2000) |
| Endpoint Latency | 6 | Individual endpoint response times |
| Throughput | 5 | Concurrent request handling (up to 16 workers) |

**Results Summary:**
- Throughput: 1241 req/s with 16 concurrent workers
- P50 latency: 12.10 ms
- P99 latency: 22.32 ms

**Report Generation:**

After tests complete, a markdown report is automatically generated at `reports/performance/gateway_service_performance.md`.

### Manual Tests

Located in `tests/manual/`. Run with `bash tests/manual/test_service_lifecycle.sh`.

Tests complete service lifecycle:
1. Verify no service running
2. Start locally with `just run`
3. Verify health endpoint responds
4. Stop with `just kill`
5. Start in Docker with `just docker-up`
6. Verify Docker container health
7. Stop with `just docker-down`

---

## Justfile Commands

### Service Level (from `services/gateway/`)

| Command | Description |
|---------|-------------|
| `just run` | Run locally with hot reload |
| `just kill` | Stop local uvicorn process |
| `just status` | Check service health and info |
| `just test` | Run all tests (unit + integration) |
| `just test-unit` | Run unit tests only |
| `just test-integration` | Run integration tests only |
| `just test-performance` | Run performance tests (generates report) |
| `just test-coverage` | Run tests with coverage report |
| `just ci` | Run all CI checks (uses unit tests) |
| `just ci-all` | Run CI including integration tests |
| `just ci-quiet` | Run CI silently (output on errors only) |
| `just docker-up` | Start this service in Docker |
| `just docker-down` | Stop this service |
| `just docker-build` | Build Docker image |
| `just docker-logs` | View Docker logs |

---

## Commits Summary

| Commit | Description |
|--------|-------------|
| `179b33d` | feat(tools): add justfile target comparison script |
| `631f85e` | feat(gateway): align justfile with embeddings service structure |
| `df38794` | feat(gateway): configure project dependencies and service settings |
| `809a40c` | feat(gateway): add core application modules |
| `f91e916` | feat(gateway): add core module for lifecycle and error handling |
| `7c7eaac` | feat(gateway): add backend service clients |
| `85b7d41` | feat(gateway): add API routers for all endpoints |
| `025439e` | feat(gateway): add test infrastructure |
| `25601b3` | feat(gateway): add comprehensive unit test suite |
| `eb06df1` | feat(gateway): add integration test suite |
| `7b5e3dd` | feat(gateway): add performance test suite |
| `a9efc6d` | feat(gateway): add manual service lifecycle test |
| `a7c5f5a` | chore(gateway): remove placeholder test file |
| `1c084b9` | fix(geometric): include uv.lock in Docker build |

---

## Statistics

| Metric | Value |
|--------|-------|
| Total Tests | 159 |
| Unit Tests | 114 |
| Integration Tests | 27 |
| Performance Tests | 18 |
| Source Files | 20 |
| Test Files | 21 |
| Lines Added | ~7,400 |
| Test Coverage | 73% |
