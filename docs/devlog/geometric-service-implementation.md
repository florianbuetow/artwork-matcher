# Developer Log: Geometric Service Implementation

**Branch:** `feature-geometric-service`

---

## Overview

This branch implements the **Geometric Service**, a classical computer vision microservice that provides ORB feature extraction and RANSAC-based geometric verification for artwork matching. It serves as the final verification stage in the artwork-matcher pipeline, confirming that candidate matches from the embedding search are geometrically consistent.

The service follows the same architecture patterns established by the Embeddings and Search services: FastAPI factory pattern, YAML configuration with Pydantic validation, structured JSON logging, and three-tier testing.

---

## Features Implemented

### Feature Extraction API

**POST /extract** - Extract ORB features from a base64-encoded image

- Accepts JPEG, PNG, and WebP formats
- Handles data URL prefixes (e.g., `data:image/jpeg;base64,...`)
- Optional `max_features` parameter to override default
- Returns keypoints with x, y, size, angle attributes
- Returns base64-encoded binary descriptors (32 bytes per keypoint)
- Includes image dimensions (width, height)
- Includes server-side processing time

### Geometric Matching API

**POST /match** - Verify geometric consistency between two images

- Accepts query image (base64) and reference (image or pre-extracted features)
- Extracts ORB features from both images
- Matches features using BFMatcher with Lowe's ratio test
- Verifies geometric consistency using RANSAC homography estimation
- Returns comprehensive match results:
  - `is_match`: Boolean decision based on inlier threshold
  - `confidence`: Combined score from inlier count and ratio
  - `inliers`: Number of geometrically consistent matches
  - `total_matches`: Feature matches before RANSAC filtering
  - `inlier_ratio`: Fraction of matches that are inliers
  - `homography`: 3x3 transformation matrix (if match found)
  - `query_features` / `reference_features`: Feature counts
- Validates minimum feature count (returns 422 if insufficient)
- Includes server-side processing time

**POST /match/batch** - Verify query against multiple references

- Accepts query image and list of references
- Each reference can provide image or pre-extracted features
- Returns individual results for each reference
- Identifies best match (highest confidence among matches)
- Continues processing even if some references fail
- Efficient for comparing one query against a candidate set

### Standard Endpoints

**GET /health** - Health check endpoint

- Returns service status with uptime (seconds and human-readable)
- Includes system timestamp

**GET /info** - Service configuration endpoint

- Exposes algorithm configuration:
  - Feature detector (ORB) and its parameters
  - Matcher (BFMatcher) with norm type and ratio threshold
  - Verification method (RANSAC) with thresholds
- Automatically redacts sensitive values

---

## Architecture

### Application Structure

```
services/geometric/
├── src/geometric_service/
│   ├── __init__.py
│   ├── app.py              # FastAPI factory function
│   ├── config.py           # YAML configuration + Pydantic validation
│   ├── schemas.py          # Request/response Pydantic models
│   ├── logging.py          # Structured JSON logging
│   ├── main.py             # Production entry point (uvicorn)
│   ├── core/
│   │   ├── state.py        # Runtime state (uptime tracking)
│   │   ├── lifespan.py     # Startup/shutdown lifecycle
│   │   └── exceptions.py   # Custom errors and HTTP handlers
│   ├── routers/
│   │   ├── health.py       # GET /health
│   │   ├── info.py         # GET /info
│   │   ├── extract.py      # POST /extract
│   │   └── match.py        # POST /match, /match/batch
│   ├── services/
│   │   ├── feature_extractor.py  # ORB feature extraction
│   │   ├── feature_matcher.py    # BFMatcher with ratio test
│   │   └── geometric_verifier.py # RANSAC homography verification
│   └── utils/
│       └── image.py        # Base64 decoding utilities
└── tests/
    ├── factories.py        # Test image generators
    ├── conftest.py         # Shared pytest fixtures
    ├── unit/               # Unit tests (mocked OpenCV)
    ├── integration/        # Integration tests (real OpenCV)
    └── performance/        # Performance benchmarks
```

### Configuration

All configuration is loaded from `config.yaml` with zero defaults in code:

```yaml
service:
  name: geometric
  version: 0.1.0

orb:
  max_features: 1000         # Maximum keypoints to detect
  scale_factor: 1.2          # Pyramid scale factor
  n_levels: 8                # Number of pyramid levels
  edge_threshold: 31         # Border exclusion zone
  patch_size: 31             # Descriptor patch size
  fast_threshold: 20         # FAST corner threshold

matching:
  ratio_threshold: 0.75      # Lowe's ratio test threshold
  cross_check: false         # Disabled for kNN matching

ransac:
  reproj_threshold: 5.0      # Max reprojection error (pixels)
  max_iters: 2000            # Maximum RANSAC iterations
  confidence: 0.995          # Required confidence level

verification:
  min_features: 50           # Minimum features per image
  min_matches: 20            # Minimum matches for verification
  min_inliers: 10            # Minimum inliers to declare match

server:
  host: "0.0.0.0"
  port: 8003
  log_level: "info"
```

Environment variable overrides: `GEOMETRIC__ORB__MAX_FEATURES=2000`

---

## Design Decisions

### 1. ORB Feature Detector

Using OpenCV's ORB (Oriented FAST and Rotated BRIEF) for feature detection.

**Why:** Patent-free, CPU-efficient (~12ms vs SIFT's ~116ms), rotation-invariant. Binary descriptors enable fast Hamming distance matching. Good balance of speed and accuracy for real-time verification.

### 2. BFMatcher with kNN and Ratio Test

Using brute-force matcher with k=2 nearest neighbors and Lowe's ratio test.

**Why:** Ratio test filters ambiguous matches where best and second-best are similar. Threshold of 0.75 is well-established default. More reliable than simple nearest-neighbor matching.

**Implementation detail:** Perfect matches (distance=0) always pass ratio test to handle self-matching and exact duplicates.

### 3. RANSAC Homography Verification

Using RANSAC to estimate homography and count inliers.

**Why:** Homography models planar transformations (rotation, scale, perspective). RANSAC robustly filters outliers from noisy feature matches. Inlier count directly indicates geometric consistency.

### 4. Confidence Score Formula

Combined score: `0.7 * min(inliers/100, 1.0) + 0.3 * inlier_ratio`

**Why:** Inlier count is more reliable than ratio alone (high ratio with few matches is suspicious). Saturates at 100 inliers to prevent runaway scores. 70/30 weighting emphasizes absolute count.

### 5. Minimum Feature Threshold

Reject images with fewer than 50 features (returns 422).

**Why:** Images with too few features cannot be reliably matched. Early rejection prevents wasted computation. Clear error message helps debugging.

### 6. Pre-extracted Features Support

Match endpoint accepts either raw image or pre-extracted features.

**Why:** Gateway service can cache reference features for repeated queries. Avoids redundant extraction. Significant speedup for batch operations.

### 7. Zero-Default Configuration

All configuration values must be explicitly specified. No defaults in code.

**Why:** Fail-fast at startup. No hidden assumptions. Aligns with CLAUDE.md project philosophy.

### 8. FastAPI Factory Pattern

Using `create_app()` factory instead of module-level `app` variable.

**Why:** Better testability (fresh app per test). Avoids import-time side effects. Cleaner lifespan management.

**Implication:** Requires `uvicorn --factory` flag.

### 9. Singleton Application State

Module-level `AppState` singleton initialized during lifespan, accessed via `get_app_state()`.

**Why:** State (uptime tracking) initialized once at startup. Accessible from request handlers. Explicit initialization makes testing easier.

### 10. Lifespan Context Manager

Using FastAPI's `lifespan` async context manager.

**Why:** Modern FastAPI pattern. Cleaner resource management. State fully initialized before accepting requests. Cleanup guaranteed on shutdown.

### 11. Structured JSON Logging

All logs are JSON with consistent schema: timestamp, level, logger, message, extra fields.

**Why:** Machine-parseable for log aggregation. Easy to filter/search. Includes service context in every message.

### 12. Sensitive Data Redaction

The `/info` endpoint automatically redacts values for keys containing: key, secret, password, token, credential, etc.

**Why:** Safe to expose configuration for debugging without leaking secrets.

### 13. Graceful Batch Failure Handling

Batch match continues processing if individual references fail.

**Why:** One bad reference shouldn't fail entire batch. Failed references return zero inliers. Errors logged for debugging.

---

## Algorithm Pipeline

### Stage 1: Feature Extraction (ORBFeatureExtractor)

1. Decode base64 image to bytes
2. Convert to OpenCV format (BGR)
3. Convert to grayscale for ORB
4. Detect keypoints using FAST corner detector
5. Compute BRIEF descriptors (32 bytes each, binary)
6. Serialize keypoints to JSON-compatible format

**Output:** List of keypoints (x, y, size, angle), numpy array of descriptors

### Stage 2: Feature Matching (BFFeatureMatcher)

1. Compute Hamming distance between all descriptor pairs
2. Find k=2 nearest neighbors for each query descriptor
3. Apply Lowe's ratio test: accept if `best < 0.75 * second`
4. Special case: accept if `best.distance == 0` (exact match)

**Output:** List of cv2.DMatch objects

### Stage 3: Geometric Verification (RANSACVerifier)

1. Extract matched point coordinates
2. Estimate homography using RANSAC:
   - Randomly sample 4 point correspondences
   - Compute homography matrix
   - Count inliers (reprojection error < threshold)
   - Repeat up to max_iters or until confidence reached
3. Count final inliers from best homography
4. Calculate confidence score
5. Decide match based on min_inliers threshold

**Output:** is_match, confidence, inliers, inlier_ratio, homography

---

## Testing

### Test Structure

Tests follow the three-tier pattern established by other services:

```
tests/
├── factories.py              # Test image generators
├── conftest.py               # Shared fixtures
├── unit/                     # Fast, mocked tests
│   ├── conftest.py           # Unit test fixtures
│   ├── test_config.py        # Configuration loading
│   ├── test_schemas.py       # Pydantic validation
│   ├── services/             # Service layer tests
│   │   ├── test_feature_extractor.py
│   │   ├── test_feature_matcher.py
│   │   └── test_geometric_verifier.py
│   └── routers/              # Endpoint tests
│       ├── test_health.py
│       ├── test_info.py
│       ├── test_extract.py
│       └── test_match.py
├── integration/              # Real OpenCV tests
│   ├── conftest.py           # Integration fixtures
│   └── test_endpoints.py     # Full API tests
└── performance/              # Benchmark tests
    ├── conftest.py           # Performance fixtures
    ├── metrics.py            # Metrics collection
    ├── generators.py         # Image generators
    └── test_performance.py   # Latency/throughput tests
```

### Unit Tests (95 tests)

Located in `tests/unit/`. Run with `just test-unit`.

**Configuration Tests (34 tests):**
- YAML loading and validation
- Environment variable overrides
- Missing/invalid configuration handling
- Sensitive data redaction patterns

**Schema Tests (35 tests):**
- Request model validation (required fields, min lengths)
- Response model construction
- Extra field rejection (Pydantic forbid)
- Keypoint, ImageSize, AlgorithmInfo models

**Service Tests (15 tests):**
- ORB extractor with mocked OpenCV
- BFMatcher with synthetic descriptors
- RANSAC verifier with synthetic keypoints
- Edge cases (empty inputs, few matches)

**Router Tests (11 tests):**
- Endpoint response formats
- Error handling paths
- Request validation

### Integration Tests (22 tests)

Located in `tests/integration/`. Run with `just test-integration`.

**Basic Endpoint Tests (6 tests):**
- Health returns correct format
- Info exposes algorithm configuration
- Extract returns features from valid image
- Extract rejects invalid image data
- Match identical images returns is_match=True
- Batch returns results for all references

**Transformation Matching Tests (5 tests):**
- Rotated artwork (15°) matches original
- Scaled artwork (80%) matches original
- Combined rotation + scale matches
- Cropped artwork (80%) matches original
- Noise image with rotation matches

**Non-Matching Tests (4 tests):**
- Different artworks don't match
- Different noise images don't match
- Artwork vs noise doesn't match
- Multiple pairwise comparisons all fail

**Edge Case Tests (5 tests):**
- Solid color images rejected (insufficient features)
- Low feature image rejected
- Large rotation (45°) processes correctly
- Very small rotation (2°) matches easily
- Slight scale change (95%) matches easily

**Batch Accuracy Tests (2 tests):**
- Correct match identified among decoys
- No false positives when query doesn't match

### Performance Tests (17 tests)

Located in `tests/performance/`. Run with `just test-performance`.

**Dimension Latency Tests:**
- Measure extraction time across image sizes (256×256 to 2048×2048)
- Track scaling behavior with resolution

**Feature Count Tests:**
- Measure impact of max_features setting (500 to 4000)
- Track actual features extracted vs limit

**Match Scenario Tests:**
- Extract-only baseline (single image)
- Full match (two images + RANSAC)
- Compare overhead of matching vs extraction

**Throughput Tests:**
- Sequential requests (baseline throughput)
- Concurrent requests (2, 4, 8 workers)
- Measure scaling efficiency

**Metrics Collected:**
- Latency: mean, min, max, std, p50, p95, p99
- Throughput: requests/second, total duration

**Report Generation:**
Markdown report automatically generated at `reports/performance/geometric_service_performance.md`

### Test Data Factories

The `tests/factories.py` module provides deterministic image generators:

```python
# Geometric patterns with corner features
create_checkerboard_base64(width, height, block_size, seed)

# Dense random features (always many keypoints)
create_noise_image_base64(width, height, seed)

# Simulated artwork with shapes/colors
create_artwork_simulation_base64(width, height, seed)

# Transform existing image
create_transformed_image_base64(
    base_image_b64,
    rotation_deg=15,    # Counterclockwise rotation
    scale=0.8,          # Resize factor
    crop_ratio=0.7      # Center crop fraction
)

# Edge case images
create_solid_color_base64(width, height, color)  # No features
create_non_image_base64()                         # Invalid data
create_invalid_base64()                           # Malformed base64
```

---

## Issues Encountered & Fixes

### OpenCV Type Stubs Incomplete

**Problem:** OpenCV's type stubs incorrectly claim `cv2.findHomography` cannot return None.

**Fix:** Use `cast("Any", result)` to override incorrect stubs, then handle None properly.

### Ratio Test Edge Case

**Problem:** Self-matching (identical images) would sometimes fail ratio test if best and second-best distances were both 0.

**Fix:** Added explicit check: if `m.distance == 0`, always accept the match regardless of ratio test.

### Grayscale Image Transformation

**Problem:** Rotating grayscale images with PIL's `fillcolor` parameter expects RGB tuple, not single value.

**Fix:** Convert grayscale to RGB before applying transformations in test factories.

### Checkerboard Images Poor for Matching

**Problem:** Checkerboard patterns have repetitive features that confuse ORB matching after rotation.

**Fix:** Use noise images or artwork simulations for transformation tests; checkerboards only for basic feature extraction tests.

### ASGI App Not Found

**Problem:** `uvicorn geometric_service.app:app` failed because code uses factory pattern.

**Fix:** Use `--factory` flag: `uvicorn geometric_service.app:create_app --factory`

### Test Isolation

**Problem:** Integration tests and unit tests could interfere via shared module state.

**Fix:** Separate test processes via justfile recipes. Clear settings cache between test modules.

---

## Justfile Commands

### Root Level

| Command | Description |
|---------|-------------|
| `just docker-up` | Start all services (Docker) |
| `just docker-down` | Stop all services |
| `just docker-logs` | View Docker logs |
| `just docker-build` | Build all Docker images |
| `just run-geometric` | Run geometric service locally |

### Service Level (from `services/geometric/`)

| Command | Description |
|---------|-------------|
| `just run` | Run locally with hot reload |
| `just kill` | Stop local uvicorn process |
| `just status` | Check service health |
| `just test` | Run all tests (unit + integration) |
| `just test-unit` | Run unit tests only |
| `just test-integration` | Run integration tests only |
| `just test-performance` | Run performance tests |
| `just ci` | Run all CI checks (verbose) |
| `just ci-quiet` | Run CI silently |
| `just docker-up` | Start this service in Docker |
| `just docker-down` | Stop this service |
| `just code-format` | Auto-fix formatting |
| `just code-style` | Check style (read-only) |
| `just code-typecheck` | Run mypy |
| `just code-lspchecks` | Run pyright (strict) |
| `just code-security` | Run bandit |

---

## API Request/Response Examples

### Extract Features

**Request:**
```bash
curl -X POST http://localhost:8003/extract \
  -H "Content-Type: application/json" \
  -d '{
    "image": "'$(base64 -i artwork.jpg | tr -d '\n')'",
    "image_id": "artwork_001"
  }'
```

**Response:**
```json
{
  "image_id": "artwork_001",
  "num_features": 847,
  "keypoints": [
    {"x": 123.5, "y": 456.2, "size": 31.0, "angle": 45.3},
    ...
  ],
  "descriptors": "base64-encoded-binary-data...",
  "image_size": {"width": 1024, "height": 768},
  "processing_time_ms": 23.45
}
```

### Match Two Images

**Request:**
```bash
curl -X POST http://localhost:8003/match \
  -H "Content-Type: application/json" \
  -d '{
    "query_image": "'$(base64 -i visitor_photo.jpg | tr -d '\n')'",
    "reference_image": "'$(base64 -i reference.jpg | tr -d '\n')'",
    "query_id": "visitor_001",
    "reference_id": "artwork_007"
  }'
```

**Response (Match Found):**
```json
{
  "is_match": true,
  "confidence": 0.85,
  "inliers": 67,
  "total_matches": 124,
  "inlier_ratio": 0.5403,
  "query_features": 892,
  "reference_features": 756,
  "homography": [
    [0.98, -0.12, 45.3],
    [0.11, 0.99, -12.7],
    [0.0001, 0.0002, 1.0]
  ],
  "query_id": "visitor_001",
  "reference_id": "artwork_007",
  "processing_time_ms": 48.72
}
```

**Response (No Match):**
```json
{
  "is_match": false,
  "confidence": 0.12,
  "inliers": 3,
  "total_matches": 45,
  "inlier_ratio": 0.0667,
  "query_features": 892,
  "reference_features": 634,
  "homography": null,
  "query_id": "visitor_001",
  "reference_id": "artwork_042",
  "processing_time_ms": 35.18
}
```

### Batch Match

**Request:**
```bash
curl -X POST http://localhost:8003/match/batch \
  -H "Content-Type: application/json" \
  -d '{
    "query_image": "'$(base64 -i query.jpg | tr -d '\n')'",
    "query_id": "visitor_photo",
    "references": [
      {"reference_id": "art_001", "reference_image": "'$(base64 -i ref1.jpg | tr -d '\n')'"},
      {"reference_id": "art_002", "reference_image": "'$(base64 -i ref2.jpg | tr -d '\n')'"},
      {"reference_id": "art_003", "reference_image": "'$(base64 -i ref3.jpg | tr -d '\n')'"}
    ]
  }'
```

**Response:**
```json
{
  "query_id": "visitor_photo",
  "query_features": 892,
  "results": [
    {"reference_id": "art_001", "is_match": false, "confidence": 0.08, "inliers": 2, "inlier_ratio": 0.04},
    {"reference_id": "art_002", "is_match": true, "confidence": 0.82, "inliers": 58, "inlier_ratio": 0.52},
    {"reference_id": "art_003", "is_match": false, "confidence": 0.11, "inliers": 4, "inlier_ratio": 0.07}
  ],
  "best_match": {
    "reference_id": "art_002",
    "confidence": 0.82
  },
  "processing_time_ms": 127.34
}
```

### Error Response (Insufficient Features)

**Response (422):**
```json
{
  "error": "insufficient_features",
  "message": "Query has only 23 features (minimum: 50)",
  "details": {"query_features": 23}
}
```

---

## Pydantic Schema Models

### Request Models

| Model | Fields | Purpose |
|-------|--------|---------|
| `ExtractRequest` | image, image_id?, max_features? | Feature extraction input |
| `MatchRequest` | query_image, reference_image?, reference_features?, query_id?, reference_id? | Single match input |
| `BatchMatchRequest` | query_image, references[], query_id? | Batch match input |
| `ReferenceInput` | reference_id, reference_image?, reference_features? | Batch reference item |
| `ReferenceFeatures` | keypoints[], descriptors | Pre-extracted features |
| `KeypointData` | x, y, size, angle | Single keypoint |

### Response Models

| Model | Fields | Purpose |
|-------|--------|---------|
| `ExtractResponse` | image_id, num_features, keypoints[], descriptors, image_size, processing_time_ms | Extraction output |
| `MatchResponse` | is_match, confidence, inliers, total_matches, inlier_ratio, query_features, reference_features, homography?, query_id?, reference_id?, processing_time_ms | Match output |
| `BatchMatchResponse` | query_id, query_features, results[], best_match?, processing_time_ms | Batch output |
| `BatchMatchResult` | reference_id, is_match, confidence, inliers, inlier_ratio | Individual batch result |
| `BestMatch` | reference_id, confidence | Best match summary |
| `HealthResponse` | status, uptime_seconds, uptime, system_time | Health check |
| `InfoResponse` | service, version, algorithm | Service info |
| `AlgorithmInfo` | feature_detector, max_features, matcher, matcher_norm, ratio_threshold, verification, ransac_reproj_threshold, min_inliers | Algorithm config |
| `ErrorResponse` | error, message, details | Standard error |

All models use `ConfigDict(extra="forbid")` to reject unexpected fields.

---

## CI Checks

All passing:

| Check | Tool | Purpose |
|-------|------|---------|
| code-format | ruff format | Auto-formatting |
| code-style | ruff check | Linting rules |
| code-typecheck | mypy | Type checking |
| code-lspchecks | pyright | Strict type checking |
| code-security | bandit | Security scan |
| code-deptry | deptry | Dependency hygiene |
| code-spell | codespell | Spelling check |
| code-semgrep | semgrep | Static analysis |
| code-audit | pip-audit | Vulnerability scan |
| test | pytest | Unit tests |

---

## Statistics

| Metric | Value |
|--------|-------|
| Total Tests | 117 |
| Unit Tests | 95 |
| Integration Tests | 22 |
| Performance Tests | 17 |
| Source Files | 21 |
| Test Files | 25 |
| Lines of Code (src) | ~1200 |
| Lines of Code (tests) | ~1800 |
