# Developer Log: Embeddings Service Implementation

**Branch:** `feature-embedding-service`

---

## Overview

This branch implements the **Embeddings Service**, the first complete microservice in the artwork-matcher system. It extracts L2-normalized visual embeddings from artwork images using Meta's DINOv2 foundation model.

---

## Features Implemented

### Embedding Extraction API

**POST /embed** - Extract embedding from a base64-encoded image

- Accepts JPEG, PNG, and WebP formats
- Handles data URL prefixes (e.g., `data:image/jpeg;base64,...`)
- Returns 768-dimensional L2-normalized embedding
- Includes processing time in response
- Optional `image_id` field for request tracking

**GET /health** - Health check endpoint

- Returns service status
- Tracks and reports uptime (seconds and human-readable format)
- Includes system timestamp

**GET /info** - Service configuration endpoint

- Exposes service name, version, model config, preprocessing config
- Automatically redacts sensitive values (keys, passwords, tokens)

---

## Architecture

### Application Structure

```
src/embeddings_service/
├── app.py           # FastAPI factory function
├── config.py        # YAML configuration loading + Pydantic validation
├── schemas.py       # Request/response Pydantic models
├── logging.py       # Structured JSON logging
├── main.py          # Production entry point (uvicorn)
├── core/
│   ├── state.py     # Runtime state (model, processor, uptime)
│   ├── lifespan.py  # Startup/shutdown lifecycle
│   └── exceptions.py# Error handling and HTTP responses
└── routers/
    ├── embed.py     # POST /embed
    ├── health.py    # GET /health
    └── info.py      # GET /info
```

### Configuration

All configuration is loaded from `config.yaml` with zero defaults in code:

```yaml
service:
  name: "embeddings"
  version: "0.1.0"

model:
  name: "facebook/dinov2-base"
  device: "auto"              # auto, cpu, cuda, mps
  embedding_dimension: 768
  cache_dir: "data/models"
  revision: "main"

preprocessing:
  image_size: 518
  normalize: true

server:
  host: "0.0.0.0"
  port: 8000

logging:
  level: "INFO"
  format: "json"
```

---

## Design Decisions

### 1. Zero-Default Configuration

All configuration values must be explicitly specified. No defaults anywhere in code.

**Why:** Fail-fast at startup instead of runtime. No hidden assumptions. Aligns with CLAUDE.md project philosophy.

### 2. FastAPI Factory Pattern

Using `create_app()` factory instead of module-level `app` variable.

**Why:** Better testability (fresh app per test), avoids import-time side effects, cleaner lifespan management.

**Implication:** Requires `uvicorn --factory` flag.

### 3. Singleton Application State

Module-level `AppState` singleton initialized during lifespan, accessed via `get_app_state()`.

**Why:** Model loading is expensive, needs to happen once at startup. State needs to be accessible from request handlers. Explicit initialization makes testing easier.

### 4. Lifespan Context Manager

Using FastAPI's `lifespan` async context manager instead of deprecated `on_startup`/`on_shutdown`.

**Why:** Modern FastAPI pattern. Cleaner resource management. Model fully loaded before accepting requests. Cleanup guaranteed on shutdown.

### 5. L2-Normalized Embeddings

All returned embeddings are L2-normalized.

**Why:** Makes cosine similarity equal to inner product (`cos(a,b) = a·b` when `||a||=||b||=1`). FAISS inner product is faster than cosine. Standard practice in image retrieval.

### 6. Structured JSON Logging

All logs are JSON with consistent schema: timestamp, level, logger, message, extra fields.

**Why:** Machine-parseable for log aggregation. Easy to filter/search. Includes service context in every message.

### 7. Mocked Unit Tests

Unit tests mock all external dependencies (model, settings, state).

**Why:** Fast execution, no network/GPU needed, tests focus on business logic.

### 8. Separate Unit and Integration Tests

Tests split into `tests/unit/` and `tests/integration/` with pytest markers.

**Why:** CI runs fast unit tests by default. Integration tests (which load the real model) run separately when needed.

### 9. Sensitive Data Redaction

The `/info` endpoint automatically redacts values for keys containing: key, secret, password, token, credential, etc.

**Why:** Safe to expose configuration for debugging without leaking secrets.

### 10. Test Factory Pattern

Using factory functions (`create_test_image_base64()`, `create_random_embedding()`) instead of static fixtures.

**Why:** Flexible, self-documenting, reusable across test types.

---

## Issues Encountered & Fixes

### ASGI App Not Found

**Problem:** `uvicorn embeddings_service.app:app` failed because code uses factory pattern.

**Fix:** Use `--factory` flag: `uvicorn embeddings_service.app:create_app --factory`

### Transformers Type Stubs

**Problem:** `transformers` library lacks complete type stubs.

**Fix:** Added mypy override to ignore missing imports for `transformers.*`.

### Deptry False Positives

**Problem:** Dev dependencies flagged as unused (they're tools, not imported).

**Fix:** Added `DEP002` ignores for dev tools in `pyproject.toml`.

### Semgrep Comment Detection

**Problem:** `nosemgrep` comments triggered Ruff's "commented-out code" rule.

**Fix:** Per-file ERA001 ignores for affected files.

---

## Testing

### Unit Tests (79 tests)

Located in `tests/unit/`. Run with `just test-unit`.

- Router tests (embed, health, info)
- Configuration loading and validation
- Schema validation
- All external dependencies mocked

### Integration Tests

Located in `tests/integration/`. Run with `just test-integration`.

- Real model loading
- End-to-end embedding extraction
- Embedding consistency verification (same image → same embedding)

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

### Service Level (from `services/embeddings/`)

| Command | Description |
|---------|-------------|
| `just run` | Run locally with hot reload |
| `just kill` | Stop local uvicorn process |
| `just test` | Run all tests |
| `just test-unit` | Run unit tests only |
| `just test-integration` | Run integration tests only |
| `just ci` | Run all CI checks (uses unit tests) |
| `just docker-up` | Start this service in Docker |
| `just docker-down` | Stop this service |
| `just status` | Check this service's health |

---

## Documentation Added

| File | Description |
|------|-------------|
| `docs/implementation-guides/fastapi_service_template.md` | Service structure and patterns |
| `docs/implementation-guides/fastapi_testing_guide.md` | Testing patterns with pytest |
| `docs/implementation-guides/common_issues_and_fixes.md` | Troubleshooting guide |
