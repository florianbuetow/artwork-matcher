# Service Testing Guide

This guide documents the testing principles and structure used across artwork-matcher microservices. The **embeddings service** serves as the reference implementation.

## Overview

Each service implements a **three-tier testing approach**:

| Tier | Purpose | Speed | Dependencies |
|------|---------|-------|--------------|
| **Unit** | Test isolated components with mocked dependencies | Fast (~seconds) | None |
| **Integration** | Test full request/response with real components | Medium (~minutes) | Real service components |
| **Performance** | Measure latency and throughput | Slow (~minutes) | Real service components |

This separation allows developers to:
- Run fast unit tests during development (`just test-unit`)
- Run integration tests before commits (`just test-integration`)
- Run performance tests for optimization work (`just test-performance`)

---

## Test Directory Structure

Every service should follow this standard layout:

```
services/<service_name>/
└── tests/
    ├── __init__.py
    ├── conftest.py              # Root conftest (minimal, shared config)
    ├── factories.py             # Test data generators
    │
    ├── unit/                    # Fast, isolated tests
    │   ├── __init__.py
    │   ├── conftest.py          # Unit test fixtures (mocks)
    │   ├── test_config.py
    │   ├── test_schemas.py
    │   └── routers/
    │       └── test_<endpoint>.py
    │
    ├── integration/             # Full stack tests
    │   ├── __init__.py
    │   ├── conftest.py          # Integration fixtures (real components)
    │   ├── test_endpoints.py
    │   └── test_<behavior>.py
    │
    └── performance/             # Latency and throughput tests
        ├── __init__.py
        ├── conftest.py
        ├── generators.py
        ├── metrics.py
        └── test_performance.py
```

**Reference:** [`services/embeddings/tests/`](../../services/embeddings/tests/)

---

## Unit Tests

### Principles

- **Fast**: Complete in seconds
- **Isolated**: All external dependencies are mocked
- **Deterministic**: Same result every run
- **Focused**: One concern per test

### What to Test

| Category | Description |
|----------|-------------|
| **Input validation** | Boundary values, missing fields, wrong types, edge cases |
| **Error handling** | Every error path the component can take |
| **Configuration** | Loading, validation, environment overrides |
| **Business logic** | Core algorithms with various inputs |
| **Response format** | Required fields present, correct types |

### Pytest Marker

```python
@pytest.mark.unit
class TestHealthRouter:
    ...
```

**Reference:** [`services/embeddings/tests/unit/`](../../services/embeddings/tests/unit/)

---

## Integration Tests

### Principles

- **Realistic**: Uses real components (models, indices, etc.)
- **End-to-end**: Tests full request/response cycle
- **Catches wiring issues**: Serialization bugs, dependency injection problems

### What to Test

| Category | Description |
|----------|-------------|
| **Happy paths** | Complete successful flow for each endpoint |
| **Error responses** | Correct status codes and error formats |
| **Data round-trips** | Save/load cycles preserve data integrity |
| **Behavioral guarantees** | Determinism (same input → same output), discrimination (different inputs → different outputs) |
| **Concurrent access** | Behavior under parallel requests |

### Pytest Marker

```python
@pytest.mark.integration
class TestEmbedEndpoint:
    ...
```

**Reference:** [`services/embeddings/tests/integration/`](../../services/embeddings/tests/integration/)

---

## Performance Tests

### Principles

- **Measurement-focused**: Collects timing data with statistical analysis
- **Parameterized**: Tests multiple scenarios (sizes, concurrency levels)
- **Pre-generated data**: Test data created before measurements to avoid contamination
- **Report generation**: Produces markdown reports with analysis

### What to Test

| Category | Description |
|----------|-------------|
| **Input variation** | How different input characteristics affect latency |
| **Sequential throughput** | Requests per second with one client |
| **Concurrent throughput** | Requests per second with multiple workers |

### Pytest Markers

```python
@pytest.mark.slow
@pytest.mark.performance
class TestThroughput:
    ...
```

**Reference:** [`services/embeddings/tests/performance/`](../../services/embeddings/tests/performance/)

---

## Test Factories

Factories provide reusable functions for creating test data. Keep them in `tests/factories.py`.

**Reference:** [`services/embeddings/tests/factories.py`](../../services/embeddings/tests/factories.py)

---

## Justfile Recipes

Every service justfile should include these testing recipes:

```makefile
# Run all tests
test:
    @uv run pytest tests/ -v

# Run unit tests only
test-unit:
    @uv run pytest tests/unit -m unit -v

# Run integration tests only
test-integration:
    @uv run pytest tests/integration -m integration -v

# Run performance tests
test-performance:
    @uv run pytest tests/performance -m performance -v -s

# CI with unit tests (fast)
ci:
    # ... other checks ...
    just test-unit

# Full CI with integration tests
ci-all:
    just ci
    just test-integration
```

**Reference:** [`services/embeddings/justfile`](../../services/embeddings/justfile)

---

## Pytest Configuration

Register markers in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "unit: Unit tests (fast, isolated)",
    "integration: Integration tests (real components)",
    "performance: Performance tests (slow)",
    "slow: Slow tests (excluded from default run)",
]
```

---

## Implementation Checklist

When implementing tests for a new service:

### Structure
- [ ] `tests/factories.py`
- [ ] `tests/unit/` with `conftest.py`
- [ ] `tests/integration/` with `conftest.py`
- [ ] `tests/performance/` with `conftest.py`

### Unit Tests
- [ ] Configuration validation
- [ ] Schema validation
- [ ] Router tests for each endpoint
- [ ] Error handling paths

### Integration Tests
- [ ] Happy path for each endpoint
- [ ] Error responses
- [ ] Determinism tests
- [ ] Discrimination tests

### Performance Tests
- [ ] Input variation latency
- [ ] Sequential throughput
- [ ] Concurrent throughput

### Justfile
- [ ] `test-unit`, `test-integration`, `test-performance`
- [ ] `ci-all` recipe

### Pytest
- [ ] Markers on all test classes
- [ ] Markers registered in `pyproject.toml`

---

## Reference Implementation

The **embeddings service** is the canonical example:

| Component | Path |
|-----------|------|
| Unit Tests | [`services/embeddings/tests/unit/`](../../services/embeddings/tests/unit/) |
| Integration Tests | [`services/embeddings/tests/integration/`](../../services/embeddings/tests/integration/) |
| Performance Tests | [`services/embeddings/tests/performance/`](../../services/embeddings/tests/performance/) |
| Factories | [`services/embeddings/tests/factories.py`](../../services/embeddings/tests/factories.py) |
| Justfile | [`services/embeddings/justfile`](../../services/embeddings/justfile) |
