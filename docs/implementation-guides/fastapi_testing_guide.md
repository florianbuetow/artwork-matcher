# Testing Guide for FastAPI Services

This document defines testing standards, patterns, and practices for all microservices in the Artwork Matcher project. It distinguishes between **unit tests** and **integration tests** — two fundamentally different testing approaches with different purposes, dependencies, and execution contexts.

---

## Table of Contents

1. [Testing Philosophy](#testing-philosophy)
2. [Unit Tests vs Integration Tests](#unit-tests-vs-integration-tests)
3. [Project Structure](#project-structure)
4. [pytest Configuration](#pytest-configuration)
5. [Unit Testing](#unit-testing)
6. [Integration Testing](#integration-testing)
7. [Fixtures and Factories](#fixtures-and-factories)
8. [Mocking Patterns](#mocking-patterns)
9. [Async Testing](#async-testing)
10. [Coverage Requirements](#coverage-requirements)
11. [CI Pipeline Integration](#ci-pipeline-integration)
12. [Complete Examples](#complete-examples)

---

## Testing Philosophy

### Guiding Principles

| Principle | Implementation |
|-----------|----------------|
| **Test behavior, not implementation** | Focus on inputs/outputs, not internal details |
| **Fail fast, fail clearly** | Tests should fail with actionable error messages |
| **Isolation** | Unit tests never touch external systems |
| **Determinism** | Tests produce the same result every time |
| **Speed** | Unit tests run in milliseconds, integration tests are slower but still bounded |
| **Independence** | Tests can run in any order without affecting each other |

### The Testing Pyramid

```
                    ┌───────────────┐
                    │   E2E Tests   │  ← Few, slow, high confidence
                    │  (Manual/CI)  │
                    ├───────────────┤
                    │  Integration  │  ← Some, medium speed
                    │     Tests     │     Tests real service interactions
                    ├───────────────┤
                    │               │
                    │  Unit Tests   │  ← Many, fast, test logic in isolation
                    │               │
                    └───────────────┘
```

For this project:
- **Unit tests**: 80% of test code — fast, isolated, run on every commit
- **Integration tests**: 20% of test code — require running services, run in CI
- **E2E tests**: Manual verification + `tools/evaluate.py`

---

## Unit Tests vs Integration Tests

### Definitions

| Aspect | Unit Tests | Integration Tests |
|--------|------------|-------------------|
| **Scope** | Single function/class/module | Multiple components working together |
| **Dependencies** | All external dependencies mocked | Real services, real databases |
| **Speed** | Milliseconds | Seconds to minutes |
| **Isolation** | Complete — no I/O, no network | Partial — controlled environment |
| **Purpose** | Verify logic correctness | Verify components integrate correctly |
| **When to run** | Every commit, pre-push | CI pipeline, pre-deploy |
| **Failure meaning** | Bug in specific code | Contract violation or env issue |

### What Each Type Tests

**Unit Tests verify:**
- Configuration parsing and validation
- Request/response schema validation
- Business logic (embedding extraction, matching algorithms)
- Error handling paths
- Edge cases and boundary conditions
- Individual route handlers (with mocked dependencies)

**Integration Tests verify:**
- HTTP endpoints respond correctly
- Services communicate properly
- Database/index operations work
- The full request → processing → response cycle
- Error propagation across service boundaries
- Health checks and readiness

### Key Distinction

```python
# UNIT TEST — Dependencies are mocked
def test_extract_embedding_returns_correct_dimension(mock_model):
    """Test embedding extraction logic with mocked model."""
    mock_model.return_value = np.zeros(768)
    
    result = extract_embedding(fake_image, model=mock_model)
    
    assert len(result) == 768
    mock_model.assert_called_once()


# INTEGRATION TEST — Real service is running
def test_embed_endpoint_returns_embedding(running_service):
    """Test /embed endpoint against running service."""
    response = httpx.post(
        f"{running_service}/embed",
        json={"image": base64_image}
    )
    
    assert response.status_code == 200
    assert len(response.json()["embedding"]) == 768
```

---

## Project Structure

```
services/<service_name>/
├── src/<service_name>/
│   └── ...
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures
│   ├── unit/                    # Unit tests
│   │   ├── __init__.py
│   │   ├── conftest.py          # Unit test fixtures
│   │   ├── test_config.py
│   │   ├── test_schemas.py
│   │   ├── test_<domain>.py
│   │   └── routers/
│   │       ├── __init__.py
│   │       ├── test_health.py
│   │       └── test_info.py
│   └── integration/             # Integration tests
│       ├── __init__.py
│       ├── conftest.py          # Integration fixtures
│       ├── test_endpoints.py
│       └── test_pipeline.py
├── config.yaml
└── pyproject.toml
```

**Root-level integration tests** (cross-service):

```
artwork-matcher/
├── services/
│   └── ...
└── tests/                       # Cross-service integration tests
    ├── __init__.py
    ├── conftest.py
    ├── test_full_pipeline.py    # Gateway → Embeddings → Search → Geometric
    └── test_service_health.py   # All services healthy
```

---

## pytest Configuration

### pyproject.toml

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
asyncio_mode = "auto"
addopts = "-v --tb=short --strict-markers"

# Custom markers
markers = [
    "unit: Unit tests (fast, isolated, no external dependencies)",
    "integration: Integration tests (require running services)",
    "slow: Tests that take more than 1 second",
]

# Default: run only unit tests (fast feedback)
# Use -m integration to run integration tests
filterwarnings = [
    "error",  # Treat warnings as errors
    "ignore::DeprecationWarning",
]
```

### Running Tests

```bash
# Run only unit tests (default, fast)
just test
# OR
uv run pytest tests/unit -m unit

# Run only integration tests (requires services running)
just test-integration
# OR
uv run pytest tests/integration -m integration

# Run all tests
uv run pytest

# Run with coverage
uv run pytest tests/unit --cov=src --cov-report=html --cov-fail-under=80

# Run specific test file
uv run pytest tests/unit/test_config.py -v

# Run tests matching pattern
uv run pytest -k "test_health" -v
```

### justfile Commands

```makefile
# Run unit tests (fast, no external dependencies)
test:
    @echo "\033[0;34m=== Running Unit Tests ===\033[0m"
    @uv run pytest tests/unit -m unit -v
    @echo "\033[0;32m✓ Unit tests passed\033[0m"

# Run unit tests with coverage
test-coverage:
    @echo "\033[0;34m=== Running Unit Tests with Coverage ===\033[0m"
    @uv run pytest tests/unit -m unit -v \
        --cov=src \
        --cov-report=html:reports/coverage/html \
        --cov-report=term \
        --cov-fail-under=80
    @echo "\033[0;32m✓ Coverage threshold met\033[0m"

# Run integration tests (requires running services)
test-integration:
    @echo "\033[0;34m=== Running Integration Tests ===\033[0m"
    @echo "Ensure services are running: just up"
    @uv run pytest tests/integration -m integration -v
    @echo "\033[0;32m✓ Integration tests passed\033[0m"

# Run all tests
test-all:
    @echo "\033[0;34m=== Running All Tests ===\033[0m"
    @uv run pytest -v
    @echo "\033[0;32m✓ All tests passed\033[0m"
```

---

## Unit Testing

### Core Principles

1. **Mock all external dependencies** — No network, no filesystem, no databases
2. **Test one thing per test** — Single assertion focus
3. **Use descriptive names** — `test_<what>_<condition>_<expected>`
4. **Arrange-Act-Assert pattern** — Clear test structure

### Testing Configuration

```python
# tests/unit/test_config.py
"""Unit tests for configuration management."""

import pytest
from pydantic import ValidationError

from embeddings_service.config import (
    Settings,
    load_yaml_config,
    is_sensitive_key,
    redact_sensitive_values,
    ConfigurationError,
)


class TestLoadYamlConfig:
    """Tests for YAML configuration loading."""

    def test_load_valid_config(self, tmp_path):
        """Valid YAML file loads successfully."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
service:
  name: test-service
  version: 0.1.0
server:
  host: localhost
  port: 8000
logging:
  level: INFO
  format: json
""")

        result = load_yaml_config(config_file)

        assert result["service"]["name"] == "test-service"
        assert result["server"]["port"] == 8000

    def test_missing_file_raises_error(self, tmp_path):
        """Missing config file raises ConfigurationError."""
        missing_file = tmp_path / "does_not_exist.yaml"

        with pytest.raises(ConfigurationError) as exc_info:
            load_yaml_config(missing_file)

        assert "not found" in str(exc_info.value)

    def test_empty_file_raises_error(self, tmp_path):
        """Empty config file raises ConfigurationError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")

        with pytest.raises(ConfigurationError) as exc_info:
            load_yaml_config(config_file)

        assert "empty" in str(exc_info.value)

    def test_invalid_yaml_raises_error(self, tmp_path):
        """Malformed YAML raises ConfigurationError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("invalid: yaml: content: [")

        with pytest.raises(ConfigurationError):
            load_yaml_config(config_file)


class TestSettings:
    """Tests for Settings validation."""

    def test_valid_config_creates_settings(self):
        """Valid configuration creates Settings instance."""
        config = {
            "service": {"name": "test", "version": "1.0.0"},
            "server": {"host": "0.0.0.0", "port": 8000},
            "logging": {"level": "INFO", "format": "json"},
        }

        settings = Settings(**config)

        assert settings.service.name == "test"
        assert settings.server.port == 8000

    def test_missing_required_field_raises_error(self):
        """Missing required field raises ValidationError."""
        config = {
            "service": {"name": "test"},  # Missing 'version'
            "server": {"host": "0.0.0.0", "port": 8000},
            "logging": {"level": "INFO", "format": "json"},
        }

        with pytest.raises(ValidationError) as exc_info:
            Settings(**config)

        assert "version" in str(exc_info.value)

    def test_extra_field_raises_error(self):
        """Extra field raises ValidationError (extra='forbid')."""
        config = {
            "service": {"name": "test", "version": "1.0.0"},
            "server": {"host": "0.0.0.0", "port": 8000, "unknown": "value"},
            "logging": {"level": "INFO", "format": "json"},
        }

        with pytest.raises(ValidationError) as exc_info:
            Settings(**config)

        assert "extra" in str(exc_info.value).lower()

    def test_invalid_port_raises_error(self):
        """Invalid port value raises ValidationError."""
        config = {
            "service": {"name": "test", "version": "1.0.0"},
            "server": {"host": "0.0.0.0", "port": "not_a_number"},
            "logging": {"level": "INFO", "format": "json"},
        }

        with pytest.raises(ValidationError):
            Settings(**config)


class TestSensitiveDataRedaction:
    """Tests for sensitive data filtering."""

    @pytest.mark.parametrize(
        "key,expected",
        [
            ("api_key", True),
            ("apikey", True),
            ("API_KEY", True),
            ("password", True),
            ("db_password", True),
            ("secret", True),
            ("client_secret", True),
            ("token", True),
            ("auth_token", True),
            ("credential", True),
            ("private_key", True),
            ("normal_field", False),
            ("username", False),
            ("host", False),
            ("port", False),
        ],
    )
    def test_is_sensitive_key(self, key, expected):
        """Correctly identifies sensitive keys."""
        assert is_sensitive_key(key) == expected

    def test_redact_sensitive_values_flat(self):
        """Redacts sensitive values in flat dict."""
        data = {
            "host": "localhost",
            "api_key": "secret123",
            "port": 8000,
        }

        result = redact_sensitive_values(data)

        assert result["host"] == "localhost"
        assert result["api_key"] == "[REDACTED]"
        assert result["port"] == 8000

    def test_redact_sensitive_values_nested(self):
        """Redacts sensitive values in nested dict."""
        data = {
            "database": {
                "host": "localhost",
                "password": "secret123",
            },
            "api": {
                "key": "abc123",
            },
        }

        result = redact_sensitive_values(data)

        assert result["database"]["host"] == "localhost"
        assert result["database"]["password"] == "[REDACTED]"
        assert result["api"]["key"] == "[REDACTED]"

    def test_redact_preserves_structure(self):
        """Redaction preserves original structure."""
        data = {
            "list_field": [1, 2, 3],
            "nested": {"deep": {"value": "keep"}},
        }

        result = redact_sensitive_values(data)

        assert result["list_field"] == [1, 2, 3]
        assert result["nested"]["deep"]["value"] == "keep"
```

### Testing Pydantic Schemas

```python
# tests/unit/test_schemas.py
"""Unit tests for request/response schemas."""

import pytest
from pydantic import ValidationError

from embeddings_service.schemas import EmbedRequest, EmbedResponse


class TestEmbedRequest:
    """Tests for embedding request schema."""

    def test_valid_request(self):
        """Valid request with required fields."""
        request = EmbedRequest(image="base64encodeddata")
        
        assert request.image == "base64encodeddata"
        assert request.image_id is None  # Optional field

    def test_valid_request_with_optional_fields(self):
        """Valid request with all fields."""
        request = EmbedRequest(
            image="base64encodeddata",
            image_id="test_001",
        )
        
        assert request.image_id == "test_001"

    def test_missing_required_field(self):
        """Missing required field raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            EmbedRequest()
        
        assert "image" in str(exc_info.value)

    def test_empty_image_rejected(self):
        """Empty image string is rejected."""
        with pytest.raises(ValidationError):
            EmbedRequest(image="")


class TestEmbedResponse:
    """Tests for embedding response schema."""

    def test_valid_response(self):
        """Valid response with all fields."""
        response = EmbedResponse(
            embedding=[0.1, 0.2, 0.3],
            dimension=3,
            image_id="test_001",
            processing_time_ms=45.2,
        )
        
        assert response.dimension == 3
        assert len(response.embedding) == 3

    def test_embedding_dimension_mismatch_allowed(self):
        """Schema doesn't validate embedding length matches dimension."""
        # This is intentional - validation happens in business logic
        response = EmbedResponse(
            embedding=[0.1, 0.2],
            dimension=768,  # Doesn't match embedding length
            processing_time_ms=45.2,
        )
        
        assert response.dimension == 768
```

### Testing Route Handlers (with Mocks)

```python
# tests/unit/routers/test_health.py
"""Unit tests for health endpoint."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from embeddings_service.app import create_app


@pytest.fixture
def mock_app_state():
    """Mock application state."""
    with patch("embeddings_service.routers.health.get_app_state") as mock:
        state = MagicMock()
        state.uptime_seconds = 123.45
        state.uptime_formatted = "2m 3s"
        mock.return_value = state
        yield state


@pytest.fixture
def client(mock_app_state, mock_settings):
    """Test client with mocked dependencies."""
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        """Health endpoint returns 200 OK."""
        response = client.get("/health")
        
        assert response.status_code == 200

    def test_health_returns_healthy_status(self, client):
        """Health endpoint returns healthy status."""
        response = client.get("/health")
        data = response.json()
        
        assert data["status"] == "healthy"

    def test_health_includes_uptime(self, client, mock_app_state):
        """Health endpoint includes uptime information."""
        response = client.get("/health")
        data = response.json()
        
        assert data["uptime_seconds"] == 123.45
        assert data["uptime"] == "2m 3s"

    def test_health_includes_system_time(self, client):
        """Health endpoint includes system time."""
        response = client.get("/health")
        data = response.json()
        
        assert "system_time" in data
        # Verify format: yyyy-mm-dd hh:mm
        import re
        assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$", data["system_time"])
```

### Testing Business Logic

```python
# tests/unit/test_embedding.py
"""Unit tests for embedding extraction logic."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from embeddings_service.model import (
    extract_embedding,
    preprocess_image,
    normalize_embedding,
)


class TestPreprocessImage:
    """Tests for image preprocessing."""

    def test_converts_rgba_to_rgb(self):
        """RGBA images are converted to RGB."""
        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        
        result = preprocess_image(rgba_image)
        
        assert result.mode == "RGB"

    def test_converts_grayscale_to_rgb(self):
        """Grayscale images are converted to RGB."""
        gray_image = Image.new("L", (100, 100), 128)
        
        result = preprocess_image(gray_image)
        
        assert result.mode == "RGB"

    def test_resizes_large_image(self):
        """Large images are resized to target size."""
        large_image = Image.new("RGB", (2000, 2000))
        
        result = preprocess_image(large_image, target_size=518)
        
        assert max(result.size) <= 518

    def test_preserves_aspect_ratio(self):
        """Aspect ratio is preserved during resize."""
        wide_image = Image.new("RGB", (1000, 500))
        
        result = preprocess_image(wide_image, target_size=518)
        
        # Original ratio is 2:1, should be preserved
        width, height = result.size
        ratio = width / height
        assert abs(ratio - 2.0) < 0.01


class TestNormalizeEmbedding:
    """Tests for embedding normalization."""

    def test_normalizes_to_unit_length(self):
        """Embedding is normalized to unit length."""
        embedding = np.array([3.0, 4.0])  # Length = 5
        
        result = normalize_embedding(embedding)
        
        length = np.linalg.norm(result)
        assert abs(length - 1.0) < 1e-6

    def test_handles_zero_vector(self):
        """Zero vector returns zero vector (no division by zero)."""
        embedding = np.array([0.0, 0.0, 0.0])
        
        result = normalize_embedding(embedding)
        
        assert np.allclose(result, [0.0, 0.0, 0.0])

    def test_preserves_direction(self):
        """Normalization preserves vector direction."""
        embedding = np.array([1.0, 2.0, 3.0])
        
        result = normalize_embedding(embedding)
        
        # Direction should be same (proportional)
        expected_direction = embedding / np.linalg.norm(embedding)
        assert np.allclose(result, expected_direction)


class TestExtractEmbedding:
    """Tests for embedding extraction."""

    @pytest.fixture
    def mock_model(self):
        """Mock DINOv2 model."""
        model = MagicMock()
        model.return_value.last_hidden_state = MagicMock()
        model.return_value.last_hidden_state.__getitem__ = MagicMock(
            return_value=MagicMock(
                squeeze=MagicMock(
                    return_value=MagicMock(
                        numpy=MagicMock(return_value=np.random.randn(768))
                    )
                )
            )
        )
        return model

    @pytest.fixture
    def mock_processor(self):
        """Mock image processor."""
        processor = MagicMock()
        processor.return_value = {"pixel_values": MagicMock()}
        return processor

    def test_returns_correct_dimension(self, mock_model, mock_processor):
        """Embedding has correct dimension."""
        image = Image.new("RGB", (100, 100))
        
        with patch.object(mock_model, "__call__"):
            # Setup mock to return 768-dim embedding
            mock_output = np.random.randn(768).astype(np.float32)
            mock_model.return_value.last_hidden_state = MagicMock()
            
            result = extract_embedding(
                image,
                model=mock_model,
                processor=mock_processor,
                dimension=768,
            )
        
        assert len(result) == 768

    def test_returns_normalized_embedding(self, mock_model, mock_processor):
        """Returned embedding is L2 normalized."""
        image = Image.new("RGB", (100, 100))
        
        result = extract_embedding(
            image,
            model=mock_model,
            processor=mock_processor,
            dimension=768,
        )
        
        length = np.linalg.norm(result)
        assert abs(length - 1.0) < 1e-5
```

---

## Integration Testing

### Core Principles

1. **Require running services** — Tests fail if services aren't available
2. **Use real HTTP calls** — No mocking of network layer
3. **Test contracts** — Verify API contracts are honored
4. **Isolate test data** — Don't pollute production data

### Integration Test Fixtures

```python
# tests/integration/conftest.py
"""Integration test fixtures."""

import os
import pytest
import httpx

# Service URLs (can be overridden via environment)
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8000")
EMBEDDINGS_URL = os.environ.get("EMBEDDINGS_URL", "http://localhost:8001")
SEARCH_URL = os.environ.get("SEARCH_URL", "http://localhost:8002")
GEOMETRIC_URL = os.environ.get("GEOMETRIC_URL", "http://localhost:8003")


def is_service_healthy(url: str, timeout: float = 5.0) -> bool:
    """Check if a service is healthy."""
    try:
        response = httpx.get(f"{url}/health", timeout=timeout)
        return response.status_code == 200
    except httpx.RequestError:
        return False


@pytest.fixture(scope="session")
def check_services():
    """
    Verify required services are running before tests.
    
    Fails fast with clear message if services aren't available.
    """
    services = {
        "gateway": GATEWAY_URL,
        "embeddings": EMBEDDINGS_URL,
        "search": SEARCH_URL,
        "geometric": GEOMETRIC_URL,
    }
    
    unavailable = []
    for name, url in services.items():
        if not is_service_healthy(url):
            unavailable.append(f"{name} ({url})")
    
    if unavailable:
        pytest.fail(
            f"Integration tests require running services.\n"
            f"Unavailable services: {', '.join(unavailable)}\n"
            f"Start services with: just up"
        )


@pytest.fixture(scope="session")
def gateway_url(check_services) -> str:
    """Gateway service URL."""
    return GATEWAY_URL


@pytest.fixture(scope="session")
def embeddings_url(check_services) -> str:
    """Embeddings service URL."""
    return EMBEDDINGS_URL


@pytest.fixture(scope="session")
def search_url(check_services) -> str:
    """Search service URL."""
    return SEARCH_URL


@pytest.fixture(scope="session")
def geometric_url(check_services) -> str:
    """Geometric service URL."""
    return GEOMETRIC_URL


@pytest.fixture(scope="session")
def http_client() -> httpx.Client:
    """Shared HTTP client for integration tests."""
    with httpx.Client(timeout=30.0) as client:
        yield client


@pytest.fixture(scope="session")
def async_http_client() -> httpx.AsyncClient:
    """Shared async HTTP client."""
    return httpx.AsyncClient(timeout=30.0)


@pytest.fixture
def sample_image_base64() -> str:
    """Base64-encoded sample image for testing."""
    import base64
    from PIL import Image
    from io import BytesIO
    
    # Create a simple test image
    image = Image.new("RGB", (100, 100), color="red")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
```

### Testing Service Health

```python
# tests/integration/test_service_health.py
"""Integration tests for service health endpoints."""

import pytest
import httpx


@pytest.mark.integration
class TestServiceHealth:
    """Verify all services are healthy and responding."""

    def test_gateway_health(self, gateway_url, http_client):
        """Gateway service is healthy."""
        response = http_client.get(f"{gateway_url}/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "uptime_seconds" in data
        assert "system_time" in data

    def test_embeddings_health(self, embeddings_url, http_client):
        """Embeddings service is healthy."""
        response = http_client.get(f"{embeddings_url}/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_search_health(self, search_url, http_client):
        """Search service is healthy."""
        response = http_client.get(f"{search_url}/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_geometric_health(self, geometric_url, http_client):
        """Geometric service is healthy."""
        response = http_client.get(f"{geometric_url}/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


@pytest.mark.integration
class TestServiceInfo:
    """Verify service info endpoints return expected configuration."""

    def test_embeddings_info_includes_model(self, embeddings_url, http_client):
        """Embeddings info includes model configuration."""
        response = http_client.get(f"{embeddings_url}/info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "embeddings"
        assert "config" in data
        # Verify expected config structure
        config = data["config"]
        assert "model" in config or "service" in config

    def test_search_info_includes_index_count(self, search_url, http_client):
        """Search info includes index statistics."""
        response = http_client.get(f"{search_url}/info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "search"
```

### Testing API Endpoints

```python
# tests/integration/test_embeddings_api.py
"""Integration tests for embeddings service API."""

import pytest
import httpx


@pytest.mark.integration
class TestEmbedEndpoint:
    """Tests for POST /embed endpoint."""

    def test_embed_returns_embedding(
        self,
        embeddings_url,
        http_client,
        sample_image_base64,
    ):
        """Embed endpoint returns valid embedding."""
        response = http_client.post(
            f"{embeddings_url}/embed",
            json={"image": sample_image_base64},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        assert "dimension" in data
        assert len(data["embedding"]) == data["dimension"]

    def test_embed_returns_768_dimensions(
        self,
        embeddings_url,
        http_client,
        sample_image_base64,
    ):
        """DINOv2-base returns 768-dimensional embeddings."""
        response = http_client.post(
            f"{embeddings_url}/embed",
            json={"image": sample_image_base64},
        )
        
        assert response.status_code == 200
        assert response.json()["dimension"] == 768

    def test_embed_returns_normalized_vector(
        self,
        embeddings_url,
        http_client,
        sample_image_base64,
    ):
        """Embedding is L2 normalized (unit length)."""
        import numpy as np
        
        response = http_client.post(
            f"{embeddings_url}/embed",
            json={"image": sample_image_base64},
        )
        
        embedding = np.array(response.json()["embedding"])
        length = np.linalg.norm(embedding)
        assert abs(length - 1.0) < 1e-5

    def test_embed_includes_processing_time(
        self,
        embeddings_url,
        http_client,
        sample_image_base64,
    ):
        """Response includes processing time."""
        response = http_client.post(
            f"{embeddings_url}/embed",
            json={"image": sample_image_base64},
        )
        
        data = response.json()
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] > 0

    def test_embed_echoes_image_id(
        self,
        embeddings_url,
        http_client,
        sample_image_base64,
    ):
        """Image ID is echoed in response."""
        response = http_client.post(
            f"{embeddings_url}/embed",
            json={
                "image": sample_image_base64,
                "image_id": "test_123",
            },
        )
        
        assert response.json()["image_id"] == "test_123"

    def test_embed_invalid_image_returns_400(
        self,
        embeddings_url,
        http_client,
    ):
        """Invalid image data returns 400 error."""
        response = http_client.post(
            f"{embeddings_url}/embed",
            json={"image": "not_valid_base64!!!"},
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"] in ("invalid_image", "decode_error")

    def test_embed_missing_image_returns_422(
        self,
        embeddings_url,
        http_client,
    ):
        """Missing required field returns 422 error."""
        response = http_client.post(
            f"{embeddings_url}/embed",
            json={},
        )
        
        assert response.status_code == 422
```

### Testing Search Service

```python
# tests/integration/test_search_api.py
"""Integration tests for search service API."""

import pytest
import httpx
import numpy as np


@pytest.mark.integration
class TestSearchEndpoint:
    """Tests for POST /search endpoint."""

    @pytest.fixture
    def sample_embedding(self) -> list[float]:
        """Generate a sample 768-dimensional embedding."""
        embedding = np.random.randn(768).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        return embedding.tolist()

    def test_search_returns_results(
        self,
        search_url,
        http_client,
        sample_embedding,
    ):
        """Search endpoint returns results array."""
        response = http_client.post(
            f"{search_url}/search",
            json={
                "embedding": sample_embedding,
                "k": 5,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_search_respects_k_parameter(
        self,
        search_url,
        http_client,
        sample_embedding,
    ):
        """Search returns at most k results."""
        response = http_client.post(
            f"{search_url}/search",
            json={
                "embedding": sample_embedding,
                "k": 3,
            },
        )
        
        data = response.json()
        assert len(data["results"]) <= 3

    def test_search_results_have_required_fields(
        self,
        search_url,
        http_client,
        sample_embedding,
    ):
        """Search results include required fields."""
        response = http_client.post(
            f"{search_url}/search",
            json={
                "embedding": sample_embedding,
                "k": 5,
            },
        )
        
        data = response.json()
        if data["results"]:  # If index has items
            result = data["results"][0]
            assert "object_id" in result
            assert "score" in result
            assert "rank" in result

    def test_search_wrong_dimension_returns_400(
        self,
        search_url,
        http_client,
    ):
        """Wrong embedding dimension returns 400 error."""
        wrong_dimension_embedding = [0.1] * 512  # Should be 768
        
        response = http_client.post(
            f"{search_url}/search",
            json={
                "embedding": wrong_dimension_embedding,
                "k": 5,
            },
        )
        
        assert response.status_code == 400
        assert response.json()["error"] == "dimension_mismatch"
```

### Testing Full Pipeline

```python
# tests/integration/test_full_pipeline.py
"""Integration tests for the complete identification pipeline."""

import base64
import pytest
import httpx
from pathlib import Path


@pytest.mark.integration
class TestIdentificationPipeline:
    """End-to-end tests for artwork identification."""

    @pytest.fixture
    def real_test_image(self) -> str:
        """Load a real test image from the test data."""
        # Look for test images in data/pictures/
        test_images_dir = Path("data/pictures")
        if test_images_dir.exists():
            for image_path in test_images_dir.glob("*.jpg"):
                return base64.b64encode(image_path.read_bytes()).decode("ascii")
        
        # Fallback to generated image
        from PIL import Image
        from io import BytesIO
        
        image = Image.new("RGB", (200, 200), color="blue")
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def test_identify_returns_match_or_no_match(
        self,
        gateway_url,
        http_client,
        real_test_image,
    ):
        """Identify endpoint returns valid response structure."""
        response = http_client.post(
            f"{gateway_url}/identify",
            json={"image": real_test_image},
            timeout=60.0,  # Pipeline can be slow
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "match" in data  # Can be None
        assert "timing" in data

    def test_identify_timing_includes_all_stages(
        self,
        gateway_url,
        http_client,
        real_test_image,
    ):
        """Timing breakdown includes all pipeline stages."""
        response = http_client.post(
            f"{gateway_url}/identify",
            json={"image": real_test_image},
            timeout=60.0,
        )
        
        timing = response.json()["timing"]
        assert "embedding_ms" in timing
        assert "search_ms" in timing
        assert "total_ms" in timing
        # geometric_ms may be 0 if no candidates

    def test_identify_with_geometric_verification(
        self,
        gateway_url,
        http_client,
        real_test_image,
    ):
        """Identify with geometric verification enabled."""
        response = http_client.post(
            f"{gateway_url}/identify",
            json={
                "image": real_test_image,
                "options": {"geometric_verification": True},
            },
            timeout=60.0,
        )
        
        assert response.status_code == 200
        data = response.json()
        if data["match"]:
            assert "verification_method" in data["match"]

    def test_pipeline_handles_invalid_image(
        self,
        gateway_url,
        http_client,
    ):
        """Pipeline returns proper error for invalid image."""
        response = http_client.post(
            f"{gateway_url}/identify",
            json={"image": "not_valid_base64"},
            timeout=60.0,
        )
        
        # Should get 400 or 502 (depending on where validation happens)
        assert response.status_code in (400, 502)
        assert "error" in response.json()
```

---

## Fixtures and Factories

### Test Data Factories

```python
# tests/factories.py
"""Test data factories for generating test fixtures."""

import base64
import numpy as np
from io import BytesIO
from PIL import Image


def create_test_image(
    width: int = 100,
    height: int = 100,
    color: str = "red",
    format: str = "JPEG",
) -> bytes:
    """Create a test image as bytes."""
    image = Image.new("RGB", (width, height), color=color)
    buffer = BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def create_test_image_base64(
    width: int = 100,
    height: int = 100,
    color: str = "red",
) -> str:
    """Create a base64-encoded test image."""
    image_bytes = create_test_image(width, height, color)
    return base64.b64encode(image_bytes).decode("ascii")


def create_random_embedding(dimension: int = 768, normalized: bool = True) -> list[float]:
    """Create a random embedding vector."""
    embedding = np.random.randn(dimension).astype(np.float32)
    if normalized:
        embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()


def create_mock_search_results(count: int = 5) -> list[dict]:
    """Create mock search results."""
    return [
        {
            "object_id": f"object_{i:03d}",
            "score": 0.95 - (i * 0.05),
            "rank": i + 1,
            "metadata": {"name": f"Artwork {i}"},
        }
        for i in range(count)
    ]
```

### Shared Unit Test Fixtures

```python
# tests/unit/conftest.py
"""Shared fixtures for unit tests."""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_settings():
    """Mock settings for unit tests."""
    with patch("embeddings_service.config.get_settings") as mock:
        settings = MagicMock()
        settings.service.name = "test-service"
        settings.service.version = "0.1.0"
        settings.server.host = "0.0.0.0"
        settings.server.port = 8000
        settings.logging.level = "INFO"
        settings.logging.format = "json"
        mock.return_value = settings
        yield settings


@pytest.fixture
def mock_logger():
    """Mock logger for unit tests."""
    with patch("embeddings_service.logging.get_logger") as mock:
        logger = MagicMock()
        mock.return_value = logger
        yield logger


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing."""
    from PIL import Image
    return Image.new("RGB", (100, 100), color="red")


@pytest.fixture
def sample_image_base64():
    """Create a base64-encoded sample image."""
    from tests.factories import create_test_image_base64
    return create_test_image_base64()
```

---

## Mocking Patterns

### Mocking External Services

```python
# Pattern: Mock HTTP clients for unit testing

from unittest.mock import AsyncMock, MagicMock, patch
import pytest


@pytest.fixture
def mock_embeddings_client():
    """Mock embeddings service HTTP client."""
    with patch("gateway.clients.embeddings_client") as mock:
        client = AsyncMock()
        
        # Setup successful response
        client.post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "embedding": [0.1] * 768,
                "dimension": 768,
                "processing_time_ms": 45.0,
            }),
        )
        
        mock.return_value = client
        yield client


def test_gateway_calls_embeddings_service(mock_embeddings_client):
    """Gateway calls embeddings service for embedding extraction."""
    # ... test implementation
    pass
```

### Mocking Models

```python
# Pattern: Mock ML models for fast unit tests

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_dinov2_model():
    """Mock DINOv2 model for unit testing."""
    with patch("embeddings_service.model.AutoModel") as mock_model_class:
        # Create mock model instance
        model = MagicMock()
        
        # Setup forward pass to return fake embeddings
        def fake_forward(**kwargs):
            result = MagicMock()
            # Shape: [batch, sequence, hidden_dim]
            result.last_hidden_state = MagicMock()
            result.last_hidden_state.__getitem__ = MagicMock(
                return_value=MagicMock(
                    squeeze=MagicMock(
                        return_value=MagicMock(
                            numpy=MagicMock(
                                return_value=np.random.randn(768).astype(np.float32)
                            )
                        )
                    )
                )
            )
            return result
        
        model.side_effect = fake_forward
        mock_model_class.from_pretrained.return_value = model
        
        yield model
```

### Mocking File System

```python
# Pattern: Use tmp_path fixture for file system tests

def test_config_loads_from_file(tmp_path):
    """Configuration loads from YAML file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
service:
  name: test
  version: 1.0.0
server:
  host: localhost
  port: 8000
logging:
  level: DEBUG
  format: json
""")
    
    from embeddings_service.config import load_yaml_config
    
    result = load_yaml_config(config_file)
    
    assert result["service"]["name"] == "test"
```

---

## Async Testing

### Testing Async Endpoints

```python
# tests/unit/test_async_endpoints.py
"""Tests for async endpoint handlers."""

import pytest
from httpx import AsyncClient
from embeddings_service.app import create_app


@pytest.fixture
async def async_client():
    """Async test client."""
    app = create_app()
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_health_async(async_client):
    """Health endpoint works with async client."""
    response = await async_client.get("/health")
    
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_embed_async(async_client, sample_image_base64):
    """Embed endpoint works with async client."""
    response = await async_client.post(
        "/embed",
        json={"image": sample_image_base64},
    )
    
    assert response.status_code == 200
```

### Testing Async Functions

```python
# tests/unit/test_async_functions.py
"""Tests for async business logic."""

import pytest
from unittest.mock import AsyncMock


@pytest.mark.asyncio
async def test_fetch_embedding_from_service():
    """Test async embedding fetch."""
    from gateway.clients import fetch_embedding
    
    mock_client = AsyncMock()
    mock_client.post.return_value.json.return_value = {
        "embedding": [0.1] * 768,
        "dimension": 768,
    }
    
    result = await fetch_embedding(
        client=mock_client,
        image_base64="test_image",
    )
    
    assert len(result) == 768
    mock_client.post.assert_called_once()
```

---

## Coverage Requirements

### Coverage Configuration

```toml
# pyproject.toml

[tool.coverage.run]
source = ["src"]
branch = true
omit = [
    "tests/*",
    "**/__init__.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
    "if __name__ == .__main__.:",
    "@abstractmethod",
]
fail_under = 80
show_missing = true
skip_covered = false

[tool.coverage.html]
directory = "reports/coverage/html"
```

### Coverage Targets

| Component | Minimum Coverage | Target |
|-----------|------------------|--------|
| Configuration | 90% | 95% |
| Schemas | 85% | 90% |
| Business Logic | 80% | 90% |
| Route Handlers | 75% | 85% |
| Error Handling | 90% | 95% |
| **Overall** | **80%** | **85%** |

### Running Coverage

```bash
# Unit test coverage
uv run pytest tests/unit \
    --cov=src \
    --cov-report=html:reports/coverage/html \
    --cov-report=term \
    --cov-fail-under=80

# View HTML report
open reports/coverage/html/index.html
```

---

## CI Pipeline Integration

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Test

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    name: Unit Tests
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v4
      
      - name: Install dependencies
        run: uv sync --all-extras
      
      - name: Run unit tests
        run: |
          uv run pytest tests/unit -m unit \
            --cov=src \
            --cov-report=xml \
            --cov-fail-under=80
      
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: coverage.xml

  integration-tests:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: unit-tests  # Only run if unit tests pass
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v4
      
      - name: Start services
        run: docker compose up -d
      
      - name: Wait for services
        run: |
          for i in {1..30}; do
            if curl -s http://localhost:8000/health > /dev/null; then
              echo "Services ready"
              break
            fi
            echo "Waiting for services... ($i/30)"
            sleep 2
          done
      
      - name: Run integration tests
        run: uv run pytest tests/integration -m integration -v
      
      - name: Stop services
        if: always()
        run: docker compose down
```

### Test Markers in CI

```yaml
# Run only fast unit tests on every push
unit-tests:
  script:
    - uv run pytest -m "unit and not slow" -v

# Run all tests including slow ones before merge
full-tests:
  script:
    - uv run pytest -m "unit" -v
    - docker compose up -d
    - uv run pytest -m "integration" -v
```

---

## Complete Examples

### Complete Unit Test File

```python
# tests/unit/test_embeddings_service.py
"""
Complete unit test suite for embeddings service.

These tests run in isolation without any external dependencies.
All I/O, network, and model inference is mocked.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from PIL import Image

from tests.factories import (
    create_test_image_base64,
    create_random_embedding,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mock_settings():
    """Mock application settings."""
    with patch("embeddings_service.config.get_settings") as mock:
        settings = MagicMock()
        settings.service.name = "embeddings"
        settings.service.version = "0.1.0"
        settings.server.host = "0.0.0.0"
        settings.server.port = 8000
        settings.logging.level = "INFO"
        settings.logging.format = "json"
        settings.model.name = "facebook/dinov2-base"
        settings.model.device = "cpu"
        settings.model.embedding_dimension = 768
        settings.preprocessing.image_size = 518
        settings.preprocessing.normalize = True
        mock.return_value = settings
        yield settings


@pytest.fixture
def mock_app_state():
    """Mock application state."""
    with patch("embeddings_service.core.state.get_app_state") as mock:
        state = MagicMock()
        state.uptime_seconds = 100.0
        state.uptime_formatted = "1m 40s"
        mock.return_value = state
        yield state


@pytest.fixture
def mock_model():
    """Mock DINOv2 model."""
    with patch("embeddings_service.model.model") as mock:
        def fake_inference(*args, **kwargs):
            result = MagicMock()
            embedding = np.random.randn(768).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            result.last_hidden_state = [[MagicMock(
                squeeze=MagicMock(return_value=MagicMock(
                    numpy=MagicMock(return_value=embedding)
                ))
            )]]
            return result
        mock.side_effect = fake_inference
        yield mock


@pytest.fixture
def client(mock_settings, mock_app_state, mock_model):
    """Test client with all dependencies mocked."""
    from embeddings_service.app import create_app
    app = create_app()
    return TestClient(app)


# ============================================================
# Health Endpoint Tests
# ============================================================

class TestHealthEndpoint:
    """Unit tests for GET /health."""

    def test_returns_200(self, client):
        """Health check returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_healthy_status(self, client):
        """Health check returns healthy status."""
        response = client.get("/health")
        assert response.json()["status"] == "healthy"

    def test_includes_uptime_seconds(self, client):
        """Response includes uptime in seconds."""
        response = client.get("/health")
        assert response.json()["uptime_seconds"] == 100.0

    def test_includes_formatted_uptime(self, client):
        """Response includes human-readable uptime."""
        response = client.get("/health")
        assert response.json()["uptime"] == "1m 40s"

    def test_includes_system_time(self, client):
        """Response includes current system time."""
        response = client.get("/health")
        system_time = response.json()["system_time"]
        # Format: yyyy-mm-dd hh:mm
        assert len(system_time) == 16
        assert system_time[4] == "-"
        assert system_time[10] == " "


# ============================================================
# Info Endpoint Tests
# ============================================================

class TestInfoEndpoint:
    """Unit tests for GET /info."""

    def test_returns_200(self, client):
        """Info endpoint returns 200 OK."""
        response = client.get("/info")
        assert response.status_code == 200

    def test_returns_service_name(self, client):
        """Info includes service name."""
        response = client.get("/info")
        assert response.json()["service"] == "embeddings"

    def test_returns_version(self, client):
        """Info includes service version."""
        response = client.get("/info")
        assert response.json()["version"] == "0.1.0"

    def test_returns_config(self, client):
        """Info includes configuration."""
        response = client.get("/info")
        assert "config" in response.json()


# ============================================================
# Embed Endpoint Tests
# ============================================================

class TestEmbedEndpoint:
    """Unit tests for POST /embed."""

    def test_returns_200_for_valid_image(self, client):
        """Valid image returns 200 OK."""
        image_b64 = create_test_image_base64()
        
        response = client.post("/embed", json={"image": image_b64})
        
        assert response.status_code == 200

    def test_returns_embedding_array(self, client):
        """Response includes embedding array."""
        image_b64 = create_test_image_base64()
        
        response = client.post("/embed", json={"image": image_b64})
        
        assert "embedding" in response.json()
        assert isinstance(response.json()["embedding"], list)

    def test_embedding_has_correct_dimension(self, client):
        """Embedding has 768 dimensions."""
        image_b64 = create_test_image_base64()
        
        response = client.post("/embed", json={"image": image_b64})
        
        assert response.json()["dimension"] == 768
        assert len(response.json()["embedding"]) == 768

    def test_embedding_is_normalized(self, client):
        """Embedding is L2 normalized."""
        image_b64 = create_test_image_base64()
        
        response = client.post("/embed", json={"image": image_b64})
        
        embedding = np.array(response.json()["embedding"])
        length = np.linalg.norm(embedding)
        assert abs(length - 1.0) < 1e-5

    def test_includes_processing_time(self, client):
        """Response includes processing time."""
        image_b64 = create_test_image_base64()
        
        response = client.post("/embed", json={"image": image_b64})
        
        assert "processing_time_ms" in response.json()
        assert response.json()["processing_time_ms"] >= 0

    def test_echoes_image_id(self, client):
        """Image ID is echoed in response."""
        image_b64 = create_test_image_base64()
        
        response = client.post(
            "/embed",
            json={"image": image_b64, "image_id": "test_123"},
        )
        
        assert response.json()["image_id"] == "test_123"

    def test_missing_image_returns_422(self, client):
        """Missing image field returns 422."""
        response = client.post("/embed", json={})
        
        assert response.status_code == 422

    def test_empty_image_returns_400(self, client):
        """Empty image string returns 400."""
        response = client.post("/embed", json={"image": ""})
        
        assert response.status_code == 400

    def test_invalid_base64_returns_400(self, client):
        """Invalid base64 returns 400."""
        response = client.post("/embed", json={"image": "not-valid!!!"})
        
        assert response.status_code == 400
        assert response.json()["error"] in ("decode_error", "invalid_image")


# ============================================================
# Error Handling Tests
# ============================================================

class TestErrorHandling:
    """Unit tests for error handling."""

    def test_unhandled_exception_returns_500(self, client, mock_model):
        """Unhandled exception returns 500 with safe message."""
        mock_model.side_effect = RuntimeError("Unexpected error")
        image_b64 = create_test_image_base64()
        
        response = client.post("/embed", json={"image": image_b64})
        
        assert response.status_code == 500
        assert response.json()["error"] == "internal_error"
        # Should not expose internal error details
        assert "Unexpected error" not in response.json()["message"]
```

### Complete Integration Test File

```python
# tests/integration/test_embeddings_integration.py
"""
Integration tests for embeddings service.

These tests require the embeddings service to be running.
Run with: just test-integration
"""

import base64
import numpy as np
import pytest
import httpx
from pathlib import Path
from PIL import Image
from io import BytesIO


# ============================================================
# Test Configuration
# ============================================================

EMBEDDINGS_URL = "http://localhost:8001"


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def service_url():
    """Get service URL and verify it's running."""
    try:
        response = httpx.get(f"{EMBEDDINGS_URL}/health", timeout=5.0)
        if response.status_code != 200:
            pytest.skip(f"Embeddings service not healthy: {response.status_code}")
    except httpx.RequestError as e:
        pytest.skip(f"Embeddings service not available: {e}")
    
    return EMBEDDINGS_URL


@pytest.fixture(scope="module")
def http_client():
    """Shared HTTP client for tests."""
    with httpx.Client(timeout=30.0) as client:
        yield client


@pytest.fixture
def test_image_base64():
    """Create a test image."""
    image = Image.new("RGB", (200, 200), color="blue")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.fixture
def real_artwork_image():
    """Load a real artwork image if available."""
    data_dir = Path("data/objects")
    if data_dir.exists():
        for image_path in data_dir.glob("*.jpg"):
            return base64.b64encode(image_path.read_bytes()).decode("ascii")
    
    # Fallback to generated image
    image = Image.new("RGB", (500, 500), color="green")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


# ============================================================
# Health & Info Tests
# ============================================================

@pytest.mark.integration
class TestServiceAvailability:
    """Tests for service health and info endpoints."""

    def test_health_endpoint_responds(self, service_url, http_client):
        """Health endpoint is accessible."""
        response = http_client.get(f"{service_url}/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_health_includes_uptime(self, service_url, http_client):
        """Health response includes uptime information."""
        response = http_client.get(f"{service_url}/health")
        data = response.json()
        
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] > 0
        assert "uptime" in data

    def test_info_endpoint_responds(self, service_url, http_client):
        """Info endpoint is accessible."""
        response = http_client.get(f"{service_url}/info")
        
        assert response.status_code == 200
        assert response.json()["service"] == "embeddings"

    def test_info_includes_model_config(self, service_url, http_client):
        """Info includes model configuration."""
        response = http_client.get(f"{service_url}/info")
        config = response.json()["config"]
        
        # Should have model configuration
        assert "model" in config or "service" in config


# ============================================================
# Embedding Extraction Tests
# ============================================================

@pytest.mark.integration
class TestEmbeddingExtraction:
    """Tests for the /embed endpoint."""

    def test_embed_returns_embedding(
        self, service_url, http_client, test_image_base64
    ):
        """Embed endpoint returns an embedding vector."""
        response = http_client.post(
            f"{service_url}/embed",
            json={"image": test_image_base64},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        assert len(data["embedding"]) > 0

    def test_embedding_dimension_is_768(
        self, service_url, http_client, test_image_base64
    ):
        """DINOv2-base produces 768-dimensional embeddings."""
        response = http_client.post(
            f"{service_url}/embed",
            json={"image": test_image_base64},
        )
        
        data = response.json()
        assert data["dimension"] == 768
        assert len(data["embedding"]) == 768

    def test_embedding_is_normalized(
        self, service_url, http_client, test_image_base64
    ):
        """Embedding vector has unit length."""
        response = http_client.post(
            f"{service_url}/embed",
            json={"image": test_image_base64},
        )
        
        embedding = np.array(response.json()["embedding"])
        length = np.linalg.norm(embedding)
        
        assert abs(length - 1.0) < 1e-5, f"Embedding length {length} != 1.0"

    def test_same_image_produces_same_embedding(
        self, service_url, http_client, test_image_base64
    ):
        """Same image produces identical embedding (deterministic)."""
        response1 = http_client.post(
            f"{service_url}/embed",
            json={"image": test_image_base64},
        )
        response2 = http_client.post(
            f"{service_url}/embed",
            json={"image": test_image_base64},
        )
        
        emb1 = np.array(response1.json()["embedding"])
        emb2 = np.array(response2.json()["embedding"])
        
        assert np.allclose(emb1, emb2, atol=1e-5)

    def test_different_images_produce_different_embeddings(
        self, service_url, http_client
    ):
        """Different images produce different embeddings."""
        # Create two different images
        image1 = Image.new("RGB", (100, 100), color="red")
        image2 = Image.new("RGB", (100, 100), color="blue")
        
        buffer1 = BytesIO()
        buffer2 = BytesIO()
        image1.save(buffer1, format="JPEG")
        image2.save(buffer2, format="JPEG")
        
        b64_1 = base64.b64encode(buffer1.getvalue()).decode()
        b64_2 = base64.b64encode(buffer2.getvalue()).decode()
        
        response1 = http_client.post(
            f"{service_url}/embed",
            json={"image": b64_1},
        )
        response2 = http_client.post(
            f"{service_url}/embed",
            json={"image": b64_2},
        )
        
        emb1 = np.array(response1.json()["embedding"])
        emb2 = np.array(response2.json()["embedding"])
        
        # Embeddings should be different (cosine similarity < 1)
        similarity = np.dot(emb1, emb2)
        assert similarity < 0.99, "Different images should have different embeddings"

    def test_embed_with_real_artwork(
        self, service_url, http_client, real_artwork_image
    ):
        """Embedding extraction works with real artwork images."""
        response = http_client.post(
            f"{service_url}/embed",
            json={"image": real_artwork_image},
        )
        
        assert response.status_code == 200
        assert response.json()["dimension"] == 768


# ============================================================
# Performance Tests
# ============================================================

@pytest.mark.integration
@pytest.mark.slow
class TestPerformance:
    """Performance-related integration tests."""

    def test_embedding_latency_under_threshold(
        self, service_url, http_client, test_image_base64
    ):
        """Embedding extraction completes within acceptable time."""
        import time
        
        start = time.perf_counter()
        response = http_client.post(
            f"{service_url}/embed",
            json={"image": test_image_base64},
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        assert response.status_code == 200
        # Should complete within 5 seconds even on CPU
        assert elapsed_ms < 5000, f"Embedding took {elapsed_ms}ms"

    def test_concurrent_requests(
        self, service_url, test_image_base64
    ):
        """Service handles concurrent requests."""
        import concurrent.futures
        
        def make_request():
            with httpx.Client(timeout=30.0) as client:
                return client.post(
                    f"{service_url}/embed",
                    json={"image": test_image_base64},
                )
        
        # Send 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            responses = [f.result() for f in futures]
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200


# ============================================================
# Error Handling Tests
# ============================================================

@pytest.mark.integration
class TestErrorHandling:
    """Tests for error handling in the service."""

    def test_invalid_base64_returns_400(self, service_url, http_client):
        """Invalid base64 data returns 400 error."""
        response = http_client.post(
            f"{service_url}/embed",
            json={"image": "not-valid-base64!!!"},
        )
        
        assert response.status_code == 400
        assert "error" in response.json()

    def test_empty_image_returns_400(self, service_url, http_client):
        """Empty image returns 400 error."""
        response = http_client.post(
            f"{service_url}/embed",
            json={"image": ""},
        )
        
        assert response.status_code == 400

    def test_missing_image_field_returns_422(self, service_url, http_client):
        """Missing required field returns 422 error."""
        response = http_client.post(
            f"{service_url}/embed",
            json={},
        )
        
        assert response.status_code == 422

    def test_corrupt_image_returns_400(self, service_url, http_client):
        """Corrupt image data returns 400 error."""
        # Valid base64 but not a valid image
        corrupt_data = base64.b64encode(b"not an image").decode()
        
        response = http_client.post(
            f"{service_url}/embed",
            json={"image": corrupt_data},
        )
        
        assert response.status_code == 400
        assert response.json()["error"] in ("invalid_image", "decode_error")
```

---

## Summary

### Quick Reference

| Test Type | Marker | Command | Dependencies |
|-----------|--------|---------|--------------|
| Unit | `@pytest.mark.unit` | `just test` | None (all mocked) |
| Integration | `@pytest.mark.integration` | `just test-integration` | Running services |
| Slow | `@pytest.mark.slow` | Excluded by default | Varies |

### Key Principles

1. **Unit tests** = Fast, isolated, mock everything external
2. **Integration tests** = Real services, real HTTP, verify contracts
3. **80% coverage minimum** for unit tests
4. **Fail fast** with clear error messages
5. **pytest markers** to separate test types
6. **Factories** for consistent test data generation

### File Organization

```
tests/
├── conftest.py           # Shared fixtures
├── factories.py          # Test data factories
├── unit/                 # Fast, isolated tests
│   ├── conftest.py       # Unit test fixtures (mocks)
│   └── test_*.py
└── integration/          # Service integration tests
    ├── conftest.py       # Integration fixtures (URLs, clients)
    └── test_*.py
```
