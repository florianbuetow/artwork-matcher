# Gateway Service Implementation Task

## Overview

Implement the **Gateway Service** for the artwork-matcher project. This is the public-facing API that orchestrates the artwork identification pipeline, coordinating requests across the internal services (Embeddings, Search, Geometric) and presenting a unified interface to clients.

**Service Identity:**
- **Name:** gateway
- **Port:** 8000
- **Role:** API orchestration, pipeline coordination, public entry point

---

## Reference Documentation

Read and follow these documents in order:

1. **@docs/api/gateway_service_api_spec.md** - Complete API specification including:
   - All endpoints: `/health`, `/info`, `/identify`, `/objects`, `/objects/{id}`, `/objects/{id}/image`
   - Request/response schemas with examples
   - Pipeline orchestration strategy
   - Error handling and propagation
   - Timeout strategy
   - Confidence score calculation
   - CORS configuration

2. **@docs/api/uniform_api_structure.md** - Common API conventions:
   - Health/info endpoint patterns
   - Error response format (`error`, `message`, `details`)
   - Processing time tracking
   - HTTP status codes

3. **@docs/implementation-guides/fastapi_service_template.md** - Service structure:
   - Directory layout (`src/gateway_service/...`)
   - Configuration management (Pydantic + YAML, no defaults in code)
   - Structured JSON logging
   - Application factory pattern
   - Router organization

4. **@docs/implementation-guides/config_pattern.md** - Configuration pattern:
   - How to load YAML and validate with Pydantic
   - Environment variable overrides
   - Fail-fast on invalid config

5. **@docs/implementation-guides/cors_implementation_guide.md** - CORS setup for browser access

6. **@CLAUDE.md** - Development rules:
   - Use `uv run` for all Python execution
   - Run tests after every change
   - No hardcoded defaults
   - Git commit guidelines

---

## Existing Files

The following already exist and should be used:
- `services/gateway/config.yaml` - Configuration (already populated)
- `services/gateway/pyproject.toml` - Dependencies including `httpx`
- `services/gateway/justfile` - Build/run commands
- `services/gateway/src/` - Directory exists but needs population

---

## Files to Create

Following the structure in `@docs/implementation-guides/fastapi_service_template.md`:

```
services/gateway/src/gateway_service/
├── __init__.py
├── main.py              # Entry point with main()
├── app.py               # FastAPI app factory
├── config.py            # Pydantic settings loading config.yaml
├── logging.py           # Structured JSON logging
├── schemas.py           # Pydantic request/response models
├── core/
│   ├── __init__.py
│   ├── lifespan.py      # Startup (validate backends) / shutdown
│   ├── state.py         # App state (uptime)
│   └── exceptions.py    # ServiceError and BackendError handlers
├── routers/
│   ├── __init__.py
│   ├── health.py        # GET /health (with backend status)
│   ├── info.py          # GET /info (with backend info)
│   ├── identify.py      # POST /identify (main pipeline)
│   └── objects.py       # GET /objects, /objects/{id}, /objects/{id}/image
└── clients/
    ├── __init__.py
    ├── base.py          # Base HTTP client with error handling
    ├── embeddings.py    # Embeddings service client
    ├── search.py        # Search service client
    └── geometric.py     # Geometric service client
```

Also create:
- `services/gateway/tests/conftest.py` - Test fixtures with mocked backends
- `services/gateway/tests/test_health.py` - Health endpoint tests
- `services/gateway/tests/test_identify.py` - Identification pipeline tests

---

## Key Implementation Details

### 1. Backend Clients (`clients/`)

Create async HTTP clients for each backend service:

```python
# clients/base.py
class BackendClient:
    """Base class for backend service clients."""

    def __init__(self, base_url: str, timeout: float, service_name: str):
        self.base_url = base_url
        self.service_name = service_name
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout, connect=5.0),
        )

    async def health_check(self) -> str:
        """Check backend health. Returns status string."""
        try:
            response = await self.client.get("/health")
            response.raise_for_status()
            return response.json()["status"]
        except Exception:
            return "unavailable"

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        """Make request with standardized error handling."""
        try:
            response = await self.client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            raise BackendError(
                error="backend_timeout",
                message=f"{self.service_name} service timed out",
                status_code=504,
                details={"backend": self.service_name},
            )
        except httpx.ConnectError:
            raise BackendError(
                error="backend_unavailable",
                message=f"{self.service_name} service is not responding",
                status_code=502,
                details={"backend": self.service_name},
            )
        except httpx.HTTPStatusError as e:
            backend_error = e.response.json()
            raise BackendError(
                error="backend_error",
                message=f"{self.service_name} service error: {backend_error['message']}",
                status_code=502,
                details={
                    "backend": self.service_name,
                    "backend_error": backend_error["error"],
                    "backend_message": backend_error["message"],
                },
            )
```

```python
# clients/embeddings.py
class EmbeddingsClient(BackendClient):
    """Client for Embeddings service."""

    async def embed(self, image_b64: str, image_id: str | None = None) -> list[float]:
        """Extract embedding from image."""
        result = await self._request(
            "POST",
            "/embed",
            json={"image": image_b64, "image_id": image_id},
        )
        return result["embedding"]

    async def get_info(self) -> dict:
        """Get service info including model details."""
        return await self._request("GET", "/info")
```

### 2. Identification Pipeline (`routers/identify.py`)

```python
async def identify_artwork(
    request: IdentifyRequest,
    embeddings_client: EmbeddingsClient,
    search_client: SearchClient,
    geometric_client: GeometricClient,
    config: PipelineConfig,
) -> IdentifyResponse:
    """
    Execute the identification pipeline:
    1. Extract embedding from visitor photo
    2. Search for similar artworks
    3. Optionally verify with geometric matching
    4. Return best match with confidence
    """
    timing = {}
    start = time.perf_counter()

    # Step 1: Extract embedding
    t0 = time.perf_counter()
    embedding = await embeddings_client.embed(request.image)
    timing["embedding_ms"] = (time.perf_counter() - t0) * 1000

    # Step 2: Search for candidates
    t0 = time.perf_counter()
    candidates = await search_client.search(
        embedding=embedding,
        k=request.options.k or config.search_k,
        threshold=request.options.threshold or config.similarity_threshold,
    )
    timing["search_ms"] = (time.perf_counter() - t0) * 1000

    if not candidates:
        return IdentifyResponse(
            success=True,
            match=None,
            message="No matching artwork found",
            timing=timing,
        )

    # Step 3: Geometric verification (optional)
    geometric_results = None
    if config.geometric_verification and request.options.geometric_verification:
        t0 = time.perf_counter()
        try:
            geometric_results = await geometric_client.match_batch(
                query_image=request.image,
                references=[...],  # Load reference images for candidates
            )
        except BackendError:
            logger.warning("Geometric verification unavailable, using embedding only")
        timing["geometric_ms"] = (time.perf_counter() - t0) * 1000

    # Step 4: Calculate confidence and select best match
    best_match = select_best_match(candidates, geometric_results, config)

    timing["total_ms"] = (time.perf_counter() - start) * 1000

    return IdentifyResponse(
        success=True,
        match=best_match,
        timing=timing,
    )
```

### 3. Confidence Score Calculation

```python
def calculate_confidence(
    similarity: float,
    geometric_score: float | None,
    verification_enabled: bool,
) -> float:
    """
    Calculate overall match confidence.

    - If geometric verification passed: weight both scores
    - If geometric verification failed: reduce confidence
    - If geometric not performed: use similarity with penalty
    """
    if geometric_score is not None:
        if geometric_score > 0.5:
            # Geometric confirmed: high confidence
            return 0.6 * similarity + 0.4 * geometric_score
        else:
            # Geometric rejected: low confidence despite similarity
            return 0.3 * similarity + 0.2 * geometric_score
    elif verification_enabled:
        # Geometric was supposed to run but didn't
        return similarity * 0.7  # Penalty for missing verification
    else:
        # Geometric intentionally skipped
        return similarity * 0.85  # Small penalty
```

### 4. Health Check with Backend Status

```python
@router.get("/health")
async def health_check(
    check_backends: bool = True,
    embeddings: EmbeddingsClient = Depends(get_embeddings_client),
    search: SearchClient = Depends(get_search_client),
    geometric: GeometricClient = Depends(get_geometric_client),
) -> HealthResponse:
    """Check gateway and backend health."""
    backends = {}

    if check_backends:
        backends = {
            "embeddings": await embeddings.health_check(),
            "search": await search.health_check(),
            "geometric": await geometric.health_check(),
        }

    # Determine overall status
    if not check_backends:
        status = "healthy"
    elif backends["embeddings"] != "healthy" or backends["search"] != "healthy":
        status = "unhealthy"  # Critical backends down
    elif backends["geometric"] != "healthy":
        status = "degraded"  # Non-critical backend down
    else:
        status = "healthy"

    return HealthResponse(status=status, backends=backends)
```

### 5. CORS Configuration

```python
# In app.py
from fastapi.middleware.cors import CORSMiddleware

def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(...)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    return app
```

### 6. Configuration (use existing `config.yaml`)

```yaml
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
  log_level: "info"
  cors_origins:
    - "*"

data:
  objects_path: "/data/objects"
  labels_path: "/data/labels.csv"
```

### 7. Objects Metadata Loading

Load artwork metadata from `labels.csv` or a metadata JSON file:

```python
class ObjectsRepository:
    """Repository for artwork metadata."""

    def __init__(self, objects_path: Path, labels_path: Path):
        self.objects_path = objects_path
        self.metadata = self._load_metadata(labels_path)

    def _load_metadata(self, labels_path: Path) -> dict[str, dict]:
        """Load metadata from labels.csv."""
        # Parse CSV: object_id, name, artist, year, etc.
        ...

    def list_objects(self) -> list[dict]:
        """List all objects."""
        return list(self.metadata.values())

    def get_object(self, object_id: str) -> dict | None:
        """Get object by ID."""
        return self.metadata.get(object_id)

    def get_image_path(self, object_id: str) -> Path | None:
        """Get path to object's reference image."""
        ...
```

### 8. Error Handling

| Error Code | HTTP Status | When |
|------------|-------------|------|
| `backend_unavailable` | 502 | Backend service is not responding |
| `backend_error` | 502 | Backend returned an error |
| `backend_timeout` | 504 | Backend didn't respond in time |
| `invalid_image` | 400 | Uploaded image cannot be processed |
| `not_found` | 404 | Object ID not found |
| `pipeline_error` | 500 | Unexpected error during pipeline |

---

## Validation Checklist

After implementation, verify:

- [ ] `just init` - Environment initializes
- [ ] `just test` - All tests pass
- [ ] `just ci` - All CI checks pass (style, types, security)
- [ ] `just run` - Service starts on port 8000
- [ ] `curl http://localhost:8000/health` returns status with backend info
- [ ] `curl http://localhost:8000/info` returns pipeline config
- [ ] `curl http://localhost:8000/objects` lists all objects
- [ ] `curl http://localhost:8000/objects/{id}` returns object details
- [ ] POST /identify returns match result (requires running backends)

---

## Testing Notes

For unit tests, mock the backend clients:

```python
# tests/conftest.py
import pytest
from unittest.mock import AsyncMock

@pytest.fixture
def mock_embeddings_client():
    client = AsyncMock()
    client.health_check.return_value = "healthy"
    client.embed.return_value = [0.1] * 768  # Fake embedding
    return client

@pytest.fixture
def mock_search_client():
    client = AsyncMock()
    client.health_check.return_value = "healthy"
    client.search.return_value = [
        {"object_id": "obj_001", "score": 0.92, "metadata": {"name": "Test Art"}}
    ]
    return client
```

For integration tests, use Docker Compose to run all services together.

---

## Reference Implementation

Use **@services/embeddings/** as a reference for the service structure. It follows the same patterns described in the template guide.

---

## Dependencies on Other Services

The Gateway requires these services to be running for full functionality:

| Service | Required For | Fallback |
|---------|--------------|----------|
| Embeddings (8001) | `/identify` | None (502 error) |
| Search (8002) | `/identify` | None (502 error) |
| Geometric (8003) | `/identify` verification | Skip verification |

For development, you can test the Gateway in isolation by mocking the backend clients.
