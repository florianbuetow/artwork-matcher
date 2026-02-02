# Search Service Implementation Task

## Overview

Implement the **Search Service** for the artwork-matcher project. This service manages a FAISS vector index for fast similarity search, storing embeddings with associated metadata and returning the most similar items for query vectors.

**Service Identity:**
- **Name:** search-service
- **Port:** 8002
- **Technology:** FAISS with `IndexFlatIP` (inner product)

---

## Reference Documentation

Read and follow these documents in order:

1. **@docs/api/search_service_api_spec.md** - Complete API specification including:
   - All endpoints: `/health`, `/info`, `/search`, `/add`, `/index/save`, `/index/load`, `DELETE /index`
   - Request/response schemas with examples
   - Error codes and handling
   - FAISS index design decisions (why `IndexFlatIP`)
   - Metadata storage pattern (separate JSON file)

2. **@docs/api/uniform_api_structure.md** - Common API conventions:
   - Health/info endpoint patterns
   - Error response format (`error`, `message`, `details`)
   - Processing time tracking
   - HTTP status codes

3. **@docs/implementation-guides/fastapi_service_template.md** - Service structure:
   - Directory layout (`src/search_service/...`)
   - Configuration management (Pydantic + YAML, no defaults in code)
   - Structured JSON logging
   - Application factory pattern
   - Router organization

4. **@docs/implementation-guides/config_pattern.md** - Configuration pattern:
   - How to load YAML and validate with Pydantic
   - Environment variable overrides
   - Fail-fast on invalid config

5. **@CLAUDE.md** - Development rules:
   - Use `uv run` for all Python execution
   - Run tests after every change
   - No hardcoded defaults
   - Git commit guidelines

---

## Existing Files

The following already exist and should be used:
- `services/search/config.yaml` - Configuration (already populated)
- `services/search/pyproject.toml` - Dependencies including `faiss-cpu`
- `services/search/justfile` - Build/run commands
- `services/search/src/search_service/__init__.py` - Package marker

---

## Files to Create

Following the structure in `@docs/implementation-guides/fastapi_service_template.md`:

```
services/search/src/search_service/
├── __init__.py          (exists)
├── main.py              # Entry point with main()
├── app.py               # FastAPI app factory
├── config.py            # Pydantic settings loading config.yaml
├── logging.py           # Structured JSON logging
├── schemas.py           # Pydantic request/response models
├── core/
│   ├── __init__.py
│   ├── lifespan.py      # Startup (load index) / shutdown
│   ├── state.py         # App state (uptime + index reference)
│   └── exceptions.py    # ServiceError and handlers
├── routers/
│   ├── __init__.py
│   ├── health.py        # GET /health
│   ├── info.py          # GET /info
│   ├── search.py        # POST /search
│   └── index.py         # POST /add, POST /index/save, POST /index/load, DELETE /index
└── services/
    ├── __init__.py
    └── faiss_index.py   # FAISS index wrapper class
```

Also create:
- `services/search/tests/conftest.py` - Test fixtures
- `services/search/tests/test_health.py` - Health endpoint tests
- `services/search/tests/test_search.py` - Search functionality tests

---

## Key Implementation Details

### 1. FAISS Index Wrapper (`services/faiss_index.py`)

```python
class FAISSIndex:
    """Wrapper around FAISS index with metadata storage."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: list[dict] = []  # Parallel list for metadata

    def add(self, object_id: str, embedding: list[float], metadata: dict) -> int:
        """Add embedding with metadata. Returns index position."""
        ...

    def search(self, embedding: list[float], k: int, threshold: float) -> list[dict]:
        """Search for similar vectors. Returns ranked results."""
        ...

    def save(self, index_path: Path, metadata_path: Path) -> None:
        """Persist index and metadata to disk."""
        ...

    def load(self, index_path: Path, metadata_path: Path) -> None:
        """Load index and metadata from disk."""
        ...

    def clear(self) -> int:
        """Clear all vectors and metadata. Returns previous count."""
        ...
```

### 2. Index Lifecycle

- **On startup:** Auto-load from configured path if `auto_load: true` and file exists
- **On `/add`:** Add to in-memory index (no auto-save)
- **On `/index/save`:** Persist to disk
- **On `/index/load`:** Reload from disk

### 3. Search Response

Return ranked results with:
- `object_id`, `score`, `rank`, `metadata`
- Apply threshold filtering
- Include `processing_time_ms`

### 4. Configuration (use existing `config.yaml`)

```yaml
faiss:
  embedding_dimension: 768
  index_type: "flat"
  metric: "inner_product"

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
  port: 8000
  log_level: "info"
```

### 5. Error Handling

| Error Code | HTTP Status | When |
|------------|-------------|------|
| `dimension_mismatch` | 400 | Vector dimension doesn't match index |
| `invalid_embedding` | 400 | Embedding is malformed (NaN, wrong type) |
| `index_not_loaded` | 503 | Index not available for search |
| `index_empty` | 422 | Search called on empty index |
| `save_failed` | 500 | Failed to write index to disk |
| `load_failed` | 500 | Failed to read index from disk |

---

## Validation Checklist

After implementation, verify:

- [ ] `just init` - Environment initializes
- [ ] `just test` - All tests pass
- [ ] `just ci` - All CI checks pass (style, types, security)
- [ ] `just run` - Service starts on port 8002
- [ ] `curl http://localhost:8002/health` returns `{"status": "healthy"}`
- [ ] `curl http://localhost:8002/info` returns index stats
- [ ] Can add embeddings via POST /add
- [ ] Can save index via POST /index/save
- [ ] Can reload index via POST /index/load
- [ ] Can search via POST /search

---

## Reference Implementation

Use **@services/embeddings/** as a reference for the service structure. It follows the same patterns described in the template guide.
