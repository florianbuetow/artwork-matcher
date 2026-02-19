# Uniform API Structure

## Overview

All services in the Artwork Matcher system follow a consistent API structure. This ensures predictable behavior, simplifies client implementation, and enables uniform monitoring and error handling.

**Services:**

| Service | Port | Purpose |
|---------|------|---------|
| Gateway | 8000 | API orchestration, public entry point |
| Embeddings | 8001 | DINOv2 embedding extraction |
| Search | 8002 | FAISS vector search |
| Geometric | 8003 | ORB + RANSAC geometric verification |
| Storage | 8004 | Binary object storage |

---

## Common Endpoints

Every service implements these two operational endpoints with identical response schemas.

### GET /health

Health check for container orchestration (Docker health checks, Kubernetes liveness/readiness probes).

**Design Principles:**
- Must respond quickly (< 100ms)
- Should not perform expensive operations (no model inference, no database queries)
- Returns current operational status

**Response: 200 OK**

```json
{
  "status": "healthy"
}
```

**Status Values:**

| Status | HTTP Code | Meaning | Orchestrator Action |
|--------|-----------|---------|---------------------|
| `healthy` | 200 | Fully operational | Route traffic |
| `degraded` | 200 | Operational with issues | Route traffic, alert |
| `unhealthy` | 503 | Not ready to serve | Stop routing, restart |

**Implementation Pattern:**

```python
from enum import Enum
from pydantic import BaseModel

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthResponse(BaseModel):
    status: HealthStatus

@app.get("/health")
async def health() -> HealthResponse:
    # Fast check - no expensive operations
    return HealthResponse(status=HealthStatus.HEALTHY)
```

**When to Return Each Status:**

| Status | Examples |
|--------|----------|
| `healthy` | All systems nominal |
| `degraded` | High latency, cache miss, non-critical dependency down |
| `unhealthy` | Model failed to load, critical dependency unreachable |

---

### GET /info

Returns service metadata. Used for service discovery, compatibility validation, and debugging.

**Response: 200 OK**

```json
{
  "service": "<service-name>",
  "version": "<semver>",
  ...service-specific configuration...
}
```

**Required Fields (all services):**

| Field | Type | Description |
|-------|------|-------------|
| `service` | string | Service identifier (e.g., "embeddings", "search") |
| `version` | string | Semantic version (e.g., "0.1.0") |

**Service-Specific Fields:**

Each service adds configuration relevant to its consumers:

**Embeddings Service:**
```json
{
  "service": "embeddings",
  "version": "0.1.0",
  "model": {
    "name": "facebook/dinov2-base",
    "embedding_dimension": 768,
    "device": "mps"
  },
  "preprocessing": {
    "image_size": 518,
    "normalize": true
  }
}
```

**Search Service:**
```json
{
  "service": "search",
  "version": "0.1.0",
  "index": {
    "type": "flat",
    "embedding_dimension": 768,
    "count": 20
  }
}
```

**Geometric Service:**
```json
{
  "service": "geometric",
  "version": "0.1.0",
  "algorithm": {
    "feature_detector": "ORB",
    "max_features": 1000,
    "matcher": "BFMatcher",
    "verification": "RANSAC"
  }
}
```

**Gateway:**
```json
{
  "service": "gateway",
  "version": "0.1.0",
  "pipeline": {
    "search_k": 5,
    "similarity_threshold": 0.7,
    "geometric_verification": true
  },
  "backends": {
    "embeddings": "http://localhost:8001",
    "search": "http://localhost:8002",
    "geometric": "http://localhost:8003",
    "storage": "http://localhost:8004"
  }
}
```

**Usage Pattern — Startup Validation:**

```python
async def validate_backend_compatibility(
    embeddings_url: str,
    search_url: str,
) -> None:
    """Validate that backend services are compatible."""
    async with httpx.AsyncClient() as client:
        emb_info = (await client.get(f"{embeddings_url}/info")).json()
        search_info = (await client.get(f"{search_url}/info")).json()
    
    emb_dim = emb_info["model"]["embedding_dimension"]
    search_dim = search_info["index"]["embedding_dimension"]
    
    if emb_dim != search_dim:
        raise ConfigurationError(
            f"Dimension mismatch: embeddings={emb_dim}, search={search_dim}"
        )
```

---

## Error Response Format

All services return errors in a consistent JSON structure. This enables uniform error handling in clients and consistent logging.

### Error Response Schema

```json
{
  "error": "<error_code>",
  "message": "<human-readable description>",
  "details": { ... }
}
```

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `error` | string | Yes | Machine-readable error code (snake_case) |
| `message` | string | Yes | Human-readable description |
| `details` | object | No | Additional context (varies by error) |

### Common Error Codes

These error codes are used across all services:

| Error Code | HTTP Status | Description |
|------------|-------------|-------------|
| `validation_error` | 422 | Request failed schema validation |
| `not_found` | 404 | Requested resource does not exist |
| `internal_error` | 500 | Unexpected server error |
| `service_unavailable` | 503 | Service temporarily unavailable |

### Service-Specific Error Codes

Services define additional error codes relevant to their domain:

**Embeddings Service:**
- `invalid_image` (400) — Image data cannot be decoded
- `unsupported_format` (400) — Image format not supported
- `decode_error` (400) — Base64 decoding failed
- `payload_too_large` (413) — Request exceeds size limit
- `model_error` (500) — Model inference failed

**Search Service:**
- `index_not_loaded` (503) — FAISS index not available
- `dimension_mismatch` (400) — Embedding dimension doesn't match index
- `invalid_embedding` (400) — Embedding vector is malformed

**Geometric Service:**
- `invalid_image` (400) — Image data cannot be decoded
- `insufficient_features` (422) — Not enough features detected for matching

**Gateway:**
- `backend_error` (502) — Backend service returned an error
- `backend_timeout` (504) — Backend service timed out
- `no_match_found` (200) — Pipeline completed but no confident match (not an error)

### Error Response Examples

**Validation Error (422):**

```json
{
  "error": "validation_error",
  "message": "Request validation failed",
  "details": {
    "fields": [
      {
        "field": "image",
        "message": "Field required"
      }
    ]
  }
}
```

**Backend Error (502):**

```json
{
  "error": "backend_error",
  "message": "Embeddings service returned an error",
  "details": {
    "backend": "embeddings",
    "backend_error": "model_error",
    "backend_message": "CUDA out of memory"
  }
}
```

### Implementation Pattern

```python
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

class ErrorResponse(BaseModel):
    error: str
    message: str
    details: dict | None = None

class ServiceError(Exception):
    """Base exception for service errors."""
    def __init__(
        self,
        error: str,
        message: str,
        status_code: int = 400,
        details: dict | None = None,
    ):
        self.error = error
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)

@app.exception_handler(ServiceError)
async def service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.error,
            message=exc.message,
            details=exc.details,
        ).model_dump(),
    )

# Usage
raise ServiceError(
    error="invalid_image",
    message="Failed to decode image: invalid JPEG header",
    status_code=400,
    details={"image_id": "test_001"},
)
```

---

## Request Conventions

### Content Type

All endpoints accept and return `application/json`.

```
Content-Type: application/json
Accept: application/json
```

### Image Data

Images are transmitted as Base64-encoded strings within JSON:

```json
{
  "image": "<base64-encoded-data>"
}
```

**Supported Formats:**
- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)

**Encoding (Python):**
```python
import base64

image_data = base64.b64encode(image_bytes).decode("ascii")
```

**Encoding (Bash):**
```bash
base64 -i image.jpg | tr -d '\n'
```

### Embedding Vectors

Embedding vectors are transmitted as JSON arrays of floats:

```json
{
  "embedding": [0.0234, -0.0891, 0.0412, ...]
}
```

**Properties:**
- L2-normalized (unit length)
- 32-bit float precision
- Dimension determined by model configuration

### Optional Request IDs

All endpoints that process data accept an optional identifier for tracing:

```json
{
  "image": "...",
  "image_id": "visitor_photo_001"
}
```

This ID is echoed in responses and included in logs, enabling request tracing across services.

---

## Response Conventions

### Success Responses

Success responses return HTTP 200 with a JSON body. The schema varies by endpoint.

**Common Patterns:**

```json
// Single result
{
  "result": { ... },
  "processing_time_ms": 45.2
}

// List of results
{
  "results": [ ... ],
  "count": 5,
  "processing_time_ms": 12.8
}
```

### Processing Time

All computation endpoints include `processing_time_ms` for performance monitoring:

```json
{
  "embedding": [...],
  "processing_time_ms": 47.3
}
```

This measures server-side processing only (excludes network latency).

### Pagination

For endpoints that could return many results, use cursor-based pagination:

```json
{
  "results": [...],
  "count": 10,
  "has_more": true,
  "cursor": "eyJvZmZzZXQiOiAxMH0="
}
```

**Note:** For this project's scale (20 objects), pagination is not required. Included here for completeness.

---

## HTTP Status Codes

### Success Codes

| Code | Meaning | Usage |
|------|---------|-------|
| 200 | OK | Successful operation |
| 201 | Created | Resource created (e.g., embedding added to index) |
| 204 | No Content | Successful operation with no response body |

### Client Error Codes

| Code | Meaning | Usage |
|------|---------|-------|
| 400 | Bad Request | Malformed request (invalid JSON, bad image data) |
| 404 | Not Found | Resource doesn't exist |
| 413 | Payload Too Large | Request body exceeds size limit |
| 422 | Unprocessable Entity | Valid JSON but failed validation |

### Server Error Codes

| Code | Meaning | Usage |
|------|---------|-------|
| 500 | Internal Server Error | Unexpected server error |
| 502 | Bad Gateway | Backend service error |
| 503 | Service Unavailable | Service not ready (starting up, index loading) |
| 504 | Gateway Timeout | Backend service timeout |

---

## Timeouts

### Recommended Timeouts

| Operation | Client Timeout | Server Timeout |
|-----------|----------------|----------------|
| Health check | 5s | N/A |
| Info | 5s | N/A |
| Embedding extraction | 30s | 25s |
| Vector search | 10s | 8s |
| Geometric matching | 30s | 25s |
| Full pipeline (gateway) | 60s | 55s |

### Implementation Pattern

```python
import httpx

# Client-side
async with httpx.AsyncClient(timeout=30.0) as client:
    response = await client.post(f"{url}/embed", json=payload)

# Server-side (FastAPI background task timeout)
from asyncio import timeout

async with timeout(25.0):
    result = await process_embedding(image)
```

---

## Versioning

### API Versioning Strategy

**Current approach: No URL versioning**

For this project's scope, API versioning is not implemented. All services expose `v1`-equivalent APIs at the root path.

**Future consideration:** If breaking changes are needed, use URL prefix versioning:

```
/v1/embed
/v2/embed
```

### Service Version

Service versions follow [Semantic Versioning](https://semver.org/):

- **MAJOR:** Breaking API changes
- **MINOR:** Backward-compatible new features
- **PATCH:** Backward-compatible bug fixes

The version is exposed via `/info`:

```json
{
  "service": "embeddings",
  "version": "0.1.0"
}
```

---

## CORS Configuration

For the Gateway service (public-facing), CORS is configured to allow browser access:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Internal services (embeddings, search, geometric, storage) do not require CORS as they are not
accessed directly from browsers.

---

## Logging

### Log Format

All services use structured JSON logging for consistency:

```json
{
  "timestamp": "2025-01-15T10:30:45.123Z",
  "level": "INFO",
  "service": "embeddings",
  "message": "Embedding extracted",
  "image_id": "object_001",
  "processing_time_ms": 47.3,
  "dimension": 768
}
```

### Request Logging

All requests are logged with:

| Field | Description |
|-------|-------------|
| `method` | HTTP method |
| `path` | Request path |
| `status_code` | Response status |
| `duration_ms` | Total request duration |
| `image_id` | Request identifier (if provided) |

### Error Logging

Errors include additional context:

```json
{
  "timestamp": "2025-01-15T10:30:45.123Z",
  "level": "ERROR",
  "service": "embeddings",
  "error": "invalid_image",
  "message": "Failed to decode image",
  "image_id": "test_001",
  "traceback": "..."
}
```

---

## Testing Endpoints

### curl Examples

```bash
# Health check (all services)
curl http://localhost:8001/health

# Service info (all services)
curl http://localhost:8001/info

# Check all services are healthy
for port in 8000 8001 8002 8003 8004; do
  echo "Port $port: $(curl -s http://localhost:$port/health | jq -r '.status')"
done
```

### Python Health Check Script

```python
import httpx

SERVICES = {
    "gateway": "http://localhost:8000",
    "embeddings": "http://localhost:8001",
    "search": "http://localhost:8002",
    "geometric": "http://localhost:8003",
    "storage": "http://localhost:8004",
}

def check_all_services() -> dict[str, str]:
    """Check health of all services."""
    results = {}
    for name, url in SERVICES.items():
        try:
            response = httpx.get(f"{url}/health", timeout=5.0)
            results[name] = response.json()["status"]
        except Exception as e:
            results[name] = f"error: {e}"
    return results

if __name__ == "__main__":
    for service, status in check_all_services().items():
        print(f"{service}: {status}")
```

---

## Summary

| Aspect | Convention |
|--------|------------|
| Content type | `application/json` |
| Health endpoint | `GET /health` → `{"status": "healthy"}` |
| Info endpoint | `GET /info` → service metadata |
| Error format | `{"error": "...", "message": "...", "details": {...}}` |
| Image encoding | Base64 in JSON |
| Embedding format | JSON array of floats, L2-normalized |
| Request tracing | Optional `image_id` / `request_id` field |
| Timing | Include `processing_time_ms` in responses |
| Versioning | Semantic versioning via `/info`, no URL prefixes |

This uniform structure ensures that:
1. Clients can interact with any service using the same patterns
2. Monitoring and alerting can be standardized
3. Error handling is consistent across the system
4. New services can be added following established conventions
