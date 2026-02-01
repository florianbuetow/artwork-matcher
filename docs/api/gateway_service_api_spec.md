# Gateway Service API Specification

## Overview

The Gateway Service is the **public-facing API** that orchestrates the artwork identification pipeline. It coordinates requests across the internal services (Embeddings, Search, Geometric) and presents a unified interface to clients.

**Service Identity:**
- **Name:** gateway
- **Default Port:** 8000
- **Protocol:** HTTP/REST + JSON
- **Role:** API orchestration, pipeline coordination

**See Also:** [Uniform API Structure](uniform_api_structure.md) for common endpoints, error handling, and conventions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Gateway Service                                 │
│                                 (Port 8000)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    ┌──────────────┐                                                          │
│    │   Client     │                                                          │
│    │  (Browser,   │                                                          │
│    │   curl, app) │                                                          │
│    └──────┬───────┘                                                          │
│           │                                                                  │
│           ▼                                                                  │
│    ┌──────────────┐         ┌─────────────────────────────────────────┐     │
│    │   Gateway    │────────▶│           Identification Pipeline        │     │
│    │   /identify  │         │                                          │     │
│    └──────────────┘         │  1. Extract embedding (Embeddings Svc)   │     │
│                             │  2. Search candidates (Search Svc)       │     │
│                             │  3. Verify geometry (Geometric Svc)      │     │
│                             │  4. Return best match                    │     │
│                             └─────────────────────────────────────────┘     │
│                                                                              │
│    Internal Service Calls:                                                   │
│    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                        │
│    │ Embeddings  │  │   Search    │  │  Geometric  │                        │
│    │   :8001     │  │    :8002    │  │    :8003    │                        │
│    └─────────────┘  └─────────────┘  └─────────────┘                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why a Gateway?**

| Concern | Without Gateway | With Gateway |
|---------|-----------------|--------------|
| Client complexity | Must call 3 services, handle coordination | Single endpoint |
| Error handling | Client handles partial failures | Gateway handles, returns clean response |
| Protocol changes | All clients must update | Only gateway updates |
| Authentication | Each service implements | Centralized |
| Rate limiting | Each service implements | Centralized |
| CORS | Each service configures | Gateway only |

---

## Design Decisions

### Pipeline Orchestration Strategy

**Decision: Sequential with early termination**

```
Query Image
    │
    ▼
┌─────────────────┐
│ 1. Embed Image  │ ──── If fails → Return error
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Search Index │ ──── If no results → Return "no match"
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│ 3. Geometric Verification   │ ──── Optional (configurable)
│    (on top-K candidates)    │
└────────┬────────────────────┘
         │
         ▼
    Best Match (or "no confident match")
```

**Why sequential (not parallel)?**

Each stage depends on the previous:
- Can't search without embedding
- Can't verify without candidates
- Early termination saves resources (if embedding fails, no point searching)

**Geometric verification is optional** because:
- Adds ~50-200ms latency per request
- For high-confidence embedding matches (>0.9), verification rarely changes the result
- Can be disabled for speed when false positives are acceptable

---

### Candidate Selection Strategy

**Decision: Top-K with threshold filtering**

```python
# Configuration
search_k: 5           # Retrieve top 5 candidates
similarity_threshold: 0.7  # Minimum embedding similarity
```

**Why top-K, not just top-1?**

1. **Embedding similarity isn't perfect** — The closest embedding might be a similar-looking different artwork
2. **Geometric verification can re-rank** — A candidate ranked #3 by embedding might have better geometric consistency
3. **Multiple artworks in photo** — Some visitor photos contain multiple recognizable works

**Candidate filtering flow:**

```
Search returns: [0.92, 0.88, 0.85, 0.71, 0.65]  (5 candidates)
                         │
Threshold filter (0.7):  [0.92, 0.88, 0.85, 0.71]  (4 pass)
                         │
Geometric verify:        [✓ 67 inliers, ✗ 8 inliers, ✓ 45 inliers, ✗ 5 inliers]
                         │
Final ranking:           #1: 0.92 (67 inliers) ← Best match
                         #2: 0.85 (45 inliers)
```

---

### Error Handling Strategy

**Decision: Graceful degradation with detailed error context**

| Backend Failure | Gateway Behavior |
|-----------------|------------------|
| Embeddings service down | Return 502 with clear message |
| Search service down | Return 502 with clear message |
| Geometric service down | Skip verification, return embedding-only results |
| Search returns empty | Return 200 with "no match found" |
| Geometric verification fails | Return match based on embedding only |

**Philosophy:** The pipeline should be as robust as possible. Geometric verification is an enhancement, not a requirement. If it fails, we still have embedding results.

```python
# Pseudo-code for graceful degradation
async def identify(image: bytes) -> IdentifyResponse:
    # Step 1: Required
    embedding = await embeddings_service.embed(image)
    
    # Step 2: Required
    candidates = await search_service.search(embedding, k=5, threshold=0.7)
    if not candidates:
        return IdentifyResponse(match=None, message="No matching artwork found")
    
    # Step 3: Optional enhancement
    if config.geometric_verification:
        try:
            verified = await geometric_service.match_batch(image, candidates)
            return IdentifyResponse(match=verified.best, verification="geometric")
        except GeometricServiceError:
            logger.warning("Geometric verification unavailable, using embedding only")
    
    # Fallback: return top embedding match
    return IdentifyResponse(match=candidates[0], verification="embedding_only")
```

---

### Timeout Strategy

**Decision: Cascading timeouts with per-service limits**

| Service | Individual Timeout | Rationale |
|---------|-------------------|-----------|
| Embeddings | 25s | Model inference can be slow on CPU |
| Search | 8s | Should be fast; if slow, something's wrong |
| Geometric | 25s | Multiple image comparisons |
| **Total pipeline** | **55s** | Sum of components + overhead |

**Implementation:**

```python
class ServiceClient:
    def __init__(self, base_url: str, timeout: float):
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout, connect=5.0),
        )
```

Client-facing timeout should be slightly higher than the sum of internal timeouts to account for network overhead and processing.

---

### Response Design

**Decision: Rich response with confidence indicators**

Rather than just returning an object ID, the response includes:
- Match confidence (embedding similarity + geometric if available)
- Verification method used
- Timing breakdown
- Alternative matches (if multiple strong candidates)

This helps clients make informed decisions about displaying results.

---

## API Endpoints

### GET /health

Health check that verifies gateway and optionally backend connectivity.

**Response: 200 OK**

```json
{
  "status": "healthy",
  "backends": {
    "embeddings": "healthy",
    "search": "healthy",
    "geometric": "healthy"
  }
}
```

**Status Logic:**

| Condition | Gateway Status | Note |
|-----------|----------------|------|
| All backends healthy | `healthy` | Full functionality |
| Geometric unavailable | `degraded` | Can still identify (no verification) |
| Embeddings OR Search unavailable | `unhealthy` | Cannot process requests |

**Query Parameter:**

```
GET /health?check_backends=false
```

Skip backend health checks for faster response (default: true).

---

### GET /info

Returns gateway configuration and backend status.

**Response: 200 OK**

```json
{
  "service": "gateway",
  "version": "0.1.0",
  "pipeline": {
    "search_k": 5,
    "similarity_threshold": 0.7,
    "geometric_verification": true,
    "confidence_threshold": 0.6
  },
  "backends": {
    "embeddings": {
      "url": "http://localhost:8001",
      "status": "healthy",
      "model": "facebook/dinov2-base",
      "embedding_dimension": 768
    },
    "search": {
      "url": "http://localhost:8002",
      "status": "healthy",
      "index_count": 20
    },
    "geometric": {
      "url": "http://localhost:8003",
      "status": "healthy",
      "algorithm": "ORB+RANSAC"
    }
  }
}
```

---

### POST /identify

**The main endpoint.** Upload a visitor photo and receive artwork identification.

**Request:**

```
POST /identify
Content-Type: application/json
```

```json
{
  "image": "<base64-encoded-visitor-photo>",
  "options": {
    "k": 5,
    "threshold": 0.7,
    "geometric_verification": true,
    "include_alternatives": true
  }
}
```

**Request Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `image` | string (base64) | Yes | — | Visitor photo to identify |
| `options.k` | integer | No | config | Number of candidates to consider |
| `options.threshold` | float | No | config | Minimum similarity threshold |
| `options.geometric_verification` | boolean | No | config | Enable geometric verification |
| `options.include_alternatives` | boolean | No | false | Include runner-up matches |

**Response: 200 OK (Match Found)**

```json
{
  "success": true,
  "match": {
    "object_id": "object_007",
    "name": "Water Lilies",
    "artist": "Claude Monet",
    "year": "1906",
    "confidence": 0.89,
    "similarity_score": 0.92,
    "geometric_score": 0.85,
    "verification_method": "geometric",
    "image_url": "/objects/object_007/image"
  },
  "alternatives": [
    {
      "object_id": "object_012",
      "name": "Bridge over a Pond of Water Lilies",
      "confidence": 0.72,
      "similarity_score": 0.85,
      "geometric_score": 0.58
    }
  ],
  "timing": {
    "embedding_ms": 47.2,
    "search_ms": 1.3,
    "geometric_ms": 156.8,
    "total_ms": 208.4
  },
  "debug": {
    "candidates_considered": 4,
    "candidates_verified": 4,
    "embedding_dimension": 768
  }
}
```

**Response: 200 OK (No Match)**

```json
{
  "success": true,
  "match": null,
  "message": "No matching artwork found with sufficient confidence",
  "timing": {
    "embedding_ms": 45.1,
    "search_ms": 1.2,
    "geometric_ms": 0,
    "total_ms": 49.8
  },
  "debug": {
    "candidates_considered": 0,
    "highest_similarity": 0.43,
    "threshold": 0.7
  }
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether the request was processed successfully |
| `match` | object \| null | Best matching artwork, or null if none found |
| `match.object_id` | string | Unique identifier |
| `match.confidence` | float | Combined confidence score (0-1) |
| `match.similarity_score` | float | Embedding similarity (0-1) |
| `match.geometric_score` | float \| null | Geometric verification score (if performed) |
| `match.verification_method` | string | "geometric" or "embedding_only" |
| `alternatives` | array | Other strong matches (if requested) |
| `timing` | object | Per-stage timing breakdown |
| `debug` | object | Diagnostic information |

**Confidence Score Calculation:**

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

---

### GET /objects

List all objects in the database.

**Response: 200 OK**

```json
{
  "objects": [
    {
      "object_id": "object_001",
      "name": "Mona Lisa",
      "artist": "Leonardo da Vinci",
      "year": "1503-1519"
    },
    {
      "object_id": "object_002",
      "name": "Starry Night",
      "artist": "Vincent van Gogh",
      "year": "1889"
    }
  ],
  "count": 20
}
```

---

### GET /objects/{object_id}

Get details for a specific object.

**Response: 200 OK**

```json
{
  "object_id": "object_007",
  "name": "Water Lilies",
  "artist": "Claude Monet",
  "year": "1906",
  "description": "Part of a series of approximately 250 oil paintings...",
  "location": "Gallery 3, East Wing",
  "image_url": "/objects/object_007/image",
  "indexed_at": "2025-01-15T10:30:00Z"
}
```

**Response: 404 Not Found**

```json
{
  "error": "not_found",
  "message": "Object 'object_999' not found in database"
}
```

---

### GET /objects/{object_id}/image

Retrieve the reference image for an object.

**Response: 200 OK**

```
Content-Type: image/jpeg

<binary image data>
```

**Use Case:** Display the matched artwork to visitors for confirmation.

---

## Error Handling

### Gateway-Specific Error Codes

| Error Code | HTTP Status | Description |
|------------|-------------|-------------|
| `backend_unavailable` | 502 | Required backend service is down |
| `backend_error` | 502 | Backend returned an error |
| `backend_timeout` | 504 | Backend didn't respond in time |
| `invalid_image` | 400 | Uploaded image cannot be processed |
| `pipeline_error` | 500 | Unexpected error during pipeline execution |

### Error Response Examples

**Backend Unavailable (502):**

```json
{
  "error": "backend_unavailable",
  "message": "Embeddings service is not responding",
  "details": {
    "backend": "embeddings",
    "url": "http://localhost:8001",
    "last_error": "Connection refused"
  }
}
```

**Backend Error (502):**

```json
{
  "error": "backend_error",
  "message": "Search service returned an error",
  "details": {
    "backend": "search",
    "backend_error": "index_not_loaded",
    "backend_message": "FAISS index is not loaded"
  }
}
```

**Backend Timeout (504):**

```json
{
  "error": "backend_timeout",
  "message": "Embeddings service timed out after 25 seconds",
  "details": {
    "backend": "embeddings",
    "timeout_seconds": 25,
    "stage": "embedding_extraction"
  }
}
```

### Error Propagation

When a backend returns an error, the gateway preserves context:

```python
async def call_embeddings_service(image: str) -> list[float]:
    try:
        response = await embeddings_client.post("/embed", json={"image": image})
        response.raise_for_status()
        return response.json()["embedding"]
    except httpx.HTTPStatusError as e:
        backend_error = e.response.json()
        raise BackendError(
            error="backend_error",
            message=f"Embeddings service returned an error: {backend_error['message']}",
            status_code=502,
            details={
                "backend": "embeddings",
                "backend_error": backend_error["error"],
                "backend_message": backend_error["message"],
            },
        )
    except httpx.TimeoutException:
        raise BackendError(
            error="backend_timeout",
            message="Embeddings service timed out",
            status_code=504,
            details={"backend": "embeddings", "timeout_seconds": 25},
        )
    except httpx.ConnectError:
        raise BackendError(
            error="backend_unavailable",
            message="Embeddings service is not responding",
            status_code=502,
            details={"backend": "embeddings"},
        )
```

---

## CORS Configuration

The gateway is the only service that receives browser requests, so CORS is configured here:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

See [CORS Implementation Guide](./cors_implementation_guide.md) for details.

---

## Configuration

### Configuration Schema

```yaml
# config.yaml
backends:
  embeddings:
    url: "http://localhost:8001"
    timeout_seconds: 25
    health_check_interval: 30
  search:
    url: "http://localhost:8002"
    timeout_seconds: 8
    health_check_interval: 30
  geometric:
    url: "http://localhost:8003"
    timeout_seconds: 25
    health_check_interval: 30

pipeline:
  search_k: 5                    # Candidates to retrieve
  similarity_threshold: 0.7      # Minimum embedding similarity
  geometric_verification: true   # Enable geometric stage
  confidence_threshold: 0.6      # Minimum confidence to return match
  fallback_on_geometric_failure: true  # Use embedding-only if geometric fails

server:
  host: "0.0.0.0"
  port: 8000
  log_level: "info"
  cors_origins: ["*"]

data:
  objects_metadata_path: "/data/objects/metadata.json"
  objects_images_path: "/data/objects"
```

### Environment Variable Overrides

```bash
GATEWAY__BACKENDS__EMBEDDINGS__URL=http://embeddings:8001
GATEWAY__BACKENDS__SEARCH__URL=http://search:8002
GATEWAY__BACKENDS__GEOMETRIC__URL=http://geometric:8003
GATEWAY__PIPELINE__SEARCH_K=10
GATEWAY__PIPELINE__GEOMETRIC_VERIFICATION=false
```

---

## Usage Examples

### curl

```bash
# Health check
curl http://localhost:8000/health

# Health check with backend status
curl "http://localhost:8000/health?check_backends=true"

# Service info
curl http://localhost:8000/info

# Identify artwork
curl -X POST http://localhost:8000/identify \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$(base64 -i visitor_photo.jpg | tr -d '\n')\"}"

# Identify with options
curl -X POST http://localhost:8000/identify \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"$(base64 -i visitor_photo.jpg | tr -d '\n')\",
    \"options\": {
      \"k\": 10,
      \"threshold\": 0.6,
      \"geometric_verification\": true,
      \"include_alternatives\": true
    }
  }"

# List objects
curl http://localhost:8000/objects

# Get object details
curl http://localhost:8000/objects/object_007

# Get object image
curl http://localhost:8000/objects/object_007/image > artwork.jpg
```

### Python Client

```python
import base64
import httpx
from pathlib import Path
from dataclasses import dataclass

@dataclass
class IdentificationResult:
    success: bool
    object_id: str | None
    name: str | None
    confidence: float | None
    timing_ms: float

class ArtworkMatcherClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.Client(timeout=60.0)
    
    def health_check(self) -> dict:
        """Check gateway and backend health."""
        response = self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def identify(
        self,
        image_path: Path,
        threshold: float = 0.7,
        geometric: bool = True,
    ) -> IdentificationResult:
        """Identify artwork in a visitor photo."""
        image_b64 = base64.b64encode(image_path.read_bytes()).decode()
        
        response = self.client.post(
            f"{self.base_url}/identify",
            json={
                "image": image_b64,
                "options": {
                    "threshold": threshold,
                    "geometric_verification": geometric,
                },
            },
        )
        response.raise_for_status()
        result = response.json()
        
        match = result.get("match")
        return IdentificationResult(
            success=result["success"],
            object_id=match["object_id"] if match else None,
            name=match.get("name") if match else None,
            confidence=match["confidence"] if match else None,
            timing_ms=result["timing"]["total_ms"],
        )
    
    def list_objects(self) -> list[dict]:
        """List all objects in the database."""
        response = self.client.get(f"{self.base_url}/objects")
        response.raise_for_status()
        return response.json()["objects"]
    
    def get_object(self, object_id: str) -> dict:
        """Get details for a specific object."""
        response = self.client.get(f"{self.base_url}/objects/{object_id}")
        response.raise_for_status()
        return response.json()


# Usage
client = ArtworkMatcherClient()

# Check system health
health = client.health_check()
print(f"System status: {health['status']}")

# Identify a photo
result = client.identify(Path("visitor_photo.jpg"))
if result.object_id:
    print(f"Match: {result.name} (confidence: {result.confidence:.2f})")
    print(f"Time: {result.timing_ms:.1f}ms")
else:
    print("No match found")
```

### Async Python Client

```python
import asyncio
import base64
import httpx
from pathlib import Path

class AsyncArtworkMatcherClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    async def identify_batch(
        self,
        image_paths: list[Path],
        max_concurrent: int = 4,
    ) -> list[dict]:
        """Identify multiple images concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def identify_one(client: httpx.AsyncClient, path: Path) -> dict:
            async with semaphore:
                image_b64 = base64.b64encode(path.read_bytes()).decode()
                response = await client.post(
                    f"{self.base_url}/identify",
                    json={"image": image_b64},
                )
                result = response.json()
                result["source_file"] = str(path)
                return result
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            tasks = [identify_one(client, path) for path in image_paths]
            return await asyncio.gather(*tasks)


# Usage
async def main():
    client = AsyncArtworkMatcherClient()
    
    photos = list(Path("visitor_photos").glob("*.jpg"))
    results = await client.identify_batch(photos)
    
    for result in results:
        match = result.get("match")
        if match:
            print(f"{result['source_file']}: {match['name']} ({match['confidence']:.2f})")
        else:
            print(f"{result['source_file']}: No match")

asyncio.run(main())
```

---

## Performance Characteristics

### End-to-End Latency

| Stage | Typical | P99 |
|-------|---------|-----|
| Request parsing + validation | 1-5 ms | 10 ms |
| Embedding extraction | 50-100 ms | 200 ms |
| Vector search | 1-5 ms | 20 ms |
| Geometric verification (5 candidates) | 100-200 ms | 400 ms |
| Response serialization | 1-5 ms | 10 ms |
| **Total (with geometric)** | **150-300 ms** | **600 ms** |
| **Total (without geometric)** | **50-120 ms** | **250 ms** |

### Throughput

| Configuration | Requests/sec | Bottleneck |
|---------------|--------------|------------|
| With geometric verification | 3-5 | Geometric service |
| Without geometric verification | 8-15 | Embeddings service |
| Embeddings on GPU | 15-30 | Network overhead |

### Scaling Considerations

For higher throughput:

1. **Horizontal scaling** — Run multiple gateway instances behind a load balancer
2. **GPU for embeddings** — 3-5× faster inference
3. **Pre-computed geometric features** — Skip feature extraction for reference images
4. **Disable geometric verification** — When embedding confidence is very high (>0.95)

---

## OpenAPI Specification

```yaml
openapi: 3.1.0
info:
  title: Artwork Matcher Gateway
  description: API for identifying museum artworks from visitor photos
  version: 0.1.0
servers:
  - url: http://localhost:8000
    description: Local development

paths:
  /health:
    get:
      operationId: health_check
      summary: Health check
      tags: [Operations]
      parameters:
        - name: check_backends
          in: query
          schema:
            type: boolean
            default: true
          description: Whether to check backend services
      responses:
        "200":
          description: Health status
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/HealthResponse"

  /info:
    get:
      operationId: get_info
      summary: Service information
      tags: [Operations]
      responses:
        "200":
          description: Gateway configuration and backend status
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/InfoResponse"

  /identify:
    post:
      operationId: identify_artwork
      summary: Identify artwork in a photo
      description: |
        Upload a visitor photo to identify which artwork it contains.
        The pipeline extracts embeddings, searches the index, and optionally
        performs geometric verification.
      tags: [Identification]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/IdentifyRequest"
      responses:
        "200":
          description: Identification result (match found or not)
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/IdentifyResponse"
        "400":
          description: Invalid request
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"
        "502":
          description: Backend service error
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"
        "504":
          description: Backend service timeout
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"

  /objects:
    get:
      operationId: list_objects
      summary: List all objects
      tags: [Objects]
      responses:
        "200":
          description: List of objects
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ObjectListResponse"

  /objects/{object_id}:
    get:
      operationId: get_object
      summary: Get object details
      tags: [Objects]
      parameters:
        - name: object_id
          in: path
          required: true
          schema:
            type: string
      responses:
        "200":
          description: Object details
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ObjectDetails"
        "404":
          description: Object not found
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"

  /objects/{object_id}/image:
    get:
      operationId: get_object_image
      summary: Get object reference image
      tags: [Objects]
      parameters:
        - name: object_id
          in: path
          required: true
          schema:
            type: string
      responses:
        "200":
          description: Object image
          content:
            image/jpeg:
              schema:
                type: string
                format: binary
        "404":
          description: Object not found
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"

components:
  schemas:
    HealthResponse:
      type: object
      required: [status]
      properties:
        status:
          type: string
          enum: [healthy, degraded, unhealthy]
        backends:
          type: object
          properties:
            embeddings:
              type: string
            search:
              type: string
            geometric:
              type: string

    InfoResponse:
      type: object
      required: [service, version, pipeline, backends]
      properties:
        service:
          type: string
          example: gateway
        version:
          type: string
          example: 0.1.0
        pipeline:
          type: object
          properties:
            search_k:
              type: integer
            similarity_threshold:
              type: number
            geometric_verification:
              type: boolean
            confidence_threshold:
              type: number
        backends:
          type: object
          additionalProperties:
            type: object

    IdentifyRequest:
      type: object
      required: [image]
      properties:
        image:
          type: string
          format: byte
          description: Base64-encoded visitor photo
        options:
          type: object
          properties:
            k:
              type: integer
              minimum: 1
              maximum: 20
              description: Number of candidates to consider
            threshold:
              type: number
              minimum: 0
              maximum: 1
              description: Minimum similarity threshold
            geometric_verification:
              type: boolean
              description: Enable geometric verification
            include_alternatives:
              type: boolean
              description: Include alternative matches in response

    IdentifyResponse:
      type: object
      required: [success]
      properties:
        success:
          type: boolean
        match:
          $ref: "#/components/schemas/Match"
        alternatives:
          type: array
          items:
            $ref: "#/components/schemas/Match"
        message:
          type: string
        timing:
          type: object
          properties:
            embedding_ms:
              type: number
            search_ms:
              type: number
            geometric_ms:
              type: number
            total_ms:
              type: number
        debug:
          type: object
          additionalProperties: true

    Match:
      type: object
      required: [object_id, confidence]
      properties:
        object_id:
          type: string
        name:
          type: string
        artist:
          type: string
        year:
          type: string
        confidence:
          type: number
          minimum: 0
          maximum: 1
        similarity_score:
          type: number
        geometric_score:
          type: number
          nullable: true
        verification_method:
          type: string
          enum: [geometric, embedding_only]
        image_url:
          type: string

    ObjectListResponse:
      type: object
      required: [objects, count]
      properties:
        objects:
          type: array
          items:
            type: object
            properties:
              object_id:
                type: string
              name:
                type: string
              artist:
                type: string
              year:
                type: string
        count:
          type: integer

    ObjectDetails:
      type: object
      required: [object_id]
      properties:
        object_id:
          type: string
        name:
          type: string
        artist:
          type: string
        year:
          type: string
        description:
          type: string
        location:
          type: string
        image_url:
          type: string
        indexed_at:
          type: string
          format: date-time

    ErrorResponse:
      type: object
      required: [error, message]
      properties:
        error:
          type: string
          enum:
            - invalid_image
            - backend_unavailable
            - backend_error
            - backend_timeout
            - not_found
            - pipeline_error
        message:
          type: string
        details:
          type: object
          additionalProperties: true
```

---

## Appendix: Why a Separate Gateway?

### Alternative: Direct Client Access

```
┌──────────┐     ┌─────────────┐
│  Client  │────▶│ Embeddings  │
│          │────▶│   Search    │
│          │────▶│  Geometric  │
└──────────┘     └─────────────┘
```

**Problems:**
- Client must understand the pipeline
- Client handles all error cases
- No central place for auth, rate limiting, CORS
- Harder to change internal architecture

### Alternative: Monolithic Service

```
┌──────────┐     ┌─────────────────────────┐
│  Client  │────▶│  All-in-One Service     │
│          │     │  (embed + search + geo) │
└──────────┘     └─────────────────────────┘
```

**Problems:**
- Can't scale components independently
- Single point of failure
- Harder to test individual components
- Memory-heavy (all models loaded together)

### Our Architecture: Gateway + Microservices

```
┌──────────┐     ┌─────────┐     ┌─────────────┐
│  Client  │────▶│ Gateway │────▶│ Embeddings  │
│          │     │         │────▶│   Search    │
│          │     │         │────▶│  Geometric  │
└──────────┘     └─────────┘     └─────────────┘
```

**Benefits:**
- Clean separation of concerns
- Independent scaling (GPU for embeddings, CPU for search)
- Central error handling and logging
- Easy to add features (caching, rate limiting)
- Each service can be developed/tested independently
- Graceful degradation (geometric can fail without breaking identification)

For a demo project, this architecture might seem like over-engineering. But it demonstrates understanding of production patterns and makes the system easier to explain and extend.
