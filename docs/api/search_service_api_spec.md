# Search Service API Specification

## Overview

The Search Service manages a FAISS vector index for fast similarity search. It stores embeddings with associated metadata and returns the most similar items for a query vector.

**Service Identity:**
- **Name:** search-service
- **Default Port:** 8002
- **Protocol:** HTTP/REST + JSON

**See Also:** [Uniform API Structure](uniform_api_structure.md) for common endpoints, error handling, and conventions.

---

## Design Decisions

### Index Type Selection

**Decision: Use `IndexFlatIP` (Flat Inner Product)**

FAISS offers many index types optimized for different scales and requirements. Understanding these options demonstrates awareness of production considerations, even when choosing the simplest option.

#### FAISS Index Types Comparison

| Index Type | Search Complexity | Accuracy | Training Required | Memory per Vector | Incremental Add |
|------------|-------------------|----------|-------------------|-------------------|-----------------|
| `IndexFlatIP` | O(n) | 100% | No | 4B × dim | ✓ |
| `IndexFlatL2` | O(n) | 100% | No | 4B × dim | ✓ |
| `IndexIVFFlat` | O(n/nlist) | 95-99% | Yes | 4B × dim + overhead | ✓ |
| `IndexIVFPQ` | O(n/nlist) | 80-95% | Yes | ~64B (compressed) | ✗ |
| `IndexHNSWFlat` | O(log n) | 95-99% | No | 4B × dim + graph | ✓ |
| `IndexLSH` | O(n/buckets) | 70-90% | No | Bit vectors | ✓ |

#### Index Type Details

**`IndexFlatIP` / `IndexFlatL2` — Brute Force (Our Choice)**

```python
index = faiss.IndexFlatIP(dimension)  # Inner product
index = faiss.IndexFlatL2(dimension)  # L2 distance
```

- **How it works:** Compares query against every vector in the index
- **Accuracy:** 100% (exact search)
- **Speed:** Linear in index size, but SIMD-optimized
- **Memory:** `n × d × 4 bytes` (32-bit floats)
- **Best for:** < 100K vectors, or when accuracy is critical

**Why we chose this:**
- 20 objects → 20 comparisons → < 0.1ms
- No configuration complexity
- Guaranteed correct results
- No training data needed

---

**`IndexIVFFlat` — Inverted File with Flat Storage**

```python
quantizer = faiss.IndexFlatIP(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
index.train(training_vectors)  # Required!
index.nprobe = 10  # How many clusters to search
```

- **How it works:** Clusters vectors into `nlist` partitions using k-means. Search only visits `nprobe` nearest clusters.
- **Accuracy:** 95-99% (depends on nprobe/nlist ratio)
- **Speed:** O(n/nlist × nprobe) — much faster for large indices
- **Memory:** Same as flat + cluster centroids
- **Best for:** 100K - 10M vectors

**Why we didn't choose this:**
- Requires training on representative data (minimum ~30 × nlist vectors)
- With 20 objects, we can't meaningfully train clusters
- Adds configuration complexity (nlist, nprobe tuning)
- Speed benefit is negligible at our scale

---

**`IndexIVFPQ` — Inverted File with Product Quantization**

```python
index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)
index.train(training_vectors)  # Required!
```

- **How it works:** Combines IVF clustering with Product Quantization (PQ) compression. Vectors are compressed to ~64 bytes regardless of dimension.
- **Accuracy:** 80-95% (lossy compression)
- **Speed:** Very fast, even for billions of vectors
- **Memory:** Dramatically reduced (~10-50× compression)
- **Best for:** 10M+ vectors, memory-constrained environments

**Why we didn't choose this:**
- Lossy compression reduces accuracy
- Requires substantial training data
- Complex tuning (nlist, m, nbits parameters)
- Overkill for 20 vectors

---

**`IndexHNSWFlat` — Hierarchical Navigable Small World**

```python
index = faiss.IndexHNSWFlat(dimension, M)  # M = graph connectivity
index.hnsw.efConstruction = 40  # Build quality
index.hnsw.efSearch = 16  # Search quality
```

- **How it works:** Builds a multi-layer graph structure. Search navigates the graph from coarse to fine layers.
- **Accuracy:** 95-99% (approximate, but high quality)
- **Speed:** O(log n) — excellent scaling
- **Memory:** Vectors + graph structure (higher than flat)
- **Best for:** 10K - 10M vectors, when query latency is critical

**Why we didn't choose this:**
- Higher memory overhead per vector
- Graph construction parameters to tune
- At 20 vectors, graph navigation overhead exceeds brute-force
- No accuracy benefit since flat is already exact

---

**`IndexLSH` — Locality Sensitive Hashing**

```python
index = faiss.IndexLSH(dimension, nbits)
```

- **How it works:** Hashes vectors into binary codes. Similar vectors likely share hash buckets.
- **Accuracy:** 70-90% (coarse approximation)
- **Speed:** Very fast for high-dimensional data
- **Memory:** Very low (bit vectors)
- **Best for:** Very high dimensions, memory-critical, approximate matching acceptable

**Why we didn't choose this:**
- Lowest accuracy of all options
- 768 dimensions isn't high enough to benefit
- Not appropriate when match quality matters

---

#### Decision Matrix for This Project

| Criterion | Requirement | IndexFlatIP | IndexIVFFlat | IndexHNSWFlat |
|-----------|-------------|-------------|--------------|---------------|
| Index size | 20 items | ✓ Perfect | ✗ Overkill | ✗ Overkill |
| Accuracy | Critical for demo | ✓ 100% | ⚠ 95-99% | ⚠ 95-99% |
| Training data | None available | ✓ Not needed | ✗ Required | ✓ Not needed |
| Configuration | Simple | ✓ Zero params | ✗ nlist, nprobe | ⚠ M, ef params |
| Search latency | Not critical | ✓ < 0.1ms | ✓ < 0.1ms | ✓ < 0.1ms |

**Conclusion:** At 20 items, all indices perform identically in terms of latency. The differentiator is complexity and accuracy. `IndexFlatIP` wins on both.

---

#### When to Choose Each Index Type

```
Is your index size < 10K vectors?
  └─ YES → Use IndexFlatIP (exact search is fast enough)
  └─ NO  → Is accuracy critical (medical, legal, financial)?
              └─ YES → Use IndexFlatIP (accept slower search)
              └─ NO  → Is your index size < 1M vectors?
                          └─ YES → Use IndexHNSWFlat (best speed/accuracy)
                          └─ NO  → Is memory constrained?
                                      └─ YES → Use IndexIVFPQ (compressed)
                                      └─ NO  → Use IndexIVFFlat (good balance)
```

---

#### Scaling Considerations (If Needed)

If this system scaled to a large museum network:

| Scale | Recommended Index | Configuration |
|-------|-------------------|---------------|
| 100 objects | `IndexFlatIP` | None |
| 10K objects | `IndexFlatIP` | None (still < 10ms) |
| 100K objects | `IndexHNSWFlat` | M=32, efSearch=64 |
| 1M objects | `IndexIVFFlat` | nlist=4096, nprobe=64 |
| 10M+ objects | `IndexIVFPQ` | nlist=65536, m=64, nbits=8 |

The current implementation uses `IndexFlatIP` but the service architecture allows swapping index types via configuration without API changes.

### Inner Product vs L2 Distance

**Decision: Inner Product (IP) with normalized vectors**

FAISS supports two distance metrics:
- `IndexFlatL2` — Euclidean (L2) distance
- `IndexFlatIP` — Inner product (dot product)

**Rationale for Inner Product:**

For L2-normalized vectors (unit length), inner product equals cosine similarity:

```
cosine_similarity(a, b) = dot(a, b) / (||a|| * ||b||)

If ||a|| = ||b|| = 1:
cosine_similarity(a, b) = dot(a, b) = inner_product(a, b)
```

The Embeddings Service returns L2-normalized vectors, so:
- IP search finds highest cosine similarity
- Scores range from -1 to 1 (1 = identical)
- Higher score = more similar

This is more intuitive than L2 distance where lower = more similar.

### Metadata Storage

**Decision: Separate JSON file alongside FAISS index**

FAISS indices store only vectors and return integer indices. We need to map these to meaningful metadata (object IDs, names, image paths).

**Approach:**

```
data/index/
├── faiss.index      # FAISS binary index file
└── metadata.json    # ID-to-metadata mapping
```

**metadata.json structure:**

```json
{
  "dimension": 768,
  "count": 20,
  "items": [
    {
      "index": 0,
      "object_id": "object_001",
      "name": "Mona Lisa",
      "image_path": "objects/001.jpg"
    },
    {
      "index": 1,
      "object_id": "object_002",
      "name": "Starry Night",
      "image_path": "objects/002.jpg"
    }
  ]
}
```

**Why not embed metadata in FAISS?**

FAISS is optimized for vector operations, not metadata storage. Keeping them separate:
- Allows metadata updates without rebuilding the index
- Makes debugging easier (metadata is human-readable JSON)
- Follows FAISS best practices

### Index Lifecycle

**Decision: Explicit save/load with auto-load on startup**

| Approach | Pros | Cons |
|----------|------|------|
| Auto-save on every add | No data loss | Slow for batch operations |
| Manual save endpoint | Fast batch adds | Risk of data loss |
| Auto-load on startup | Simple deployment | Startup delay if index is large |

**Our approach:**

1. **Startup:** Auto-load index from configured path if it exists
2. **Runtime adds:** Accumulate in memory (for index building)
3. **Explicit save:** Call `/index/save` to persist
4. **Explicit load:** Call `/index/load` to reload from disk

This balances safety with performance. The `build_index.py` tool:
1. Calls `/add` for each object (fast, in-memory)
2. Calls `/index/save` once at the end (single disk write)

---

## API Endpoints

### GET /health

Health check endpoint for container orchestration.

**Response: 200 OK**

```json
{
  "status": "healthy"
}
```

**Status Logic:**

| Condition | Status |
|-----------|--------|
| Service running, index loaded | `healthy` |
| Service running, index empty/not loaded | `healthy` (empty index is valid) |
| Index file corrupted | `unhealthy` |

---

### GET /info

Returns service metadata including index statistics.

**Response: 200 OK**

```json
{
  "service": "search",
  "version": "0.1.0",
  "index": {
    "type": "flat",
    "metric": "inner_product",
    "embedding_dimension": 768,
    "count": 20,
    "is_loaded": true
  },
  "config": {
    "index_path": "/data/index/faiss.index",
    "metadata_path": "/data/index/metadata.json",
    "default_k": 5
  }
}
```

**Field Descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| `index.type` | string | FAISS index type (flat, ivf, hnsw) |
| `index.metric` | string | Distance metric (inner_product, l2) |
| `index.embedding_dimension` | integer | Expected vector dimension |
| `index.count` | integer | Number of vectors in index |
| `index.is_loaded` | boolean | Whether index is ready for search |

**Usage — Startup Validation:**

```python
def validate_search_service(search_url: str, expected_dimension: int) -> None:
    """Validate search service compatibility."""
    response = httpx.get(f"{search_url}/info")
    info = response.json()
    
    if info["index"]["embedding_dimension"] != expected_dimension:
        raise ConfigurationError("Dimension mismatch")
    
    if not info["index"]["is_loaded"]:
        raise ServiceNotReady("Index not loaded")
```

---

### POST /search

Search for similar vectors in the index.

**Request:**

```
POST /search
Content-Type: application/json
```

```json
{
  "embedding": [0.0234, -0.0891, 0.0412, ...],
  "k": 5,
  "threshold": 0.7
}
```

**Request Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `embedding` | array[float] | Yes | — | Query vector (must match index dimension) |
| `k` | integer | No | 5 | Maximum results to return |
| `threshold` | float | No | 0.0 | Minimum similarity score (0-1 for normalized vectors) |

**Response: 200 OK**

```json
{
  "results": [
    {
      "object_id": "object_007",
      "score": 0.923,
      "rank": 1,
      "metadata": {
        "name": "Water Lilies",
        "image_path": "objects/007.jpg"
      }
    },
    {
      "object_id": "object_012",
      "score": 0.847,
      "rank": 2,
      "metadata": {
        "name": "Bridge over a Pond",
        "image_path": "objects/012.jpg"
      }
    }
  ],
  "count": 2,
  "query_dimension": 768,
  "processing_time_ms": 0.8
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `results` | array | Ranked list of matches |
| `results[].object_id` | string | Unique identifier for the matched object |
| `results[].score` | float | Similarity score (higher = more similar) |
| `results[].rank` | integer | 1-indexed rank in results |
| `results[].metadata` | object | Associated metadata |
| `count` | integer | Number of results returned |
| `query_dimension` | integer | Dimension of query vector (for validation) |
| `processing_time_ms` | float | Search time in milliseconds |

**Score Interpretation (for normalized vectors):**

| Score | Interpretation |
|-------|----------------|
| 0.95 - 1.0 | Near-identical (same image or trivial variation) |
| 0.85 - 0.95 | Very similar (same artwork, different photo) |
| 0.70 - 0.85 | Similar (likely match, verify with geometric) |
| 0.50 - 0.70 | Weak similarity (probably not a match) |
| < 0.50 | Different content |

---

### POST /add

Add an embedding to the index.

**Request:**

```
POST /add
Content-Type: application/json
```

```json
{
  "object_id": "object_001",
  "embedding": [0.0234, -0.0891, 0.0412, ...],
  "metadata": {
    "name": "Mona Lisa",
    "image_path": "objects/001.jpg"
  }
}
```

**Request Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `object_id` | string | Yes | Unique identifier for this object |
| `embedding` | array[float] | Yes | Embedding vector |
| `metadata` | object | No | Additional metadata to store |

**Response: 201 Created**

```json
{
  "object_id": "object_001",
  "index_position": 0,
  "index_count": 1
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `object_id` | string | Echo of input object_id |
| `index_position` | integer | Position in the FAISS index |
| `index_count` | integer | Total items in index after add |

**Notes:**

- Adding an embedding with an existing `object_id` creates a duplicate. The service does not enforce uniqueness (FAISS doesn't support updates).
- The embedding is added to memory only. Call `/index/save` to persist.
- Vector dimension must match configured `embedding_dimension`.

---

### POST /index/save

Persist the current index to disk.

**Request:**

```
POST /index/save
Content-Type: application/json
```

```json
{}
```

Or with custom path:

```json
{
  "path": "/data/index/custom.index"
}
```

**Request Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `path` | string | No | config value | Custom save path |

**Response: 200 OK**

```json
{
  "index_path": "/data/index/faiss.index",
  "metadata_path": "/data/index/metadata.json",
  "count": 20,
  "size_bytes": 61440
}
```

---

### POST /index/load

Load an index from disk.

**Request:**

```
POST /index/load
Content-Type: application/json
```

```json
{}
```

Or with custom path:

```json
{
  "path": "/data/index/custom.index"
}
```

**Response: 200 OK**

```json
{
  "index_path": "/data/index/faiss.index",
  "metadata_path": "/data/index/metadata.json",
  "count": 20,
  "dimension": 768
}
```

---

### DELETE /index

Clear the index (remove all vectors and metadata).

**Request:**

```
DELETE /index
```

**Response: 200 OK**

```json
{
  "previous_count": 20,
  "current_count": 0
}
```

**Use Case:** Rebuilding the index from scratch without restarting the service.

---

## Error Handling

### Error Codes

| Error Code | HTTP Status | Description | Client Action |
|------------|-------------|-------------|---------------|
| `dimension_mismatch` | 400 | Vector dimension doesn't match index | Check embedding service config |
| `invalid_embedding` | 400 | Embedding is malformed (NaN, wrong type) | Validate embedding |
| `index_not_loaded` | 503 | Index not available for search | Call /index/load or wait for startup |
| `index_empty` | 422 | Search called on empty index | Add embeddings first |
| `save_failed` | 500 | Failed to write index to disk | Check disk space/permissions |
| `load_failed` | 500 | Failed to read index from disk | Check file exists/permissions |
| `internal_error` | 500 | Unexpected error | Retry; report if persistent |

### Error Response Examples

**Dimension Mismatch (400):**

```json
{
  "error": "dimension_mismatch",
  "message": "Embedding dimension 1024 does not match index dimension 768",
  "details": {
    "expected": 768,
    "received": 1024
  }
}
```

**Index Not Loaded (503):**

```json
{
  "error": "index_not_loaded",
  "message": "Index is not loaded. Call POST /index/load or wait for startup.",
  "details": {
    "index_path": "/data/index/faiss.index"
  }
}
```

---

## Size Estimations

### Index Size

FAISS `IndexFlatIP` stores vectors as 32-bit floats:

```
size = n_vectors × dimension × 4 bytes
```

| Vectors | Dimension | Index Size |
|---------|-----------|------------|
| 20 | 768 | 60 KB |
| 1,000 | 768 | 3 MB |
| 100,000 | 768 | 300 MB |
| 1,000,000 | 768 | 3 GB |

For this project (20 objects), the index is trivially small.

### Metadata Size

Metadata is stored as JSON. Estimate ~200 bytes per item:

| Items | Metadata Size |
|-------|---------------|
| 20 | 4 KB |
| 1,000 | 200 KB |
| 100,000 | 20 MB |

### Request/Response Sizes

| Operation | Request Size | Response Size |
|-----------|--------------|---------------|
| Search (768-dim) | ~15 KB | ~2 KB (5 results) |
| Add (768-dim) | ~15 KB | ~100 bytes |
| Save/Load | ~100 bytes | ~200 bytes |

### Memory Usage

| Component | Memory |
|-----------|--------|
| FAISS index (20 items) | ~100 KB |
| FAISS index (100K items) | ~350 MB |
| Metadata (20 items) | ~10 KB |
| Service overhead | ~50 MB |
| **Recommended container limit** | **512 MB - 1 GB** |

---

## Configuration

### Configuration Schema

```yaml
# config.yaml
faiss:
  embedding_dimension: 768
  index_type: "flat"         # flat | ivf | hnsw
  metric: "inner_product"    # inner_product | l2

index:
  path: "/data/index/faiss.index"
  metadata_path: "/data/index/metadata.json"
  auto_load: true            # Load index on startup if exists

search:
  default_k: 5
  max_k: 100
  default_threshold: 0.0

server:
  host: "0.0.0.0"
  port: 8000
  log_level: "info"
```

### Environment Variable Overrides

```bash
SEARCH__FAISS__EMBEDDING_DIMENSION=1024
SEARCH__INDEX__PATH=/custom/path/faiss.index
SEARCH__SEARCH__DEFAULT_K=10
```

---

## Usage Examples

### curl

```bash
# Health check
curl http://localhost:8002/health

# Service info
curl http://localhost:8002/info

# Search (using jq to format embedding)
curl -X POST http://localhost:8002/search \
  -H "Content-Type: application/json" \
  -d '{
    "embedding": [0.1, 0.2, ...],
    "k": 5,
    "threshold": 0.7
  }'

# Add embedding
curl -X POST http://localhost:8002/add \
  -H "Content-Type: application/json" \
  -d '{
    "object_id": "object_001",
    "embedding": [0.1, 0.2, ...],
    "metadata": {"name": "Test Object"}
  }'

# Save index
curl -X POST http://localhost:8002/index/save \
  -H "Content-Type: application/json" \
  -d '{}'

# Clear index
curl -X DELETE http://localhost:8002/index
```

### Python — Building an Index

```python
import httpx
from pathlib import Path

EMBEDDINGS_URL = "http://localhost:8001"
SEARCH_URL = "http://localhost:8002"

def build_index(objects_dir: Path, labels: dict[str, str]) -> None:
    """Build search index from object images."""
    
    with httpx.Client(timeout=60.0) as client:
        # Clear existing index
        client.delete(f"{SEARCH_URL}/index")
        
        for image_path in sorted(objects_dir.glob("*.jpg")):
            object_id = image_path.stem
            
            # Get embedding from embeddings service
            image_b64 = base64.b64encode(image_path.read_bytes()).decode()
            emb_response = client.post(
                f"{EMBEDDINGS_URL}/embed",
                json={"image": image_b64, "image_id": object_id}
            )
            embedding = emb_response.json()["embedding"]
            
            # Add to search index
            client.post(
                f"{SEARCH_URL}/add",
                json={
                    "object_id": object_id,
                    "embedding": embedding,
                    "metadata": {
                        "name": labels.get(object_id, "Unknown"),
                        "image_path": str(image_path)
                    }
                }
            )
            print(f"Added {object_id}")
        
        # Persist index
        save_response = client.post(f"{SEARCH_URL}/index/save", json={})
        print(f"Saved index: {save_response.json()['count']} items")
```

### Python — Searching

```python
async def search_similar(
    embedding: list[float],
    k: int = 5,
    threshold: float = 0.7,
) -> list[dict]:
    """Search for similar objects."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SEARCH_URL}/search",
            json={
                "embedding": embedding,
                "k": k,
                "threshold": threshold,
            }
        )
        response.raise_for_status()
        return response.json()["results"]
```

---

## Performance Characteristics

### Search Latency

For `IndexFlatIP` (exact search):

| Index Size | Search Time (k=5) |
|------------|-------------------|
| 20 | < 0.1 ms |
| 1,000 | < 1 ms |
| 100,000 | ~10-50 ms |
| 1,000,000 | ~100-500 ms |

For this project (20 objects), search is effectively instantaneous.

### Add Latency

| Operation | Time |
|-----------|------|
| Add single vector | < 0.1 ms |
| Add 1000 vectors (sequential) | ~10-50 ms |

### Save/Load Latency

| Index Size | Save Time | Load Time |
|------------|-----------|-----------|
| 20 items | < 10 ms | < 10 ms |
| 100K items | ~500 ms | ~200 ms |

---

## OpenAPI Specification

```yaml
openapi: 3.1.0
info:
  title: Search Service
  description: FAISS vector similarity search for artwork embeddings
  version: 0.1.0
servers:
  - url: http://localhost:8002
    description: Local development

paths:
  /health:
    get:
      operationId: health_check
      summary: Health check
      tags: [Operations]
      responses:
        "200":
          description: Service status
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
          description: Service information with index stats
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/InfoResponse"

  /search:
    post:
      operationId: search
      summary: Search for similar vectors
      tags: [Search]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/SearchRequest"
      responses:
        "200":
          description: Search results
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/SearchResponse"
        "400":
          description: Invalid request
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"
        "503":
          description: Index not loaded
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"

  /add:
    post:
      operationId: add_embedding
      summary: Add embedding to index
      tags: [Index Management]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/AddRequest"
      responses:
        "201":
          description: Embedding added
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/AddResponse"
        "400":
          description: Invalid request
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"

  /index/save:
    post:
      operationId: save_index
      summary: Persist index to disk
      tags: [Index Management]
      requestBody:
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/SaveRequest"
      responses:
        "200":
          description: Index saved
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/SaveResponse"
        "500":
          description: Save failed
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"

  /index/load:
    post:
      operationId: load_index
      summary: Load index from disk
      tags: [Index Management]
      requestBody:
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/LoadRequest"
      responses:
        "200":
          description: Index loaded
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/LoadResponse"
        "500":
          description: Load failed
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"

  /index:
    delete:
      operationId: clear_index
      summary: Clear the index
      tags: [Index Management]
      responses:
        "200":
          description: Index cleared
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ClearResponse"

components:
  schemas:
    HealthResponse:
      type: object
      required: [status]
      properties:
        status:
          type: string
          enum: [healthy, degraded, unhealthy]

    InfoResponse:
      type: object
      required: [service, version, index]
      properties:
        service:
          type: string
          example: search
        version:
          type: string
          example: 0.1.0
        index:
          type: object
          properties:
            type:
              type: string
              example: flat
            metric:
              type: string
              example: inner_product
            embedding_dimension:
              type: integer
              example: 768
            count:
              type: integer
              example: 20
            is_loaded:
              type: boolean
              example: true

    SearchRequest:
      type: object
      required: [embedding]
      properties:
        embedding:
          type: array
          items:
            type: number
            format: float
          description: Query embedding vector
        k:
          type: integer
          minimum: 1
          maximum: 100
          default: 5
          description: Maximum results to return
        threshold:
          type: number
          format: float
          minimum: 0
          maximum: 1
          default: 0
          description: Minimum similarity score

    SearchResponse:
      type: object
      required: [results, count]
      properties:
        results:
          type: array
          items:
            $ref: "#/components/schemas/SearchResult"
        count:
          type: integer
        query_dimension:
          type: integer
        processing_time_ms:
          type: number
          format: float

    SearchResult:
      type: object
      required: [object_id, score, rank]
      properties:
        object_id:
          type: string
        score:
          type: number
          format: float
        rank:
          type: integer
        metadata:
          type: object
          additionalProperties: true

    AddRequest:
      type: object
      required: [object_id, embedding]
      properties:
        object_id:
          type: string
          description: Unique identifier for this object
        embedding:
          type: array
          items:
            type: number
            format: float
        metadata:
          type: object
          additionalProperties: true

    AddResponse:
      type: object
      required: [object_id, index_position, index_count]
      properties:
        object_id:
          type: string
        index_position:
          type: integer
        index_count:
          type: integer

    SaveRequest:
      type: object
      properties:
        path:
          type: string
          description: Custom save path (optional)

    SaveResponse:
      type: object
      properties:
        index_path:
          type: string
        metadata_path:
          type: string
        count:
          type: integer
        size_bytes:
          type: integer

    LoadRequest:
      type: object
      properties:
        path:
          type: string
          description: Custom load path (optional)

    LoadResponse:
      type: object
      properties:
        index_path:
          type: string
        metadata_path:
          type: string
        count:
          type: integer
        dimension:
          type: integer

    ClearResponse:
      type: object
      properties:
        previous_count:
          type: integer
        current_count:
          type: integer

    ErrorResponse:
      type: object
      required: [error, message]
      properties:
        error:
          type: string
        message:
          type: string
        details:
          type: object
          additionalProperties: true
```

---

## Appendix: Why FAISS?

FAISS (Facebook AI Similarity Search) was selected for this project:

1. **Industry standard** — Used by Meta, Spotify, and many production systems for billion-scale search.

2. **Python bindings** — `pip install faiss-cpu` provides full functionality. GPU version available for larger scale.

3. **Appropriate complexity** — For 20-100K items, FAISS with flat index is simpler than managed vector databases (Pinecone, Weaviate, Qdrant).

4. **Control** — Full control over index type, persistence, and search parameters. No external dependencies or API costs.

5. **Performance** — Even on CPU, sub-millisecond search for museum-scale collections.

**Alternatives considered:**

| Option | Pros | Cons |
|--------|------|------|
| Qdrant | Great API, built-in persistence | Additional service to manage |
| ChromaDB | Simple, good for prototypes | Less control, slower at scale |
| Pinecone | Managed, scalable | External dependency, cost |
| NumPy brute force | No dependencies | Slower, no index optimizations |

For this demo, FAISS provides the right balance of simplicity, performance, and control.
