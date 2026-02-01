# Search Service

Manages a FAISS vector index for fast similarity search. Stores embeddings with metadata and returns the most similar artworks for a query vector.

## Quick Start

```bash
# From repository root
just run-search

# Or from this directory
just run
```

The service will be available at `http://localhost:8002`.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/info` | GET | Index statistics and configuration |
| `/search` | POST | Find similar vectors |
| `/add` | POST | Add embedding to index |
| `/index/save` | POST | Persist index to disk |
| `/index/load` | POST | Load index from disk |
| `/index` | DELETE | Clear the index |

### Example: Search

```bash
curl -X POST http://localhost:8002/search \
  -H "Content-Type: application/json" \
  -d '{"embedding": [0.1, 0.2, ...], "k": 5, "threshold": 0.7}'
```

Response:
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
    }
  ],
  "count": 1,
  "processing_time_ms": 0.8
}
```

### Example: Add to Index

```bash
curl -X POST http://localhost:8002/add \
  -H "Content-Type: application/json" \
  -d '{
    "object_id": "object_001",
    "embedding": [0.1, 0.2, ...],
    "metadata": {"name": "Mona Lisa", "artist": "Leonardo da Vinci"}
  }'
```

## Configuration

Configuration is loaded from `config.yaml` with environment variable overrides.

```yaml
# config.yaml
faiss:
  embedding_dimension: 768
  index_type: "flat"         # flat, ivf, hnsw
  metric: "inner_product"    # inner_product, l2

index:
  path: "/data/index/faiss.index"
  metadata_path: "/data/index/metadata.json"
  auto_load: true

search:
  default_k: 5
  max_k: 100

server:
  host: "0.0.0.0"
  port: 8000
  log_level: "info"
```

### Environment Overrides

```bash
SEARCH__FAISS__EMBEDDING_DIMENSION=1024
SEARCH__INDEX__PATH=/custom/path/faiss.index
SEARCH__SEARCH__DEFAULT_K=10
```

## Development

```bash
# Initialize environment
just init

# Run with hot reload
just run

# Run tests
just test

# Run all CI checks
just ci
```

## Index Types

| Index | Best For | Accuracy | Speed |
|-------|----------|----------|-------|
| `flat` | < 100K vectors | 100% | O(n) |
| `ivf` | 100K - 10M vectors | 95-99% | O(n/clusters) |
| `hnsw` | 10K - 10M vectors | 95-99% | O(log n) |

For 20 objects, `flat` (brute force) is ideal — exact results in < 0.1ms.

## API Specification

For complete API documentation, see [Search Service API Spec](../../docs/api/search_service_api_spec.md).