# Embeddings Service

Extracts visual embedding vectors from artwork images using DINOv2 foundation models. These embeddings enable fast similarity search against the museum's collection.

## Quick Start

```bash
# From repository root
just run-embeddings

# Or from this directory
just run
```

The service will be available at `http://localhost:8001`.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/info` | GET | Model configuration and version |
| `/embed` | POST | Extract embedding from image |

### Example: Extract Embedding

```bash
curl -X POST http://localhost:8001/embed \
  -H "Content-Type: application/json" \
  -d '{"image": "'$(base64 -i artwork.jpg | tr -d '\n')'", "image_id": "test_001"}'
```

Response:
```json
{
  "embedding": [0.0234, -0.0891, 0.0412, ...],
  "dimension": 768,
  "image_id": "test_001",
  "processing_time_ms": 47.3
}
```

## Configuration

Configuration is loaded from `config.yaml` with environment variable overrides.

```yaml
# config.yaml
model:
  name: "facebook/dinov2-base"
  device: "auto"  # auto, cpu, cuda, mps
  embedding_dimension: 768

preprocessing:
  image_size: 518
  normalize: true

server:
  host: "0.0.0.0"
  port: 8000
  log_level: "info"
```

### Environment Overrides

```bash
EMBEDDINGS__MODEL__NAME=facebook/dinov2-large
EMBEDDINGS__MODEL__DEVICE=cuda
EMBEDDINGS__MODEL__EMBEDDING_DIMENSION=1024
```

### Supported Models

| Model | Dimension | Memory | Speed |
|-------|-----------|--------|-------|
| `facebook/dinov2-small` | 384 | ~200 MB | Fast |
| `facebook/dinov2-base` | 768 | ~400 MB | Medium |
| `facebook/dinov2-large` | 1024 | ~1.2 GB | Slow |

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

## Why DINOv2?

- **Artwork-validated**: Meta benchmarked DINOv2 on the Met Museum dataset (+34% mAP improvement)
- **Self-supervised**: Learns from visual patterns without text bias
- **Robust**: Handles lighting, viewpoint, and partial occlusion well
- **Practical**: Single `pip install transformers` to use

## API Specification

For complete API documentation, see [Embeddings Service API Spec](../../docs/api/embeddings_service_api_spec.md).