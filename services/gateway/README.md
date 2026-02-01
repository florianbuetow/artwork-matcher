# Gateway Service

The public-facing API that orchestrates the artwork identification pipeline. It coordinates requests across internal services (Embeddings, Search, Geometric) and presents a unified interface to clients.

## Quick Start

```bash
# From repository root
just run-gateway

# Or from this directory
just run
```

The service will be available at `http://localhost:8000`.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (includes backend status) |
| `/info` | GET | Service configuration and version |
| `/identify` | POST | Upload photo, receive matched artwork |
| `/objects` | GET | List all artworks in database |
| `/objects/{id}` | GET | Get artwork details |
| `/objects/{id}/image` | GET | Get artwork reference image |

### Example: Identify Artwork

```bash
curl -X POST http://localhost:8000/identify \
  -H "Content-Type: application/json" \
  -d '{"image": "'$(base64 -i photo.jpg | tr -d '\n')'"}'
```

## Configuration

Configuration is loaded from `config.yaml` with environment variable overrides.

```yaml
# config.yaml
backends:
  embeddings_url: "http://localhost:8001"
  search_url: "http://localhost:8002"
  geometric_url: "http://localhost:8003"

pipeline:
  search_k: 5                    # Candidates to retrieve
  similarity_threshold: 0.7      # Minimum embedding similarity
  geometric_verification: true   # Enable geometric stage

server:
  host: "0.0.0.0"
  port: 8000
  log_level: "info"
```

### Environment Overrides

```bash
GATEWAY__BACKENDS__EMBEDDINGS_URL=http://embeddings:8001
GATEWAY__PIPELINE__SEARCH_K=10
GATEWAY__PIPELINE__GEOMETRIC_VERIFICATION=false
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

## Architecture

```
Client Request
      │
      ▼
┌─────────────────┐
│ 1. Embed Image  │ ──→ Embeddings Service (:8001)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Search Index │ ──→ Search Service (:8002)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Verify Match │ ──→ Geometric Service (:8003)
└────────┬────────┘
         │
         ▼
   Best Match Response
```

## API Specification

For complete API documentation, see [Gateway Service API Spec](../../docs/api/gateway_service_api_spec.md).