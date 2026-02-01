# artwork-matcher

A microservices-based museum artwork recognition system that identifies artworks from visitor photos using deep learning embeddings and geometric verification.

## Overview

Museum visitors photograph artworks during their visit. This system automatically identifies which artwork they photographed by comparing against a reference database of museum objects.

### Approach

**Two-stage retrieval pipeline:**

1. **Fast candidate retrieval** — DINOv2 embeddings + FAISS vector search finds visually similar artworks in <10ms
2. **Geometric verification** — ORB feature matching + RANSAC confirms spatial consistency, eliminating false positives

### Why this architecture?

| Stage | Method | Purpose |
|-------|--------|---------|
| Embeddings | DINOv2 (Meta) | State-of-the-art visual similarity, validated on Met Museum dataset |
| Search | FAISS IndexFlatIP | Exact nearest neighbor search, no training required |
| Verification | ORB + RANSAC | Classical CV confirms local features align spatially |
| Gateway | FastAPI | Orchestrates pipeline, single public API |

This mirrors production systems like Smartify and Google Arts & Culture.

## Prerequisites

- **Python 3.12+** - Programming language
- **uv** - Python package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))
- **just** - Command runner ([installation guide](https://github.com/casey/just#installation))
- **Docker** - Container runtime ([installation guide](https://docs.docker.com/get-docker/))

## Quick Start

1. **Download the data**

   > **🚧 WORK IN PROGRESS** — Data download instructions coming soon.

   - `objects/` → `data/objects/`
   - `pictures/` → `data/pictures/`
   - `labels.csv` → `data/labels.csv`

2. **Initialize and start services:**
   ```bash
   just init-all
   just up
   ```

3. **Build the search index:**
   ```bash
   just build-index
   ```

4. **Test identification:**
   ```bash
   curl -X POST http://localhost:8000/identify \
     -H "Content-Type: application/json" \
     -d '{"image": "'$(base64 -i data/pictures/001.jpg | tr -d '\n')'"}'
   ```

5. **Evaluate accuracy:**
   ```bash
   just evaluate
   ```

## API Endpoints

All endpoints are exposed through the Gateway service (port 8000).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check (includes backend status) |
| `/info` | GET | Service configuration and version info |
| `/identify` | POST | Upload visitor photo, receive matched artwork |
| `/objects` | GET | List all artworks in the database |
| `/objects/{id}` | GET | Get details for a specific artwork |
| `/objects/{id}/image` | GET | Retrieve artwork reference image |

### Example: Identify an Artwork

```bash
curl -X POST http://localhost:8000/identify \
  -H "Content-Type: application/json" \
  -d '{"image": "'$(base64 -i visitor_photo.jpg | tr -d '\n')'"}'
```

Response:
```json
{
  "success": true,
  "match": {
    "object_id": "object_007",
    "name": "Water Lilies",
    "artist": "Claude Monet",
    "confidence": 0.89,
    "image_url": "/objects/object_007/image"
  },
  "timing": {
    "embedding_ms": 47.2,
    "search_ms": 1.3,
    "geometric_ms": 156.8,
    "total_ms": 208.4
  }
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client                                   │
│                    (Browser / curl / App)                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Gateway Service (:8000)                       │
│                                                                  │
│   POST /identify ──────────────────────────────────────────┐    │
│                                                             │    │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐     │    │
│   │ Embeddings  │   │   Search    │   │  Geometric  │     │    │
│   │   :8001     │──▶│    :8002    │──▶│    :8003    │     │    │
│   │  (DINOv2)   │   │   (FAISS)   │   │ (ORB+RANSAC)│     │    │
│   └─────────────┘   └─────────────┘   └─────────────┘     │    │
│                                                             │    │
│   ◀─────────────────── Best Match ─────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| [Gateway](services/gateway/README.md) | 8000 | Public API, orchestrates identification pipeline |
| [Embeddings](services/embeddings/README.md) | 8001 | DINOv2 embedding extraction |
| [Search](services/search/README.md) | 8002 | FAISS vector similarity search |
| [Geometric](services/geometric/README.md) | 8003 | ORB + RANSAC geometric verification |

### Documentation

| Document | Description |
|----------|-------------|
| [API Overview](docs/api/README.md) | System architecture, sequence diagrams, and data flows |
| [Gateway API Spec](docs/api/gateway_service_api_spec.md) | Public API endpoints and orchestration |
| [Embeddings API Spec](docs/api/embeddings_service_api_spec.md) | DINOv2 embedding extraction |
| [Search API Spec](docs/api/search_service_api_spec.md) | FAISS vector similarity search |
| [Geometric API Spec](docs/api/geometric_service_api_spec.md) | ORB + RANSAC verification |
| [Uniform API Structure](docs/api/uniform_api_structure.md) | Common conventions across all services |

## Development

### All-in-One Commands

- `just init-all` - Initialize all service environments
- `just up` - Start all services (Docker)
- `just up-dev` - Start services with hot reload
- `just down` - Stop all services
- `just logs [service]` - View service logs
- `just build` - Build all Docker images
- `just destroy-all` - Remove all virtual environments
- `just help` - Show all available commands

### Running Individual Services

- `just run-embeddings` - Run embeddings service locally
- `just run-search` - Run search service locally
- `just run-geometric` - Run geometric service locally
- `just run-gateway` - Run gateway service locally

### Code Quality & Testing

| Command | Description |
|---------|-------------|
| `just test-all` | Run tests for all services |
| `just format-all` | Auto-format code across all services |
| `just ci-all` | Run all validation checks (verbose) |
| `just ci-all-quiet` | Run all checks (silent, fail-fast) |

#### Per-Service Quality Checks
Run these from the root as `just <command> <service>` or within a service directory as `just <command>`:
- `code-style` - Check formatting (read-only)
- `code-format` - Auto-fix formatting
- `code-typecheck` - Mypy type checking
- `code-lspchecks` - Pyright strict type checking
- `code-security` - Bandit security scan
- `code-deptry` - Dependency hygiene
- `code-spell` - Spell check
- `code-audit` - Vulnerability scan
- `code-semgrep` - Static analysis

### Data Pipeline

- `just build-index` - Build FAISS index from object images
- `just evaluate` - Evaluate accuracy against labels.csv

## Repository Structure

```
artwork-matcher/
├── AGENTS.md               # AI agent development rules
├── CLAUDE.md               # Symlink to AGENTS.md
├── README.md               # This file
├── justfile                # Root task runner
├── docker-compose.yml      # Production Docker configuration
├── docker-compose.dev.yml  # Development Docker configuration
├── .gitignore              # Git ignore patterns
│
├── services/
│   ├── embeddings/         # DINOv2 embedding extraction (port 8001)
│   │   ├── README.md       # Service documentation
│   │   ├── config.yaml     # Service configuration
│   │   ├── pyproject.toml  # Dependencies
│   │   ├── justfile        # Service-specific commands
│   │   ├── src/embeddings_service/
│   │   └── tests/
│   │
│   ├── search/             # FAISS vector search (port 8002)
│   │   ├── README.md
│   │   └── ...
│   │
│   ├── geometric/          # ORB + RANSAC verification (port 8003)
│   │   ├── README.md
│   │   └── ...
│   │
│   └── gateway/            # API orchestration (port 8000)
│       ├── README.md
│       └── ...
│
├── tools/
│   ├── justfile            # Tools task runner
│   ├── pyproject.toml      # Tools dependencies
│   ├── build_index.py      # Build FAISS index from object images
│   └── evaluate.py         # Evaluate accuracy against labels.csv
│
├── tests/                  # Integration tests (cross-service)
│   ├── conftest.py
│   └── test_integration.py
│
├── data/
│   ├── objects/            # Museum object reference images
│   ├── pictures/           # Visitor test photos
│   ├── index/              # Generated FAISS index files
│   ├── features/           # Pre-computed ORB features (optional)
│   └── labels.csv          # Ground truth mappings
│
└── config/
    ├── semgrep/            # Custom static analysis rules
    └── codespell/          # Spell-check ignore list
```

## Project Rules

See [AGENTS.md](AGENTS.md) for detailed development guidelines including:
- Python execution rules (use `uv run` exclusively)
- Git commit guidelines (no AI attribution)
- Testing requirements
- Project structure conventions

## License

MIT License - See LICENSE file for details.
