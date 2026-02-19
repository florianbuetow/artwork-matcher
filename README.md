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
| Storage | File system | Binary object storage for reference images and metadata |
| Gateway | FastAPI | Orchestrates pipeline, single public API |

This mirrors production systems like Smartify and Google Arts & Culture.

## Prerequisites

- **just** - Command runner ([installation guide](https://github.com/casey/just#installation))
- **uv** - Python package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Python 3.12+** - Programming language
- **Docker** - Container runtime ([installation guide](https://docs.docker.com/get-docker/))
- **curl** - HTTP client (pre-installed on most systems)
- **jq** - JSON processor ([installation guide](https://jqlang.github.io/jq/download/))
- **lsof** - Process inspector (pre-installed on most systems)

## Quick Start

### 1. Check prerequisites

Verify that all required tools are installed:

```bash
just check
```

### 2. Initialize and start services

```bash
just init-all
just start-all
```

### 3. Prepare evaluation data

The evaluation pipeline expects a `data/evaluation/` directory with the following structure:

```
data/evaluation/
├── objects/        # Reference images of museum artworks (one per artwork)
├── pictures/       # Visitor photos to match against the reference collection
└── labels.csv      # Ground truth: maps each visitor photo to its correct artwork(s)
```

- **`objects/`** — Contains one image file per artwork in the museum's collection. The filename (without extension) is the artwork's unique ID.
- **`pictures/`** — Contains visitor photos. Each photo should depict one of the artworks from `objects/`.
- **`labels.csv`** — A CSV file mapping visitor photos to their correct artwork matches. Format:

  ```
  picture_image_path, painting_image_paths
  photo_001.jpg, artwork_A.jpg
  photo_002.jpg, artwork_B.jpg;artwork_C.jpg
  ```

  - `picture_image_path` — filename of the visitor photo (relative to `pictures/`)
  - `painting_image_paths` — one or more artwork filenames (relative to `objects/`), separated by `;` when multiple artworks are valid matches

Populate this directory with your own dataset. The project does not ship evaluation data.

### 4. Run the evaluation pipeline

Index the evaluation dataset and run the end-to-end accuracy and performance test:

```bash
just evaluate
```

### 5. Open the web application

Open the gateway web UI in your browser at [http://localhost:8000](http://localhost:8000), or run:

```bash
just demo
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
│                         Client                                  │
│                    (Browser / curl / App)                       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Gateway Service (:8000)                      │
│                                                                 │
│   POST /identify ──────────────────────────────────────────┐    │
│                                                            │    │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐      │    │
│   │ Embeddings  │   │   Search    │   │  Geometric  │      │    │
│   │   :8001     │──▶│    :8002    │──▶│    :8003    │      │    │
│   │  (DINOv2)   │   │   (FAISS)   │   │ (ORB+RANSAC)│      │    │
│   └─────────────┘   └──────┬──────┘   └─────────────┘      │    │
│                             │                              │    │
│                      ┌──────┴──────┐                       │    │
│                      │   Storage   │                       │    │
│                      │    :8004    │                       │    │
│                      └─────────────┘                       │    │
│                                                            │    │
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
| [Storage](services/storage/README.md) | 8004 | Binary object storage |

### Documentation

| Document | Description |
|----------|-------------|
| [API Overview](docs/api/README.md) | System architecture, sequence diagrams, and data flows |
| [Gateway API Spec](docs/api/gateway_service_api_spec.md) | Public API endpoints and orchestration |
| [Embeddings API Spec](docs/api/embeddings_service_api_spec.md) | DINOv2 embedding extraction |
| [Search API Spec](docs/api/search_service_api_spec.md) | FAISS vector similarity search |
| [Geometric API Spec](docs/api/geometric_service_api_spec.md) | ORB + RANSAC verification |
| [Uniform API Structure](docs/api/uniform_api_structure.md) | Common conventions across all services |
| [Confidence Scoring Penalties](docs/decisions/confidence-scoring-penalties.md) | Decision record and tuning plan for confidence scoring |
| [Performance Testing Methodology](docs/performance/testing-methodology.md) | Strategy for service benchmarks and end-to-end validation |
| [Evaluation Report](reports/evaluation/evaluation_report.md) | E2E accuracy evaluation results against test dataset |
| [Embeddings Performance](reports/performance/embedding_service_performance.md) | Embeddings service latency and throughput benchmarks |
| [Search Performance](reports/performance/search_service_performance.md) | Search service latency and throughput benchmarks |
| [Geometric Performance](reports/performance/geometric_service_performance.md) | Geometric service latency and throughput benchmarks |
| [Gateway Performance](reports/performance/gateway_service_performance.md) | Gateway service latency and throughput benchmarks |

## Development

### Docker (All Services)

- `just docker-up` - Start all services (production mode, detached)
- `just docker-up-dev` - Start all services (dev mode with hot reload)
- `just docker-down` - Stop all services
- `just docker-logs [service]` - View service logs
- `just docker-build` - Build all Docker images

### Docker (Single Service)

From within a service directory (e.g., `cd services/embeddings`):
- `just docker-up` - Start this service (production mode, detached)
- `just docker-up-dev` - Start this service (dev mode with hot reload)
- `just docker-down` - Stop this service
- `just docker-logs` - View logs for this service
- `just docker-build` - Build Docker image for this service

### Local Development (No Docker)

- `just init-all` - Initialize all service environments
- `just start-all` - Start all services in background
- `just start-embeddings` - Start embeddings service locally
- `just start-search` - Start search service locally
- `just start-geometric` - Start geometric service locally
- `just start-storage` - Start storage service locally
- `just start-gateway` - Start gateway service locally
- `just stop-all` - Stop all locally running services
- `just status` - Check health status of all services
- `just destroy-all` - Remove all virtual environments

Or from within a service directory:
```bash
cd services/embeddings
just init  # First time only
just run   # Runs locally with hot reload
```

### General

- `just help` - Show all available commands

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

### Ingestion

- `just download-batch` - Download diverse batch from Rijksmuseum
- `just download <args>` - Download with custom options
- `just build-index` - Build FAISS index from downloaded images
- `just delete-index` - Delete the FAISS index

### Evaluation

- `just build-eval-index` - Build FAISS index from evaluation object images
- `just evaluate` - Full E2E evaluation pipeline (local)
- `just docker-evaluate` - Full E2E evaluation pipeline (Docker)
- `just test-perf-all` - Run performance tests for all services
- `just test-perf-embeddings` - Run performance tests for embeddings
- `just test-perf-search` - Run performance tests for search
- `just test-perf-geometric` - Run performance tests for geometric
- `just test-perf-gateway` - Run performance tests for gateway

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
│   ├── storage/            # Binary object storage (port 8004)
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
│   ├── evaluate.py         # Evaluate accuracy against labels.csv
│   ├── run_evaluation.py   # Full E2E evaluation pipeline
│   ├── verify_evaluation.py # Verify evaluation pipeline wiring
│   ├── evaluation/         # Evaluation client, metrics, models
│   └── downloader/         # Rijksmuseum data downloader
│       ├── config.yaml     # Download configuration
│       └── download_data.py
│
├── tests/                  # Integration tests (cross-service)
│   ├── conftest.py
│   └── test_integration.py
│
├── data/
│   ├── downloads/          # Downloaded Rijksmuseum data (output)
│   ├── evaluation/         # Evaluation dataset
│   │   ├── objects/        # Reference images for evaluation
│   │   ├── pictures/       # Visitor test photos for evaluation
│   │   └── labels.csv      # Ground truth mappings
│   ├── objects/            # Museum object reference images (production)
│   ├── pictures/           # Visitor test photos (production)
│   ├── index/              # Generated FAISS index files (output)
│   ├── models/             # Cached model weights
│   └── features/           # Pre-computed ORB features (optional)
│
└── config/
    ├── semgrep/            # Custom static analysis rules
    └── codespell/          # Spell-check ignore list
```

## Project Rules

See [AGENTS.md](AGENTS.md) for detailed development guidelines including:
- Python execution rules (use `uv run` exclusively)
- Git commit guidelines
- Testing requirements
- Project structure conventions

## License

MIT License - See [LICENSE](LICENSE) for details.
