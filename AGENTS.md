# Artwork Matcher - AI Agent Instructions

## Project Overview

Artwork Matcher is a Python microservices project that matches visitor photos of artworks to a museum's reference collection. It uses DINOv2 embeddings, FAISS vector search, and ORB+RANSAC geometric verification, orchestrated through a FastAPI gateway.

## Build & Run

```bash
just help             # Show all available commands
just init-all         # Initialize all service environments
just start-all        # Start all services in background
just stop-all         # Stop all locally running services
just status           # Check health status of all services
just test-all         # Run all tests
just ci-all           # Run all CI checks (verbose)
just ci-all-quiet     # Run all CI checks (quiet)
just destroy-all      # Remove all virtual environments
```

### Docker

```bash
just docker-up        # Start all services (Docker)
just docker-up-dev    # Start with hot reload
just docker-down      # Stop all services
just docker-logs [service]  # View logs
just docker-build     # Build all Docker images
```

### Data & Evaluation

```bash
just download-batch   # Download diverse batch from Rijksmuseum
just build-index      # Build FAISS index from downloaded images
just delete-index     # Delete the FAISS index
just build-eval-index # Build FAISS index from evaluation images
just evaluate         # Full E2E evaluation pipeline (local)
just docker-evaluate  # Full E2E evaluation pipeline (Docker)
```

### Per-Service Commands

Run from within `services/<name>/`:

```bash
just init             # Initialize service environment
just destroy          # Remove virtual environment
just run              # Run service locally with hot reload
just test             # Run unit tests
just ci               # Run all CI checks (verbose)
just ci-quiet         # Run all CI checks (quiet)
just code-format      # Auto-fix formatting
just code-style       # Check style (read-only)
just code-typecheck   # Run mypy
just code-lspchecks   # Run pyright (strict)
just code-security    # Run bandit
just code-deptry      # Check dependencies
just code-spell       # Check spelling
just code-semgrep     # Run custom rules
just code-audit       # Vulnerability scan
```

## Testing

- After **every change** to the code, the tests must be executed
- Always verify the program runs correctly with `just run` after modifications
- Always run `just test-all` or `just ci-all-quiet` to verify changes before claiming they work
- **Tests are acceptance tests — do NOT modify existing test files.** Add new test files to cover new or additional requirements instead.

## Architecture

```
services/
  gateway/            Public API, orchestration (port 8000)
  embeddings/         DINOv2 embedding extraction (port 8001)
  search/             FAISS vector search (port 8002)
  geometric/          ORB + RANSAC verification (port 8003)
  storage/            Binary object storage (port 8004)
tools/
  build_index.py      Index building script
  evaluate.py         Accuracy evaluation script
  run_evaluation.py   Full E2E evaluation pipeline
  downloader/         Rijksmuseum data downloader
```

### Service Layout

Each service in `services/` follows this layout:

```
services/<service_name>/
├── config.yaml                    # Service configuration (YAML)
├── justfile                       # Service-specific commands
├── pyproject.toml                 # Dependencies and tool config
├── pyrightconfig.json             # Strict type checking config
├── Dockerfile                     # Container definition
├── src/<service_name>/
│   ├── __init__.py
│   ├── config.py                  # Pydantic Settings (loads config.yaml)
│   ├── app.py                     # FastAPI application
│   ├── schemas.py                 # Pydantic request/response models
│   └── ...                        # Service-specific modules
└── tests/
    ├── __init__.py
    └── test_*.py
```

### Key design principles

- Services communicate via HTTP/JSON using `httpx` for async clients
- All endpoints follow the uniform API structure
- Health checks: `GET /health` returns `{"status": "healthy"}`
- Service info: `GET /info` returns configuration and version
- All services return errors in a consistent JSON format with `error`, `message`, and `details` fields

## Git Rules

- **Never use `git -C <path>`** to operate on other worktrees. Always use the full `git` command from the current working directory.

## Code Style

### General

- **Never assume any default values anywhere** — always be explicit about values, paths, and configurations
- If a value is not provided, handle it explicitly (raise error, use null, or prompt for input)
- **Never create Python files in the project root**

### Python Execution

- Python code must be executed **only** via `uv run ...`
  - Example: `uv run uvicorn embeddings_service.app:app --reload`
  - **Never** use: `python`, `python3`, or direct script execution
- Virtual environments are created via `uv sync`
- Each service has its **own** virtual environment in `services/<name>/.venv/`
- **Never** use: `pip install`, `python -m pip`, or `uv pip`
- All dependencies declared in service's `pyproject.toml`

### Configuration

- **Never hardcode configuration values** in application code
- **Never use default parameter values** for configurable settings
- **All config must come from config.yaml** or environment variables
- **Fail fast** — invalid configuration crashes at startup, not runtime
- **Type-safe access** — use `settings.section.key`, never `config["section"]["key"]`

```python
# WRONG - hardcoded default
def create_index(dimension: int = 768):
    ...

# WRONG - buried default
model_name = config.get("model_name", "facebook/dinov2-base")

# CORRECT - explicit from config
from embeddings_service.config import settings

def create_index(dimension: int):  # No default
    ...

create_index(dimension=settings.model.embedding_dimension)
```

### Error Handling

- Scripts should continue processing other items even if one fails
- Failed/invalid outputs should be handled gracefully
- Scripts should track and report success/failure counts
- Exit with code 1 if any items failed, 0 if all succeeded

### HTTP Status Codes

- `400` - Bad request (invalid input)
- `404` - Resource not found
- `422` - Validation error (valid JSON, invalid content)
- `500` - Internal server error
- `502` - Backend service error (Gateway only)
- `503` - Service unavailable
- `504` - Backend timeout (Gateway only)

## Files to never edit directly

- `data/index/` - generated FAISS index (output)
- `data/features/` - pre-computed features (output)
- `data/downloads/` - downloaded Rijksmuseum data (output)
- `data/models/` - cached model weights
- `reports/` - generated test artifacts

## Common workflows

### Working on a single service

1. Navigate to service: `cd services/<name>`
2. Initialize environment: `just init`
3. Run locally with hot reload: `just run`
4. Run tests: `just test`
5. Run all CI checks: `just ci`

### Adding a dependency

1. Edit `pyproject.toml` in the service directory to add the dependency
2. Run `uv sync --all-extras` from the service directory

### Fixing a bug

1. Write a failing test first
2. Verify it fails with `just test-all`
3. Implement the fix
4. Verify it passes with `just test-all`

## Delegating Work

See [DELEGATE.md](DELEGATE.md) for instructions on delegating work to the Codex agent via tmux.

## Ticket Management

Every feature request or bug fix must have a corresponding test ticket that blocks it. The test ticket describes how to write a failing test that confirms the feature is not yet implemented or the bug still exists. The implementation ticket depends on the test ticket — no implementation work begins until the failing test is written and verified.

### Workflow

1. Create a test ticket: "Write acceptance tests for: \<feature/bug summary\>"
2. Create the implementation ticket: "\<feature/bug summary\>"
3. Add a dependency: implementation ticket depends on test ticket (`bd dep add <impl> <test>`)
4. Write the failing test first, verify it fails
5. Close the test ticket
6. Implement the feature/fix, verify the test passes
7. Close the implementation ticket

### Rules

- **No implementation without a failing test** — every implementation ticket must be blocked by a test ticket
- **Tests must fail first** — a test ticket is only closed once the test exists and fails against current code
- **Test describes the "what", not the "how"** — test tickets describe observable behavior to assert, not implementation details
