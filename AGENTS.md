# Development Rules for artwork-matcher

This file provides guidance to AI agents and AI-assisted development tools when working with this project. This includes Claude Code, Cursor IDE, GitHub Copilot, Windsurf, and any other AI coding assistants.

## General Coding Principles
- **Never assume any default values anywhere**
- Always be explicit about values, paths, and configurations
- If a value is not provided, handle it explicitly (raise error, use null, or prompt for input)

## Git Commit Guidelines

**IMPORTANT:** When creating git commits in this repository:
- **NEVER include AI attribution in commit messages**
- **NEVER add "Generated with [AI tool name]" or similar phrases**
- **NEVER add "Co-Authored-By: [AI name]" or similar attribution**
- **NEVER run `git add -A` or `git add .` - always stage files explicitly**
- Keep commit messages professional and focused on the changes made
- Commit messages should describe what changed and why, without mentioning AI assistance
- **ALWAYS run `git push` after creating a commit to push changes to the remote repository**

## Testing
- After **every change** to the code, the tests must be executed
- Always verify the program runs correctly with `just run` after modifications

## Python Execution Rules

### Execution
- Python code must be executed **only** via `uv run ...`
  - Example: `uv run uvicorn embeddings_service.app:app --reload`
  - **Never** use: `python`, `python3`, or direct script execution

### Dependencies
- Virtual environments are created via `uv sync`
- Each service has its **own** virtual environment in `services/<name>/.venv/`
- **Never** use: `pip install`, `python -m pip`, or `uv pip`
- All dependencies declared in service's `pyproject.toml`

### Adding a Dependency
```bash
# 1. Edit pyproject.toml to add the dependency
# 2. Sync the environment
cd services/embeddings
uv sync --all-extras
```

## Justfile Rules

### Root Justfile Commands
- `just` or `just help` - Show all available commands
- `just init-all` - Initialize all service environments
- `just destroy-all` - Remove all virtual environments
- `just docker-up` - Start all services (Docker)
- `just docker-up-dev` - Start with hot reload
- `just docker-down` - Stop all services
- `just docker-logs [service]` - View logs
- `just docker-build` - Build all Docker images
- `just test-all` - Run all tests
- `just ci-all` - Run all CI checks (verbose)
- `just ci-all-quiet` - Run all CI checks (quiet)
- `just build-index` - Build FAISS index
- `just evaluate` - Run accuracy evaluation

### Per-Service Justfile Commands
Run from within `services/<name>/`:

- `just init` - Initialize service environment
- `just destroy` - Remove virtual environment
- `just run` - Run service locally with hot reload
- `just test` - Run unit tests
- `just ci` - Run all CI checks (verbose)
- `just ci-quiet` - Run all CI checks (quiet)
- `just code-format` - Auto-fix formatting
- `just code-style` - Check style (read-only)
- `just code-typecheck` - Run mypy
- `just code-lspchecks` - Run pyright (strict)
- `just code-security` - Run bandit
- `just code-deptry` - Check dependencies
- `just code-spell` - Check spelling
- `just code-semgrep` - Run custom rules
- `just code-audit` - Vulnerability scan

## Project Structure

This is a **microservices** project with four services. Each service is self-contained with its own dependencies, tests, and configuration.

### Root Level
- `justfile` - Root task runner (orchestrates all services)
- `docker-compose.yml` - Production deployment
- `docker-compose.dev.yml` - Development with hot reload
- **Never create Python files in the project root**

### Services Structure
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

### Data Directory
```
data/
├── objects/       # Museum reference images (input, read-only)
├── pictures/      # Visitor test photos (input, read-only)
├── index/         # Generated FAISS index (output)
├── features/      # Pre-computed features (output, optional)
└── labels.csv     # Ground truth for evaluation (input)
```

### Tools Directory
```
tools/
├── justfile           # Tools task runner
├── pyproject.toml     # Tools dependencies
├── build_index.py     # Index building script
└── evaluate.py        # Accuracy evaluation script
```

## Configuration Rules

### YAML + Pydantic Pattern
Every service uses YAML configuration files validated by Pydantic Settings:

1. **`config.yaml`** - Human-readable defaults with comments
2. **`config.py`** - Pydantic models that validate and load the YAML
3. **Environment overrides** - `SERVICE__SECTION__KEY=value` pattern

### Configuration Principles
- **Never hardcode configuration values** in application code
- **Never use default parameter values** for configurable settings
- **All config must come from config.yaml** or environment variables
- **Fail fast** - Invalid configuration crashes at startup, not runtime
- **Type-safe access** - Use `settings.section.key`, never `config["section"]["key"]`

### Example Pattern
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

# Called with config value
create_index(dimension=settings.model.embedding_dimension)
```

### Config File Location
- Default: `config.yaml` in service root
- Override via: `CONFIG_PATH` environment variable
- Docker: Mounted or baked into image

## Microservices Development

### Working on a Single Service
```bash
# Navigate to service
cd services/embeddings

# Initialize environment
just init

# Run locally (with hot reload)
just run

# Run tests
just test

# Run all CI checks
just ci
```

### Service Ports
| Service | Port | Purpose |
|---------|------|---------|
| Gateway | 8000 | Public API, orchestration |
| Embeddings | 8001 | DINOv2 embedding extraction |
| Search | 8002 | FAISS vector search |
| Geometric | 8003 | ORB + RANSAC verification |

### Adding Dependencies
```bash
# In the service directory
cd services/embeddings

# Add to pyproject.toml, then:
uv sync --all-extras
```

### Cross-Service Communication
- Services communicate via HTTP/JSON
- Use `httpx` for async HTTP clients
- All endpoints follow the uniform API structure (see API specs)
- Health checks: `GET /health` returns `{"status": "healthy"}`
- Service info: `GET /info` returns configuration and version

## Error Handling

### General Principles
- Scripts should continue processing other items even if one fails
- Failed/invalid outputs should be handled gracefully
- Scripts should track and report success/failure counts
- Exit with code 1 if any items failed, 0 if all succeeded

### Service Error Responses
All services return errors in a consistent JSON format:
```json
{
  "error": "error_code",
  "message": "Human-readable description",
  "details": {}
}
```

### HTTP Status Codes
- `400` - Bad request (invalid input)
- `404` - Resource not found
- `422` - Validation error (valid JSON, invalid content)
- `500` - Internal server error
- `502` - Backend service error (Gateway only)
- `503` - Service unavailable
- `504` - Backend timeout (Gateway only)

## Optimization

### Index Building (`build_index.py`)
- Check if index already exists before rebuilding
- Allow `--force` flag to rebuild anyway
- Report count of objects indexed

### Evaluation (`evaluate.py`)
- Cache embeddings if running multiple evaluations
- Report per-image timing for performance analysis

### General
- **Skip processing if output already exists** - Don't reprocess unnecessarily
- Check if output file exists before starting expensive operations
- Track skipped items separately in summary reports
- Allow users to force reprocessing via flags or by deleting output files
