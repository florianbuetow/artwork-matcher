# Project Scaffolding Setup Guide

This document provides complete instructions for setting up the folder structure, configuration files, and justfiles for the Artwork Matcher microservices project. **No application code is included** — only scaffolding and placeholder files.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Complete Folder Structure](#complete-folder-structure)
3. [Step 1: Clean Existing Structure](#step-1-clean-existing-structure)
4. [Step 2: Create Directory Structure](#step-2-create-directory-structure)
5. [Step 3: Root Level Files](#step-3-root-level-files)
6. [Step 4: Embeddings Service](#step-4-embeddings-service)
7. [Step 5: Search Service](#step-5-search-service)
8. [Step 6: Geometric Service](#step-6-geometric-service)
9. [Step 7: Gateway Service](#step-7-gateway-service)
10. [Step 8: Tools Directory](#step-8-tools-directory)
11. [Step 9: Data Directory](#step-9-data-directory)
12. [Step 10: Integration Tests](#step-10-integration-tests)
13. [Verification](#verification)

---

## Project Overview

This is a microservices-based museum artwork recognition system with four services:

| Service | Purpose | Port |
|---------|---------|------|
| **embeddings** | DINOv2 embedding extraction | 8001 |
| **search** | FAISS vector search | 8002 |
| **geometric** | ORB + RANSAC geometric verification | 8003 |
| **gateway** | API orchestration | 8000 |

---

## Complete Folder Structure

```
artwork-matcher/
├── AGENTS.md
├── CLAUDE.md -> AGENTS.md
├── README.md
├── justfile
├── docker-compose.yml
├── docker-compose.dev.yml
├── .gitignore
│
├── config/
│   ├── codespell/
│   │   └── ignore.txt
│   └── semgrep/
│       ├── no-default-values.yml
│       ├── no-noqa.yml
│       ├── no-sneaky-fallbacks.yml
│       ├── no_type_suppression.yml
│       └── python-constants.yml
│
├── services/
│   ├── embeddings/
│   │   ├── Dockerfile
│   │   ├── justfile
│   │   ├── pyproject.toml
│   │   ├── pyrightconfig.json
│   │   ├── config.yaml
│   │   ├── src/
│   │   │   └── embeddings_service/
│   │   │       └── __init__.py
│   │   └── tests/
│   │       ├── __init__.py
│   │       └── test_placeholder.py
│   │
│   ├── search/
│   │   ├── Dockerfile
│   │   ├── justfile
│   │   ├── pyproject.toml
│   │   ├── pyrightconfig.json
│   │   ├── config.yaml
│   │   ├── src/
│   │   │   └── search_service/
│   │   │       └── __init__.py
│   │   └── tests/
│   │       ├── __init__.py
│   │       └── test_placeholder.py
│   │
│   ├── geometric/
│   │   ├── Dockerfile
│   │   ├── justfile
│   │   ├── pyproject.toml
│   │   ├── pyrightconfig.json
│   │   ├── config.yaml
│   │   ├── src/
│   │   │   └── geometric_service/
│   │   │       └── __init__.py
│   │   └── tests/
│   │       ├── __init__.py
│   │       └── test_placeholder.py
│   │
│   └── gateway/
│       ├── Dockerfile
│       ├── justfile
│       ├── pyproject.toml
│       ├── pyrightconfig.json
│       ├── config.yaml
│       ├── src/
│       │   └── gateway/
│       │       └── __init__.py
│       └── tests/
│           ├── __init__.py
│           └── test_placeholder.py
│
├── tools/
│   ├── justfile
│   ├── pyproject.toml
│   ├── build_index.py
│   └── evaluate.py
│
├── data/
│   ├── .gitkeep
│   ├── objects/
│   │   └── .gitkeep
│   ├── pictures/
│   │   └── .gitkeep
│   ├── index/
│   │   └── .gitkeep
│   └── features/
│       └── .gitkeep
│
└── tests/
    ├── __init__.py
    ├── conftest.py
    └── test_integration.py
```

---

## Step 1: Clean Existing Structure

Remove these files and directories if they exist:

```bash
rm -rf src/
rm -f pyproject.toml
rm -f uv.lock
rm -f pyrightconfig.json
```

Keep these files:
- `AGENTS.md`
- `CLAUDE.md` (symlink)
- `README.md`
- `config/` directory and contents
- `reports/` directory

---

## Step 2: Create Directory Structure

```bash
# Services
mkdir -p services/embeddings/src/embeddings_service
mkdir -p services/embeddings/tests
mkdir -p services/search/src/search_service
mkdir -p services/search/tests
mkdir -p services/geometric/src/geometric_service
mkdir -p services/geometric/tests
mkdir -p services/gateway/src/gateway
mkdir -p services/gateway/tests

# Tools
mkdir -p tools

# Data
mkdir -p data/objects
mkdir -p data/pictures
mkdir -p data/index
mkdir -p data/features

# Integration tests
mkdir -p tests
```

---

## Step 3: Root Level Files

### File: `justfile`

```makefile
# Default recipe: show available commands
_default:
    @just --list

# Show help information
help:
    @echo ""
    @clear
    @echo ""
    @echo "\033[0;34m=== Artwork Matcher ===\033[0m"
    @echo ""
    @echo "Available commands:"
    @just --list
    @echo ""

# === Full Stack (Docker) ===

# Start all services with Docker Compose
up:
    @echo ""
    @echo "\033[0;34m=== Starting All Services ===\033[0m"
    docker compose up -d
    @echo "\033[0;32m✓ Services started\033[0m"
    @echo ""

# Start all services in development mode (with hot reload)
up-dev:
    @echo ""
    @echo "\033[0;34m=== Starting All Services (Dev Mode) ===\033[0m"
    docker compose -f docker-compose.yml -f docker-compose.dev.yml up
    @echo ""

# Stop all services
down:
    @echo ""
    @echo "\033[0;34m=== Stopping All Services ===\033[0m"
    docker compose down
    @echo "\033[0;32m✓ Services stopped\033[0m"
    @echo ""

# View logs (optionally for a specific service)
logs service="":
    docker compose logs -f {{service}}

# Build all Docker images
build:
    @echo ""
    @echo "\033[0;34m=== Building All Images ===\033[0m"
    docker compose build
    @echo "\033[0;32m✓ Build complete\033[0m"
    @echo ""

# === Individual Services (Local Development) ===

# Run embeddings service locally
run-embeddings:
    cd services/embeddings && just run

# Run search service locally
run-search:
    cd services/search && just run

# Run geometric service locally
run-geometric:
    cd services/geometric && just run

# Run gateway service locally
run-gateway:
    cd services/gateway && just run

# === Initialize All Services ===

# Initialize all service environments
init-all:
    @echo ""
    @echo "\033[0;34m=== Initializing All Services ===\033[0m"
    cd services/embeddings && just init
    cd services/search && just init
    cd services/geometric && just init
    cd services/gateway && just init
    cd tools && just init
    @echo "\033[0;32m✓ All services initialized\033[0m"
    @echo ""

# Destroy all virtual environments
destroy-all:
    @echo ""
    @echo "\033[0;34m=== Destroying All Virtual Environments ===\033[0m"
    cd services/embeddings && just destroy
    cd services/search && just destroy
    cd services/geometric && just destroy
    cd services/gateway && just destroy
    cd tools && just destroy
    @echo "\033[0;32m✓ All virtual environments removed\033[0m"
    @echo ""

# === Testing ===

# Run tests for all services
test-all:
    @echo ""
    @echo "\033[0;34m=== Running All Tests ===\033[0m"
    cd services/embeddings && just test
    cd services/search && just test
    cd services/geometric && just test
    cd services/gateway && just test
    @echo "\033[0;32m✓ All tests passed\033[0m"
    @echo ""

# Run tests for a specific service
test service:
    cd services/{{service}} && just test

# === CI ===

# Run CI checks for all services (verbose)
ci-all:
    #!/usr/bin/env bash
    set -e
    echo ""
    echo "\033[0;34m=== Running CI for All Services ===\033[0m"
    echo ""
    cd services/embeddings && just ci
    cd services/search && just ci
    cd services/geometric && just ci
    cd services/gateway && just ci
    echo ""
    echo "\033[0;32m✓ All CI checks passed\033[0m"
    echo ""

# Run CI checks for all services (quiet mode)
ci-all-quiet:
    #!/usr/bin/env bash
    set -e
    echo "\033[0;34m=== Running CI for All Services (Quiet Mode) ===\033[0m"
    
    echo "Checking embeddings..."
    cd services/embeddings && just ci-quiet
    
    echo "Checking search..."
    cd services/search && just ci-quiet
    
    echo "Checking geometric..."
    cd services/geometric && just ci-quiet
    
    echo "Checking gateway..."
    cd services/gateway && just ci-quiet
    
    echo ""
    echo "\033[0;32m✓ All CI checks passed\033[0m"
    echo ""

# Run CI for a specific service
ci service:
    cd services/{{service}} && just ci

# === Code Quality (Root Level) ===

# Format all services
format-all:
    cd services/embeddings && just code-format
    cd services/search && just code-format
    cd services/geometric && just code-format
    cd services/gateway && just code-format

# === Data Pipeline ===

# Build the FAISS index from object images
build-index:
    cd tools && just build-index

# Evaluate accuracy against labels.csv
evaluate:
    cd tools && just evaluate
```

### File: `docker-compose.yml`

```yaml
services:
  embeddings:
    build: ./services/embeddings
    ports:
      - "8001:8000"
    volumes:
      - ./data:/data:ro
    environment:
      - CONFIG_PATH=/app/config.yaml
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  search:
    build: ./services/search
    ports:
      - "8002:8000"
    volumes:
      - ./data:/data
    environment:
      - CONFIG_PATH=/app/config.yaml
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  geometric:
    build: ./services/geometric
    ports:
      - "8003:8000"
    volumes:
      - ./data:/data:ro
    environment:
      - CONFIG_PATH=/app/config.yaml
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  gateway:
    build: ./services/gateway
    ports:
      - "8000:8000"
    volumes:
      - ./data:/data:ro
    environment:
      - CONFIG_PATH=/app/config.yaml
      - EMBEDDINGS_URL=http://embeddings:8000
      - SEARCH_URL=http://search:8000
      - GEOMETRIC_URL=http://geometric:8000
    depends_on:
      embeddings:
        condition: service_healthy
      search:
        condition: service_healthy
      geometric:
        condition: service_healthy
```

### File: `docker-compose.dev.yml`

```yaml
services:
  embeddings:
    build:
      context: ./services/embeddings
    volumes:
      - ./services/embeddings/src:/app/src
      - ./data:/data:ro
    command: uv run uvicorn embeddings_service.app:app --reload --host 0.0.0.0

  search:
    build:
      context: ./services/search
    volumes:
      - ./services/search/src:/app/src
      - ./data:/data
    command: uv run uvicorn search_service.app:app --reload --host 0.0.0.0

  geometric:
    build:
      context: ./services/geometric
    volumes:
      - ./services/geometric/src:/app/src
      - ./data:/data:ro
    command: uv run uvicorn geometric_service.app:app --reload --host 0.0.0.0

  gateway:
    build:
      context: ./services/gateway
    volumes:
      - ./services/gateway/src:/app/src
      - ./data:/data:ro
    command: uv run uvicorn gateway.app:app --reload --host 0.0.0.0
```

### File: `.gitignore`

Append these lines to your existing `.gitignore`:

```gitignore
# === Microservices ===

# Per-service artifacts
services/*/reports/
services/*/.venv/
services/*/uv.lock
services/*/__pycache__/
services/*/.pytest_cache/
services/*/.mypy_cache/
services/*/.ruff_cache/

# Tools
tools/.venv/
tools/uv.lock
tools/__pycache__/

# Data (keep structure, ignore contents)
data/objects/*
data/pictures/*
data/index/*
data/features/*
data/*.csv
!data/.gitkeep
!data/**/.gitkeep
```

---

## Step 4: Embeddings Service

### File: `services/embeddings/justfile`

```makefile
# Default recipe: show available commands
_default:
    @just --list

# === Service Configuration ===
SERVICE_NAME := "embeddings_service"
PORT := "8001"

# Show help information
help:
    @echo ""
    @clear
    @echo ""
    @echo "\033[0;34m=== Embeddings Service ===\033[0m"
    @echo ""
    @echo "Available commands:"
    @just --list
    @echo ""

# === Environment Management ===

# Initialize the development environment
init:
    @echo ""
    @echo "\033[0;34m=== Initializing Development Environment ===\033[0m"
    @mkdir -p reports/coverage
    @mkdir -p reports/security
    @mkdir -p reports/pyright
    @mkdir -p reports/deptry
    @echo "Installing Python dependencies..."
    @uv sync --all-extras
    @echo "\033[0;32m✓ Development environment ready\033[0m"
    @echo ""

# Destroy the virtual environment
destroy:
    @echo ""
    @echo "\033[0;34m=== Destroying Virtual Environment ===\033[0m"
    @rm -rf .venv
    @rm -rf reports
    @echo "\033[0;32m✓ Virtual environment removed\033[0m"
    @echo ""

# === Run Service ===

# Run the service with hot reload
run:
    @echo ""
    @echo "\033[0;34m=== Running Embeddings Service on port 8001 ===\033[0m"
    @uv run uvicorn embeddings_service.app:app --reload --host 0.0.0.0 --port 8001

# === Code Quality ===

# Check code style and formatting (read-only)
code-style:
    @echo ""
    @echo "\033[0;34m=== Checking Code Style ===\033[0m"
    @uv run ruff check .
    @echo ""
    @uv run ruff format --check .
    @echo ""
    @echo "\033[0;32m✓ Style checks passed\033[0m"
    @echo ""

# Auto-fix code style and formatting
code-format:
    @echo ""
    @echo "\033[0;34m=== Formatting Code ===\033[0m"
    @uv run ruff check . --fix
    @echo ""
    @uv run ruff format .
    @echo ""
    @echo "\033[0;32m✓ Code formatted\033[0m"
    @echo ""

# Run static type checking with mypy
code-typecheck:
    @echo ""
    @echo "\033[0;34m=== Running Type Checks ===\033[0m"
    @uv run mypy src/
    @echo ""
    @echo "\033[0;32m✓ Type checks passed\033[0m"
    @echo ""

# Run strict type checking with Pyright (LSP-based)
code-lspchecks:
    @echo ""
    @echo "\033[0;34m=== Running Pyright Type Checks ===\033[0m"
    @mkdir -p reports/pyright
    @uv run pyright --project pyrightconfig.json > reports/pyright/pyright.txt 2>&1 || true
    @uv run pyright --project pyrightconfig.json
    @echo ""
    @echo "\033[0;32m✓ Pyright checks passed\033[0m"
    @echo "  Report: reports/pyright/pyright.txt"
    @echo ""

# Run security checks with bandit
code-security:
    @echo ""
    @echo "\033[0;34m=== Running Security Checks ===\033[0m"
    @mkdir -p reports/security
    @uv run bandit -c pyproject.toml -r src -f txt -o reports/security/bandit.txt || true
    @uv run bandit -c pyproject.toml -r src
    @echo ""
    @echo "\033[0;32m✓ Security checks passed\033[0m"
    @echo ""

# Check dependency hygiene with deptry
code-deptry:
    @echo ""
    @echo "\033[0;34m=== Checking Dependencies ===\033[0m"
    @mkdir -p reports/deptry
    @uv run deptry src
    @echo ""
    @echo "\033[0;32m✓ Dependency checks passed\033[0m"
    @echo ""

# Generate code statistics with pygount
code-stats:
    @echo ""
    @echo "\033[0;34m=== Code Statistics ===\033[0m"
    @mkdir -p reports
    @uv run pygount src/ tests/ --suffix=py,md,txt,toml,yaml,yml --format=summary
    @echo ""
    @uv run pygount src/ tests/ --suffix=py,md,txt,toml,yaml,yml --format=summary > reports/code-stats.txt
    @echo "\033[0;32m✓ Report saved to reports/code-stats.txt\033[0m"
    @echo ""

# Check spelling in code and documentation
code-spell:
    @echo ""
    @echo "\033[0;34m=== Checking Spelling ===\033[0m"
    @uv run codespell src tests *.md *.toml --ignore-words=../../config/codespell/ignore.txt
    @echo ""
    @echo "\033[0;32m✓ Spelling checks passed\033[0m"
    @echo ""

# Scan dependencies for known vulnerabilities
code-audit:
    @echo ""
    @echo "\033[0;34m=== Scanning Dependencies for Vulnerabilities ===\033[0m"
    @uv run pip-audit
    @echo ""
    @echo "\033[0;32m✓ No known vulnerabilities found\033[0m"
    @echo ""

# Run Semgrep static analysis (uses root config)
code-semgrep:
    @echo ""
    @echo "\033[0;34m=== Running Semgrep Static Analysis ===\033[0m"
    @uv run semgrep --config ../../config/semgrep/ --error src
    @echo ""
    @echo "\033[0;32m✓ Semgrep checks passed\033[0m"
    @echo ""

# === Testing ===

# Run unit tests only (fast)
test:
    @echo ""
    @echo "\033[0;34m=== Running Unit Tests ===\033[0m"
    @uv run pytest tests/ -v
    @echo ""

# Run unit tests with coverage report and threshold check
test-coverage: init
    @echo ""
    @echo "\033[0;34m=== Running Unit Tests with Coverage ===\033[0m"
    @uv run pytest tests/ -v \
        --cov=src \
        --cov-report=html:reports/coverage/html \
        --cov-report=term \
        --cov-report=xml:reports/coverage/coverage.xml \
        --cov-fail-under=80
    @echo ""
    @echo "\033[0;32m✓ Coverage threshold met\033[0m"
    @echo "  HTML: reports/coverage/html/index.html"
    @echo ""

# === CI Pipelines ===

# Run ALL validation checks (verbose)
ci:
    #!/usr/bin/env bash
    set -e
    echo ""
    echo "\033[0;34m=== Running CI Checks for Embeddings Service ===\033[0m"
    echo ""
    just init
    just code-format
    just code-style
    just code-typecheck
    just code-security
    just code-deptry
    just code-spell
    just code-semgrep
    just code-audit
    just test
    just code-lspchecks
    echo ""
    echo "\033[0;32m✓ All CI checks passed\033[0m"
    echo ""

# Run ALL validation checks silently (only show output on errors)
ci-quiet:
    #!/usr/bin/env bash
    set -e
    echo "\033[0;34m=== Running CI Checks (Quiet Mode) ===\033[0m"
    TMPFILE=$(mktemp)
    trap "rm -f $TMPFILE" EXIT

    just init > $TMPFILE 2>&1 || { echo "\033[0;31m✗ Init failed\033[0m"; cat $TMPFILE; exit 1; }
    echo "\033[0;32m✓ Init passed\033[0m"

    just code-format > $TMPFILE 2>&1 || { echo "\033[0;31m✗ Code-format failed\033[0m"; cat $TMPFILE; exit 1; }
    echo "\033[0;32m✓ Code-format passed\033[0m"

    just code-style > $TMPFILE 2>&1 || { echo "\033[0;31m✗ Code-style failed\033[0m"; cat $TMPFILE; exit 1; }
    echo "\033[0;32m✓ Code-style passed\033[0m"

    just code-typecheck > $TMPFILE 2>&1 || { echo "\033[0;31m✗ Code-typecheck failed\033[0m"; cat $TMPFILE; exit 1; }
    echo "\033[0;32m✓ Code-typecheck passed\033[0m"

    just code-security > $TMPFILE 2>&1 || { echo "\033[0;31m✗ Code-security failed\033[0m"; cat $TMPFILE; exit 1; }
    echo "\033[0;32m✓ Code-security passed\033[0m"

    just code-deptry > $TMPFILE 2>&1 || { echo "\033[0;31m✗ Code-deptry failed\033[0m"; cat $TMPFILE; exit 1; }
    echo "\033[0;32m✓ Code-deptry passed\033[0m"

    just code-spell > $TMPFILE 2>&1 || { echo "\033[0;31m✗ Code-spell failed\033[0m"; cat $TMPFILE; exit 1; }
    echo "\033[0;32m✓ Code-spell passed\033[0m"

    just code-semgrep > $TMPFILE 2>&1 || { echo "\033[0;31m✗ Code-semgrep failed\033[0m"; cat $TMPFILE; exit 1; }
    echo "\033[0;32m✓ Code-semgrep passed\033[0m"

    just code-audit > $TMPFILE 2>&1 || { echo "\033[0;31m✗ Code-audit failed\033[0m"; cat $TMPFILE; exit 1; }
    echo "\033[0;32m✓ Code-audit passed\033[0m"

    just test > $TMPFILE 2>&1 || { echo "\033[0;31m✗ Test failed\033[0m"; cat $TMPFILE; exit 1; }
    echo "\033[0;32m✓ Test passed\033[0m"

    just code-lspchecks > $TMPFILE 2>&1 || { echo "\033[0;31m✗ Code-lspchecks failed\033[0m"; cat $TMPFILE; exit 1; }
    echo "\033[0;32m✓ Code-lspchecks passed\033[0m"

    echo ""
    echo "\033[0;32m✓ All CI checks passed\033[0m"
    echo ""
```

### File: `services/embeddings/pyproject.toml`

```toml
[project]
name = "embeddings-service"
version = "0.1.0"
description = "DINOv2 embedding extraction service"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn>=0.34.0",
    "pillow>=11.0.0",
    "torch>=2.5.0",
    "transformers>=4.47.0",
    "pydantic>=2.10.0",
    "pydantic-settings>=2.7.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=6.0.0",
    "pytest-asyncio>=0.24.0",
    "httpx>=0.28.0",
    "ruff>=0.8.0",
    "mypy>=1.13.0",
    "pyright>=1.1.390",
    "bandit>=1.7.0",
    "deptry>=0.21.0",
    "codespell>=2.3.0",
    "semgrep>=1.99.0",
    "pip-audit>=2.7.0",
    "pygount>=1.8.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/embeddings_service"]

# === Ruff ===
[tool.ruff]
target-version = "py312"
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # Pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "UP",     # pyupgrade
    "ARG",    # flake8-unused-arguments
    "SIM",    # flake8-simplify
    "TCH",    # flake8-type-checking
    "PTH",    # flake8-use-pathlib
    "ERA",    # eradicate (commented out code)
    "PL",     # pylint
    "RUF",    # Ruff-specific rules
]
ignore = [
    "PLR0913",  # Too many arguments
    "PLR2004",  # Magic value comparison
]

[tool.ruff.lint.isort]
known-first-party = ["embeddings_service"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"

# === Mypy ===
[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_configs = true
show_error_codes = true
files = ["src"]

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

# === Bandit ===
[tool.bandit]
exclude_dirs = ["tests", ".venv"]
skips = ["B101"]

# === Pytest ===
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
asyncio_mode = "auto"
addopts = "-v --tb=short"
markers = [
    "unit: Unit tests",
    "integration: Integration tests (require running services)",
]

# === Coverage ===
[tool.coverage.run]
source = ["src"]
branch = true
omit = ["tests/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
    "if __name__ == .__main__.:",
]
fail_under = 80
show_missing = true

# === Deptry ===
[tool.deptry]
extend_exclude = [".venv", "tests"]
```

### File: `services/embeddings/pyrightconfig.json`

```json
{
  "include": ["src"],
  "exclude": ["**/__pycache__", ".venv"],
  "typeCheckingMode": "strict",
  "pythonVersion": "3.12",
  "reportMissingTypeStubs": false,
  "reportUnknownMemberType": false,
  "reportUnknownArgumentType": false,
  "reportUnknownVariableType": false
}
```

### File: `services/embeddings/config.yaml`

```json
{
    "model_name": "facebook/dinov2-base",
    "device": "auto",
    "embedding_dimension": 768,
    "image_size": 518,
    "host": "0.0.0.0",
    "port": 8000
}
```

### File: `services/embeddings/Dockerfile`

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install uv

COPY pyproject.toml .
RUN uv sync --frozen --no-dev

COPY config.yaml .
COPY src/ src/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "embeddings_service.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### File: `services/embeddings/src/embeddings_service/__init__.py`

```python
"""
Embeddings Service - DINOv2 embedding extraction.

TODO: Implement the following modules:
    - config.py: Configuration management
    - model.py: DINOv2 model wrapper
    - schemas.py: Pydantic request/response models
    - app.py: FastAPI application
"""

print("=" * 60)
print("Welcome to the Embeddings Service!")
print("This service is not yet implemented.")
print("TODO: Implement DINOv2 embedding extraction")
print("=" * 60)
```

### File: `services/embeddings/tests/__init__.py`

```python
"""Tests for embeddings service."""
```

### File: `services/embeddings/tests/test_placeholder.py`

```python
"""Placeholder tests for embeddings service."""


def test_placeholder() -> None:
    """Placeholder test to verify pytest works."""
    assert True, "Placeholder test should pass"
```

---

## Step 5: Search Service

### File: `services/search/justfile`

Copy the embeddings justfile and change:
- Line 5: `SERVICE_NAME := "search_service"`
- Line 6: `PORT := "8002"`
- Line 12: `@echo "\033[0;34m=== Search Service ===\033[0m"`
- Line 25: `@echo "\033[0;34m=== Running Search Service on port 8002 ===\033[0m"`
- Line 26: `@uv run uvicorn search_service.app:app --reload --host 0.0.0.0 --port 8002`
- CI section: `@echo "\033[0;34m=== Running CI Checks for Search Service ===\033[0m"`

### File: `services/search/pyproject.toml`

```toml
[project]
name = "search-service"
version = "0.1.0"
description = "FAISS vector search service"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn>=0.34.0",
    "faiss-cpu>=1.9.0",
    "numpy>=2.0.0",
    "pydantic>=2.10.0",
    "pydantic-settings>=2.7.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=6.0.0",
    "pytest-asyncio>=0.24.0",
    "httpx>=0.28.0",
    "ruff>=0.8.0",
    "mypy>=1.13.0",
    "pyright>=1.1.390",
    "bandit>=1.7.0",
    "deptry>=0.21.0",
    "codespell>=2.3.0",
    "semgrep>=1.99.0",
    "pip-audit>=2.7.0",
    "pygount>=1.8.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/search_service"]

# === Ruff ===
[tool.ruff]
target-version = "py312"
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E", "W", "F", "I", "B", "C4", "UP", "ARG", "SIM", "TCH", "PTH", "ERA", "PL", "RUF",
]
ignore = ["PLR0913", "PLR2004"]

[tool.ruff.lint.isort]
known-first-party = ["search_service"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"

# === Mypy ===
[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_configs = true
show_error_codes = true
files = ["src"]

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

# === Bandit ===
[tool.bandit]
exclude_dirs = ["tests", ".venv"]
skips = ["B101"]

# === Pytest ===
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
asyncio_mode = "auto"
addopts = "-v --tb=short"
markers = ["unit: Unit tests", "integration: Integration tests"]

# === Coverage ===
[tool.coverage.run]
source = ["src"]
branch = true
omit = ["tests/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover", "def __repr__", "raise NotImplementedError",
    "if TYPE_CHECKING:", "if __name__ == .__main__.:",
]
fail_under = 80
show_missing = true

# === Deptry ===
[tool.deptry]
extend_exclude = [".venv", "tests"]
```

### File: `services/search/pyrightconfig.json`

```json
{
  "include": ["src"],
  "exclude": ["**/__pycache__", ".venv"],
  "typeCheckingMode": "strict",
  "pythonVersion": "3.12",
  "reportMissingTypeStubs": false,
  "reportUnknownMemberType": false,
  "reportUnknownArgumentType": false,
  "reportUnknownVariableType": false
}
```

### File: `services/search/config.yaml`

```json
{
    "embedding_dimension": 768,
    "index_path": "/data/index/faiss.index",
    "metadata_path": "/data/index/metadata.json",
    "default_k": 5,
    "host": "0.0.0.0",
    "port": 8000
}
```

### File: `services/search/Dockerfile`

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install uv

COPY pyproject.toml .
RUN uv sync --frozen --no-dev

COPY config.yaml .
COPY src/ src/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "search_service.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### File: `services/search/src/search_service/__init__.py`

```python
"""
Search Service - FAISS vector search.

TODO: Implement the following modules:
    - config.py: Configuration management
    - index.py: FAISS index operations
    - schemas.py: Pydantic request/response models
    - app.py: FastAPI application
"""

print("=" * 60)
print("Welcome to the Search Service!")
print("This service is not yet implemented.")
print("TODO: Implement FAISS vector search")
print("=" * 60)
```

### File: `services/search/tests/__init__.py`

```python
"""Tests for search service."""
```

### File: `services/search/tests/test_placeholder.py`

```python
"""Placeholder tests for search service."""


def test_placeholder() -> None:
    """Placeholder test to verify pytest works."""
    assert True, "Placeholder test should pass"
```

---

## Step 6: Geometric Service

### File: `services/geometric/justfile`

Copy the embeddings justfile and change:
- Line 5: `SERVICE_NAME := "geometric_service"`
- Line 6: `PORT := "8003"`
- Line 12: `@echo "\033[0;34m=== Geometric Service ===\033[0m"`
- Line 25: `@echo "\033[0;34m=== Running Geometric Service on port 8003 ===\033[0m"`
- Line 26: `@uv run uvicorn geometric_service.app:app --reload --host 0.0.0.0 --port 8003`
- CI section: `@echo "\033[0;34m=== Running CI Checks for Geometric Service ===\033[0m"`

### File: `services/geometric/pyproject.toml`

```toml
[project]
name = "geometric-service"
version = "0.1.0"
description = "ORB feature extraction and geometric verification"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn>=0.34.0",
    "opencv-python-headless>=4.10.0",
    "numpy>=2.0.0",
    "pillow>=11.0.0",
    "pydantic>=2.10.0",
    "pydantic-settings>=2.7.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=6.0.0",
    "pytest-asyncio>=0.24.0",
    "httpx>=0.28.0",
    "ruff>=0.8.0",
    "mypy>=1.13.0",
    "pyright>=1.1.390",
    "bandit>=1.7.0",
    "deptry>=0.21.0",
    "codespell>=2.3.0",
    "semgrep>=1.99.0",
    "pip-audit>=2.7.0",
    "pygount>=1.8.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/geometric_service"]

# === Ruff ===
[tool.ruff]
target-version = "py312"
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E", "W", "F", "I", "B", "C4", "UP", "ARG", "SIM", "TCH", "PTH", "ERA", "PL", "RUF",
]
ignore = ["PLR0913", "PLR2004"]

[tool.ruff.lint.isort]
known-first-party = ["geometric_service"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"

# === Mypy ===
[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_configs = true
show_error_codes = true
files = ["src"]

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

# === Bandit ===
[tool.bandit]
exclude_dirs = ["tests", ".venv"]
skips = ["B101"]

# === Pytest ===
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
asyncio_mode = "auto"
addopts = "-v --tb=short"
markers = ["unit: Unit tests", "integration: Integration tests"]

# === Coverage ===
[tool.coverage.run]
source = ["src"]
branch = true
omit = ["tests/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover", "def __repr__", "raise NotImplementedError",
    "if TYPE_CHECKING:", "if __name__ == .__main__.:",
]
fail_under = 80
show_missing = true

# === Deptry ===
[tool.deptry]
extend_exclude = [".venv", "tests"]
```

### File: `services/geometric/pyrightconfig.json`

```json
{
  "include": ["src"],
  "exclude": ["**/__pycache__", ".venv"],
  "typeCheckingMode": "strict",
  "pythonVersion": "3.12",
  "reportMissingTypeStubs": false,
  "reportUnknownMemberType": false,
  "reportUnknownArgumentType": false,
  "reportUnknownVariableType": false
}
```

### File: `services/geometric/config.yaml`

```json
{
    "orb_features": 1000,
    "ransac_threshold": 5.0,
    "min_inliers": 10,
    "match_ratio": 0.75,
    "host": "0.0.0.0",
    "port": 8000
}
```

### File: `services/geometric/Dockerfile`

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install uv

COPY pyproject.toml .
RUN uv sync --frozen --no-dev

COPY config.yaml .
COPY src/ src/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "geometric_service.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### File: `services/geometric/src/geometric_service/__init__.py`

```python
"""
Geometric Service - ORB feature extraction and RANSAC verification.

TODO: Implement the following modules:
    - config.py: Configuration management
    - features.py: ORB feature extraction
    - matching.py: RANSAC geometric matching
    - schemas.py: Pydantic request/response models
    - app.py: FastAPI application
"""

print("=" * 60)
print("Welcome to the Geometric Service!")
print("This service is not yet implemented.")
print("TODO: Implement ORB + RANSAC geometric verification")
print("=" * 60)
```

### File: `services/geometric/tests/__init__.py`

```python
"""Tests for geometric service."""
```

### File: `services/geometric/tests/test_placeholder.py`

```python
"""Placeholder tests for geometric service."""


def test_placeholder() -> None:
    """Placeholder test to verify pytest works."""
    assert True, "Placeholder test should pass"
```

---

## Step 7: Gateway Service

### File: `services/gateway/justfile`

Copy the embeddings justfile and change:
- Line 5: `SERVICE_NAME := "gateway"`
- Line 6: `PORT := "8000"`
- Line 12: `@echo "\033[0;34m=== Gateway Service ===\033[0m"`
- Line 25: `@echo "\033[0;34m=== Running Gateway Service on port 8000 ===\033[0m"`
- Line 26: `@uv run uvicorn gateway.app:app --reload --host 0.0.0.0 --port 8000`
- CI section: `@echo "\033[0;34m=== Running CI Checks for Gateway Service ===\033[0m"`

### File: `services/gateway/pyproject.toml`

```toml
[project]
name = "gateway"
version = "0.1.0"
description = "API gateway for artwork matching"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn>=0.34.0",
    "httpx>=0.28.0",
    "pydantic>=2.10.0",
    "pydantic-settings>=2.7.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=6.0.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.8.0",
    "mypy>=1.13.0",
    "pyright>=1.1.390",
    "bandit>=1.7.0",
    "deptry>=0.21.0",
    "codespell>=2.3.0",
    "semgrep>=1.99.0",
    "pip-audit>=2.7.0",
    "pygount>=1.8.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/gateway"]

# === Ruff ===
[tool.ruff]
target-version = "py312"
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E", "W", "F", "I", "B", "C4", "UP", "ARG", "SIM", "TCH", "PTH", "ERA", "PL", "RUF",
]
ignore = ["PLR0913", "PLR2004"]

[tool.ruff.lint.isort]
known-first-party = ["gateway"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"

# === Mypy ===
[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_configs = true
show_error_codes = true
files = ["src"]

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

# === Bandit ===
[tool.bandit]
exclude_dirs = ["tests", ".venv"]
skips = ["B101"]

# === Pytest ===
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
asyncio_mode = "auto"
addopts = "-v --tb=short"
markers = ["unit: Unit tests", "integration: Integration tests"]

# === Coverage ===
[tool.coverage.run]
source = ["src"]
branch = true
omit = ["tests/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover", "def __repr__", "raise NotImplementedError",
    "if TYPE_CHECKING:", "if __name__ == .__main__.:",
]
fail_under = 80
show_missing = true

# === Deptry ===
[tool.deptry]
extend_exclude = [".venv", "tests"]
```

### File: `services/gateway/pyrightconfig.json`

```json
{
  "include": ["src"],
  "exclude": ["**/__pycache__", ".venv"],
  "typeCheckingMode": "strict",
  "pythonVersion": "3.12",
  "reportMissingTypeStubs": false,
  "reportUnknownMemberType": false,
  "reportUnknownArgumentType": false,
  "reportUnknownVariableType": false
}
```

### File: `services/gateway/config.yaml`

```json
{
    "embeddings_url": "http://localhost:8001",
    "search_url": "http://localhost:8002",
    "geometric_url": "http://localhost:8003",
    "search_k": 5,
    "similarity_threshold": 0.7,
    "geometric_verification": true,
    "host": "0.0.0.0",
    "port": 8000
}
```

### File: `services/gateway/Dockerfile`

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install uv

COPY pyproject.toml .
RUN uv sync --frozen --no-dev

COPY config.yaml .
COPY src/ src/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "gateway.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### File: `services/gateway/src/gateway/__init__.py`

```python
"""
Gateway - API orchestration service.

TODO: Implement the following modules:
    - config.py: Configuration management
    - clients.py: HTTP clients for internal services
    - pipeline.py: Identification pipeline orchestration
    - schemas.py: Pydantic request/response models
    - app.py: FastAPI application
"""

print("=" * 60)
print("Welcome to the Gateway Service!")
print("This service is not yet implemented.")
print("TODO: Implement API orchestration")
print("=" * 60)
```

### File: `services/gateway/tests/__init__.py`

```python
"""Tests for gateway service."""
```

### File: `services/gateway/tests/test_placeholder.py`

```python
"""Placeholder tests for gateway service."""


def test_placeholder() -> None:
    """Placeholder test to verify pytest works."""
    assert True, "Placeholder test should pass"
```

---

## Step 8: Tools Directory

### File: `tools/justfile`

```makefile
# Default recipe: show available commands
_default:
    @just --list

# Initialize the tools environment
init:
    @echo ""
    @echo "\033[0;34m=== Initializing Tools Environment ===\033[0m"
    @uv sync
    @echo "\033[0;32m✓ Tools environment ready\033[0m"
    @echo ""

# Destroy the virtual environment
destroy:
    @echo ""
    @echo "\033[0;34m=== Destroying Virtual Environment ===\033[0m"
    @rm -rf .venv
    @echo "\033[0;32m✓ Virtual environment removed\033[0m"
    @echo ""

# Build the FAISS index from object images
build-index:
    @echo ""
    @echo "\033[0;34m=== Building Index ===\033[0m"
    @uv run python build_index.py
    @echo ""

# Evaluate accuracy against labels.csv
evaluate:
    @echo ""
    @echo "\033[0;34m=== Evaluating Accuracy ===\033[0m"
    @uv run python evaluate.py
    @echo ""
```

### File: `tools/pyproject.toml`

```toml
[project]
name = "artwork-matcher-tools"
version = "0.1.0"
description = "CLI tools for artwork matcher"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.28.0",
    "pillow>=11.0.0",
    "pandas>=2.2.0",
    "rich>=13.9.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### File: `tools/build_index.py`

```python
"""
Build FAISS index from object images.

TODO: Implement index building:
    1. Load images from data/objects/
    2. Call embeddings service for each image
    3. Call search service /add for each embedding
    4. Report progress and final count
"""

print("=" * 60)
print("Build Index Tool")
print("=" * 60)
print()
print("TODO: Implement index building")
print()
print("This tool will:")
print("  1. Load images from data/objects/")
print("  2. Call embeddings service for each image")
print("  3. Call search service /add for each embedding")
print("  4. Report progress and final count")
print()
print("=" * 60)
```

### File: `tools/evaluate.py`

```python
"""
Evaluate accuracy against labels.csv.

TODO: Implement evaluation:
    1. Load labels.csv
    2. For each picture, call gateway /identify
    3. Compare top match to ground truth
    4. Report accuracy metrics (precision, recall, etc.)
"""

print("=" * 60)
print("Evaluate Tool")
print("=" * 60)
print()
print("TODO: Implement evaluation")
print()
print("This tool will:")
print("  1. Load labels.csv")
print("  2. For each picture, call gateway /identify")
print("  3. Compare top match to ground truth")
print("  4. Report accuracy metrics")
print()
print("=" * 60)
```

---

## Step 9: Data Directory

### File: `data/.gitkeep`

```
# This file ensures the data directory is tracked by git.
# Actual data files are ignored via .gitignore.
```

### File: `data/objects/.gitkeep`

```
# Place museum object reference images here.
```

### File: `data/pictures/.gitkeep`

```
# Place visitor photos here.
```

### File: `data/index/.gitkeep`

```
# FAISS index and metadata files will be stored here.
# Generated by: just build-index
```

### File: `data/features/.gitkeep`

```
# Pre-computed ORB features will be stored here (optional).
```

---

## Step 10: Integration Tests

### File: `tests/__init__.py`

```python
"""Integration tests for artwork matcher."""
```

### File: `tests/conftest.py`

```python
"""
Shared fixtures for integration tests.

TODO: Add fixtures for:
    - Service URLs
    - Test images
    - HTTP client
"""

import pytest


@pytest.fixture
def gateway_url() -> str:
    """Gateway URL for integration tests."""
    return "http://localhost:8000"


@pytest.fixture
def embeddings_url() -> str:
    """Embeddings service URL for integration tests."""
    return "http://localhost:8001"


@pytest.fixture
def search_url() -> str:
    """Search service URL for integration tests."""
    return "http://localhost:8002"


@pytest.fixture
def geometric_url() -> str:
    """Geometric service URL for integration tests."""
    return "http://localhost:8003"
```

### File: `tests/test_integration.py`

```python
"""
Integration tests - require all services running.

TODO: Implement integration tests:
    - Test full identification pipeline
    - Test service health endpoints
    - Test error handling
"""

import pytest


@pytest.mark.integration
def test_placeholder() -> None:
    """Placeholder integration test."""
    print()
    print("=" * 60)
    print("Integration Tests")
    print("=" * 60)
    print()
    print("TODO: Implement integration tests")
    print()
    print("Run services first with: just up")
    print()
    print("=" * 60)
    assert True
```

---

## Verification

After completing all steps, verify the setup:

### 1. Check directory structure

```bash
tree -L 4 --dirsfirst
```

### 2. Initialize a single service

```bash
cd services/embeddings
just init
just test
```

### 3. Initialize all services

```bash
# From repository root
just init-all
```

### 4. Run tests for all services

```bash
just test-all
```

### 5. Run CI for a single service

```bash
cd services/embeddings
just ci
```

---

## Summary Checklist

After setup, verify you have:

**Root level:**
- [ ] `justfile`
- [ ] `docker-compose.yml`
- [ ] `docker-compose.dev.yml`
- [ ] Updated `.gitignore`

**Each service (`embeddings`, `search`, `geometric`, `gateway`):**
- [ ] `justfile` (with correct SERVICE_NAME and PORT)
- [ ] `pyproject.toml` (with correct package name and isort config)
- [ ] `pyrightconfig.json`
- [ ] `config.yaml`
- [ ] `Dockerfile`
- [ ] `src/<service_name>/__init__.py`
- [ ] `tests/__init__.py`
- [ ] `tests/test_placeholder.py`

**Tools:**
- [ ] `tools/justfile`
- [ ] `tools/pyproject.toml`
- [ ] `tools/build_index.py`
- [ ] `tools/evaluate.py`

**Data:**
- [ ] `data/.gitkeep`
- [ ] `data/objects/.gitkeep`
- [ ] `data/pictures/.gitkeep`
- [ ] `data/index/.gitkeep`
- [ ] `data/features/.gitkeep`

**Integration tests:**
- [ ] `tests/__init__.py`
- [ ] `tests/conftest.py`
- [ ] `tests/test_integration.py`
