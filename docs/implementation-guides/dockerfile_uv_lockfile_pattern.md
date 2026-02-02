# Dockerfile uv Lockfile Pattern

## Overview

When using `uv` as the Python package manager in Docker builds, you **must** copy both `pyproject.toml` and `uv.lock` before running `uv sync --frozen`.

## The Problem

If you only copy `pyproject.toml` without `uv.lock`, the build will fail:

```dockerfile
# BROKEN - Missing uv.lock
COPY pyproject.toml .
RUN uv sync --frozen --no-dev
```

Error:
```
error: Unable to find lockfile at `uv.lock`, but `--frozen` was provided.
To create a lockfile, run `uv lock` or `uv sync` without the flag.
```

## The Solution

Always copy both files together:

```dockerfile
# CORRECT - Copy both pyproject.toml and uv.lock
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
```

## Complete Dockerfile Pattern

Here's the complete pattern for a service using `uv`:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install uv

# Copy dependency files FIRST (both are required for --frozen)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy configuration and source code
COPY config.yaml .
COPY src/ src/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Why This Matters

1. **`--frozen` flag** - Ensures reproducible builds by using exact versions from lockfile
2. **Docker layer caching** - Dependencies are only reinstalled when `pyproject.toml` or `uv.lock` change
3. **Build reliability** - Without the lockfile, builds will fail immediately

## Checklist for New Services

When creating a new service Dockerfile:

- [ ] Copy `pyproject.toml` AND `uv.lock` together
- [ ] Use `uv sync --frozen --no-dev` for production builds
- [ ] Ensure `uv.lock` exists locally (run `uv sync` first)
- [ ] Add `uv.lock` to version control (do NOT gitignore it)

## Services That Need This Pattern

All services in this project use `uv` and require this pattern:

| Service | Port | Dockerfile Status |
|---------|------|-------------------|
| Gateway | 8000 | Fixed |
| Embeddings | 8001 | Correct |
| Search | 8002 | Correct |
| Geometric | 8003 | **Needs fix** |

## Related Files

- Each service's `Dockerfile` in `services/<name>/Dockerfile`
- Each service's `uv.lock` in `services/<name>/uv.lock`
- Root `docker-compose.yml` that builds all services
