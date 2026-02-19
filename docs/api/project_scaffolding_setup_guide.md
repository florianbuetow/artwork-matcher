# Project Scaffolding Setup Guide

This guide describes the **current** project scaffolding and setup flow for Artwork Matcher.

It reflects the repository as implemented today, including all five services:
- `gateway` (includes storage integration)
- `embeddings`
- `search`
- `geometric`
- `storage`

No placeholder application modules or placeholder tests are part of the expected scaffold.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Current Repository Layout](#current-repository-layout)
3. [Bootstrap From a Fresh Checkout](#bootstrap-from-a-fresh-checkout)
4. [Per-Service Scaffold Requirements](#per-service-scaffold-requirements)
5. [Root-Level Requirements](#root-level-requirements)
6. [Validation Checklist](#validation-checklist)

---

## Architecture Overview

| Service | Purpose | Port |
|---------|---------|------|
| `gateway` | Public API orchestration, including object/image access via storage integration | `8000` |
| `embeddings` | DINOv2 embedding extraction | `8001` |
| `search` | FAISS vector similarity search | `8002` |
| `geometric` | ORB + RANSAC geometric verification | `8003` |
| `storage` | Binary object storage and retrieval | `8004` |

---

## Current Repository Layout

```text
artwork-matcher/
├── AGENTS.md
├── CLAUDE.md
├── README.md
├── justfile
├── docker-compose.yml
├── docker-compose.dev.yml
├── docs/
├── services/
│   ├── gateway/
│   ├── embeddings/
│   ├── search/
│   ├── geometric/
│   └── storage/
├── tools/
├── tests/
├── data/
└── reports/
```

### Service Layout Pattern

Each service follows the same structural pattern:

```text
services/<service>/
├── config.yaml
├── justfile
├── pyproject.toml
├── pyrightconfig.json
├── Dockerfile
├── src/<package_name>/
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   ├── schemas.py
│   └── ... service-specific modules
└── tests/
    ├── __init__.py
    ├── unit/
    ├── integration/
    └── performance/
```

---

## Bootstrap From a Fresh Checkout

### 1. Initialize environments

```bash
just init-all
```

### 2. Validate quality gates

```bash
just ci-all-quiet
```

### 3. Run complete test suite

```bash
just test-all
```

### 4. Optional: start services locally

```bash
just start-all
just status
just stop-all
```

### 5. Optional: start services with Docker

```bash
just docker-up
just docker-logs
just docker-down
```

---

## Per-Service Scaffold Requirements

Each service should include:

- `justfile` with `init`, `run`, `test`, `ci`, `ci-quiet`
- Strict typing and lint configuration in `pyproject.toml` and `pyrightconfig.json`
- `config.yaml` with explicit required values (no hidden defaults in code)
- App factory in `src/<package>/app.py`
- Centralized settings in `src/<package>/config.py`
- Unit, integration, and performance tests in `tests/`
- Dockerfile for container execution

### Service Package Names

| Service dir | Package |
|-------------|---------|
| `services/gateway` | `gateway` |
| `services/embeddings` | `embeddings_service` |
| `services/search` | `search_service` |
| `services/geometric` | `geometric_service` |
| `services/storage` | `storage_service` |

---

## Root-Level Requirements

Root `justfile` should provide orchestration commands for:

- lifecycle: `start-all`, `stop-all`, `status`
- quality: `test-all`, `ci-all`, `ci-all-quiet`
- performance: `test-perf-all` and per-service perf commands
- data/evaluation: `download-batch`, `build-index`, `evaluate`, `docker-evaluate`

Core integration files:

- `docker-compose.yml` and `docker-compose.dev.yml`
- `tools/` scripts for index building and evaluation
- `docs/api/` specifications matching implemented behavior

---

## Validation Checklist

Use this checklist after scaffold changes:

- [ ] `just init-all` succeeds
- [ ] `just ci-all-quiet` succeeds
- [ ] `just test-all` succeeds
- [ ] `just status` reports expected healthy services when running
- [ ] Each service exposes `/health` and `/info`
- [ ] Gateway `/info` and `/health` reflect storage integration state
- [ ] Performance tests exist under each service's `tests/performance/`

---

## Notes

- This guide is intentionally current-state and implementation-aligned.
- For historical planning artifacts, use files under `docs/plans/` and `docs/devlog/`.
- For endpoint contracts, use `docs/api/*_service_api_spec.md` and `docs/api/uniform_api_structure.md`.
