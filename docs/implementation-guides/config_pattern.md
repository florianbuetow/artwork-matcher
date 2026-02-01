# Configuration Pattern for Artwork Matcher Services

## Design Principles

1. **Fail fast** — Invalid config crashes on startup, not at runtime
2. **Type-safe** — IDE autocompletion, no stringly-typed access
3. **Environment override** — YAML defaults, env vars for deployment
4. **Single source of truth** — One Settings object, imported everywhere
5. **Explicit over implicit** — No hidden defaults scattered in code

---

## Structure Per Service

```
services/embeddings/
├── config.yaml              # Default configuration
├── src/embeddings_service/
│   ├── config.py            # Pydantic Settings model
│   └── app.py               # Uses config
```

---

## Example: Embeddings Service

### `config.yaml`

```yaml
# Embeddings Service Configuration
# All values can be overridden via environment variables
# e.g., EMBEDDINGS__MODEL_NAME=facebook/dinov2-large

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

### `src/embeddings_service/config.py`

```python
"""
Configuration management for Embeddings Service.

Loads from config.yaml with environment variable overrides.
All configuration is validated at startup - fail fast on invalid config.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    """DINOv2 model configuration."""

    name: str = Field(
        description="HuggingFace model identifier",
        examples=["facebook/dinov2-base", "facebook/dinov2-large"],
    )
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        default="auto",
        description="Compute device. 'auto' selects best available.",
    )
    embedding_dimension: int = Field(
        gt=0,
        description="Output embedding dimension. Must match model.",
    )

    @field_validator("name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Ensure model name looks valid."""
        if not v.startswith(("facebook/dinov2", "google/", "openai/")):
            # Allow but warn - could be a custom model
            pass
        return v


class PreprocessingConfig(BaseModel):
    """Image preprocessing configuration."""

    image_size: int = Field(
        gt=0,
        le=1024,
        description="Target image size (square). DINOv2 expects 518.",
    )
    normalize: bool = Field(
        default=True,
        description="Apply ImageNet normalization.",
    )


class ServerConfig(BaseModel):
    """HTTP server configuration."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: Literal["debug", "info", "warning", "error"] = Field(
        default="info"
    )


class Settings(BaseSettings):
    """
    Root configuration for Embeddings Service.

    Configuration is loaded from:
    1. config.yaml (defaults)
    2. Environment variables (overrides)

    Environment variables use double underscore for nesting:
        EMBEDDINGS__MODEL__NAME=facebook/dinov2-large
        EMBEDDINGS__SERVER__PORT=8001
    """

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDINGS__",
        env_nested_delimiter="__",
        extra="forbid",  # Fail on unknown keys
    )

    model: ModelConfig
    preprocessing: PreprocessingConfig
    server: ServerConfig


def load_yaml_config(config_path: Path) -> dict:
    """Load and parse YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Expected location: {config_path.absolute()}"
        )

    with config_path.open() as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Configuration file is empty: {config_path}")

    return config


@lru_cache
def get_settings() -> Settings:
    """
    Load and validate configuration.

    Cached to ensure single instance across application.
    Called once at startup - fails fast on invalid config.

    Returns:
        Validated Settings instance

    Raises:
        FileNotFoundError: Config file missing
        ValidationError: Invalid configuration values
    """
    # Determine config path
    # Priority: CONFIG_PATH env var > default location
    import os

    config_path_str = os.environ.get("CONFIG_PATH", "config.yaml")
    config_path = Path(config_path_str)

    # Load YAML defaults
    yaml_config = load_yaml_config(config_path)

    # Create Settings with YAML as defaults, env vars as overrides
    return Settings(**yaml_config)


# Module-level convenience for direct import
# Usage: from embeddings_service.config import settings
settings = get_settings()
```

### Usage in `app.py`

```python
"""FastAPI application for Embeddings Service."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from embeddings_service.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup/shutdown lifecycle."""
    # Config is already validated by import
    # Log effective configuration
    print(f"Starting Embeddings Service")
    print(f"  Model: {settings.model.name}")
    print(f"  Device: {settings.model.device}")
    print(f"  Embedding dim: {settings.model.embedding_dimension}")
    yield


app = FastAPI(
    title="Embeddings Service",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


# In endpoints, access config directly:
# settings.model.name
# settings.preprocessing.image_size
```

---

## Why This Pattern?

### 1. **Pydantic validates everything**

```python
# This fails at startup with a clear error:
# config.yaml has: embedding_dimension: "not_a_number"
#
# pydantic_core._pydantic_core.ValidationError: 1 validation error
# model.embedding_dimension
#   Input should be a valid integer [type=int_type]
```

### 2. **Environment overrides for deployment**

```bash
# In Docker/production, override via env:
docker run -e EMBEDDINGS__MODEL__DEVICE=cuda ...

# In docker-compose.yml:
environment:
  - EMBEDDINGS__MODEL__DEVICE=cuda
  - EMBEDDINGS__SERVER__LOG_LEVEL=warning
```

### 3. **Type-safe access**

```python
# IDE knows the types, provides autocompletion
settings.model.name          # str
settings.model.device        # Literal["auto", "cpu", "cuda", "mps"]
settings.preprocessing.image_size  # int

# Typos caught by IDE/mypy:
settings.modle.name  # Error: "Settings" has no attribute "modle"
```

### 4. **Fail fast, explicit errors**

```python
# Extra keys in YAML are rejected (extra="forbid"):
# config.yaml has: unknown_key: value
#
# ValidationError: Extra inputs are not permitted

# Missing required fields:
# config.yaml missing: model.name
#
# ValidationError: Field required [model.name]
```

---

## Per-Service Config Files

### Gateway `config.yaml`

```yaml
services:
  embeddings_url: "http://localhost:8001"
  search_url: "http://localhost:8002"
  geometric_url: "http://localhost:8003"

pipeline:
  search_k: 5
  similarity_threshold: 0.7
  geometric_verification: true
  timeout_seconds: 30.0

server:
  host: "0.0.0.0"
  port: 8000
  log_level: "info"
```

### Search `config.yaml`

```yaml
faiss:
  embedding_dimension: 768
  index_type: "flat"  # flat, ivf, hnsw
  index_path: "/data/index/faiss.index"
  metadata_path: "/data/index/metadata.json"

search:
  default_k: 5
  max_k: 100

server:
  host: "0.0.0.0"
  port: 8000
  log_level: "info"
```

### Geometric `config.yaml`

```yaml
orb:
  n_features: 1000
  scale_factor: 1.2
  n_levels: 8

matching:
  ratio_threshold: 0.75
  min_matches: 10

ransac:
  reproj_threshold: 5.0
  max_iters: 1000
  confidence: 0.99

server:
  host: "0.0.0.0"
  port: 8000
  log_level: "info"
```

---

## Shared Pattern: Base Config Module

To reduce duplication, create a shared pattern each service can adapt:

```python
# Each service's config.py follows same structure:
# 1. Define nested Pydantic models for each section
# 2. Define root Settings(BaseSettings) class
# 3. load_yaml_config() function
# 4. get_settings() with @lru_cache
# 5. Module-level `settings = get_settings()`
```

---

## Testing Configuration

```python
# tests/test_config.py

import pytest
from pydantic import ValidationError

from embeddings_service.config import Settings, load_yaml_config


def test_valid_config_loads(tmp_path):
    """Valid YAML produces valid Settings."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
model:
  name: facebook/dinov2-base
  device: auto
  embedding_dimension: 768
preprocessing:
  image_size: 518
  normalize: true
server:
  host: 0.0.0.0
  port: 8000
  log_level: info
""")
    
    yaml_config = load_yaml_config(config_file)
    settings = Settings(**yaml_config)
    
    assert settings.model.name == "facebook/dinov2-base"
    assert settings.model.embedding_dimension == 768


def test_invalid_embedding_dimension_rejected(tmp_path):
    """Non-integer embedding_dimension fails validation."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
model:
  name: facebook/dinov2-base
  device: auto
  embedding_dimension: "not_a_number"
preprocessing:
  image_size: 518
server:
  host: 0.0.0.0
  port: 8000
""")
    
    yaml_config = load_yaml_config(config_file)
    
    with pytest.raises(ValidationError) as exc_info:
        Settings(**yaml_config)
    
    assert "embedding_dimension" in str(exc_info.value)


def test_missing_required_field_rejected(tmp_path):
    """Missing required field fails validation."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
model:
  device: auto
  embedding_dimension: 768
preprocessing:
  image_size: 518
server:
  host: 0.0.0.0
  port: 8000
""")
    
    yaml_config = load_yaml_config(config_file)
    
    with pytest.raises(ValidationError) as exc_info:
        Settings(**yaml_config)
    
    assert "name" in str(exc_info.value)


def test_extra_keys_rejected(tmp_path):
    """Unknown configuration keys are rejected."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
model:
  name: facebook/dinov2-base
  device: auto
  embedding_dimension: 768
  unknown_key: should_fail
preprocessing:
  image_size: 518
server:
  host: 0.0.0.0
  port: 8000
""")
    
    yaml_config = load_yaml_config(config_file)
    
    with pytest.raises(ValidationError) as exc_info:
        Settings(**yaml_config)
    
    assert "extra" in str(exc_info.value).lower()
```

---

## Summary

| Aspect | Choice | Why |
|--------|--------|-----|
| Format | YAML | Human-readable, supports comments, standard for config |
| Validation | Pydantic | Already in stack, excellent errors, type-safe |
| Override | Env vars | Standard for Docker/K8s deployment |
| Loading | Eager at startup | Fail fast, clear errors |
| Access | Module singleton | `from service.config import settings` |

This pattern demonstrates:
- **Defensive programming** — validate early, fail clearly
- **Production-awareness** — env var overrides for deployment
- **Maintainability** — type-safe access, IDE support
- **Testability** — config loading is easily unit tested
