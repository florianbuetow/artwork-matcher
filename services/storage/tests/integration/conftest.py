"""
Fixtures for integration tests.

These tests use the real application with actual file storage.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from storage_service.app import create_app
from storage_service.config import clear_settings_cache
from storage_service.core.state import reset_app_state

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(scope="module")
def integration_client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    """
    Create test client with the real application.

    Uses a temporary directory for storage to avoid polluting real data.
    """
    tmp_dir = tmp_path_factory.mktemp("storage_integration")
    storage_dir = tmp_dir / "objects"
    config_file = tmp_dir / "config.yaml"
    config_file.write_text(f"""
service:
  name: storage
  version: 0.1.0

storage:
  path: "{storage_dir}"
  content_type: "application/octet-stream"

server:
  host: "0.0.0.0"
  port: 8004
  log_level: "info"

logging:
  level: "INFO"
  format: "json"
""")
    os.environ["CONFIG_PATH"] = str(config_file)

    clear_settings_cache()
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client

    clear_settings_cache()
    reset_app_state()
    os.environ.pop("CONFIG_PATH", None)


@pytest.fixture
def client(integration_client: TestClient) -> Iterator[TestClient]:
    """
    Per-test client that clears storage before each test.
    """
    integration_client.delete("/objects")
    yield integration_client
