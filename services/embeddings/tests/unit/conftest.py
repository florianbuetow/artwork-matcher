"""
Shared fixtures for unit tests.

All external dependencies (model, settings, app state) are mocked
to ensure tests run in isolation without I/O or network access.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from tests.factories import create_random_embedding, create_test_image_base64

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def mock_settings() -> Iterator[MagicMock]:
    """
    Mock application settings.

    Provides a complete mock of the Settings object with all
    configuration values typically loaded from config.yaml.
    """
    with patch("embeddings_service.config.get_settings") as mock:
        settings = MagicMock()

        # Service config
        settings.service.name = "embeddings"
        settings.service.version = "0.1.0"

        # Model config
        settings.model.name = "facebook/dinov2-base"
        settings.model.device = "cpu"
        settings.model.embedding_dimension = 768
        settings.model.cache_dir = ".cache/huggingface"
        settings.model.revision = "main"

        # Preprocessing config
        settings.preprocessing.image_size = 518
        settings.preprocessing.normalize = True

        # Server config
        settings.server.host = "0.0.0.0"
        settings.server.port = 8001

        # Logging config
        settings.logging.level = "INFO"
        settings.logging.format = "json"

        mock.return_value = settings
        yield settings


@pytest.fixture
def mock_app_state() -> Iterator[MagicMock]:
    """
    Mock application state.

    Provides a mock AppState with predictable uptime values
    and mock model/processor references.
    """
    with patch("embeddings_service.core.state.get_app_state") as mock:
        state = MagicMock()
        state.start_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        state.uptime_seconds = 123.45
        state.uptime_formatted = "2m 3s"
        state.device = torch.device("cpu")

        # Model and processor are available
        state.model = MagicMock()
        state.processor = MagicMock()

        mock.return_value = state
        yield state


@pytest.fixture
def mock_model() -> Iterator[MagicMock]:
    """
    Mock DINOv2 model with fake embedding output.

    Returns a normalized 768-dimensional embedding for any input.
    """
    with patch("embeddings_service.routers.embed.extract_dino_embedding") as mock:
        # Return a normalized random embedding
        embedding = np.array(create_random_embedding(768, normalized=True), dtype=np.float32)
        mock.return_value = embedding
        yield mock


@pytest.fixture
def mock_logger() -> Iterator[MagicMock]:
    """Mock logger for unit tests."""
    with patch("embeddings_service.logging.get_logger") as mock:
        logger = MagicMock()
        mock.return_value = logger
        yield logger


@pytest.fixture
def sample_image_base64() -> str:
    """Create a base64-encoded sample JPEG image."""
    return create_test_image_base64(100, 100, "red", "JPEG")


@pytest.fixture
def sample_png_base64() -> str:
    """Create a base64-encoded sample PNG image."""
    return create_test_image_base64(100, 100, "blue", "PNG")


@pytest.fixture
def sample_webp_base64() -> str:
    """Create a base64-encoded sample WebP image."""
    return create_test_image_base64(100, 100, "green", "WEBP")
