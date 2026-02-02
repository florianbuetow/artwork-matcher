"""
Shared fixtures for unit tests.

All external dependencies (FAISS index, settings, app state) are mocked
to ensure tests run in isolation without I/O or network access.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


def create_random_embedding(dimension: int, normalized: bool) -> list[float]:
    """Create a random embedding vector for testing."""
    arr = np.random.randn(dimension).astype(np.float32)
    if normalized:
        arr = arr / np.linalg.norm(arr)
    return arr.tolist()


@pytest.fixture
def mock_settings() -> Iterator[MagicMock]:
    """
    Mock application settings.

    Provides a complete mock of the Settings object with all
    configuration values typically loaded from config.yaml.
    """
    with patch("search_service.config.get_settings") as mock:
        settings = MagicMock()

        # Service config
        settings.service.name = "search"
        settings.service.version = "0.1.0"

        # FAISS config
        settings.faiss.embedding_dimension = 768
        settings.faiss.index_type = "flat"
        settings.faiss.metric = "inner_product"

        # Index config
        settings.index.path = "/data/index/faiss.index"
        settings.index.metadata_path = "/data/index/metadata.json"
        settings.index.auto_load = True
        settings.index.allowed_path_base = None  # Defaults to parent of index.path

        # Search config
        settings.search.default_k = 5
        settings.search.max_k = 100
        settings.search.default_threshold = 0.0

        # Server config
        settings.server.host = "0.0.0.0"
        settings.server.port = 8002

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
    and a mock FAISS index reference.
    """
    with patch("search_service.core.state.get_app_state") as mock:
        state = MagicMock()
        state.start_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        state.uptime_seconds = 123.45
        state.uptime_formatted = "2m 3s"
        state.index_loaded = True
        state.index_count = 20
        state.index_load_error = None  # No load error by default

        # Mock FAISS index
        faiss_index = MagicMock()
        faiss_index.dimension = 768
        faiss_index.count = 20
        faiss_index.is_empty = False
        state.faiss_index = faiss_index

        mock.return_value = state
        yield state


@pytest.fixture
def mock_faiss_index() -> Iterator[MagicMock]:
    """
    Mock FAISS index with search functionality.

    Returns mock search results for any query.
    """
    with patch("search_service.services.faiss_index.FAISSIndex") as mock_class:
        faiss_index = MagicMock()
        faiss_index.dimension = 768
        faiss_index.count = 20
        faiss_index.is_empty = False

        # Mock search results
        from search_service.services.faiss_index import SearchResult

        faiss_index.search.return_value = [
            SearchResult(
                object_id="object_001",
                score=0.95,
                rank=1,
                metadata={"name": "Test Object 1"},
            ),
            SearchResult(
                object_id="object_002",
                score=0.87,
                rank=2,
                metadata={"name": "Test Object 2"},
            ),
        ]

        mock_class.return_value = faiss_index
        yield faiss_index


@pytest.fixture
def mock_logger() -> Iterator[MagicMock]:
    """Mock logger for unit tests."""
    with patch("search_service.logging.get_logger") as mock:
        logger = MagicMock()
        mock.return_value = logger
        yield logger


@pytest.fixture
def sample_embedding_768() -> list[float]:
    """Create a sample 768-dimensional embedding."""
    return create_random_embedding(768, normalized=True)
