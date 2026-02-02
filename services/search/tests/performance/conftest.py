"""
Shared fixtures for performance tests.

Performance tests use the real application with an actual FAISS index.
Embeddings are pre-generated to avoid contaminating latency measurements.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient

from search_service.app import create_app
from search_service.config import clear_settings_cache
from search_service.core.state import reset_app_state

from .generators import create_normalized_embedding
from .metrics import PerformanceReport

if TYPE_CHECKING:
    from collections.abc import Iterator


# Service root directory (services/search/)
SERVICE_ROOT = Path(__file__).parent.parent.parent

# Test configuration constants
ITERATIONS_PER_SCENARIO = 30
INDEX_SIZES = [100, 500, 1000, 5000, 10000, 50000]
K_VALUES = [1, 5, 10, 25, 50, 100]
THROUGHPUT_REQUESTS = 250
CONCURRENCY_LEVELS = [2, 4, 8, 16]

# Embedding dimension (must match config.yaml faiss.embedding_dimension)
EMBEDDING_DIMENSION = 768

# Report output path
REPORTS_DIR = SERVICE_ROOT / "reports" / "performance"
REPORT_PATH = REPORTS_DIR / "search_service_performance.md"


@pytest.fixture(scope="session")
def performance_report() -> Iterator[PerformanceReport]:
    """
    Session-scoped fixture that collects test results and writes report.

    The report is written to reports/performance/search_service_performance.md
    when all tests complete.
    """
    report = PerformanceReport()
    yield report

    # Write report after all tests complete
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report.generate_markdown())
    print(f"\n\nPerformance report written to: {REPORT_PATH}")


@pytest.fixture(scope="module")
def performance_client() -> Iterator[TestClient]:
    """
    Create test client with the real application.

    Uses context manager to trigger lifespan events (index creation).
    This client is shared across all tests in the module to avoid
    recreating the app for each test.
    """
    clear_settings_cache()
    reset_app_state()
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    # Clean up after all tests in this module
    clear_settings_cache()
    reset_app_state()


@pytest.fixture
def client(performance_client: TestClient) -> TestClient:
    """Alias for performance_client for test compatibility."""
    return performance_client


@pytest.fixture(scope="module")
def pregenerated_data() -> dict[str, Any]:
    """
    Pre-generate all embeddings before tests run.

    This ensures embedding generation time is NOT included in latency
    measurements. All tests use identical pre-generated embeddings.

    Returns:
        Dictionary with:
        - "embeddings": List of 50,000 normalized embeddings
        - "query": Single query embedding for consistent comparison
    """
    max_index_size = max(INDEX_SIZES)

    print(f"\nPre-generating {max_index_size} embeddings...")
    embeddings = []
    for i in range(max_index_size):
        emb = create_normalized_embedding(EMBEDDING_DIMENSION, seed=i)
        embeddings.append(emb)
        if (i + 1) % 10000 == 0:
            print(f"  Generated {i + 1}/{max_index_size}")

    # Query embedding uses different seed space to ensure it's not in the index
    query = create_normalized_embedding(EMBEDDING_DIMENSION, seed=999999)

    print(f"Total pre-generated: {len(embeddings)} embeddings + 1 query")

    return {
        "embeddings": embeddings,
        "query": query,
    }
