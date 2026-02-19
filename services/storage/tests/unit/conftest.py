"""
Shared fixtures for unit tests.

Provides test isolation fixtures to ensure clean state between tests.
"""

from __future__ import annotations

import pytest

from storage_service.config import clear_settings_cache


@pytest.fixture(autouse=True)
def reset_settings_cache() -> None:
    """
    Ensure settings cache is cleared before and after each test.

    This prevents test pollution where one test's configuration
    affects another test's behavior.
    """
    clear_settings_cache()
    yield
    clear_settings_cache()
