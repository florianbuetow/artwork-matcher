"""Pytest fixtures for router unit tests."""

from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True)
def _clear_module_cache() -> None:
    """Clear cached geometric_service modules before each test.

    This ensures that mocking works correctly when importing modules
    inside the test's patch context.
    """
    modules_to_remove = [key for key in sys.modules if key.startswith("geometric_service")]
    for module in modules_to_remove:
        del sys.modules[module]
