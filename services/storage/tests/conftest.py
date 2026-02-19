"""
Shared test configuration and fixtures.

This file contains pytest configuration that applies to all tests.
Test-type-specific fixtures are defined in their respective conftest.py files.
"""

from __future__ import annotations


# The root conftest.py is intentionally minimal.
# Test-specific fixtures are defined in:
# - tests/unit/conftest.py for unit tests (with mocks)
# - tests/integration/conftest.py for integration tests (real app)
