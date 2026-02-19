"""
Test data generators for storage service performance testing.

Provides functions to generate random binary blobs of controlled sizes.
"""

from __future__ import annotations

import os


def create_random_blob(size_bytes: int) -> bytes:
    """
    Generate a random binary blob of the specified size.

    Uses os.urandom for incompressible random data,
    giving worst-case file sizes on disk.

    Args:
        size_bytes: Exact size of the blob in bytes

    Returns:
        Random bytes of the specified length
    """
    return os.urandom(size_bytes)


def format_size(size_bytes: int) -> str:
    """
    Format byte count as human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string like "1 KB", "512 KB", "1 MB"
    """
    if size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.0f} MB"
    return f"{size_bytes / 1024:.0f} KB"
