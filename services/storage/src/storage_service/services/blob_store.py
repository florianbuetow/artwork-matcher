"""
File-backed binary object storage.

Stores objects as files on the filesystem. Each object is stored as
{key}.dat in a configurable root directory.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class FileBlobStore:
    """
    File-backed binary object store.

    Objects are stored as {key}.dat files in the root directory.
    """

    def __init__(self, root_dir: Path) -> None:
        self.root = root_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, key: str, data: bytes) -> None:
        """Store an object."""
        (self.root / f"{key}.dat").write_bytes(data)

    def get(self, key: str) -> bytes | None:
        """Retrieve an object. Returns None if not found."""
        path = self.root / f"{key}.dat"
        return path.read_bytes() if path.exists() else None

    def delete(self, key: str) -> bool:
        """Delete a single object. Returns True if it existed."""
        path = self.root / f"{key}.dat"
        if path.exists():
            path.unlink()
            return True
        return False

    def delete_all(self) -> int:
        """Delete all objects. Returns the count of deleted objects."""
        count = self.count()
        shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        return count

    def count(self) -> int:
        """Return the number of stored objects."""
        return sum(1 for _ in self.root.glob("*.dat"))
