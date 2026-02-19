"""Unit tests for FileBlobStore."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from storage_service.services.blob_store import FileBlobStore

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.unit
class TestFileBlobStore:
    """Tests for FileBlobStore operations."""

    def test_put_and_get(self, tmp_path: Path) -> None:
        """Store and retrieve an object."""
        store = FileBlobStore(root_dir=tmp_path / "store")
        store.put("key1", b"hello world")

        result = store.get("key1")

        assert result == b"hello world"

    def test_get_missing_returns_none(self, tmp_path: Path) -> None:
        """Getting a non-existent key returns None."""
        store = FileBlobStore(root_dir=tmp_path / "store")

        result = store.get("nonexistent")

        assert result is None

    def test_put_overwrites_existing(self, tmp_path: Path) -> None:
        """Putting to an existing key overwrites the data."""
        store = FileBlobStore(root_dir=tmp_path / "store")
        store.put("key1", b"original")
        store.put("key1", b"updated")

        result = store.get("key1")

        assert result == b"updated"

    def test_delete_existing_returns_true(self, tmp_path: Path) -> None:
        """Deleting an existing key returns True."""
        store = FileBlobStore(root_dir=tmp_path / "store")
        store.put("key1", b"data")

        result = store.delete("key1")

        assert result is True
        assert store.get("key1") is None

    def test_delete_missing_returns_false(self, tmp_path: Path) -> None:
        """Deleting a non-existent key returns False."""
        store = FileBlobStore(root_dir=tmp_path / "store")

        result = store.delete("nonexistent")

        assert result is False

    def test_delete_all_clears_store(self, tmp_path: Path) -> None:
        """delete_all removes all objects and returns count."""
        store = FileBlobStore(root_dir=tmp_path / "store")
        store.put("key1", b"data1")
        store.put("key2", b"data2")
        store.put("key3", b"data3")

        count = store.delete_all()

        assert count == 3
        assert store.count() == 0

    def test_delete_all_empty_returns_zero(self, tmp_path: Path) -> None:
        """delete_all on empty store returns 0."""
        store = FileBlobStore(root_dir=tmp_path / "store")

        count = store.delete_all()

        assert count == 0

    def test_count(self, tmp_path: Path) -> None:
        """count returns number of stored objects."""
        store = FileBlobStore(root_dir=tmp_path / "store")

        assert store.count() == 0

        store.put("key1", b"data1")
        assert store.count() == 1

        store.put("key2", b"data2")
        assert store.count() == 2

    def test_stores_as_dat_files(self, tmp_path: Path) -> None:
        """Objects are stored as {key}.dat files."""
        store_dir = tmp_path / "store"
        store = FileBlobStore(root_dir=store_dir)
        store.put("my-object", b"content")

        assert (store_dir / "my-object.dat").exists()
        assert (store_dir / "my-object.dat").read_bytes() == b"content"

    def test_creates_directory_on_init(self, tmp_path: Path) -> None:
        """Store directory is created on initialization."""
        store_dir = tmp_path / "new" / "nested" / "store"
        FileBlobStore(root_dir=store_dir)

        assert store_dir.exists()

    def test_stores_binary_data(self, tmp_path: Path) -> None:
        """Handles arbitrary binary data."""
        store = FileBlobStore(root_dir=tmp_path / "store")
        binary_data = bytes(range(256))
        store.put("binary", binary_data)

        result = store.get("binary")

        assert result == binary_data
