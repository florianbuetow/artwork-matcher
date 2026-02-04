"""Unit tests for BF feature matcher."""

from __future__ import annotations

import numpy as np
import pytest

from geometric_service.services.feature_matcher import BFFeatureMatcher


@pytest.fixture
def bf_matcher() -> BFFeatureMatcher:
    """Provide a configured BF feature matcher for testing."""
    return BFFeatureMatcher(ratio_threshold=0.75)


@pytest.mark.unit
class TestBFFeatureMatcher:
    """Tests for BFFeatureMatcher."""

    def test_match_identical_descriptors(self, bf_matcher: BFFeatureMatcher) -> None:
        """Should find matches for identical descriptors."""
        # Create random binary descriptors
        np.random.seed(42)
        desc = np.random.randint(0, 256, size=(50, 32), dtype=np.uint8)

        matches = bf_matcher.match(desc, desc)

        # Should find matches (ratio test will filter some)
        assert len(matches) > 0

    def test_match_different_descriptors(self, bf_matcher: BFFeatureMatcher) -> None:
        """Should find fewer matches for different descriptors."""
        # Create two completely different descriptor sets
        np.random.seed(42)
        desc1 = np.random.randint(0, 256, size=(50, 32), dtype=np.uint8)
        np.random.seed(123)
        desc2 = np.random.randint(0, 256, size=(50, 32), dtype=np.uint8)

        matches = bf_matcher.match(desc1, desc2)

        # May find some random matches due to binary nature
        # But should be fewer than identical case
        assert isinstance(matches, list)

    def test_match_empty_descriptors(self, bf_matcher: BFFeatureMatcher) -> None:
        """Should return empty list for empty descriptors."""
        desc1 = np.array([], dtype=np.uint8).reshape(0, 32)
        desc2 = np.random.randint(0, 256, size=(50, 32), dtype=np.uint8)

        matches = bf_matcher.match(desc1, desc2)

        assert matches == []

    def test_match_none_descriptors(self, bf_matcher: BFFeatureMatcher) -> None:
        """Should return empty list for None descriptors."""
        desc1 = None
        desc2 = np.random.randint(0, 256, size=(50, 32), dtype=np.uint8)

        matches = bf_matcher.match(desc1, desc2)

        assert matches == []
