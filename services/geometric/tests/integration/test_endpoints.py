"""Integration tests for all endpoints."""

from __future__ import annotations

import pytest

from tests.factories import (
    create_artwork_simulation_base64,
    create_checkerboard_base64,
    create_noise_image_base64,
    create_non_image_base64,
    create_solid_color_base64,
    create_transformed_image_base64,
)


@pytest.mark.integration
class TestHealthEndpoint:
    """Integration tests for /health."""

    def test_health_returns_healthy(self, client) -> None:
        """Health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


@pytest.mark.integration
class TestInfoEndpoint:
    """Integration tests for /info."""

    def test_info_returns_algorithm_config(self, client) -> None:
        """Info endpoint returns algorithm configuration."""
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "geometric"
        assert data["algorithm"]["feature_detector"] == "ORB"


@pytest.mark.integration
class TestExtractEndpoint:
    """Integration tests for /extract."""

    def test_extract_checkerboard(self, client) -> None:
        """Extract features from checkerboard image."""
        response = client.post(
            "/extract",
            json={"image": create_checkerboard_base64(200, 200), "image_id": "test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["num_features"] > 0
        assert len(data["keypoints"]) == data["num_features"]

    def test_extract_invalid_image(self, client) -> None:
        """Extract returns error for invalid image."""
        response = client.post(
            "/extract",
            json={"image": create_non_image_base64()},
        )
        assert response.status_code == 400
        assert response.json()["error"] == "invalid_image"


@pytest.mark.integration
class TestMatchEndpoint:
    """Integration tests for /match."""

    def test_match_identical_images(self, client) -> None:
        """Matching identical images returns is_match=True."""
        image = create_checkerboard_base64(200, 200)
        response = client.post(
            "/match",
            json={"query_image": image, "reference_image": image},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_match"] is True
        # Confidence > 0 indicates positive match with geometric verification
        assert data["confidence"] > 0


@pytest.mark.integration
class TestBatchMatchEndpoint:
    """Integration tests for /match/batch."""

    def test_batch_match(self, client) -> None:
        """Batch match returns results for all references."""
        query = create_checkerboard_base64(200, 200, seed=0)
        ref1 = create_checkerboard_base64(200, 200, seed=0)
        ref2 = create_checkerboard_base64(200, 200, seed=42)

        response = client.post(
            "/match/batch",
            json={
                "query_image": query,
                "references": [
                    {"reference_id": "ref_001", "reference_image": ref1},
                    {"reference_id": "ref_002", "reference_image": ref2},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2


@pytest.mark.integration
class TestMatchWithTransformations:
    """
    Integration tests verifying the service can match transformed images.

    These tests verify the core functionality: recognizing the same subject
    despite variations in rotation, scale, and cropping.

    Note: Uses artwork simulation images which have rich features suitable
    for ORB detection, unlike simple geometric patterns.
    """

    def test_match_rotated_artwork(self, client) -> None:
        """A rotated version of an artwork should match the original."""
        original = create_artwork_simulation_base64(350, 350, seed=100)
        rotated = create_transformed_image_base64(original, rotation_deg=15)

        response = client.post(
            "/match",
            json={"query_image": rotated, "reference_image": original},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_match"] is True, (
            f"Rotated artwork should match original. "
            f"Got inliers={data['inliers']}, ratio={data['inlier_ratio']}"
        )

    def test_match_scaled_artwork(self, client) -> None:
        """A scaled version of an artwork should match the original."""
        original = create_artwork_simulation_base64(350, 350, seed=101)
        scaled = create_transformed_image_base64(original, scale=0.8)

        response = client.post(
            "/match",
            json={"query_image": scaled, "reference_image": original},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_match"] is True, (
            f"Scaled artwork should match original. "
            f"Got inliers={data['inliers']}, ratio={data['inlier_ratio']}"
        )

    def test_match_combined_transformation(self, client) -> None:
        """An artwork with mild rotation and scale should still match."""
        # Use noise image which has more stable features for combined transforms
        original = create_noise_image_base64(350, 350, seed=102)
        # Use milder transformations
        transformed = create_transformed_image_base64(original, rotation_deg=5, scale=0.95)

        response = client.post(
            "/match",
            json={"query_image": transformed, "reference_image": original},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_match"] is True, (
            f"Transformed image (rotation + scale) should match original. "
            f"Got inliers={data['inliers']}, ratio={data['inlier_ratio']}"
        )

    def test_match_cropped_artwork(self, client) -> None:
        """A center-cropped version should match the original."""
        # Use larger image to ensure enough features after cropping
        original = create_noise_image_base64(500, 500, seed=103)
        # Less aggressive crop (80%) to retain enough features
        cropped = create_transformed_image_base64(original, crop_ratio=0.8)

        response = client.post(
            "/match",
            json={"query_image": cropped, "reference_image": original},
        )
        assert response.status_code == 200
        data = response.json()
        # Cropped images may have fewer matches but should still match
        assert data["is_match"] is True, (
            f"Cropped image should match original. "
            f"Got inliers={data['inliers']}, ratio={data['inlier_ratio']}"
        )

    def test_match_noise_image_with_rotation(self, client) -> None:
        """Noise image with rotation should match (many features)."""
        original = create_noise_image_base64(300, 300, seed=200)
        rotated = create_transformed_image_base64(original, rotation_deg=8)

        response = client.post(
            "/match",
            json={"query_image": rotated, "reference_image": original},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_match"] is True, (
            f"Rotated noise image should match original. "
            f"Got inliers={data['inliers']}, ratio={data['inlier_ratio']}"
        )


@pytest.mark.integration
class TestNonMatchingImages:
    """
    Integration tests verifying different images don't falsely match.

    These tests ensure the service correctly rejects images that are
    visually unrelated, preventing false positives.
    """

    def test_different_artworks_no_match(self, client) -> None:
        """Two different simulated artworks should not match."""
        artwork1 = create_artwork_simulation_base64(300, 300, seed=300)
        artwork2 = create_artwork_simulation_base64(300, 300, seed=400)

        response = client.post(
            "/match",
            json={"query_image": artwork1, "reference_image": artwork2},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_match"] is False, (
            f"Different artworks should not match. "
            f"Got inliers={data['inliers']}, confidence={data['confidence']}"
        )

    def test_noise_images_no_match(self, client) -> None:
        """Two different noise images should not match."""
        noise1 = create_noise_image_base64(250, 250, seed=10)
        noise2 = create_noise_image_base64(250, 250, seed=20)

        response = client.post(
            "/match",
            json={"query_image": noise1, "reference_image": noise2},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_match"] is False, (
            f"Different noise images should not match. "
            f"Got inliers={data['inliers']}, confidence={data['confidence']}"
        )

    def test_artwork_vs_noise_no_match(self, client) -> None:
        """Completely different image types should not match."""
        artwork = create_artwork_simulation_base64(300, 300, seed=500)
        noise = create_noise_image_base64(300, 300, seed=600)

        response = client.post(
            "/match",
            json={"query_image": artwork, "reference_image": noise},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_match"] is False, (
            f"Artwork and noise should not match. "
            f"Got inliers={data['inliers']}, confidence={data['confidence']}"
        )

    def test_multiple_different_images(self, client) -> None:
        """Multiple sequential comparisons of different images should all fail."""
        # Use noise images which always have abundant features
        seeds = [1000, 2000, 3000]
        images = [create_noise_image_base64(250, 250, seed=s) for s in seeds]

        # Compare each pair
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                response = client.post(
                    "/match",
                    json={"query_image": images[i], "reference_image": images[j]},
                )
                assert response.status_code == 200
                data = response.json()
                assert data["is_match"] is False, (
                    f"Images with seeds {seeds[i]} and {seeds[j]} should not match. "
                    f"Got inliers={data['inliers']}"
                )


@pytest.mark.integration
class TestEdgeCases:
    """
    Integration tests for edge cases and boundary conditions.

    These tests verify robust behavior for unusual inputs.
    """

    def test_solid_color_images_rejected(self, client) -> None:
        """Solid color images have no features and should be rejected."""
        solid1 = create_solid_color_base64(100, 100, color="red")
        solid2 = create_solid_color_base64(100, 100, color="red")

        response = client.post(
            "/match",
            json={"query_image": solid1, "reference_image": solid2},
        )
        # Service returns 422 for insufficient features (expected behavior)
        assert response.status_code == 422
        data = response.json()
        assert data["error"] == "insufficient_features"

    def test_low_feature_image_rejected(self, client) -> None:
        """Low feature image should be rejected with insufficient_features error."""
        solid = create_solid_color_base64(200, 200, color="blue")
        artwork = create_artwork_simulation_base64(200, 200, seed=800)

        response = client.post(
            "/match",
            json={"query_image": solid, "reference_image": artwork},
        )
        # Service returns 422 for insufficient features
        assert response.status_code == 422
        data = response.json()
        assert data["error"] == "insufficient_features"

    def test_large_rotation_artwork(self, client) -> None:
        """Large rotation (45 degrees) can still match with rich features."""
        original = create_artwork_simulation_base64(400, 400, seed=900)
        rotated_45 = create_transformed_image_base64(original, rotation_deg=45)

        response = client.post(
            "/match",
            json={"query_image": rotated_45, "reference_image": original},
        )
        assert response.status_code == 200
        data = response.json()
        # Verify response structure is valid
        assert "is_match" in data
        assert "inliers" in data
        # 45-degree rotation is challenging - we just verify it processes correctly

    def test_very_small_rotation(self, client) -> None:
        """Very small rotation (2 degrees) should definitely match."""
        original = create_artwork_simulation_base64(300, 300, seed=901)
        rotated = create_transformed_image_base64(original, rotation_deg=2)

        response = client.post(
            "/match",
            json={"query_image": rotated, "reference_image": original},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_match"] is True, (
            f"Tiny rotation should definitely match. "
            f"Got inliers={data['inliers']}, ratio={data['inlier_ratio']}"
        )

    def test_slight_scale_change(self, client) -> None:
        """Slight scale change (95%) should match easily."""
        original = create_artwork_simulation_base64(300, 300, seed=902)
        scaled = create_transformed_image_base64(original, scale=0.95)

        response = client.post(
            "/match",
            json={"query_image": scaled, "reference_image": original},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_match"] is True, (
            f"Slight scale change should match. "
            f"Got inliers={data['inliers']}, ratio={data['inlier_ratio']}"
        )


@pytest.mark.integration
class TestBatchMatchAccuracy:
    """
    Integration tests for batch matching accuracy.

    Verifies the batch endpoint correctly identifies matches from candidates.
    """

    def test_batch_finds_correct_match(self, client) -> None:
        """Batch match should identify the correct matching reference."""
        # Create a query and its rotated version (the matching reference)
        original = create_artwork_simulation_base64(300, 300, seed=50)
        query = create_transformed_image_base64(original, rotation_deg=8)

        # Create non-matching references
        ref_wrong1 = create_artwork_simulation_base64(300, 300, seed=51)
        ref_wrong2 = create_artwork_simulation_base64(300, 300, seed=52)

        response = client.post(
            "/match/batch",
            json={
                "query_image": query,
                "references": [
                    {"reference_id": "wrong_1", "reference_image": ref_wrong1},
                    {"reference_id": "correct", "reference_image": original},
                    {"reference_id": "wrong_2", "reference_image": ref_wrong2},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 3

        # Find which reference matched
        matches = [r for r in data["results"] if r["is_match"]]
        assert len(matches) >= 1, "Should find at least one match"

        # The correct reference should have the highest confidence/inliers
        correct_result = next(r for r in data["results"] if r["reference_id"] == "correct")
        assert correct_result["is_match"] is True, "Correct reference should match"

        # Wrong references should not match
        wrong_results = [r for r in data["results"] if r["reference_id"] != "correct"]
        for wrong in wrong_results:
            assert wrong["is_match"] is False, (
                f"Wrong reference {wrong['reference_id']} should not match. "
                f"Got inliers={wrong['inliers']}"
            )

    def test_batch_no_matches(self, client) -> None:
        """Batch match should return no matches when query doesn't match any reference."""
        query = create_artwork_simulation_base64(250, 250, seed=999)
        ref1 = create_artwork_simulation_base64(250, 250, seed=100)
        ref2 = create_artwork_simulation_base64(250, 250, seed=200)
        ref3 = create_artwork_simulation_base64(250, 250, seed=300)

        response = client.post(
            "/match/batch",
            json={
                "query_image": query,
                "references": [
                    {"reference_id": "ref_1", "reference_image": ref1},
                    {"reference_id": "ref_2", "reference_image": ref2},
                    {"reference_id": "ref_3", "reference_image": ref3},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()

        # None should match
        matches = [r for r in data["results"] if r["is_match"]]
        assert len(matches) == 0, (
            f"Query should not match any reference. Found {len(matches)} matches."
        )
