# Geometric Service Implementation Plan - Part 2: API Layer, Testing, and Validation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Prerequisites:** Complete [Part 1: Core Infrastructure and Domain Logic](geometric-service-implementation-plan-part1.md) (Tasks 1-9) before starting this part.

**Goal:** Implement the API layer, application factory, tests, and validation for the Geometric Service.

**Tech Stack:** FastAPI, OpenCV (opencv-python-headless), NumPy, Pydantic, PyYAML

---

## Task 10: Routers - Health Endpoint

**Files:**
- Create: `services/geometric/src/geometric_service/routers/__init__.py`
- Create: `services/geometric/src/geometric_service/routers/health.py`
- Test: `services/geometric/tests/unit/routers/__init__.py`
- Test: `services/geometric/tests/unit/routers/test_health.py`

**Step 1: Write failing test**

Create `services/geometric/tests/unit/routers/__init__.py`:
```python
"""Unit tests for routers."""
```

Create `services/geometric/tests/unit/routers/test_health.py`:
```python
"""Unit tests for health endpoint."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.mark.unit
class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self) -> None:
        """Health endpoint returns 200 OK."""
        with (
            patch("geometric_service.config.get_settings") as mock_settings,
            patch("geometric_service.app.lifespan"),
            patch("geometric_service.routers.health.get_app_state") as mock_state,
        ):
            # Mock settings
            settings = MagicMock()
            settings.service.name = "geometric"
            settings.service.version = "0.1.0"
            mock_settings.return_value = settings

            # Mock app state
            state = MagicMock()
            state.start_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
            state.uptime_seconds = 123.45
            state.uptime_formatted = "2m 3s"
            mock_state.return_value = state

            from geometric_service.app import create_app

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/health")

            assert response.status_code == 200

    def test_health_returns_healthy_status(self) -> None:
        """Health endpoint returns healthy status."""
        with (
            patch("geometric_service.config.get_settings") as mock_settings,
            patch("geometric_service.app.lifespan"),
            patch("geometric_service.routers.health.get_app_state") as mock_state,
        ):
            settings = MagicMock()
            settings.service.name = "geometric"
            settings.service.version = "0.1.0"
            mock_settings.return_value = settings

            state = MagicMock()
            state.uptime_seconds = 123.45
            state.uptime_formatted = "2m 3s"
            mock_state.return_value = state

            from geometric_service.app import create_app

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/health")
            data = response.json()

            assert data["status"] == "healthy"

    def test_health_includes_uptime(self) -> None:
        """Health endpoint includes uptime information."""
        with (
            patch("geometric_service.config.get_settings") as mock_settings,
            patch("geometric_service.app.lifespan"),
            patch("geometric_service.routers.health.get_app_state") as mock_state,
        ):
            settings = MagicMock()
            settings.service.name = "geometric"
            settings.service.version = "0.1.0"
            mock_settings.return_value = settings

            state = MagicMock()
            state.uptime_seconds = 123.45
            state.uptime_formatted = "2m 3s"
            mock_state.return_value = state

            from geometric_service.app import create_app

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/health")
            data = response.json()

            assert data["uptime_seconds"] == 123.45
            assert data["uptime"] == "2m 3s"

    def test_health_system_time_format(self) -> None:
        """System time is in yyyy-mm-dd hh:mm format."""
        with (
            patch("geometric_service.config.get_settings") as mock_settings,
            patch("geometric_service.app.lifespan"),
            patch("geometric_service.routers.health.get_app_state") as mock_state,
        ):
            settings = MagicMock()
            settings.service.name = "geometric"
            settings.service.version = "0.1.0"
            mock_settings.return_value = settings

            state = MagicMock()
            state.uptime_seconds = 123.45
            state.uptime_formatted = "2m 3s"
            mock_state.return_value = state

            from geometric_service.app import create_app

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/health")
            data = response.json()

            # Format: yyyy-mm-dd hh:mm
            pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$"
            assert re.match(pattern, data["system_time"])
```

**Step 2: Run test to verify it fails**

Run: `cd services/geometric && uv run pytest tests/unit/routers/test_health.py -v -m unit`
Expected: FAIL

**Step 3: Create the health router**

Create `services/geometric/src/geometric_service/routers/__init__.py`:
```python
"""API routers for the geometric service."""

from geometric_service.routers import extract, health, info, match

__all__ = ["extract", "health", "info", "match"]
```

Create `services/geometric/src/geometric_service/routers/health.py`:
```python
"""
Health check endpoint.

Provides service health status for container orchestration
(Docker health checks, Kubernetes probes).
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from geometric_service.core.state import get_app_state
from geometric_service.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Check service health.

    Returns:
        Health status with uptime and system time
    """
    state = get_app_state()

    # System time in yyyy-mm-dd hh:mm format
    system_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

    return HealthResponse(
        status="healthy",
        uptime_seconds=state.uptime_seconds,
        uptime=state.uptime_formatted,
        system_time=system_time,
    )
```

**Step 4: Run test to verify it passes**

Run: `cd services/geometric && uv run pytest tests/unit/routers/test_health.py -v -m unit`
Expected: PASS (after creating remaining files)

**Step 5: Commit (after creating all routers)**

---

## Task 11: Routers - Info Endpoint

**Files:**
- Create: `services/geometric/src/geometric_service/routers/info.py`
- Test: `services/geometric/tests/unit/routers/test_info.py`

**Step 1: Write failing test**

Create `services/geometric/tests/unit/routers/test_info.py`:
```python
"""Unit tests for info endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.mark.unit
class TestInfoEndpoint:
    """Tests for GET /info."""

    def test_info_returns_200(self) -> None:
        """Info endpoint returns 200 OK."""
        with (
            patch("geometric_service.config.get_settings") as mock_settings,
            patch("geometric_service.app.lifespan"),
            patch("geometric_service.routers.health.get_app_state"),
        ):
            settings = MagicMock()
            settings.service.name = "geometric"
            settings.service.version = "0.1.0"
            settings.orb.max_features = 1000
            settings.matching.ratio_threshold = 0.75
            settings.ransac.reproj_threshold = 5.0
            settings.verification.min_inliers = 10
            mock_settings.return_value = settings

            from geometric_service.app import create_app

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/info")

            assert response.status_code == 200

    def test_info_returns_algorithm_config(self) -> None:
        """Info endpoint returns algorithm configuration."""
        with (
            patch("geometric_service.config.get_settings") as mock_settings,
            patch("geometric_service.app.lifespan"),
            patch("geometric_service.routers.health.get_app_state"),
        ):
            settings = MagicMock()
            settings.service.name = "geometric"
            settings.service.version = "0.1.0"
            settings.orb.max_features = 1000
            settings.matching.ratio_threshold = 0.75
            settings.ransac.reproj_threshold = 5.0
            settings.verification.min_inliers = 10
            mock_settings.return_value = settings

            from geometric_service.app import create_app

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/info")
            data = response.json()

            assert data["service"] == "geometric"
            assert data["algorithm"]["feature_detector"] == "ORB"
            assert data["algorithm"]["max_features"] == 1000
            assert data["algorithm"]["ratio_threshold"] == 0.75
```

**Step 2: Create the info router**

Create `services/geometric/src/geometric_service/routers/info.py`:
```python
"""
Service information endpoint.

Exposes service configuration and metadata.
"""

from __future__ import annotations

from fastapi import APIRouter

from geometric_service.config import get_settings
from geometric_service.schemas import AlgorithmInfo, InfoResponse

router = APIRouter()


@router.get("/info", response_model=InfoResponse)
async def get_info() -> InfoResponse:
    """
    Get service information and configuration.

    Returns:
        Service metadata and algorithm configuration
    """
    settings = get_settings()

    algorithm = AlgorithmInfo(
        feature_detector="ORB",
        max_features=settings.orb.max_features,
        matcher="BFMatcher",
        matcher_norm="HAMMING",
        ratio_threshold=settings.matching.ratio_threshold,
        verification="RANSAC",
        ransac_reproj_threshold=settings.ransac.reproj_threshold,
        min_inliers=settings.verification.min_inliers,
    )

    return InfoResponse(
        service=settings.service.name,
        version=settings.service.version,
        algorithm=algorithm,
    )
```

---

## Task 12: Routers - Extract Endpoint

**Files:**
- Create: `services/geometric/src/geometric_service/routers/extract.py`
- Test: `services/geometric/tests/unit/routers/test_extract.py`

**Step 1: Write failing test**

Create `services/geometric/tests/unit/routers/test_extract.py`:
```python
"""Unit tests for extract endpoint."""

from __future__ import annotations

import base64
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


def create_checkerboard_image() -> str:
    """Create a checkerboard image with features as base64."""
    img = np.zeros((200, 200), dtype=np.uint8)
    block_size = 20
    for i in range(0, 200, block_size * 2):
        for j in range(0, 200, block_size * 2):
            img[i : i + block_size, j : j + block_size] = 255
            img[i + block_size : i + block_size * 2, j + block_size : j + block_size * 2] = 255
    pil_image = Image.fromarray(img)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.mark.unit
class TestExtractEndpoint:
    """Tests for POST /extract."""

    def test_extract_returns_200(self) -> None:
        """Extract endpoint returns 200 OK for valid image."""
        with (
            patch("geometric_service.config.get_settings") as mock_settings,
            patch("geometric_service.app.lifespan"),
            patch("geometric_service.routers.health.get_app_state"),
        ):
            settings = MagicMock()
            settings.service.name = "geometric"
            settings.service.version = "0.1.0"
            settings.orb.max_features = 1000
            settings.orb.scale_factor = 1.2
            settings.orb.n_levels = 8
            settings.orb.edge_threshold = 31
            settings.orb.patch_size = 31
            settings.orb.fast_threshold = 20
            settings.matching.ratio_threshold = 0.75
            settings.ransac.reproj_threshold = 5.0
            settings.verification.min_features = 50
            settings.verification.min_inliers = 10
            mock_settings.return_value = settings

            from geometric_service.app import create_app

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post(
                "/extract",
                json={"image": create_checkerboard_image(), "image_id": "test_001"},
            )

            assert response.status_code == 200

    def test_extract_returns_keypoints(self) -> None:
        """Extract endpoint returns keypoints."""
        with (
            patch("geometric_service.config.get_settings") as mock_settings,
            patch("geometric_service.app.lifespan"),
            patch("geometric_service.routers.health.get_app_state"),
        ):
            settings = MagicMock()
            settings.service.name = "geometric"
            settings.service.version = "0.1.0"
            settings.orb.max_features = 1000
            settings.orb.scale_factor = 1.2
            settings.orb.n_levels = 8
            settings.orb.edge_threshold = 31
            settings.orb.patch_size = 31
            settings.orb.fast_threshold = 20
            settings.matching.ratio_threshold = 0.75
            settings.ransac.reproj_threshold = 5.0
            settings.verification.min_features = 50
            settings.verification.min_inliers = 10
            mock_settings.return_value = settings

            from geometric_service.app import create_app

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post(
                "/extract",
                json={"image": create_checkerboard_image()},
            )
            data = response.json()

            assert "keypoints" in data
            assert "descriptors" in data
            assert "num_features" in data
            assert data["num_features"] > 0

    def test_extract_invalid_image(self) -> None:
        """Extract endpoint returns 400 for invalid image."""
        with (
            patch("geometric_service.config.get_settings") as mock_settings,
            patch("geometric_service.app.lifespan"),
            patch("geometric_service.routers.health.get_app_state"),
        ):
            settings = MagicMock()
            settings.service.name = "geometric"
            settings.service.version = "0.1.0"
            settings.orb.max_features = 1000
            settings.orb.scale_factor = 1.2
            settings.orb.n_levels = 8
            settings.orb.edge_threshold = 31
            settings.orb.patch_size = 31
            settings.orb.fast_threshold = 20
            settings.matching.ratio_threshold = 0.75
            settings.ransac.reproj_threshold = 5.0
            settings.verification.min_features = 50
            settings.verification.min_inliers = 10
            mock_settings.return_value = settings

            from geometric_service.app import create_app

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            # Send non-image data
            response = client.post(
                "/extract",
                json={"image": base64.b64encode(b"not an image").decode()},
            )

            assert response.status_code == 400
            assert response.json()["error"] == "invalid_image"
```

**Step 2: Create the extract router**

Create `services/geometric/src/geometric_service/routers/extract.py`:
```python
"""
Feature extraction endpoint.

Extracts ORB features from base64-encoded images.
"""

from __future__ import annotations

import base64
import binascii
import time

from fastapi import APIRouter

from geometric_service.config import get_settings
from geometric_service.core.exceptions import ServiceError
from geometric_service.logging import get_logger
from geometric_service.schemas import (
    ExtractRequest,
    ExtractResponse,
    ImageSize,
    KeypointData,
)
from geometric_service.services.feature_extractor import ORBFeatureExtractor

router = APIRouter()


def decode_base64_image(base64_string: str) -> bytes:
    """
    Decode base64 string to bytes.

    Args:
        base64_string: Base64-encoded image data

    Returns:
        Raw image bytes

    Raises:
        ServiceError: If base64 decoding fails
    """
    # Handle potential data URL prefix (e.g., "data:image/jpeg;base64,...")
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]

    try:
        return base64.b64decode(base64_string)
    except binascii.Error as e:
        raise ServiceError(
            error="decode_error",
            message=f"Invalid Base64 encoding: {e}",
            status_code=400,
            details=None,
        ) from e


@router.post("/extract", response_model=ExtractResponse)
async def extract_features(request: ExtractRequest) -> ExtractResponse:
    """
    Extract ORB features from an image.

    Args:
        request: Request containing base64-encoded image

    Returns:
        Extracted keypoints and descriptors
    """
    logger = get_logger()
    start_time = time.perf_counter()

    settings = get_settings()

    # Decode base64
    image_bytes = decode_base64_image(request.image)

    # Create extractor with config
    max_features = request.max_features or settings.orb.max_features
    extractor = ORBFeatureExtractor(
        max_features=max_features,
        scale_factor=settings.orb.scale_factor,
        n_levels=settings.orb.n_levels,
        edge_threshold=settings.orb.edge_threshold,
        patch_size=settings.orb.patch_size,
        fast_threshold=settings.orb.fast_threshold,
    )

    # Extract features
    keypoints, descriptors, image_size = extractor.extract(image_bytes)

    # Check minimum features
    if len(keypoints) < settings.verification.min_features:
        raise ServiceError(
            error="insufficient_features",
            message=f"Only {len(keypoints)} features extracted (minimum: {settings.verification.min_features})",
            status_code=422,
            details={
                "image_id": request.image_id,
                "features_found": len(keypoints),
                "minimum_required": settings.verification.min_features,
            },
        )

    # Encode descriptors as base64
    if descriptors is not None:
        descriptors_b64 = base64.b64encode(descriptors.tobytes()).decode("ascii")
    else:
        descriptors_b64 = ""

    # Convert keypoints to schema format
    kp_list = [
        KeypointData(x=kp["x"], y=kp["y"], size=kp["size"], angle=kp["angle"])
        for kp in keypoints
    ]

    processing_time_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "Features extracted",
        extra={
            "image_id": request.image_id,
            "num_features": len(keypoints),
            "processing_time_ms": round(processing_time_ms, 2),
        },
    )

    return ExtractResponse(
        image_id=request.image_id,
        num_features=len(keypoints),
        keypoints=kp_list,
        descriptors=descriptors_b64,
        image_size=ImageSize(width=image_size[0], height=image_size[1]),
        processing_time_ms=round(processing_time_ms, 2),
    )
```

---

## Task 13: Routers - Match Endpoint

**Files:**
- Create: `services/geometric/src/geometric_service/routers/match.py`
- Test: `services/geometric/tests/unit/routers/test_match.py`

**Step 1: Write failing test**

Create `services/geometric/tests/unit/routers/test_match.py`:
```python
"""Unit tests for match endpoints."""

from __future__ import annotations

import base64
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


def create_checkerboard_image(seed: int = 0) -> str:
    """Create a checkerboard image with features as base64."""
    np.random.seed(seed)
    img = np.zeros((200, 200), dtype=np.uint8)
    block_size = 20
    for i in range(0, 200, block_size * 2):
        for j in range(0, 200, block_size * 2):
            img[i : i + block_size, j : j + block_size] = 255
            img[i + block_size : i + block_size * 2, j + block_size : j + block_size * 2] = 255
    pil_image = Image.fromarray(img)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def create_mock_settings() -> MagicMock:
    """Create mock settings for tests."""
    settings = MagicMock()
    settings.service.name = "geometric"
    settings.service.version = "0.1.0"
    settings.orb.max_features = 1000
    settings.orb.scale_factor = 1.2
    settings.orb.n_levels = 8
    settings.orb.edge_threshold = 31
    settings.orb.patch_size = 31
    settings.orb.fast_threshold = 20
    settings.matching.ratio_threshold = 0.75
    settings.ransac.reproj_threshold = 5.0
    settings.ransac.max_iters = 2000
    settings.ransac.confidence = 0.995
    settings.verification.min_features = 10  # Lower for tests
    settings.verification.min_matches = 4
    settings.verification.min_inliers = 4
    return settings


@pytest.mark.unit
class TestMatchEndpoint:
    """Tests for POST /match."""

    def test_match_same_image_returns_match(self) -> None:
        """Matching same image should return is_match=True."""
        with (
            patch("geometric_service.config.get_settings") as mock_settings,
            patch("geometric_service.app.lifespan"),
            patch("geometric_service.routers.health.get_app_state"),
        ):
            mock_settings.return_value = create_mock_settings()

            from geometric_service.app import create_app

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            image = create_checkerboard_image()
            response = client.post(
                "/match",
                json={
                    "query_image": image,
                    "reference_image": image,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["is_match"] is True
            assert data["inliers"] > 0

    def test_match_returns_processing_time(self) -> None:
        """Match endpoint should return processing time."""
        with (
            patch("geometric_service.config.get_settings") as mock_settings,
            patch("geometric_service.app.lifespan"),
            patch("geometric_service.routers.health.get_app_state"),
        ):
            mock_settings.return_value = create_mock_settings()

            from geometric_service.app import create_app

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            image = create_checkerboard_image()
            response = client.post(
                "/match",
                json={"query_image": image, "reference_image": image},
            )

            data = response.json()
            assert "processing_time_ms" in data
            assert data["processing_time_ms"] > 0


@pytest.mark.unit
class TestBatchMatchEndpoint:
    """Tests for POST /match/batch."""

    def test_batch_match_returns_results(self) -> None:
        """Batch match should return results for all references."""
        with (
            patch("geometric_service.config.get_settings") as mock_settings,
            patch("geometric_service.app.lifespan"),
            patch("geometric_service.routers.health.get_app_state"),
        ):
            mock_settings.return_value = create_mock_settings()

            from geometric_service.app import create_app

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            query = create_checkerboard_image(seed=0)
            ref1 = create_checkerboard_image(seed=0)  # Same as query
            ref2 = create_checkerboard_image(seed=42)  # Different

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
            assert data["results"][0]["reference_id"] == "ref_001"
            assert data["results"][1]["reference_id"] == "ref_002"
```

**Step 2: Create the match router**

Create `services/geometric/src/geometric_service/routers/match.py`:
```python
"""
Geometric matching endpoints.

Verifies spatial consistency between image pairs using ORB + RANSAC.
"""

from __future__ import annotations

import base64
import binascii
import time

import numpy as np
from fastapi import APIRouter

from geometric_service.config import get_settings
from geometric_service.core.exceptions import ServiceError
from geometric_service.logging import get_logger
from geometric_service.schemas import (
    BatchMatchRequest,
    BatchMatchResponse,
    BatchMatchResult,
    BestMatch,
    MatchRequest,
    MatchResponse,
)
from geometric_service.services.feature_extractor import ORBFeatureExtractor
from geometric_service.services.feature_matcher import BFFeatureMatcher
from geometric_service.services.geometric_verifier import RANSACVerifier

router = APIRouter()


def decode_base64_image(base64_string: str) -> bytes:
    """Decode base64 string to bytes."""
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]

    try:
        return base64.b64decode(base64_string)
    except binascii.Error as e:
        raise ServiceError(
            error="decode_error",
            message=f"Invalid Base64 encoding: {e}",
            status_code=400,
            details=None,
        ) from e


def decode_descriptors(descriptors_b64: str, num_features: int) -> np.ndarray:
    """Decode base64 descriptors to numpy array."""
    try:
        desc_bytes = base64.b64decode(descriptors_b64)
        return np.frombuffer(desc_bytes, dtype=np.uint8).reshape(num_features, 32)
    except Exception as e:
        raise ServiceError(
            error="invalid_features",
            message=f"Failed to decode descriptors: {e}",
            status_code=400,
            details=None,
        ) from e


@router.post("/match", response_model=MatchResponse)
async def match_images(request: MatchRequest) -> MatchResponse:
    """
    Geometrically verify two images.

    Args:
        request: Request with query and reference images

    Returns:
        Match result with confidence score
    """
    logger = get_logger()
    start_time = time.perf_counter()

    settings = get_settings()

    # Create extractor
    extractor = ORBFeatureExtractor(
        max_features=settings.orb.max_features,
        scale_factor=settings.orb.scale_factor,
        n_levels=settings.orb.n_levels,
        edge_threshold=settings.orb.edge_threshold,
        patch_size=settings.orb.patch_size,
        fast_threshold=settings.orb.fast_threshold,
    )

    # Extract query features
    query_bytes = decode_base64_image(request.query_image)
    query_kp, query_desc, _ = extractor.extract(query_bytes)
    query_cv_kp = extractor.keypoints_to_cv(query_kp)

    # Get reference features
    if request.reference_image is not None:
        ref_bytes = decode_base64_image(request.reference_image)
        ref_kp, ref_desc, _ = extractor.extract(ref_bytes)
        ref_cv_kp = extractor.keypoints_to_cv(ref_kp)
    elif request.reference_features is not None:
        ref_kp = [kp.model_dump() for kp in request.reference_features.keypoints]
        ref_desc = decode_descriptors(
            request.reference_features.descriptors,
            len(request.reference_features.keypoints),
        )
        ref_cv_kp = extractor.keypoints_to_cv(ref_kp)
    else:
        raise ServiceError(
            error="validation_error",
            message="Either reference_image or reference_features must be provided",
            status_code=400,
            details=None,
        )

    # Check minimum features
    if len(query_kp) < settings.verification.min_features:
        raise ServiceError(
            error="insufficient_features",
            message=f"Query has only {len(query_kp)} features (minimum: {settings.verification.min_features})",
            status_code=422,
            details={"query_features": len(query_kp)},
        )

    if len(ref_kp) < settings.verification.min_features:
        raise ServiceError(
            error="insufficient_features",
            message=f"Reference has only {len(ref_kp)} features (minimum: {settings.verification.min_features})",
            status_code=422,
            details={"reference_features": len(ref_kp)},
        )

    # Match features
    matcher = BFFeatureMatcher(ratio_threshold=settings.matching.ratio_threshold)
    matches = matcher.match(query_desc, ref_desc)

    # Verify geometry
    verifier = RANSACVerifier(
        reproj_threshold=settings.ransac.reproj_threshold,
        max_iters=settings.ransac.max_iters,
        confidence=settings.ransac.confidence,
        min_inliers=settings.verification.min_inliers,
    )
    result = verifier.verify(query_cv_kp, ref_cv_kp, matches)

    processing_time_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "Match completed",
        extra={
            "query_id": request.query_id,
            "reference_id": request.reference_id,
            "is_match": result["is_match"],
            "inliers": result["inliers"],
            "confidence": result["confidence"],
            "processing_time_ms": round(processing_time_ms, 2),
        },
    )

    return MatchResponse(
        is_match=result["is_match"],
        confidence=result["confidence"],
        inliers=result["inliers"],
        total_matches=result["total_matches"],
        inlier_ratio=result["inlier_ratio"],
        query_features=len(query_kp),
        reference_features=len(ref_kp),
        homography=result["homography"],
        query_id=request.query_id,
        reference_id=request.reference_id,
        processing_time_ms=round(processing_time_ms, 2),
    )


@router.post("/match/batch", response_model=BatchMatchResponse)
async def batch_match(request: BatchMatchRequest) -> BatchMatchResponse:
    """
    Verify query against multiple references.

    Args:
        request: Request with query and list of references

    Returns:
        Match results for all references with best match
    """
    logger = get_logger()
    start_time = time.perf_counter()

    settings = get_settings()

    # Create extractor
    extractor = ORBFeatureExtractor(
        max_features=settings.orb.max_features,
        scale_factor=settings.orb.scale_factor,
        n_levels=settings.orb.n_levels,
        edge_threshold=settings.orb.edge_threshold,
        patch_size=settings.orb.patch_size,
        fast_threshold=settings.orb.fast_threshold,
    )

    # Extract query features once
    query_bytes = decode_base64_image(request.query_image)
    query_kp, query_desc, _ = extractor.extract(query_bytes)
    query_cv_kp = extractor.keypoints_to_cv(query_kp)

    # Create matcher and verifier
    matcher = BFFeatureMatcher(ratio_threshold=settings.matching.ratio_threshold)
    verifier = RANSACVerifier(
        reproj_threshold=settings.ransac.reproj_threshold,
        max_iters=settings.ransac.max_iters,
        confidence=settings.ransac.confidence,
        min_inliers=settings.verification.min_inliers,
    )

    # Match against each reference
    results: list[BatchMatchResult] = []
    best_match: BestMatch | None = None
    best_confidence = 0.0

    for ref in request.references:
        try:
            # Get reference features
            if ref.reference_image is not None:
                ref_bytes = decode_base64_image(ref.reference_image)
                ref_kp, ref_desc, _ = extractor.extract(ref_bytes)
                ref_cv_kp = extractor.keypoints_to_cv(ref_kp)
            elif ref.reference_features is not None:
                ref_kp = [kp.model_dump() for kp in ref.reference_features.keypoints]
                ref_desc = decode_descriptors(
                    ref.reference_features.descriptors,
                    len(ref.reference_features.keypoints),
                )
                ref_cv_kp = extractor.keypoints_to_cv(ref_kp)
            else:
                # Skip if no image or features
                results.append(
                    BatchMatchResult(
                        reference_id=ref.reference_id,
                        is_match=False,
                        confidence=0.0,
                        inliers=0,
                        inlier_ratio=0.0,
                    )
                )
                continue

            # Match and verify
            matches = matcher.match(query_desc, ref_desc)
            result = verifier.verify(query_cv_kp, ref_cv_kp, matches)

            results.append(
                BatchMatchResult(
                    reference_id=ref.reference_id,
                    is_match=result["is_match"],
                    confidence=result["confidence"],
                    inliers=result["inliers"],
                    inlier_ratio=result["inlier_ratio"],
                )
            )

            # Track best match
            if result["is_match"] and result["confidence"] > best_confidence:
                best_confidence = result["confidence"]
                best_match = BestMatch(
                    reference_id=ref.reference_id,
                    confidence=result["confidence"],
                )

        except ServiceError:
            # Log but continue with other references
            results.append(
                BatchMatchResult(
                    reference_id=ref.reference_id,
                    is_match=False,
                    confidence=0.0,
                    inliers=0,
                    inlier_ratio=0.0,
                )
            )

    processing_time_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "Batch match completed",
        extra={
            "query_id": request.query_id,
            "num_references": len(request.references),
            "best_match": best_match.reference_id if best_match else None,
            "processing_time_ms": round(processing_time_ms, 2),
        },
    )

    return BatchMatchResponse(
        query_id=request.query_id,
        query_features=len(query_kp),
        results=results,
        best_match=best_match,
        processing_time_ms=round(processing_time_ms, 2),
    )
```

**Step 3: Commit all routers**

```bash
git add services/geometric/src/geometric_service/routers/*.py services/geometric/tests/unit/routers/*.py
git commit -m "feat(geometric): add API routers for health, info, extract, and match"
git push
```

---

## Task 14: Application Factory and Entry Point

**Files:**
- Create: `services/geometric/src/geometric_service/app.py`
- Create: `services/geometric/src/geometric_service/main.py`
- Update: `services/geometric/src/geometric_service/__init__.py`

**Step 1: Create the app factory**

Create `services/geometric/src/geometric_service/app.py`:
```python
"""
FastAPI application factory.

Creates and configures the FastAPI application instance.
"""

from __future__ import annotations

from fastapi import FastAPI

from geometric_service.config import get_settings
from geometric_service.core.exceptions import register_exception_handlers
from geometric_service.core.lifespan import lifespan
from geometric_service.routers import extract, health, info, match


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        Configured FastAPI instance
    """
    # Load settings (validates configuration)
    settings = get_settings()

    # Create app with lifespan management
    app = FastAPI(
        title=f"{settings.service.name} Service",
        version=settings.service.version,
        lifespan=lifespan,
    )

    # Register exception handlers
    register_exception_handlers(app)

    # Register routers
    app.include_router(health.router, tags=["Operations"])
    app.include_router(info.router, tags=["Operations"])
    app.include_router(extract.router, tags=["Features"])
    app.include_router(match.router, tags=["Matching"])

    return app
```

**Step 2: Create the entry point**

Create `services/geometric/src/geometric_service/main.py`:
```python
"""
Service entry point.

This module provides the main() function for running the service
with production-grade configuration.
"""

from __future__ import annotations

import sys

import uvicorn

from geometric_service.app import create_app
from geometric_service.config import ConfigurationError, get_settings


def main() -> int:
    """
    Run the service with uvicorn.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Validate configuration before starting
        settings = get_settings()
    except ConfigurationError as e:
        # Print to stderr - logging isn't configured yet
        print(f"FATAL: Configuration error\n{e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"FATAL: Unexpected error during configuration\n{e}", file=sys.stderr)
        return 1

    # Create application
    app = create_app()

    # Run with uvicorn
    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
        # Production settings
        log_level="warning",  # Uvicorn logs (our JSON logger handles app logs)
        access_log=False,  # Disable uvicorn access log (use middleware if needed)
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 3: Update the __init__.py**

Update `services/geometric/src/geometric_service/__init__.py`:
```python
"""
Geometric Service - ORB feature extraction and RANSAC verification.

This service provides local feature extraction and geometric verification
using classical computer vision techniques for artwork matching.
"""

__version__ = "0.1.0"
```

**Step 4: Commit**

```bash
git add services/geometric/src/geometric_service/app.py services/geometric/src/geometric_service/main.py services/geometric/src/geometric_service/__init__.py
git commit -m "feat(geometric): add FastAPI app factory and entry point"
git push
```

---

## Task 15: Update Config.yaml

**Files:**
- Update: `services/geometric/config.yaml`

**Step 1: Verify and update config.yaml**

The config.yaml should already have:
```yaml
# Geometric Service Configuration
# Environment variable overrides use prefix: GEOMETRIC__
# Example: GEOMETRIC__ORB__MAX_FEATURES=2000

service:
  name: "geometric"
  version: "0.1.0"

orb:
  max_features: 1000
  scale_factor: 1.2
  n_levels: 8
  edge_threshold: 31
  patch_size: 31
  fast_threshold: 20

matching:
  ratio_threshold: 0.75             # Lowe's ratio test threshold
  cross_check: false

ransac:
  reproj_threshold: 5.0             # Maximum reprojection error (pixels)
  max_iters: 2000
  confidence: 0.995

verification:
  min_features: 50                  # Minimum features required per image
  min_matches: 20                   # Minimum matches before RANSAC
  min_inliers: 10                   # Minimum inliers to declare match

server:
  host: "0.0.0.0"
  port: 8003
  log_level: "info"
```

If the config needs updating:
```bash
git add services/geometric/config.yaml
git commit -m "chore(geometric): update configuration with service identity"
git push
```

---

## Task 16: Test Setup - Conftest and Factories

**Files:**
- Update: `services/geometric/tests/conftest.py`
- Create: `services/geometric/tests/factories.py`
- Update: `services/geometric/tests/unit/conftest.py`

**Step 1: Create test factories**

Create `services/geometric/tests/factories.py`:
```python
"""
Test data factories for generating test fixtures.

Provides reusable functions for creating test images with features.
"""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from PIL import Image


def create_checkerboard_image(
    width: int = 200,
    height: int = 200,
    block_size: int = 20,
    seed: int | None = None,
) -> bytes:
    """
    Create a checkerboard image with detectable features.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        block_size: Size of checkerboard squares
        seed: Random seed for reproducibility

    Returns:
        Raw image bytes (PNG format)
    """
    if seed is not None:
        np.random.seed(seed)

    img = np.zeros((height, width), dtype=np.uint8)
    for i in range(0, height, block_size * 2):
        for j in range(0, width, block_size * 2):
            img[i : i + block_size, j : j + block_size] = 255
            img[i + block_size : i + block_size * 2, j + block_size : j + block_size * 2] = 255

    pil_image = Image.fromarray(img)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def create_checkerboard_base64(
    width: int = 200,
    height: int = 200,
    block_size: int = 20,
    seed: int | None = None,
) -> str:
    """Create a checkerboard image as base64."""
    image_bytes = create_checkerboard_image(width, height, block_size, seed)
    return base64.b64encode(image_bytes).decode("ascii")


def create_solid_color_image(
    width: int = 100,
    height: int = 100,
    color: str = "red",
) -> bytes:
    """Create a solid color image (no features)."""
    image = Image.new("RGB", (width, height), color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def create_solid_color_base64(
    width: int = 100,
    height: int = 100,
    color: str = "red",
) -> str:
    """Create a solid color image as base64."""
    image_bytes = create_solid_color_image(width, height, color)
    return base64.b64encode(image_bytes).decode("ascii")


def create_invalid_base64() -> str:
    """Create an invalid base64 string for error testing."""
    return "not-valid-base64!!!"


def create_non_image_base64() -> str:
    """Create valid base64 that is not an image."""
    return base64.b64encode(b"not an image file").decode("ascii")
```

**Step 2: Update conftest files**

Update `services/geometric/tests/conftest.py`:
```python
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
```

Create `services/geometric/tests/unit/conftest.py`:
```python
"""
Shared fixtures for unit tests.

All external dependencies (settings, app state) are mocked
to ensure tests run in isolation without I/O or network access.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from tests.factories import create_checkerboard_base64

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def mock_settings() -> Iterator[MagicMock]:
    """
    Mock application settings.

    Provides a complete mock of the Settings object with all
    configuration values typically loaded from config.yaml.
    """
    with patch("geometric_service.config.get_settings") as mock:
        settings = MagicMock()

        # Service config
        settings.service.name = "geometric"
        settings.service.version = "0.1.0"

        # ORB config
        settings.orb.max_features = 1000
        settings.orb.scale_factor = 1.2
        settings.orb.n_levels = 8
        settings.orb.edge_threshold = 31
        settings.orb.patch_size = 31
        settings.orb.fast_threshold = 20

        # Matching config
        settings.matching.ratio_threshold = 0.75
        settings.matching.cross_check = False

        # RANSAC config
        settings.ransac.reproj_threshold = 5.0
        settings.ransac.max_iters = 2000
        settings.ransac.confidence = 0.995

        # Verification config
        settings.verification.min_features = 50
        settings.verification.min_matches = 20
        settings.verification.min_inliers = 10

        # Server config
        settings.server.host = "0.0.0.0"
        settings.server.port = 8003
        settings.server.log_level = "info"

        mock.return_value = settings
        yield settings


@pytest.fixture
def mock_app_state() -> Iterator[MagicMock]:
    """
    Mock application state.

    Provides a mock AppState with predictable uptime values.
    """
    with patch("geometric_service.core.state.get_app_state") as mock:
        state = MagicMock()
        state.start_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        state.uptime_seconds = 123.45
        state.uptime_formatted = "2m 3s"

        mock.return_value = state
        yield state


@pytest.fixture
def sample_image_base64() -> str:
    """Create a base64-encoded checkerboard image with features."""
    return create_checkerboard_base64(200, 200, 20)
```

**Step 3: Remove placeholder test**

```bash
rm services/geometric/tests/test_placeholder.py
```

**Step 4: Commit**

```bash
git add services/geometric/tests/conftest.py services/geometric/tests/factories.py services/geometric/tests/unit/conftest.py
git rm services/geometric/tests/test_placeholder.py 2>/dev/null || true
git commit -m "test(geometric): add test fixtures and factories"
git push
```

---

## Task 17: Integration Tests

**Files:**
- Create: `services/geometric/tests/integration/__init__.py`
- Create: `services/geometric/tests/integration/conftest.py`
- Create: `services/geometric/tests/integration/test_endpoints.py`

**Step 1: Create integration test structure**

Create `services/geometric/tests/integration/__init__.py`:
```python
"""Integration tests for geometric service."""
```

Create `services/geometric/tests/integration/conftest.py`:
```python
"""
Fixtures for integration tests.

These tests use the real application with actual OpenCV processing.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from geometric_service.app import create_app
from geometric_service.config import clear_settings_cache


@pytest.fixture
def client() -> TestClient:
    """Create test client with real app."""
    clear_settings_cache()
    app = create_app()
    return TestClient(app)
```

Create `services/geometric/tests/integration/test_endpoints.py`:
```python
"""Integration tests for all endpoints."""

from __future__ import annotations

import pytest

from tests.factories import create_checkerboard_base64, create_non_image_base64


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
        assert data["confidence"] > 0.5


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
```

**Step 2: Commit**

```bash
git add services/geometric/tests/integration/__init__.py services/geometric/tests/integration/conftest.py services/geometric/tests/integration/test_endpoints.py
git commit -m "test(geometric): add integration tests for all endpoints"
git push
```

---

## Task 18: Run Full CI and Fix Issues

**Step 1: Initialize environment**

Run: `cd services/geometric && just init`

**Step 2: Run unit tests**

Run: `cd services/geometric && just test-unit`

**Step 3: Run all CI checks**

Run: `cd services/geometric && just ci`

**Step 4: Fix any issues**

Address any failures from:
- `code-style` - Fix with `just code-format`
- `code-typecheck` - Fix type annotations
- `code-security` - Fix security issues
- Other checks as needed

**Step 5: Run full test suite**

Run: `cd services/geometric && just test`

**Step 6: Commit fixes**

```bash
git add -A
git commit -m "fix(geometric): address CI check failures"
git push
```

---

## Task 19: Manual Verification

**Step 1: Start the service**

Run: `cd services/geometric && just run`

**Step 2: Test health endpoint**

Run: `curl http://localhost:8003/health`
Expected: `{"status":"healthy",...}`

**Step 3: Test info endpoint**

Run: `curl http://localhost:8003/info`
Expected: Service configuration including algorithm info

**Step 4: Test extract endpoint**

Run:
```bash
curl -X POST http://localhost:8003/extract \
  -H "Content-Type: application/json" \
  -d '{"image": "'$(base64 -i /path/to/test/image.jpg | tr -d '\n')'", "image_id": "test"}'
```
Expected: Feature extraction response

**Step 5: Test match endpoint**

Run:
```bash
curl -X POST http://localhost:8003/match \
  -H "Content-Type: application/json" \
  -d '{
    "query_image": "'$(base64 -i /path/to/image1.jpg | tr -d '\n')'",
    "reference_image": "'$(base64 -i /path/to/image2.jpg | tr -d '\n')'"
  }'
```
Expected: Match result with confidence score

**Step 6: Stop service**

Run: `cd services/geometric && just kill`

---

## Task 20: Final Commit

**Step 1: Create final commit**

```bash
git add -A
git commit -m "feat(geometric): complete geometric service implementation

Implements the Geometric Service for artwork-matcher with:
- ORB feature extraction
- Brute-force matching with Lowe's ratio test
- RANSAC homography verification
- Full API: /health, /info, /extract, /match, /match/batch
- Comprehensive unit and integration tests
- Full CI compliance"
git push
```

---

## Summary

This plan creates the Geometric Service following the same patterns as the embeddings service:

**Core Infrastructure (Tasks 1-5):**
- Configuration management with Pydantic + YAML
- Structured JSON logging
- Application state tracking
- Exception handlers
- Lifespan management

**Domain Logic (Tasks 6-9):**
- Pydantic schemas for API
- ORB feature extractor
- BF feature matcher with ratio test
- RANSAC geometric verifier

**API Layer (Tasks 10-14):**
- Health endpoint
- Info endpoint
- Extract endpoint
- Match/batch endpoints
- Application factory

**Testing (Tasks 16-17):**
- Test factories
- Unit tests with mocks
- Integration tests

**Validation (Tasks 18-19):**
- CI checks
- Manual verification

Each task is atomic and can be committed independently, following TDD principles where applicable.
