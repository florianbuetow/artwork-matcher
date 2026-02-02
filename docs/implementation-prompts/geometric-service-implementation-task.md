# Geometric Service Implementation Task

## Overview

Implement the **Geometric Service** for the artwork-matcher project. This service provides local feature extraction and geometric verification using classical computer vision techniques. It extracts ORB features from images and verifies spatial consistency between image pairs using RANSAC-based homography estimation.

**Service Identity:**
- **Name:** geometric-service
- **Port:** 8003
- **Technology:** OpenCV (ORB feature detection, BFMatcher, RANSAC homography)

---

## Reference Documentation

Read and follow these documents in order:

1. **@docs/api/geometric_service_api_spec.md** - Complete API specification including:
   - All endpoints: `/health`, `/info`, `/extract`, `/match`, `/match/batch`
   - Request/response schemas with examples
   - Error codes and handling
   - Feature detector comparison (why ORB)
   - Matching strategy (Lowe's ratio test)
   - RANSAC homography verification
   - Confidence score calculation

2. **@docs/api/uniform_api_structure.md** - Common API conventions:
   - Health/info endpoint patterns
   - Error response format (`error`, `message`, `details`)
   - Processing time tracking
   - HTTP status codes

3. **@docs/implementation-guides/fastapi_service_template.md** - Service structure:
   - Directory layout (`src/geometric_service/...`)
   - Configuration management (Pydantic + YAML, no defaults in code)
   - Structured JSON logging
   - Application factory pattern
   - Router organization

4. **@docs/implementation-guides/config_pattern.md** - Configuration pattern:
   - How to load YAML and validate with Pydantic
   - Environment variable overrides
   - Fail-fast on invalid config

5. **@CLAUDE.md** - Development rules:
   - Use `uv run` for all Python execution
   - Run tests after every change
   - No hardcoded defaults
   - Git commit guidelines

6. **@docs/implementation-guides/service_testing_guide.md** - Testing patterns:
   - Three-tier testing (unit, integration, performance)
   - What to test in each tier
   - Justfile recipes and pytest markers
   - Reference: embeddings service tests

---

## Existing Files

The following already exist and should be used:
- `services/geometric/config.yaml` - Configuration (already populated)
- `services/geometric/pyproject.toml` - Dependencies including `opencv-python`
- `services/geometric/justfile` - Build/run commands
- `services/geometric/src/geometric_service/__init__.py` - Package marker

---

## Files to Create

Following the structure in `@docs/implementation-guides/fastapi_service_template.md`:

```
services/geometric/src/geometric_service/
├── __init__.py          (exists)
├── main.py              # Entry point with main()
├── app.py               # FastAPI app factory
├── config.py            # Pydantic settings loading config.yaml
├── logging.py           # Structured JSON logging
├── schemas.py           # Pydantic request/response models
├── core/
│   ├── __init__.py
│   ├── lifespan.py      # Startup / shutdown
│   ├── state.py         # App state (uptime)
│   └── exceptions.py    # ServiceError and handlers
├── routers/
│   ├── __init__.py
│   ├── health.py        # GET /health
│   ├── info.py          # GET /info
│   ├── extract.py       # POST /extract
│   └── match.py         # POST /match, POST /match/batch
└── services/
    ├── __init__.py
    ├── feature_extractor.py  # ORB feature extraction
    ├── feature_matcher.py    # BFMatcher + ratio test
    └── geometric_verifier.py # RANSAC homography
```

Also create:
- `services/geometric/tests/conftest.py` - Test fixtures
- `services/geometric/tests/test_health.py` - Health endpoint tests
- `services/geometric/tests/test_extract.py` - Feature extraction tests
- `services/geometric/tests/test_match.py` - Matching tests

---

## Key Implementation Details

### 1. Feature Extractor (`services/feature_extractor.py`)

```python
class ORBFeatureExtractor:
    """Extract ORB features from images."""

    def __init__(self, config: ORBConfig):
        self.orb = cv2.ORB_create(
            nfeatures=config.max_features,
            scaleFactor=config.scale_factor,
            nlevels=config.n_levels,
            edgeThreshold=config.edge_threshold,
            patchSize=config.patch_size,
            fastThreshold=config.fast_threshold,
        )

    def extract(self, image_bytes: bytes) -> tuple[list[dict], np.ndarray, tuple[int, int]]:
        """
        Extract features from image bytes.

        Returns:
            - keypoints: List of {x, y, size, angle}
            - descriptors: numpy array (N, 32)
            - image_size: (width, height)
        """
        ...
```

### 2. Feature Matcher (`services/feature_matcher.py`)

```python
class BFFeatureMatcher:
    """Match features using brute-force with ratio test."""

    def __init__(self, ratio_threshold: float):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.ratio_threshold = ratio_threshold

    def match(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
    ) -> list[cv2.DMatch]:
        """
        Match descriptors using kNN + Lowe's ratio test.

        Returns:
            List of good matches after ratio test filtering.
        """
        ...
```

### 3. Geometric Verifier (`services/geometric_verifier.py`)

```python
class RANSACVerifier:
    """Verify geometric consistency using RANSAC homography."""

    def __init__(self, config: RANSACConfig, verification_config: VerificationConfig):
        self.reproj_threshold = config.reproj_threshold
        self.max_iters = config.max_iters
        self.confidence = config.confidence
        self.min_inliers = verification_config.min_inliers

    def verify(
        self,
        kp1: list[cv2.KeyPoint],
        kp2: list[cv2.KeyPoint],
        matches: list[cv2.DMatch],
    ) -> dict:
        """
        Verify geometric consistency.

        Returns:
            {
                "is_match": bool,
                "inliers": int,
                "total_matches": int,
                "inlier_ratio": float,
                "homography": list[list[float]] | None,
                "confidence": float,
            }
        """
        ...
```

### 4. Confidence Score Calculation

```python
def calculate_confidence(inliers: int, inlier_ratio: float) -> float:
    """
    Combine inlier count and ratio into confidence score.

    - High inliers + high ratio = high confidence
    - High inliers + low ratio = medium confidence
    - Low inliers + high ratio = low confidence
    """
    # Normalize inlier count (saturates at 100)
    inlier_score = min(inliers / 100, 1.0)

    # Weight: 70% inlier count, 30% inlier ratio
    confidence = 0.7 * inlier_score + 0.3 * inlier_ratio

    return round(confidence, 2)
```

### 5. Configuration (use existing `config.yaml`)

```yaml
orb:
  max_features: 1000
  scale_factor: 1.2
  n_levels: 8
  edge_threshold: 31
  patch_size: 31
  fast_threshold: 20

matching:
  ratio_threshold: 0.75
  cross_check: false

ransac:
  reproj_threshold: 5.0
  max_iters: 2000
  confidence: 0.995

verification:
  min_features: 50
  min_matches: 20
  min_inliers: 10

server:
  host: "0.0.0.0"
  port: 8000
  log_level: "info"
```

### 6. Error Handling

| Error Code | HTTP Status | When |
|------------|-------------|------|
| `invalid_image` | 400 | Image cannot be decoded |
| `decode_error` | 400 | Base64 decoding failed |
| `insufficient_features` | 422 | Too few features for matching |
| `no_matches` | 422 | No feature matches found |
| `homography_failed` | 422 | RANSAC could not find valid homography |
| `invalid_features` | 400 | Pre-extracted features are malformed |

### 7. Image Handling

```python
def decode_image(image_b64: str) -> np.ndarray:
    """Decode base64 image to OpenCV format."""
    image_bytes = base64.b64decode(image_b64)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ServiceError(
            error="invalid_image",
            message="Failed to decode image data",
            status_code=400,
        )
    return image
```

### 8. Descriptor Serialization

ORB descriptors are binary (32 bytes each). Serialize for API responses:

```python
# Encode
descriptors_b64 = base64.b64encode(descriptors.tobytes()).decode()

# Decode
desc_bytes = base64.b64decode(descriptors_b64)
descriptors = np.frombuffer(desc_bytes, dtype=np.uint8).reshape(-1, 32)
```

---

## Validation Checklist

After implementation, verify:

- [ ] `just init` - Environment initializes
- [ ] `just test` - All tests pass
- [ ] `just ci` - All CI checks pass (style, types, security)
- [ ] `just run` - Service starts on port 8003
- [ ] `curl http://localhost:8003/health` returns `{"status": "healthy"}`
- [ ] `curl http://localhost:8003/info` returns algorithm config
- [ ] Can extract features from an image via POST /extract
- [ ] Can match two images via POST /match
- [ ] Can batch match via POST /match/batch
- [ ] Matching returns `is_match: true` for same artwork, `is_match: false` for different

---

## Reference Implementation

Use **@services/embeddings/** as a reference for the service structure. It follows the same patterns described in the template guide.

---

## Testing Notes

For testing, you'll need sample images. Create fixtures that:
1. Load a small test image (can be a simple geometric pattern)
2. Verify feature extraction returns keypoints and descriptors
3. Verify matching same image returns high inliers
4. Verify matching different images returns low/no inliers

Consider using `pytest.mark.parametrize` for different image scenarios.
