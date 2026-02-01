# Geometric Service

Provides local feature extraction and geometric verification using classical computer vision. Extracts ORB features and verifies spatial consistency between image pairs using RANSAC homography estimation.

## Quick Start

```bash
# From repository root
just run-geometric

# Or from this directory
just run
```

The service will be available at `http://localhost:8003`.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/info` | GET | Algorithm configuration |
| `/extract` | POST | Extract ORB features from image |
| `/match` | POST | Verify geometric consistency between two images |
| `/match/batch` | POST | Verify query against multiple references |

### Example: Match Two Images

```bash
curl -X POST http://localhost:8003/match \
  -H "Content-Type: application/json" \
  -d '{
    "query_image": "'$(base64 -i visitor_photo.jpg | tr -d '\n')'",
    "reference_image": "'$(base64 -i reference.jpg | tr -d '\n')'"
  }'
```

Response:
```json
{
  "is_match": true,
  "confidence": 0.85,
  "inliers": 67,
  "total_matches": 124,
  "inlier_ratio": 0.54,
  "processing_time_ms": 48.7
}
```

## Configuration

Configuration is loaded from `config.yaml` with environment variable overrides.

```yaml
# config.yaml
orb:
  max_features: 1000
  scale_factor: 1.2
  n_levels: 8

matching:
  ratio_threshold: 0.75    # Lowe's ratio test
  min_matches: 20

ransac:
  reproj_threshold: 5.0
  max_iters: 2000
  confidence: 0.995

verification:
  min_inliers: 10          # Minimum to declare match

server:
  host: "0.0.0.0"
  port: 8000
  log_level: "info"
```

### Environment Overrides

```bash
GEOMETRIC__ORB__MAX_FEATURES=2000
GEOMETRIC__MATCHING__RATIO_THRESHOLD=0.7
GEOMETRIC__VERIFICATION__MIN_INLIERS=15
```

## Development

```bash
# Initialize environment
just init

# Run with hot reload
just run

# Run tests
just test

# Run all CI checks
just ci
```

## How It Works

1. **Feature Extraction**: ORB detects keypoints and computes binary descriptors
2. **Feature Matching**: Brute-force matcher with Lowe's ratio test filters matches
3. **Geometric Verification**: RANSAC estimates homography, counts inliers
4. **Decision**: If inliers >= threshold, images show the same artwork

### Why ORB + RANSAC?

| Method | Speed | Accuracy | Patent-Free |
|--------|-------|----------|-------------|
| ORB | ~12ms | Good | Yes |
| SIFT | ~116ms | Excellent | Yes (expired) |
| SURF | ~60ms | Very Good | No |
| SuperPoint | ~30ms (GPU) | Excellent | Yes |

ORB provides the best speed/accuracy tradeoff for CPU-only deployment.

## API Specification

For complete API documentation, see [Geometric Service API Spec](../../docs/api/geometric_service_api_spec.md).