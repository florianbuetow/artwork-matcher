# Embeddings Service API Specification

## Overview

The Embeddings Service extracts visual embedding vectors from artwork images using DINOv2 foundation models. It is a single-purpose microservice designed for high reliability and simple integration.

**Service Identity:**
- **Name:** embeddings-service
- **Default Port:** 8001
- **Protocol:** HTTP/REST + JSON

**See Also:** [Uniform API Structure](uniform_api_structure.md) for common endpoints, error handling, and conventions.

---

## Design Decisions

### Single Image vs Batch Processing

**Decision: Single image per request**

We evaluated two approaches for the `/embed` endpoint:

| Approach | Pros | Cons |
|----------|------|------|
| Single image | Simple error handling, predictable memory, easy testing | More HTTP round-trips for bulk operations |
| Batch images | Fewer round-trips, better GPU utilization | Complex partial failure handling, memory spikes, payload size limits |

**Rationale for single-image:**

1. **Use case alignment** — Runtime identification processes one visitor photo at a time. Batch processing is only needed for index building, which is an offline operation run once.

2. **Payload size constraints** — Base64 encoding adds ~33% overhead. High-resolution museum photos can be large:

   | Image Type | Raw Size | Base64 Encoded |
   |------------|----------|----------------|
   | 1080p JPEG (quality 85) | ~500 KB | ~670 KB |
   | 4K JPEG (quality 85) | ~3-5 MB | ~4-7 MB |
   | Batch of 20 × 4K images | ~60-100 MB | ~80-140 MB |

   Common infrastructure limits:
   - nginx default: 1 MB (configurable)
   - AWS API Gateway: 10 MB hard limit
   - Azure API Management: 4 MB default
   - Many cloud load balancers: 10-32 MB

   A batch of high-resolution images easily exceeds these limits.

3. **Error handling simplicity** — With single images, one request = one success or one failure. Batch processing requires handling partial failures: "images 1, 3, 7 succeeded; images 2, 4 failed." This adds complexity to both service and client code.

4. **Memory predictability** — Single-image requests have bounded memory usage. Batch requests can spike unpredictably based on image sizes and count.

5. **Project scope** — With 20 reference objects, index building makes 20 sequential requests taking ~10-20 seconds total. This is not a bottleneck worth optimizing.

**When to reconsider:** If scaling to millions of artworks, add a separate `/embed/batch` endpoint or use async job processing. Keep the simple `/embed` endpoint for runtime use.

### Data Format: Base64 in JSON

**Decision: Base64-encoded images in JSON request body**

Alternatives considered:

| Format | Pros | Cons |
|--------|------|------|
| multipart/form-data | Native binary, no encoding overhead | Harder to test with curl, complex parsing |
| Base64 in JSON | Simple, curl-friendly, uniform request/response | 33% size overhead |
| URL reference | Tiny payload | Requires image hosting, adds latency, security concerns |

**Rationale for Base64:**

1. **Testing simplicity** — Easy to test with curl:
   ```bash
   curl -X POST http://localhost:8001/embed \
     -H "Content-Type: application/json" \
     -d "{\"image\": \"$(base64 -i artwork.jpg)\"}"
   ```

2. **Uniform API** — All endpoints use `application/json` for both request and response. No special handling for different content types.

3. **Acceptable overhead** — For single images under 5 MB, the 33% Base64 overhead is negligible compared to model inference time (~50-200ms).

4. **Gateway compatibility** — JSON payloads pass through API gateways, proxies, and logging systems without special configuration.

### Model Configurability

**Decision: Model selection is deployment-time configuration, not runtime parameter**

The model (and thus embedding dimension) is set in `config.yaml` and fixed for the service lifetime. Clients discover the active configuration via `/info`.

**Rationale:**

1. **Resource management** — Loading a DINOv2 model requires ~500MB-2GB of memory. Runtime model switching would require either:
   - Keeping multiple models in memory (expensive)
   - Loading models on-demand (slow, ~10-30 seconds)

2. **Index compatibility** — FAISS indices are built for a specific embedding dimension. Mixing embeddings from different models corrupts the index.

3. **Operational simplicity** — One model per deployment. Different models = different service instances.

4. **Explicit over implicit** — The `/info` endpoint exposes the active model. Clients can validate compatibility at startup rather than discovering mismatches at runtime.

---

## API Endpoints

### GET /health

Health check endpoint for container orchestration (Docker health checks, Kubernetes probes).

**Response: 200 OK**

```json
{
  "status": "healthy"
}
```

**Status Values:**

| Status | Meaning | HTTP Code |
|--------|---------|-----------|
| `healthy` | Service fully operational | 200 |
| `degraded` | Operational but with issues (e.g., slow responses) | 200 |
| `unhealthy` | Not ready to serve requests | 503 |

**Design Note:** Health checks should be fast and not load the model. This endpoint only verifies the HTTP server is responding.

---

### GET /info

Returns service metadata including model configuration. Used by other services to validate compatibility.

**Response: 200 OK**

```json
{
  "service": "embeddings",
  "version": "0.1.0",
  "model": {
    "name": "facebook/dinov2-base",
    "embedding_dimension": 768,
    "device": "mps"
  },
  "preprocessing": {
    "image_size": 518,
    "normalize": true
  }
}
```

**Field Descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| `service` | string | Service identifier |
| `version` | string | Service version (semver) |
| `model.name` | string | HuggingFace model identifier |
| `model.embedding_dimension` | integer | Output vector dimension |
| `model.device` | string | Compute device (cpu, cuda, mps) |
| `preprocessing.image_size` | integer | Input image resize target |
| `preprocessing.normalize` | boolean | Whether ImageNet normalization is applied |

**Usage Example — Startup Validation:**

```python
def validate_embeddings_service(embeddings_url: str, expected_dimension: int) -> None:
    """Validate embeddings service compatibility at startup."""
    response = httpx.get(f"{embeddings_url}/info")
    response.raise_for_status()
    info = response.json()
    
    actual_dimension = info["model"]["embedding_dimension"]
    if actual_dimension != expected_dimension:
        raise ConfigurationError(
            f"Embedding dimension mismatch: service provides {actual_dimension}, "
            f"but search index expects {expected_dimension}"
        )
```

---

### POST /embed

Extract a normalized embedding vector from an image.

**Request:**

```
POST /embed
Content-Type: application/json
```

```json
{
  "image": "<base64-encoded-image-data>",
  "image_id": "object_001"
}
```

**Request Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | string (base64) | Yes | Base64-encoded image data |
| `image_id` | string | No | Optional identifier for logging/tracing |

**Supported Image Formats:**
- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)

**Response: 200 OK**

```json
{
  "embedding": [0.0234, -0.0891, 0.0412, ...],
  "dimension": 768,
  "image_id": "object_001",
  "processing_time_ms": 47.3
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `embedding` | array[float] | L2-normalized embedding vector |
| `dimension` | integer | Vector dimension (for client validation) |
| `image_id` | string \| null | Echo of input image_id |
| `processing_time_ms` | float | Server-side processing time |

**Embedding Properties:**
- **Normalization:** L2-normalized (unit length), suitable for cosine similarity via inner product
- **Dimension:** Determined by model configuration (768 for dinov2-base, 1024 for dinov2-large)
- **Data type:** 32-bit floats

---

## Error Handling

All errors return a consistent JSON structure:

```json
{
  "error": "error_code",
  "message": "Human-readable description",
  "details": {}
}
```

### Error Codes

| Error Code | HTTP Status | Description | Client Action |
|------------|-------------|-------------|---------------|
| `invalid_image` | 400 | Image data cannot be decoded | Verify image file is valid |
| `unsupported_format` | 400 | Image format not supported | Convert to JPEG/PNG/WebP |
| `decode_error` | 400 | Base64 decoding failed | Check Base64 encoding |
| `payload_too_large` | 413 | Request exceeds size limit | Reduce image size/quality |
| `model_error` | 500 | Model inference failed | Retry; report if persistent |
| `internal_error` | 500 | Unexpected server error | Retry with backoff |

### Error Response Examples

**Invalid Image (400):**

```json
{
  "error": "invalid_image",
  "message": "Failed to decode image: cannot identify image file",
  "details": {
    "image_id": "object_001"
  }
}
```

**Unsupported Format (400):**

```json
{
  "error": "unsupported_format",
  "message": "Image format 'image/tiff' is not supported. Use JPEG, PNG, or WebP.",
  "details": {
    "detected_format": "image/tiff",
    "supported_formats": ["image/jpeg", "image/png", "image/webp"]
  }
}
```

**Base64 Decode Error (400):**

```json
{
  "error": "decode_error",
  "message": "Invalid Base64 encoding: incorrect padding",
  "details": {}
}
```

---

## Size Estimations

### Request Payload Sizes

| Scenario | Image Size | Base64 Size | With JSON Overhead |
|----------|------------|-------------|-------------------|
| Typical museum photo (1080p JPEG) | 500 KB | 667 KB | ~670 KB |
| High-res scan (4K JPEG) | 3 MB | 4 MB | ~4 MB |
| Maximum recommended | 10 MB | 13.3 MB | ~13.5 MB |

**Recommendation:** Configure a 20 MB request size limit at the reverse proxy level. This accommodates high-resolution images while preventing abuse.

### Response Payload Sizes

| Model | Dimension | Embedding Size (JSON) |
|-------|-----------|----------------------|
| dinov2-base | 768 | ~15 KB |
| dinov2-large | 1024 | ~20 KB |

Response payloads are small and consistent regardless of input image size.

### Memory Usage

| Component | Memory |
|-----------|--------|
| DINOv2-base model | ~400 MB |
| DINOv2-large model | ~1.2 GB |
| Per-request overhead | ~50-100 MB (image + tensors) |
| **Recommended container limit** | **2-4 GB** |

---

## Configuration

The service is configured via `config.yaml` with environment variable overrides.

### Configuration Schema

```yaml
# config.yaml
model:
  name: "facebook/dinov2-base"      # HuggingFace model identifier
  device: "auto"                     # auto | cpu | cuda | mps
  embedding_dimension: 768           # Must match model output

preprocessing:
  image_size: 518                    # DINOv2 native size
  normalize: true                    # Apply ImageNet normalization

server:
  host: "0.0.0.0"
  port: 8000
  max_request_size_mb: 20
  log_level: "info"
```

### Environment Variable Overrides

Environment variables use double-underscore nesting with `EMBEDDINGS__` prefix:

```bash
EMBEDDINGS__MODEL__NAME=facebook/dinov2-large
EMBEDDINGS__MODEL__DEVICE=cuda
EMBEDDINGS__SERVER__PORT=8001
```

### Supported Models

| Model | Dimension | Memory | Speed | Recommendation |
|-------|-----------|--------|-------|----------------|
| `facebook/dinov2-small` | 384 | ~200 MB | Fast | Development/testing |
| `facebook/dinov2-base` | 768 | ~400 MB | Medium | **Default choice** |
| `facebook/dinov2-large` | 1024 | ~1.2 GB | Slow | Maximum accuracy |

---

## Usage Examples

### curl

```bash
# Health check
curl http://localhost:8001/health

# Service info
curl http://localhost:8001/info

# Extract embedding (macOS/Linux)
curl -X POST http://localhost:8001/embed \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$(base64 -i artwork.jpg | tr -d '\n')\", \"image_id\": \"test_001\"}"

# Extract embedding with jq for pretty output
curl -s -X POST http://localhost:8001/embed \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$(base64 -i artwork.jpg | tr -d '\n')\"}" | jq '.dimension'
```

### Python (httpx)

```python
import base64
import httpx
from pathlib import Path

def extract_embedding(image_path: Path, base_url: str = "http://localhost:8001") -> list[float]:
    """Extract embedding from an image file."""
    image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    
    response = httpx.post(
        f"{base_url}/embed",
        json={
            "image": image_data,
            "image_id": image_path.stem,
        },
        timeout=30.0,
    )
    response.raise_for_status()
    
    result = response.json()
    return result["embedding"]


# Usage
embedding = extract_embedding(Path("artwork.jpg"))
print(f"Extracted {len(embedding)}-dimensional embedding")
```

### Python (async with httpx)

```python
import asyncio
import base64
import httpx
from pathlib import Path

async def extract_embeddings_batch(
    image_paths: list[Path],
    base_url: str = "http://localhost:8001",
    max_concurrent: int = 4,
) -> dict[str, list[float]]:
    """Extract embeddings from multiple images with controlled concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def extract_one(client: httpx.AsyncClient, path: Path) -> tuple[str, list[float]]:
        async with semaphore:
            image_data = base64.b64encode(path.read_bytes()).decode("ascii")
            response = await client.post(
                f"{base_url}/embed",
                json={"image": image_data, "image_id": path.stem},
            )
            response.raise_for_status()
            return path.stem, response.json()["embedding"]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [extract_one(client, path) for path in image_paths]
        results = await asyncio.gather(*tasks)
    
    return dict(results)
```

---

## Performance Characteristics

### Expected Latencies

| Operation | Typical | P99 |
|-----------|---------|-----|
| Image decode + preprocess | 5-20 ms | 50 ms |
| Model inference (CPU) | 200-500 ms | 1000 ms |
| Model inference (MPS/Apple Silicon) | 50-100 ms | 200 ms |
| Model inference (CUDA) | 20-50 ms | 100 ms |
| **Total (MPS)** | **60-120 ms** | **250 ms** |

### Throughput Estimates

| Device | Requests/sec (sustained) |
|--------|-------------------------|
| CPU (4 cores) | 2-4 |
| Apple M1/M2 (MPS) | 8-15 |
| NVIDIA T4 (CUDA) | 20-40 |
| NVIDIA A100 (CUDA) | 50-100 |

For this project (20 objects + 17 test images), any device completes the full workload in under 30 seconds.

---

## OpenAPI Specification

```yaml
openapi: 3.1.0
info:
  title: Embeddings Service
  description: DINOv2 embedding extraction for artwork images
  version: 0.1.0
  contact:
    name: Artwork Matcher
servers:
  - url: http://localhost:8001
    description: Local development

paths:
  /health:
    get:
      operationId: health_check
      summary: Health check
      description: Returns service health status for orchestration probes.
      tags: [Operations]
      responses:
        "200":
          description: Service status
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/HealthResponse"

  /info:
    get:
      operationId: get_info
      summary: Service information
      description: Returns service metadata including model configuration.
      tags: [Operations]
      responses:
        "200":
          description: Service information
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/InfoResponse"

  /embed:
    post:
      operationId: extract_embedding
      summary: Extract embedding
      description: |
        Extracts a normalized embedding vector from the provided image.
        The embedding dimension depends on the configured model.
      tags: [Embeddings]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/EmbedRequest"
      responses:
        "200":
          description: Embedding extracted successfully
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/EmbedResponse"
        "400":
          description: Invalid request
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"
              examples:
                invalid_image:
                  summary: Invalid image data
                  value:
                    error: invalid_image
                    message: "Failed to decode image: cannot identify image file"
                    details: {}
                unsupported_format:
                  summary: Unsupported format
                  value:
                    error: unsupported_format
                    message: "Image format 'image/tiff' is not supported"
                    details:
                      detected_format: image/tiff
                      supported_formats: [image/jpeg, image/png, image/webp]
        "413":
          description: Payload too large
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"
        "500":
          description: Server error
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"

components:
  schemas:
    HealthResponse:
      type: object
      required: [status]
      properties:
        status:
          type: string
          enum: [healthy, degraded, unhealthy]
          example: healthy

    InfoResponse:
      type: object
      required: [service, version, model, preprocessing]
      properties:
        service:
          type: string
          example: embeddings
        version:
          type: string
          example: 0.1.0
        model:
          type: object
          required: [name, embedding_dimension, device]
          properties:
            name:
              type: string
              example: facebook/dinov2-base
            embedding_dimension:
              type: integer
              example: 768
            device:
              type: string
              example: mps
        preprocessing:
          type: object
          required: [image_size, normalize]
          properties:
            image_size:
              type: integer
              example: 518
            normalize:
              type: boolean
              example: true

    EmbedRequest:
      type: object
      required: [image]
      properties:
        image:
          type: string
          format: byte
          description: Base64-encoded image (JPEG, PNG, or WebP)
        image_id:
          type: string
          description: Optional identifier for logging/tracing
          example: object_001

    EmbedResponse:
      type: object
      required: [embedding, dimension]
      properties:
        embedding:
          type: array
          items:
            type: number
            format: float
          description: L2-normalized embedding vector
        dimension:
          type: integer
          description: Embedding dimension
          example: 768
        image_id:
          type: string
          nullable: true
          description: Echo of input image_id
          example: object_001
        processing_time_ms:
          type: number
          format: float
          description: Processing time in milliseconds
          example: 47.3

    ErrorResponse:
      type: object
      required: [error, message]
      properties:
        error:
          type: string
          enum:
            - invalid_image
            - unsupported_format
            - decode_error
            - payload_too_large
            - model_error
            - internal_error
        message:
          type: string
          description: Human-readable error description
        details:
          type: object
          additionalProperties: true
          description: Additional error context
```

---

## Appendix: Why DINOv2?

DINOv2 was selected for this project based on the following criteria:

1. **Artwork-validated performance** — Meta specifically benchmarked DINOv2 on the Met Museum dataset, achieving +34% mAP improvement over baselines.

2. **Self-supervised training** — Unlike CLIP, DINOv2 learns from visual patterns alone without text supervision. This avoids biases from image-caption datasets that may not represent fine art well.

3. **Robustness to variations** — DINOv2 handles lighting changes, viewpoint differences, and partial occlusions well—exactly the challenges in visitor photos.

4. **Practical deployment** — Available via HuggingFace Transformers with a single `pip install`. No custom training or fine-tuning required.

5. **Appropriate model sizes** — The base model (768-dim) provides excellent accuracy in ~400MB. Larger models available if needed.

DINOv2-base demonstrates awareness of current best practices while being practical to implement and evaluate.
