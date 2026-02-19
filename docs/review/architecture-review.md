# Beyond SOLID Architecture Review

> Scope: Full system — all 5 services, shared library, tools, infrastructure
> Method: Beyond SOLID — 10 system-level architecture principles

## Findings

---

### [Separation of Concerns] Gateway owns metadata data access — Severity: MEDIUM

**Location:** `services/gateway/src/gateway/routers/objects.py`, lines 25-70

**Issue:** The gateway loads artwork metadata by directly parsing a CSV file (`labels.csv`). This mixes data access concerns into the API orchestration layer. Metadata about museum objects is a data management responsibility that belongs in the storage service, which already owns binary object data. The gateway should retrieve metadata through the storage service API, not read files directly.

**Suggestion:** Add a metadata endpoint to the storage service (e.g., `GET /objects/{id}/metadata`) and have the gateway proxy to it, matching the pattern already used for images (`/objects/{id}/image`).

---

### [Single Responsibility] Gateway owns confidence scoring domain logic — Severity: LOW

**Location:** `services/gateway/src/gateway/routers/identify.py`, lines 75-108

**Issue:** The gateway owns confidence scoring domain logic (`calculate_confidence`) that combines embedding similarity with geometric verification scores using configurable weights. This is genuine domain logic, not orchestration. However, it is the *only* place that can logically own this computation since it combines outputs from multiple services. At current scale, this is pragmatic.

**Suggestion:** No immediate action needed. If scoring evolves into a complex model (ML-based reranking, additional verification stages), extract it into a dedicated scoring module or service. The config-driven approach (`ScoringConfig`) already enables tuning without code changes, which is good design.

---

### [DRY] Embedding dimension configured independently in two services — Severity: MEDIUM

**Location:** `services/embeddings/config.yaml` (embedding_dimension: 768) and `services/search/config.yaml` (embedding_dimension: 768)

**Issue:** The embedding dimension (768) is independently configured in both the embeddings and search services. This is genuine knowledge duplication — if DINOv2 is swapped for a different model producing different-dimension vectors, both configs must be updated in lockstep. A mismatch would cause silent 400 errors at runtime (`dimension_mismatch` from search service). The `build_index.py` tool also does not validate dimensional consistency between services.

**Suggestion:** This is an acceptable bounded-context trade-off (each service validates its own dimension). However, add a startup or health-check assertion in the gateway that queries both `/info` endpoints and verifies `embeddings.embedding_dimension == search.embedding_dimension`. This acts as an architectural fitness function catching drift at startup rather than at query time.

---

### [Resilience] No server-side operation timeouts on backend services — Severity: MEDIUM

**Location:** `services/embeddings/src/embeddings_service/routers/embed.py`, lines 179-205; `services/geometric/` (all matching endpoints)

**Issue:** Backend services have no server-side timeout on their own computationally expensive operations. The embeddings service performs DINOv2 inference and the geometric service performs ORB extraction + RANSAC verification without any server-side time bounds. If the model hangs or RANSAC enters a pathological case, the request blocks indefinitely — only the gateway's 30-second client timeout provides protection. This creates a scenario where the backend holds resources (GPU memory, CPU threads) for uncontrolled duration.

**Suggestion:** Add server-side request timeouts using FastAPI middleware or `asyncio.wait_for()` around inference and verification operations. Set these slightly below the gateway's client timeout (e.g., 25 seconds) so the backend can return a clean error rather than being killed by a client disconnect.

---

### [Resilience] No distributed tracing or request correlation — Severity: MEDIUM

**Location:** System-wide

**Issue:** Each service logs independently with structured JSON, but there is no request correlation across the pipeline. When a `/identify` request traverses gateway, embeddings, search, geometric, and storage, there is no way to trace a single user request across service logs. Debugging production issues requires manually matching timestamps. The optional `image_id` field exists in some schemas but is not propagated consistently as a correlation ID.

**Suggestion:** Add a middleware in the gateway that generates a `X-Request-ID` header and propagates it to all downstream calls. Each service should extract and include this ID in all log entries. This is a low-cost, high-value observability improvement that does not require full OpenTelemetry adoption.

---

### [POLA] Inconsistent error response format in gateway objects endpoints — Severity: LOW

**Location:** `services/gateway/src/gateway/routers/objects.py`, lines 119-127

**Issue:** The `/objects/{object_id}` endpoint raises `HTTPException` with a `detail` dict, while all other endpoints across all services raise `ServiceError`. FastAPI wraps `HTTPException.detail` in a `{"detail": {...}}` envelope, producing a response structure that differs from the uniform `{"error": ..., "message": ..., "details": ...}` pattern used everywhere else. A client parsing gateway errors must handle two different error shapes.

**Suggestion:** Replace `HTTPException` with `ServiceError` (or the gateway's `BackendError`) to maintain the uniform error response format. This is a one-line fix per endpoint.

---

### [Evolvability] No API versioning — Severity: LOW

**Location:** System-wide

**Issue:** No API versioning exists across any service. The `docs/api/uniform_api_structure.md` acknowledges this as a future concern. For internal APIs owned by a single team, this is acceptable. However, the gateway exposes a public-facing API (`/identify`, `/objects`) that external clients or a future web frontend will consume. Breaking changes would require coordinated client updates.

**Suggestion:** No immediate action needed for internal services. When the web frontend ships, consider adding versioning to the gateway's public endpoints (`/v1/identify`) or use additive-only change policies with deprecation headers.

---

### [Coupling] Sequential pipeline with batch geometric verification — Severity: LOW

**Location:** `services/gateway/src/gateway/routers/identify.py`, lines 229-271

**Issue:** The identification pipeline executes as a sequential synchronous chain: embed, search, geometric. While the sequential dependency between steps is inherent (each step needs the previous output), the geometric verification of multiple candidates (`match_batch`) sends all references to the geometric service in a single batch call. If the batch grows large (many candidates), this becomes a latency bottleneck. The gateway has no ability to parallelize or time-bound individual candidate verification.

**Suggestion:** The current batch approach is well-designed for the expected workload (k defaults to 5 candidates). If k increases significantly, consider streaming results or setting a geometric timeout budget per candidate.

---

### [KISS] No violation

**Location:** System-wide

The five-service decomposition is well-justified: each service has genuinely different dependencies (PyTorch, FAISS, OpenCV, file I/O, httpx) that would bloat a monolith. Technology choices are appropriately sized — flat FAISS index for exact search, file-based storage, FastAPI. The circuit breaker and retry infrastructure in the gateway is moderately sophisticated but well-implemented with clean configuration.

---

### [YAGNI] No violation

**Location:** System-wide

No speculative generality detected. No unused extension points, single-implementation interfaces, or infrastructure beyond what the system requires. The per-request `IdentifyOptions` override is justified by the evaluation pipeline. The `service-commons` shared library contains only genuinely shared infrastructure.

---

### [Law of Demeter] No violation

**Location:** System-wide

Services interact through well-defined HTTP APIs with clean client abstractions. No transitive service dependencies — the gateway orchestrates all calls directly. No schema reach-through or database sharing between services.

---

## Summary

| Principle | HIGH | MEDIUM | LOW |
|-----------|------|--------|-----|
| Separation of Concerns | 0 | 1 | 0 |
| Single Responsibility | 0 | 0 | 1 |
| DRY | 0 | 1 | 0 |
| Law of Demeter | 0 | 0 | 0 |
| Loose Coupling | 0 | 0 | 1 |
| Evolvability | 0 | 0 | 1 |
| Resilience | 0 | 2 | 0 |
| KISS | 0 | 0 | 0 |
| POLA | 0 | 0 | 1 |
| YAGNI | 0 | 0 | 0 |
| **Total** | **0** | **4** | **4** |

## Top 3 Priorities

1. **Distributed tracing / correlation IDs** (Resilience, MEDIUM) — Highest ROI for the lowest cost. A single middleware generating `X-Request-ID` and propagating it through all downstream calls dramatically improves debuggability. Without it, tracing a production issue across 5 services requires manual timestamp correlation.

2. **Server-side operation timeouts** (Resilience, MEDIUM) — Backend services should not rely solely on gateway client timeouts for protection. Adding `asyncio.wait_for()` around model inference and RANSAC verification prevents resource exhaustion from pathological inputs and gives backends control over their own failure modes.

3. **Embedding dimension consistency check** (DRY, MEDIUM) — Add a gateway startup assertion that cross-validates `embeddings_service.embedding_dimension == search_service.embedding_dimension` via their `/info` endpoints. This catches configuration drift as an architectural fitness function rather than allowing silent runtime failures.

## Overall Assessment

The Artwork Matcher has strong architectural health. The service decomposition follows business capabilities cleanly, cross-cutting concerns are properly centralized in `service-commons`, and the gateway implements a mature resilience stack (circuit breakers, retries, graceful degradation). The codebase is notably free of over-engineering — technology choices are appropriate to the problem scale and there is no speculative generality. The most significant gap is in observability: the lack of distributed tracing will become a real pain point as the system moves toward production. The two resilience findings are defensive improvements that would harden an already-solid design. Overall, this is a well-structured system that should evolve gracefully.
