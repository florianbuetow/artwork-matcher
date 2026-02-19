# Artwork Matcher - Distributed System Tradeoff Analysis

> Scope: All five services (gateway, embeddings, search, geometric, storage), Docker Compose topology, tooling, and deployment documentation.

---

## Table of Contents

- [Consistency \& Availability](#consistency--availability)
- [Latency \& Throughput](#latency--throughput)
- [Data Distribution](#data-distribution)
- [Transaction Boundaries \& Coordination](#transaction-boundaries--coordination)
- [Resilience \& Failure Isolation](#resilience--failure-isolation)
- [Observability, Security \& Cost](#observability-security--cost)
- [Cross-Axis Synthesis](#cross-axis-synthesis)

---

## Consistency & Availability

**Position:** Availability-first with implicit single-node consistency. The system prioritizes request-serving availability through graceful degradation, circuit breakers, and retry logic, while relying on single-instance, in-process state for data consistency -- a stance that is largely emergent from framework defaults rather than a deliberate distributed consistency strategy.

**Confidence:** HIGH

### Evidence

1. **Gateway graceful degradation for geometric and storage backends** -- Tier B
   File: `services/gateway/src/gateway/routers/health.py`, lines ~58-63
   Embeddings and search are "critical"; geometric and storage are "optional." When optional backends are down, the gateway reports `"degraded"` rather than `"unhealthy"`. **Deliberate.**

2. **Identify pipeline catches geometric backend errors and continues** -- Tier B
   File: `services/gateway/src/gateway/routers/identify.py`, lines ~261-270
   When geometric verification is unavailable, the pipeline sets `degraded=True` and applies a `geometric_missing_penalty: 0.7`, returning lower-confidence but usable results. **Deliberate.**

3. **Circuit breaker pattern on all backend clients** -- Tier B
   File: `services/gateway/src/gateway/clients/base.py`, lines ~62-108; Config: `services/gateway/config.yaml`, lines ~16-21
   Custom circuit breaker (closed/open/half_open) with `failure_threshold: 5`, `recovery_timeout_seconds: 15.0`. **Deliberate** -- values explicitly configured.

4. **Retry with exponential backoff and jitter** -- Tier B
   File: `services/gateway/src/gateway/clients/base.py`, lines ~166-198
   Config: `max_attempts: 3`, `initial_backoff: 0.1s`, `max_backoff: 1.0s`, `jitter: 0.05s`. **Deliberate** -- tuned values.

5. **Single-instance FAISS index with in-memory state** -- Tier B
   File: `services/search/src/search_service/services/faiss_index.py`, lines ~77-105
   FAISS index lives in a single process's memory. Adds are in-memory until explicitly saved. No replication, no WAL, no distributed consistency protocol. **Default.**

6. **Storage service is a single-node filesystem blob store** -- Tier B
   File: `services/storage/src/storage_service/services/blob_store.py`, lines ~17-54
   `FileBlobStore` writes objects as `{key}.dat` files via `write_bytes()`. No journaling, no fsync, no atomic rename, no replication. Last-write-wins. **Default.**

7. **Gateway startup blocks on critical backends** -- Tier A
   File: `docker-compose.yml`, lines ~77-84
   Gateway `depends_on` embeddings, search, and geometric with `condition: service_healthy`. Storage is intentionally omitted. **Deliberate.**

8. **IdentifyResponse schema encodes degradation semantics** -- Tier A
   File: `services/gateway/src/gateway/schemas.py`, lines ~194-204
   Schema includes `geometric_skipped`, `geometric_skip_reason`, `degraded`, `degradation_reason`. The API contract formally encodes partial availability. **Deliberate.**

9. **No caching layer between services** -- Tier B
   No application-level caching between services found. Consistency-preserving but **default** -- caching was never introduced rather than freshness being a deliberate choice.

10. **No conflict resolution, vector clocks, or versioning** -- Tier B
    No distributed consistency primitives anywhere. The problem is not addressed because all services are single-instance. **Default.**

### Assessment

The system achieves what looks like "strong consistency" through the simplest mechanism: single instances with no replication. There are no replicas to disagree, no partitions between data copies, and therefore no CAP tradeoff at the data layer. The deliberate availability engineering (circuit breakers, degradation, penalty scoring) is well-crafted, but the absence of any distributed consistency mechanism means the moment this system scales to multiple instances, silent data divergence will immediately surface.

### Risks & Recommendations

**Risk -- Severity: HIGH**
Location: `services/search/src/search_service/services/faiss_index.py` (save method)
Issue: Two-file save (FAISS binary + JSON metadata) without atomicity. A crash between writes corrupts state.
Recommendation: Write-to-temp-then-rename pattern. Consider periodic auto-save or WAL.

**Risk -- Severity: HIGH**
Location: `services/storage/src/storage_service/services/blob_store.py`
Issue: `put()` uses `write_bytes()` -- not crash-safe. No locking for concurrent writes. `delete_all()` via `shutil.rmtree` is non-atomic.
Recommendation: Atomic write (temp + fsync + rename). Advisory lock for concurrent access.

**Risk -- Severity: MEDIUM**
Location: `docker-compose.yml` -- all services
Issue: Every service runs as exactly one instance with zero redundancy. A single process crash in embeddings or search makes the entire system unhealthy.
Recommendation: For production, define replicas and address the single-instance data assumptions.

---

## Latency & Throughput

**Position:** Latency-aware but unoptimized -- the system has latency visibility (per-stage timing, p50/p95/p99 measurement) and resilience mechanisms, but lacks throughput-optimizing patterns (batching, concurrent fanout, worker scaling). The architecture accepts higher per-request latency in exchange for implementation simplicity.

**Confidence:** MEDIUM

### Evidence

1. **Sequential four-stage pipeline with no parallelism** -- Tier B
   File: `services/gateway/src/gateway/routers/identify.py`, lines ~180-272
   Embeddings -> search -> storage fetches -> geometric verification as sequential `await` calls. Zero `asyncio.gather` anywhere in the gateway. **Default.**

2. **Sequential reference image fetching (N+1 pattern)** -- Tier B
   File: `services/gateway/src/gateway/routers/identify.py`, lines ~55-66
   `build_geometric_references()` fetches 5-10 reference images sequentially. These are independent and could be parallelized. **Default.**

3. **Sequential geometric batch processing** -- Tier B
   File: `services/geometric/src/geometric_service/routers/match.py`, lines ~165-227
   Each reference processed in a sequential loop. OpenCV releases the GIL, so parallelization is viable. **Default.**

4. **Unified 30-second timeout across all backends** -- Tier B
   File: `services/gateway/config.yaml`, line ~14
   Same timeout for FAISS search (<1ms), storage reads (~1ms), and GPU inference (50-500ms). **Default.**

5. **Per-stage timing instrumentation** -- Tier B
   File: `services/gateway/src/gateway/schemas.py`, lines ~138-151
   `TimingInfo` model breaks down `embedding_ms`, `search_ms`, `geometric_ms`, `total_ms`. **Deliberate.**

6. **Single uvicorn worker, no worker scaling** -- Tier B
   All Dockerfiles run `uvicorn` with no `--workers` flag (defaults to 1). CPU-bound services can only process one request at a time. **Default.**

7. **FAISS IndexFlatIP (brute-force)** -- Tier B
   File: `services/search/config.yaml`, line ~11
   Flat index with O(n) complexity. Config supports `ivf` and `hnsw` alternatives. **Deliberate** for current scale.

8. **Performance test suite with percentile tracking** -- Tier C
   All five services have performance tests measuring p50/p95/p99 at concurrency levels [2, 4, 8, 16]. No SLO thresholds. **Deliberate** but observational.

9. **No rate limiting, backpressure, or load shedding** -- Tier C
   No rate limiters, admission control, or request queuing anywhere. **Default.**

### Assessment

The system is designed with latency awareness but not latency optimization. The per-stage timing instrumentation and comprehensive performance test suite show the team understands latency as a concern. However, the sequential pipeline (7-12 serial HTTP calls per request), single-worker processes, undifferentiated timeouts, and absence of concurrency all point to a system optimized for correctness and clarity over performance -- reasonable for the current development stage.

### Risks & Recommendations

**Risk -- Severity: HIGH**
Location: `services/gateway/src/gateway/routers/identify.py`, `build_geometric_references()`
Issue: Sequential storage fetches add O(k * latency) to every request. With `search_k=10`, that's 10 sequential HTTP round-trips.
Recommendation: Replace with `asyncio.gather()` for concurrent fetches.

**Risk -- Severity: HIGH**
Location: All Dockerfiles
Issue: Single uvicorn worker for CPU-bound services (embeddings, geometric). Requests queue behind each other.
Recommendation: Use `--workers N` or gunicorn wrapper for CPU-bound services.

**Risk -- Severity: MEDIUM**
Location: `services/gateway/config.yaml`, line ~14
Issue: 30-second timeout for all backends regardless of expected latency profile.
Recommendation: Per-backend timeouts: storage 5s, search 5s, embeddings 15s, geometric 15s.

**Risk -- Severity: MEDIUM**
Location: Gateway -- no admission control
Issue: Unlimited concurrent requests all compete for the single-worker embeddings service, causing latency spikes.
Recommendation: Add `asyncio.Semaphore` on the identify endpoint, return 503 with `Retry-After` when full.

---

## Data Distribution

**Position:** Monolithic data locality with functionally-partitioned service ownership -- no sharding, no replication, heavily read-optimized with pre-computed indexes and stateful in-memory data structures.

**Confidence:** HIGH

### Evidence

1. **Shared `./data` volume mount** -- Tier B
   File: `docker-compose.yml`
   All five services mount the same host `./data` directory. Search and storage mount read-write; others read-only. Single-host shared filesystem -- no distributed data layer. **Default.**

2. **FAISS IndexFlatIP loaded entirely into process memory** -- Tier B
   File: `services/search/src/search_service/services/faiss_index.py`, lines ~77-106
   Entire vector index in one process. No partitioning, no distributed FAISS. **Deliberate** -- documented as sufficient for museum scale (<100K items).

3. **Single-node filesystem blob store** -- Tier B
   File: `services/storage/src/storage_service/services/blob_store.py`
   Objects stored as `{key}.dat` files in a local directory. No distributed storage backend. **Default.**

4. **Sequential, centralized index build pipeline** -- Tier B
   File: `tools/build_index.py`
   Single sequential loop: read image -> embed -> add to index -> upload to storage. No batch parallelism. **Default.**

5. **Gateway orchestrates scatter-gather read pipeline** -- Tier B
   File: `services/gateway/src/gateway/routers/identify.py`
   Sequential fan-out across all services. No caching, no pre-joined read model. **Deliberate.**

6. **AWS deployment guide specifies EFS for shared index** -- Tier A
   File: `docs/deployment/AWS_DEPLOYMENT_GUIDE.md`
   Production plan: S3 for images, EFS for FAISS index (shared filesystem, not distributed). **Deliberate** -- intentionally defers data distribution complexity.

### Assessment

This position is deliberate and appropriate for the current scale. The initial research explicitly concludes that FAISS IndexFlatIP is sufficient for museum-scale datasets. The system is read-optimized by design: pre-computed embeddings stored in a flat index, batch offline builds, read-only query pipeline. The AWS deployment guide proposes shared storage (EFS) rather than sharded indexes, intentionally deferring distribution complexity.

### Risks & Recommendations

**Risk -- Severity: MEDIUM**
Location: `services/search/src/search_service/services/faiss_index.py`
Issue: Entire index must fit in one process's memory. Reload from disk during crashes means 503s during recovery.
Recommendation: Startup probe delaying traffic until index loads; maintain 2+ instances with staggered rolling deployments.

**Risk -- Severity: MEDIUM**
Location: `services/storage/src/storage_service/services/blob_store.py`
Issue: Filesystem blob store cannot scale horizontally. Multiple instances writing to shared filesystem introduces contention.
Recommendation: Replace with S3-backed implementation for production (as deployment guide suggests).

---

## Transaction Boundaries & Coordination

**Position:** Stateless synchronous orchestration with no distributed transactions -- the system uses a fire-and-forget pipeline model where the gateway orchestrates a sequential, read-only workflow across independent services with graceful degradation as the sole failure-handling strategy.

**Confidence:** HIGH

### Evidence

1. **Five independently deployed services** -- Tier B
   File: `docker-compose.yml`
   Each service has its own container, port, and healthcheck. Decomposition follows clear domain boundaries. **Deliberate.**

2. **Synchronous orchestration pipeline with no compensation** -- Tier A
   File: `services/gateway/src/gateway/routers/identify.py`, lines ~137-336
   Sequential pipeline: embed -> search -> geometric verify -> score -> return. Zero compensation handlers, no outbox, no saga, no rollback logic (confirmed by global grep). **Deliberate** -- the pipeline is read-only, so there is nothing to compensate.

3. **No databases anywhere** -- Tier A
   No Postgres, SQLite, Redis, MongoDB, Celery, RabbitMQ, or Kafka. No migration tooling. The system is stateless at the service level. **Deliberate.**

4. **No event/messaging infrastructure** -- Tier B
   No message broker, no event bus, no async communication. All inter-service communication is synchronous HTTP via httpx. **Deliberate** -- consistent with read-only pipeline design.

5. **Non-atomic two-file save for FAISS index** -- Tier B
   File: `services/search/src/search_service/services/faiss_index.py`, lines ~267-309
   Writes FAISS binary and metadata JSON as two sequential file operations without atomicity. **Default/accidental.**

6. **Shared library (`service-commons`) with editable local path dependency** -- Tier B
   All services depend on `service-commons` for config, exceptions, logging. Small, infrastructure-focused. **Deliberate** -- controlled coupling surface.

### Assessment

The absence of distributed transactions is not a gap but a coherent design decision. The artwork identification pipeline is fundamentally a read-only query workflow with no multi-service write path, no user state to maintain, and no business invariant spanning service boundaries. The graceful degradation pattern (embedding-only results when geometric is unavailable) is a first-class design concept with dedicated response schema fields. This represents a deliberate choice to prioritize availability and partial results over strict correctness -- appropriate for a "best match" use case.

### Risks & Recommendations

**Risk -- Severity: MEDIUM**
Location: `services/search/src/search_service/services/faiss_index.py`, save()
Issue: Non-atomic two-file save. Crash between writes leaves inconsistent state.
Recommendation: Write-to-temp-then-rename pattern.

**Risk -- Severity: LOW**
Location: `services/gateway/src/gateway/routers/objects.py`
Issue: Unbounded `/objects` endpoint with no pagination. Will cause memory/latency issues at scale.
Recommendation: Add cursor-based pagination.

**Risk -- Severity: LOW**
Location: API surface
Issue: No API versioning. Will create backward-compatibility challenges when the contract evolves.
Recommendation: Introduce versioning strategy before external consumers depend on the API.

---

## Resilience & Failure Isolation

**Position:** Mid-maturity resilience -- well-implemented circuit breakers, retries with backoff/jitter, and explicit graceful degradation in the orchestration layer. Lacks rate limiting, bulkhead isolation, chaos engineering, and operational observability.

**Confidence:** HIGH

### Evidence

1. **Per-dependency circuit breaker with full state machine** -- Tier A
   File: `services/gateway/src/gateway/clients/base.py`, lines ~67-165
   Complete Closed -> Open -> Half-Open circuit breaker with async locking. Each backend gets its own independent instance. **Deliberate.**

2. **Bounded retry with exponential backoff and jitter** -- Tier B
   Config: `max_attempts: 3`, `initial_backoff: 0.1s`, `max_backoff: 1.0s`, `jitter: 0.05s`. Retries suppressed in half-open state to prevent cascading. **Deliberate** -- mature retry engineering.

3. **Explicit failure domain classification** -- Tier A
   File: `services/gateway/src/gateway/routers/health.py`, lines ~57-63
   Embeddings/search = critical; geometric/storage = optional. Manifests consistently across health endpoint, Docker Compose dependencies, and pipeline error handling. **Deliberate.**

4. **Graceful degradation in identify pipeline** -- Tier A
   File: `services/gateway/src/gateway/routers/identify.py`, lines ~229-270
   Backend failures produce `degraded=True` responses with machine-readable reasons. First-class API contract. **Deliberate.**

5. **Uniform circuit breaker/retry config across all backends** -- Tier B
   File: `services/gateway/src/gateway/core/lifespan.py`, lines ~71-125
   All four backends share identical parameters. Embeddings (GPU inference) gets the same 30s timeout as storage (file I/O). **Default.**

6. **StorageClient bypasses retry logic** -- Tier B
   File: `services/gateway/src/gateway/clients/storage.py`, lines ~26-104
   `get_image_bytes()` does not use the base class `_request()` method, bypassing retries entirely. Only gets circuit breaker protection. **Likely accidental.**

7. **Integration tests verify degradation behavior** -- Tier C
   File: `services/gateway/tests/integration/test_storage_integration.py`
   Tests explicitly verify the pipeline returns 200 even during backend failures. **Deliberate.**

### Assessment

The resilience story is concentrated in the gateway-to-backend communication layer and is well-crafted: per-dependency circuit breakers, the critical/optional backend classification manifesting across three layers, and degradation encoded in the API schema. However, there is no rate limiting, no bulkhead isolation, no chaos engineering, no progressive delivery, and no monitoring/alerting. The system relies on single uvicorn workers with no horizontal scaling.

### Risks & Recommendations

**Risk -- Severity: MEDIUM**
Location: `services/gateway/src/gateway/core/lifespan.py` and `config.yaml`
Issue: Uniform timeout/retry/circuit-breaker config for all backends despite vastly different workload profiles.
Recommendation: Per-backend configuration. Storage: 5-10s timeout, threshold 3. Embeddings: 45-60s timeout, fewer retries.

**Risk -- Severity: MEDIUM**
Location: Gateway -- no rate limiting
Issue: Unlimited requests fan out to multiple backends, amplifying load. Retry mechanism (3 attempts) further amplifies during partial outages.
Recommendation: Add rate limiting and concurrency limit for the identify endpoint.

**Risk -- Severity: MEDIUM**
Location: `services/gateway/src/gateway/clients/storage.py`
Issue: StorageClient bypasses base class retry logic, unlike all other clients.
Recommendation: Refactor to use `_request()` for consistency, or explicitly document the exception.

**Risk -- Severity: LOW**
Location: All Dockerfiles and `docker-compose.yml`
Issue: No resource limits, no restart policies. Embeddings service could consume unbounded memory.
Recommendation: Add `mem_limit`, `cpus`, and `restart: unless-stopped`.

---

## Observability, Security & Cost

**Position:** Early-stage system with deliberate code-quality hygiene but minimal operational observability and security infrastructure. Heavy investment in static analysis and developer-time guardrails; runtime observability, authentication, and encryption entirely absent.

**Confidence:** HIGH

### Evidence

1. **Structured JSON logging** -- Tier B
   File: `libs/service-commons/src/service_commons/logging.py`, all service config.yaml files (`logging.format: "json"`)
   Shared `JSONFormatter` outputs structured JSON logs to stdout across all services. No log aggregation pipeline. **Deliberate** foundation, but no operationalization.

2. **No distributed tracing** -- Tier C
   Zero OpenTelemetry, Jaeger, Zipkin, or any tracing SDK in dependencies. No correlation IDs propagated between services. **Default.**

3. **No authentication or authorization** -- Tier B
   File: `services/gateway/src/gateway/app.py`, lines ~46-52
   No JWT, OAuth, API key, or any auth middleware. All services listen on `0.0.0.0` with no access control. **Deliberate** for development stage.

4. **CORS wildcard configuration** -- Tier B
   File: `services/gateway/config.yaml`, line ~44
   `cors_origins: ["*"]` with `allow_credentials: True`. Most permissive CORS configuration possible. **Default.**

5. **No TLS/mTLS between services** -- Tier B
   All inter-service communication uses plaintext HTTP. No certificate management. **Default.**

6. **Deliberate sensitive value redaction** -- Tier A
   File: `libs/service-commons/src/service_commons/config.py`
   Comprehensive `SENSITIVE_KEYWORDS` set with recursive `redact_sensitive_values()` for `/info` endpoints. Thoroughly tested. **Deliberate.**

7. **Comprehensive static security analysis in CI** -- Tier B
   Every service CI runs: bandit, pip-audit, semgrep, mypy strict, pyright strict, ruff. **Deliberate.**

8. **Custom semgrep rules preventing sneaky defaults** -- Tier B
   Five custom rule files prohibit: default parameter values, `dict.get()` with defaults, `getattr()` with defaults, all type suppression. CI-blocking. **Exceptionally deliberate.**

9. **Path traversal protection (tested)** -- Tier C
   File: `services/search/tests/integration/test_critical_issues.py`, lines ~55-82
   Explicit `TestPathTraversalProtection` class verifying `..` attacks are rejected. **Deliberate.**

10. **No SLOs, error budgets, or alerting** -- Tier C
    No Prometheus metrics, no alerting rules, no SLO definitions. Per-request `TimingInfo` exists but is not aggregated. **Default.**

11. **Uvicorn access logging disabled** -- Tier B
    All services run with `access_log=False`. No replacement access logging middleware configured. **Default/accidental.**

### Assessment

The system occupies an unusual but coherent position: heavy investment in developer-time security (static analysis, type safety, custom semgrep rules, fail-fast config philosophy) with near-zero runtime operational infrastructure. This split is deliberate -- no half-implemented tracing, no abandoned monitoring configs. The system is cleanly pre-operational: exceptional code hygiene, zero production infrastructure.

### Risks & Recommendations

**Risk -- Severity: HIGH**
Location: All service-to-service communication
Issue: No distributed tracing or correlation IDs across 5 services. Cannot follow a request through the pipeline.
Recommendation: Add OpenTelemetry with W3C Trace Context. Export to local Jaeger/Tempo.

**Risk -- Severity: HIGH**
Location: Gateway `cors_origins: ["*"]` and all services on `0.0.0.0` without auth
Issue: No authentication, wildcard CORS with credentials. Any client can access any endpoint including `DELETE /index`.
Recommendation: API key auth on gateway, restrict CORS origins, shared secret for inter-service communication.

**Risk -- Severity: MEDIUM**
Location: All services `main.py` (`access_log=False`)
Issue: HTTP access logs completely disabled. No record of requests.
Recommendation: Add structured access logging middleware.

**Risk -- Severity: MEDIUM**
Location: `docker-compose.yml`
Issue: All 5 service ports (8000-8004) published to host. Internal services don't need external access.
Recommendation: Only expose gateway port; use Docker internal networking for inter-service communication.

---

## Cross-Axis Synthesis

### Tradeoff Profile

> **Availability-first with single-node simplicity, latency-aware but unoptimized, no data distribution, stateless read-only orchestration, mid-maturity resilience at the gateway layer, exceptional code-time security with zero runtime operations.**

### Maturity Signals

The system shows a **bimodal maturity pattern**:

- **High maturity (deliberate asymmetry):** The critical/optional backend classification is consistent across three layers (health endpoint, Docker Compose dependencies, pipeline error handling). The circuit breaker is hand-rolled with tuned parameters. The degradation semantics are encoded in the API schema. The custom semgrep rules enforcing no-default-values are exceptionally disciplined.

- **Low maturity (uniform defaults):** Uniform 30-second timeout for all backends. Single uvicorn worker everywhere. No per-backend tuning of retry/circuit-breaker parameters. Sequential processing where parallelism is obvious. No runtime observability infrastructure.

This bimodal pattern is characteristic of a system designed by a team that thinks architecturally but has not yet faced production traffic. The patterns they chose to invest in (circuit breakers, graceful degradation, fail-fast config) reveal what they're worried about. The gaps (parallelism, scaling, observability) reveal what they haven't needed yet.

### Risk Count Table

| Axis                           | HIGH | MEDIUM | LOW |
|--------------------------------|------|--------|-----|
| Consistency & Availability     | 2    | 1      | 0   |
| Latency & Throughput           | 2    | 2      | 0   |
| Data Distribution              | 0    | 2      | 0   |
| Transaction Boundaries         | 0    | 1      | 2   |
| Resilience & Failure Isolation | 0    | 3      | 1   |
| Observability, Security & Cost | 2    | 2      | 0   |
| **Total**                      | **6**| **11** | **3**|

### Top 3 Priorities

1. **Sequential storage fetches and single-worker processes** (Latency, HIGH) -- The N+1 sequential fetch pattern in `build_geometric_references()` and single-worker uvicorn are the lowest-hanging performance fruit. Switching to `asyncio.gather()` and adding workers could cut identify latency by 50%+ with minimal code change. These are pre-production blockers.

2. **No authentication and wildcard CORS** (Security, HIGH) -- Any client can access all endpoints including destructive ones (`DELETE /index`, `DELETE /objects`). This must be addressed before any network-accessible deployment. API key auth and CORS restriction are table-stakes.

3. **No distributed tracing** (Observability, HIGH) -- With a 5-service sequential pipeline, debugging latency or failure issues without tracing means correlating JSON logs across services by timestamp. OpenTelemetry with W3C Trace Context would make the existing `TimingInfo` data dramatically more useful.

### Cross-Axis Tensions

1. **Availability vs. Latency:** The circuit breaker and graceful degradation patterns protect availability but the sequential pipeline architecture means every request pays the full latency cost of all services. The system cannot sacrifice an optional verification step early to save time -- it must attempt it, fail, and then degrade. The 30-second uniform timeout exacerbates this: a slow storage fetch blocks the entire pipeline for up to 30 seconds before triggering degradation.

2. **Resilience vs. Data Distribution:** The resilience patterns (circuit breakers, retries) are designed for a single-instance topology. The moment services scale to multiple instances, the circuit breaker state (which lives in the gateway process memory) will not be shared across gateway replicas. Each gateway instance will independently detect failures, potentially leading to inconsistent circuit states and traffic oscillation.

3. **Code-Time Security vs. Runtime Security:** The exceptional static analysis discipline (custom semgrep rules, sensitive value redaction, path traversal tests) creates a false sense of security maturity. At runtime, there is zero authentication, zero encryption, zero access logging, and zero alerting. A team reviewing the CI pipeline would conclude this is a well-secured system; a team reviewing the runtime would conclude it is completely unprotected. This gap needs to be explicitly acknowledged and prioritized.

4. **Transaction Simplicity vs. Data Durability:** The absence of distributed transactions is correct for the read-only query pipeline, but the non-atomic FAISS save (two-file write without journaling) means the one write operation the system does perform -- persisting the index -- has no transactional guarantee. The consistency and transaction axes both flag this same artifact, confirming it is a genuine cross-cutting risk rather than a single-axis concern.
