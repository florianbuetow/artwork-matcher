# SOLID Principles Audit

**Scope:** Full project (all services + tools)
**Principles checked:** SRP, OCP, LSP, ISP, DIP

## Findings

### [SRP] Violation — Severity: MEDIUM

**Location:** `services/gateway/src/gateway/clients/base.py`, class `BackendClient`, lines 21-441

**Issue:** `BackendClient` mixes three distinct concerns: HTTP communication, retry logic with exponential backoff, and circuit breaker state management. The circuit breaker tracks `_consecutive_failures`, `_circuit_state`, `_circuit_opened_at`, and has its own async lock — this is a full state machine embedded within an HTTP client. Changes to circuit breaker policy (e.g., sliding window vs consecutive failures) would require modifying the same class as changes to retry logic or HTTP request handling.

**Suggestion:** Extract `CircuitBreaker` as its own class that manages the open/half-open/closed state machine. The `BackendClient` would then accept it as a collaborator: `self._circuit_breaker = CircuitBreaker(threshold, recovery_timeout)`. Retry logic could similarly be extracted into a `RetryPolicy` if it ever grows. This would make circuit breaker behavior independently testable.

---

### [SRP] Violation — Severity: MEDIUM

**Location:** `services/geometric/src/geometric_service/routers/match.py`, functions `match_images()` and `batch_match()`, lines 30-247

**Issue:** These endpoint functions handle orchestration AND object instantiation. On every request, they create fresh `ORBFeatureExtractor`, `BFFeatureMatcher`, and `RANSACVerifier` instances from config, then orchestrate the pipeline. The instantiation boilerplate is duplicated between `match_images()` and `batch_match()` (lines 38-45 and 140-147 are identical). This is both a SRP issue (endpoint does construction + orchestration) and a DRY violation.

**Suggestion:** Initialize `ORBFeatureExtractor`, `BFFeatureMatcher`, and `RANSACVerifier` once during service startup in the lifespan context (like the gateway does with its clients), storing them in `AppState`. Endpoints then pull them from state rather than constructing them.

---

### [SRP] Violation — Severity: LOW

**Location:** `services/gateway/src/gateway/routers/objects.py`, function `load_metadata()`, lines 25-70

**Issue:** `load_metadata()` handles both CSV parsing AND column-name normalization (mapping `"id"` to `"object_id"`, `"title"` to `"name"`, `"date"` to `"year"`). This metadata loading is also called on every single request to `/objects` and `/objects/{id}` — there is no caching, meaning the CSV is re-parsed per request.

**Suggestion:** Move metadata loading into the lifespan (load once at startup, store in `AppState`). The column-name normalization logic could be a small data-mapping function. This simultaneously fixes the per-request re-parsing performance issue.

---

### [OCP] Violation — Severity: LOW

**Location:** `services/gateway/src/gateway/routers/identify.py`, function `calculate_confidence()`, lines 75-107

**Issue:** The confidence scoring strategy is a single function with hardcoded branching logic (`if geometric_score is not None`, `elif geometric_enabled`, `else`). If a new verification method were added (e.g., deep feature matching), this function would need modification rather than extension.

**Suggestion:** This is a minor OCP concern for this project's scope. The project currently has exactly two verification paths (geometric + embedding-only), and the branching is clear. Only worth refactoring if a third verification method is planned. If so, a `ConfidenceStrategy` protocol with implementations per verification method would make this extensible.

---

### [DIP] Violation — Severity: MEDIUM

**Location:** `services/storage/src/storage_service/services/blob_store.py`, class `FileBlobStore`, lines 17-54 — referenced from `services/storage/src/storage_service/core/lifespan.py` and `services/storage/src/storage_service/routers/objects.py`

**Issue:** There is no abstraction boundary for storage. `FileBlobStore` is a concrete class used directly throughout the storage service. If you wanted to swap to S3, GCS, or an in-memory store for testing, you would need to modify every consumer or use monkeypatching. The project's README already lists "Storage" as a service boundary, suggesting this may eventually need alternative backends.

**Suggestion:** Define a `BlobStore` protocol:

```python
class BlobStore(Protocol):
    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes | None: ...
    def delete(self, key: str) -> bool: ...
    def delete_all(self) -> int: ...
    def count(self) -> int: ...
```

`FileBlobStore` implements it. The rest of the service depends on the protocol. This enables easy swapping and testing.

---

### [DIP] Violation — Severity: LOW

**Location:** `services/search/src/search_service/services/faiss_index.py`, class `FAISSIndex`, lines 77-394

**Issue:** `FAISSIndex` directly imports and instantiates `faiss.IndexFlatIP` in its constructor (line 104). The index type is hardcoded despite config having fields for `index_type` (flat/ivf/hnsw) and `metric` (inner_product/l2). This means the config options for alternative index types cannot actually be used without modifying `FAISSIndex`.

**Suggestion:** This is a known gap — the config models support multiple index types but the implementation only uses `IndexFlatIP`. When it is time to support IVF or HNSW, extract index creation into a factory function that reads the config and returns the appropriate FAISS index type. The `FAISSIndex` class would accept any `faiss.Index` via constructor injection rather than creating its own.

---

### [LSP] — No violations found

The project uses inheritance sparingly and correctly. The `BackendClient` hierarchy (`EmbeddingsClient`, `SearchClient`, `GeometricClient`, `StorageClient`) all properly extend `BackendClient` by adding domain-specific methods without overriding or restricting base class behavior. Subclasses add `embed()`, `search()`, `match_batch()`, etc. — all additive. No methods are stubbed out or throw `NotImplementedError`.

---

### [ISP] — No violations found

Interfaces are kept naturally small. The services communicate via HTTP/JSON (no shared interfaces), and within each service, classes expose focused APIs. `FileBlobStore` has 5 methods, all used by its consumers. `ORBFeatureExtractor`, `BFFeatureMatcher`, and `RANSACVerifier` each have a single primary method. The Pydantic schemas use separate models for each endpoint's request/response rather than a single shared schema.

---

## Summary

| Principle | HIGH | MEDIUM | LOW |
|-----------|------|--------|-----|
| SRP       | 0    | 2      | 1   |
| OCP       | 0    | 0      | 1   |
| LSP       | 0    | 0      | 0   |
| ISP       | 0    | 0      | 0   |
| DIP       | 0    | 1      | 1   |

### Top 3 Priorities

1. **Extract circuit breaker from BackendClient (SRP, MEDIUM)** — The circuit breaker is a non-trivial state machine with its own async lock. Extracting it would make it independently testable and keep `BackendClient` focused on HTTP communication. This is the most impactful refactoring because `BackendClient` is the most complex class in the project (~440 lines).

2. **Initialize geometric service components at startup (SRP, MEDIUM)** — The duplicated construction of `ORBFeatureExtractor`/`BFFeatureMatcher`/`RANSACVerifier` in every request handler is both a responsibility issue and a performance concern (though construction is cheap). Moving to the startup pattern used by other services improves consistency across the codebase.

3. **Add BlobStore protocol (DIP, MEDIUM)** — This is the only architectural boundary in the project that lacks an abstraction. Given the project's microservices architecture is otherwise well-structured, adding a protocol here would complete the pattern and make the storage service testable without filesystem access.

### Overall Assessment

The codebase is structurally healthy and well-organized. The microservices architecture naturally enforces many SOLID principles — each service has a single responsibility, services communicate via HTTP (no tight coupling), and the consistent module layout (`app.py`, `config.py`, `schemas.py`, `core/`, `routers/`, `services/`) makes navigation intuitive. Configuration management is exemplary: zero defaults, pydantic validation, fail-fast startup. The findings above are refinements rather than structural concerns — the most significant is the circuit breaker extraction in `BackendClient`, which would improve testability of a complex component. No HIGH severity violations were found.
