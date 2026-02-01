# Common Issues and Fixes

## ASGI App Attribute Not Found

**Error:**
```
ERROR: Error loading ASGI app. Attribute "app" not found in module "embeddings_service.app".
```

**Problem:** The `just run` command used `uvicorn embeddings_service.app:app` but there was no `app` variable at module level - only a `create_app()` factory function.

**Fix:** Changed the justfile to use the `--factory` flag:
```bash
uvicorn embeddings_service.app:create_app --factory --reload --host 0.0.0.0 --port 8001
```

**Verification:** Service starts successfully, health endpoint returns `{"status": "healthy"}`, all 79 unit tests pass.
