# CORS Implementation Guide

## What Is CORS?

**CORS (Cross-Origin Resource Sharing)** is a browser security mechanism. It controls whether JavaScript running on one domain (e.g., `museum-app.com`) can make requests to an API on a different domain (e.g., `api.museum-app.com:8000`).

This is a **browser-only** concern. Backend services, curl, Postman, and Python scripts are unaffected.

---

## Do We Need It?

| Scenario | CORS Needed? |
|----------|--------------|
| Backend-to-backend calls (services talking to each other) | No |
| curl / Postman / Python scripts | No |
| Browser JS on same origin as API | No |
| Browser JS on different origin (e.g., React dev server on :3000 calling API on :8000) | **Yes** |
| Static HTML page making fetch() calls to the API | **Yes** |

**For this project:**

- If demonstrating via curl or server-rendered pages → CORS not needed
- If building a browser-based UI (React, Vue, plain HTML + fetch) → CORS required

---

## What Happens Without CORS?

If you call the API from browser JavaScript without CORS enabled:

```javascript
// From a page served on localhost:3000
fetch('http://localhost:8000/identify', { 
  method: 'POST', 
  body: JSON.stringify({ image: '...' })
})
```

The browser console shows this error:

```
Access to fetch at 'http://localhost:8000/identify' from origin 'http://localhost:3000' 
has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header is present.
```

**Important:** The request actually reaches the server and succeeds. The server processes it and returns a response. But the browser refuses to show the response to JavaScript. This is browser-enforced security, not a server issue.

This confuses many developers because:
1. The server logs show a successful request
2. The Network tab shows a response
3. But JavaScript can't access the response data

---

## Implementation

### FastAPI (Our Stack)

Adding CORS is trivial—4 lines of code:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

That's it. No additional configuration required.

### Configuration Options

| Option | Development | Production |
|--------|-------------|------------|
| `allow_origins` | `["*"]` | `["https://museum-app.com"]` |
| `allow_methods` | `["*"]` | `["GET", "POST"]` |
| `allow_headers` | `["*"]` | `["Content-Type"]` |
| `allow_credentials` | `False` | `True` (if using cookies) |

**Development settings** (permissive):

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Production settings** (restrictive):

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://museum-app.com",
        "https://www.museum-app.com",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
    allow_credentials=True,
)
```

---

## Which Services Need CORS?

| Service | CORS Needed? | Reason |
|---------|--------------|--------|
| Gateway (port 8000) | **Yes** | Public-facing, may receive browser requests |
| Embeddings (port 8001) | No | Internal only, called by gateway |
| Search (port 8002) | No | Internal only, called by gateway |
| Geometric (port 8003) | No | Internal only, called by gateway |

**Only the Gateway needs CORS.** Internal services are never called directly from browsers.

---

## Recommendation for This Project

Add permissive CORS to the Gateway service:

```python
# gateway/app.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Artwork Matcher Gateway")

# Enable CORS for browser-based clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Why:**
1. Zero cost—no performance impact, no complexity
2. Removes friction when building or demoing a browser UI
3. Allows easy testing with a web frontend
4. Can be tightened for production later

---

## Verifying CORS Works

### Check Response Headers

```bash
curl -I -X OPTIONS http://localhost:8000/identify \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: POST"
```

With CORS enabled, you should see:

```
HTTP/1.1 200 OK
access-control-allow-origin: *
access-control-allow-methods: *
access-control-allow-headers: *
```

### Test from Browser Console

Open browser dev tools on any page and run:

```javascript
fetch('http://localhost:8000/health')
  .then(r => r.json())
  .then(console.log)
  .catch(console.error)
```

- **With CORS:** Logs `{ status: "healthy" }`
- **Without CORS:** Throws CORS error

---

## Common CORS Mistakes

### 1. Adding CORS to Internal Services

Don't add CORS to embeddings, search, or geometric services. They're internal and adding CORS:
- Provides no benefit (browsers don't call them)
- Slightly increases response size (extra headers)
- Creates false impression they're meant for direct browser access

### 2. Forgetting Preflight Requests

Browsers send an `OPTIONS` request before the actual request for "non-simple" requests (POST with JSON, custom headers, etc.). FastAPI's CORS middleware handles this automatically—no extra code needed.

### 3. Credentials with Wildcard Origin

This configuration is **invalid**:

```python
# WRONG - browsers reject this combination
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,  # Can't use with "*" origins
)
```

If you need credentials (cookies, auth headers), you must specify explicit origins:

```python
# CORRECT
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
)
```

For this project, we don't use credentials, so `allow_origins=["*"]` is fine.

---

## Summary

| Question | Answer |
|----------|--------|
| What is CORS? | Browser security preventing cross-origin requests |
| Do we need it? | Only if building a browser-based UI |
| Which service? | Gateway only |
| How hard? | 4 lines of code |
| Development settings? | `allow_origins=["*"]` (permissive) |
| Production settings? | Explicit origin list (restrictive) |

**Bottom line:** Add CORS middleware to the Gateway. It costs nothing and removes potential friction when demoing the project.
