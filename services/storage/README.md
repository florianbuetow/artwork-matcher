# Storage Service

Binary object storage backed by the local filesystem. Stores, retrieves, and deletes opaque binary objects identified by string keys. Files are stored as `{id}.dat` in a configurable directory.

## Quick Start

```bash
# From repository root
just start-storage

# Or from this directory
just run
```

The service will be available at `http://localhost:8004`.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/info` | GET | Storage configuration and object count |
| `/objects/{id}` | PUT | Store binary object (raw bytes in request body) |
| `/objects/{id}` | GET | Retrieve binary object |
| `/objects/{id}` | DELETE | Delete a single object |
| `/objects` | DELETE | Delete all stored objects |

### Example: Store and Retrieve an Object

```bash
# Store a file
curl -X PUT --data-binary @photo.jpg http://localhost:8004/objects/my-photo

# Retrieve it
curl http://localhost:8004/objects/my-photo -o retrieved.dat

# Delete it
curl -X DELETE http://localhost:8004/objects/my-photo

# Delete all objects
curl -X DELETE http://localhost:8004/objects
```

### Object ID Rules

IDs must match `^[a-zA-Z0-9_-]+$` (alphanumeric, hyphens, underscores). Invalid IDs return 400.

## Configuration

Configuration is loaded from `config.yaml`.

```yaml
service:
  name: "storage"
  version: "0.1.0"

storage:
  path: "./data/objects"
  content_type: "application/octet-stream"

server:
  host: "0.0.0.0"
  port: 8004
  log_level: "info"

logging:
  level: "INFO"
  format: "json"
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
