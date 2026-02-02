# Service Lifecycle Test Pattern

This document describes the pattern for testing service lifecycle operations: starting, health verification, and stopping services in both local and Docker environments.

---

## Table of Contents

1. [Overview](#overview)
2. [When to Use](#when-to-use)
3. [Test Structure](#test-structure)
4. [Helper Functions](#helper-functions)
5. [Test Sequence](#test-sequence)
6. [Complete Example](#complete-example)
7. [Running Lifecycle Tests](#running-lifecycle-tests)
8. [Troubleshooting](#troubleshooting)

---

## Overview

Service lifecycle tests verify that:

1. Services start correctly (both locally and in Docker)
2. Health endpoints respond as expected
3. Services stop cleanly without leaving orphan processes
4. justfile commands (`just run`, `just kill`, `just docker-up`, `just docker-down`) work correctly

These are **manual tests** that validate the operational aspects of services, separate from unit and integration tests.

### Test Location

```
services/<service_name>/
└── tests/
    └── manual/
        └── test_service_lifecycle.sh
```

---

## When to Use

Run lifecycle tests when:

- Adding a new service to the project
- Modifying justfile commands for starting/stopping services
- Changing Docker configuration
- Debugging startup or shutdown issues
- Before deploying to production

---

## Test Structure

### Script Header

Every lifecycle test script should include:

```bash
#!/usr/bin/env bash
#
# Manual Test: Service Lifecycle (Local + Docker)
#
# This script tests the complete lifecycle of the <service_name> service:
# - Starting/stopping locally with just run/kill
# - Starting/stopping with Docker
# - Verifying status at each step
#
# Usage:
#   ./tests/manual/test_service_lifecycle.sh
#
# Prerequisites:
#   - Docker must be running
#   - No services should be running on port <PORT>
#   - Run from the service directory (services/<service_name>/)
#

set -e  # Exit on first error
```

### Color Output and Counters

```bash
# Colors for readable output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
```

---

## Helper Functions

### Output Functions

```bash
# Print section headers
print_header() {
    printf "\n${BLUE}=== %s ===${NC}\n\n" "$1"
}

# Record passed test
pass() {
    printf "${GREEN}✓ PASS${NC}: %s\n" "$1"
    ((TESTS_PASSED++))
}

# Record failed test and exit
fail() {
    printf "${RED}✗ FAIL${NC}: %s\n" "$1"
    printf "${RED}Reason${NC}: %s\n" "$2"
    ((TESTS_FAILED++))
    print_summary
    exit 1
}

# Print final summary
print_summary() {
    printf "\n${BLUE}=== Test Summary ===${NC}\n"
    printf "Passed: ${GREEN}%d${NC}\n" "$TESTS_PASSED"
    printf "Failed: ${RED}%d${NC}\n" "$TESTS_FAILED"
}
```

### Service Status Checks

```bash
# Check if service is responding on a port
check_service_responding() {
    local port=$1
    curl -s --connect-timeout 2 "http://localhost:${port}/health" > /dev/null 2>&1
}

# Check if service is NOT responding on a port
check_service_not_responding() {
    local port=$1
    ! curl -s --connect-timeout 2 "http://localhost:${port}/health" > /dev/null 2>&1
}
```

### Wait Functions

```bash
# Wait for service to become healthy
wait_for_healthy() {
    local port=$1
    local max_attempts=$2
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if check_service_responding "$port"; then
            return 0
        fi
        printf "  Waiting for service to start (attempt %d/%d)...\n" "$attempt" "$max_attempts"
        sleep 2
        ((attempt++))
    done
    return 1
}

# Wait for service to stop
wait_for_stopped() {
    local port=$1
    local max_attempts=$2
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if check_service_not_responding "$port"; then
            return 0
        fi
        printf "  Waiting for service to stop (attempt %d/%d)...\n" "$attempt" "$max_attempts"
        sleep 2
        ((attempt++))
    done
    return 1
}
```

---

## Test Sequence

A complete lifecycle test follows this sequence:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SERVICE LIFECYCLE TEST                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Step 1: Verify Clean Initial State                            │
│   └── Port should be free, no services running                  │
│                                                                  │
│   ┌─────────────── LOCAL LIFECYCLE ─────────────────┐           │
│   │                                                  │           │
│   │   Step 2: Start Service Locally                 │           │
│   │   └── just run (background)                     │           │
│   │                                                  │           │
│   │   Step 3: Verify Service Running                │           │
│   │   └── Health endpoint responds                  │           │
│   │   └── Returns expected fields                   │           │
│   │                                                  │           │
│   │   Step 4: Stop Service                          │           │
│   │   └── just kill                                 │           │
│   │                                                  │           │
│   │   Step 5: Verify Service Stopped                │           │
│   │   └── Port is free again                        │           │
│   │                                                  │           │
│   └──────────────────────────────────────────────────┘           │
│                                                                  │
│   ┌─────────────── DOCKER LIFECYCLE ────────────────┐           │
│   │                                                  │           │
│   │   Step 6: Start Docker Container                │           │
│   │   └── just docker-up                            │           │
│   │                                                  │           │
│   │   Step 7: Verify Container Running              │           │
│   │   └── Health endpoint responds                  │           │
│   │   └── Info endpoint returns version             │           │
│   │                                                  │           │
│   │   Step 8: Stop Docker Container                 │           │
│   │   └── just docker-down                          │           │
│   │                                                  │           │
│   │   Step 9: Verify Container Stopped              │           │
│   │   └── Port is free again                        │           │
│   │                                                  │           │
│   └──────────────────────────────────────────────────┘           │
│                                                                  │
│   Final: Print Summary                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Step Details

| Step | Purpose | Success Criteria | Timeout |
|------|---------|------------------|---------|
| 1 | Verify clean state | Port not in use | - |
| 2 | Start local | Process spawns | - |
| 3 | Verify local running | `/health` returns 200 | 30s |
| 4 | Stop local | `just kill` succeeds | - |
| 5 | Verify local stopped | Port not responding | 10s |
| 6 | Start Docker | Container starts | - |
| 7 | Verify Docker running | `/health` returns 200 | 60s |
| 8 | Stop Docker | Container stops | - |
| 9 | Verify Docker stopped | Port not responding | 20s |

---

## Complete Example

```bash
#!/usr/bin/env bash
#
# Manual Test: Service Lifecycle (Local + Docker)
#
# This script tests the complete lifecycle of the embeddings service:
# - Starting/stopping locally with just run/kill
# - Starting/stopping with Docker
# - Verifying status at each step
#
# Usage:
#   ./tests/manual/test_service_lifecycle.sh
#
# Prerequisites:
#   - Docker must be running
#   - No services should be running on ports 8001
#   - Run from the repository root directory
#

set -e  # Exit on first error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TESTS_PASSED=0
TESTS_FAILED=0

# Test result tracking
print_header() {
    printf "\n${BLUE}=== %s ===${NC}\n\n" "$1"
}

pass() {
    printf "${GREEN}✓ PASS${NC}: %s\n" "$1"
    ((TESTS_PASSED++))
}

fail() {
    printf "${RED}✗ FAIL${NC}: %s\n" "$1"
    printf "${RED}Reason${NC}: %s\n" "$2"
    ((TESTS_FAILED++))
    print_summary
    exit 1
}

print_summary() {
    printf "\n${BLUE}=== Test Summary ===${NC}\n"
    printf "Passed: ${GREEN}%d${NC}\n" "$TESTS_PASSED"
    printf "Failed: ${RED}%d${NC}\n" "$TESTS_FAILED"
}

# Check if service is responding on a port
check_service_responding() {
    local port=$1
    curl -s --connect-timeout 2 "http://localhost:${port}/health" > /dev/null 2>&1
}

# Check if service is NOT responding on a port
check_service_not_responding() {
    local port=$1
    ! curl -s --connect-timeout 2 "http://localhost:${port}/health" > /dev/null 2>&1
}

# Wait for service to become healthy
wait_for_healthy() {
    local port=$1
    local max_attempts=$2
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if check_service_responding "$port"; then
            return 0
        fi
        printf "  Waiting for service to start (attempt %d/%d)...\n" "$attempt" "$max_attempts"
        sleep 2
        ((attempt++))
    done
    return 1
}

# Wait for service to stop
wait_for_stopped() {
    local port=$1
    local max_attempts=$2
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if check_service_not_responding "$port"; then
            return 0
        fi
        printf "  Waiting for service to stop (attempt %d/%d)...\n" "$attempt" "$max_attempts"
        sleep 2
        ((attempt++))
    done
    return 1
}

# =============================================================================
# MAIN TEST SEQUENCE
# =============================================================================

print_header "Service Lifecycle Test"

# Ensure we're in the right directory
if [ ! -f "justfile" ]; then
    fail "Directory check" "Must run from repository root (justfile not found)"
fi

# -----------------------------------------------------------------------------
# Step 1: Verify no services running initially
# -----------------------------------------------------------------------------
print_header "Step 1: Verify no services running"

if check_service_not_responding 8001; then
    pass "Embeddings service not responding on port 8001"
else
    fail "Initial state check" "Port 8001 is already in use. Stop any running services first."
fi

# -----------------------------------------------------------------------------
# Step 2: Start app locally
# -----------------------------------------------------------------------------
print_header "Step 2: Start app locally"

cd services/embeddings
just run > /dev/null 2>&1 &
LOCAL_PID=$!
cd ../..

printf "  Started local service (background PID: %s)\n" "$LOCAL_PID"

# -----------------------------------------------------------------------------
# Step 3: Verify app is running
# -----------------------------------------------------------------------------
print_header "Step 3: Verify app is running"

if wait_for_healthy 8001 15; then
    pass "Embeddings service is healthy after local start"
else
    fail "Local service startup" "Service did not become healthy within timeout"
fi

# Verify we get expected response fields
response=$(curl -s "http://localhost:8001/health")
if echo "$response" | grep -q '"status":"healthy"'; then
    pass "Health endpoint returns healthy status"
else
    fail "Health response" "Expected 'healthy' status, got: $response"
fi

# -----------------------------------------------------------------------------
# Step 4: Kill the app
# -----------------------------------------------------------------------------
print_header "Step 4: Kill the app"

cd services/embeddings
kill_output=$(just kill 2>&1)
cd ../..

if echo "$kill_output" | grep -q "Service stopped"; then
    pass "just kill reports service stopped"
else
    fail "Kill command" "Expected 'Service stopped' message, got: $kill_output"
fi

# -----------------------------------------------------------------------------
# Step 5: Verify app is stopped
# -----------------------------------------------------------------------------
print_header "Step 5: Verify app is stopped"

if wait_for_stopped 8001 5; then
    pass "Embeddings service stopped after just kill"
else
    fail "Local service shutdown" "Service still responding after kill"
fi

# -----------------------------------------------------------------------------
# Step 6: Start Docker container
# -----------------------------------------------------------------------------
print_header "Step 6: Start Docker container"

cd services/embeddings
docker_up_output=$(just docker-up 2>&1)
cd ../..

if echo "$docker_up_output" | grep -q "Service running"; then
    pass "Docker container started"
else
    fail "Docker startup" "Expected 'Service running' message, got: $docker_up_output"
fi

# -----------------------------------------------------------------------------
# Step 7: Verify Docker container is running
# -----------------------------------------------------------------------------
print_header "Step 7: Verify Docker container is running"

if wait_for_healthy 8001 30; then
    pass "Embeddings service is healthy in Docker"
else
    fail "Docker service startup" "Service did not become healthy within timeout"
fi

# Verify info endpoint works
info_response=$(curl -s "http://localhost:8001/info")
if echo "$info_response" | grep -q '"version"'; then
    pass "Info endpoint returns version"
else
    fail "Info response" "Expected version in response, got: $info_response"
fi

# -----------------------------------------------------------------------------
# Step 8: Stop Docker container
# -----------------------------------------------------------------------------
print_header "Step 8: Stop Docker container"

cd services/embeddings
docker_down_output=$(just docker-down 2>&1)
cd ../..

if echo "$docker_down_output" | grep -q "Service stopped"; then
    pass "Docker container stopped"
else
    fail "Docker shutdown" "Expected 'Service stopped' message, got: $docker_down_output"
fi

# -----------------------------------------------------------------------------
# Step 9: Verify Docker container is stopped
# -----------------------------------------------------------------------------
print_header "Step 9: Verify Docker container is stopped"

if wait_for_stopped 8001 10; then
    pass "Embeddings service not responding after Docker stop"
else
    fail "Docker service shutdown" "Service still responding after docker-down"
fi

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print_header "All Tests Passed!"
print_summary

exit 0
```

---

## Running Lifecycle Tests

### Prerequisites

Before running lifecycle tests:

1. **Docker must be running**
   ```bash
   docker info  # Should not error
   ```

2. **Target port must be free**
   ```bash
   lsof -i :8001  # Should return nothing
   ```

3. **Service dependencies initialized**
   ```bash
   cd services/<service_name>
   just init
   ```

### Execution

```bash
# From repository root
cd services/embeddings
chmod +x tests/manual/test_service_lifecycle.sh
./tests/manual/test_service_lifecycle.sh
```

### Expected Output

```
=== Service Lifecycle Test ===

=== Step 1: Verify no services running ===

✓ PASS: Embeddings service not responding on port 8001

=== Step 2: Start app locally ===

  Started local service (background PID: 12345)

=== Step 3: Verify app is running ===

  Waiting for service to start (attempt 1/15)...
  Waiting for service to start (attempt 2/15)...
✓ PASS: Embeddings service is healthy after local start
✓ PASS: Health endpoint returns healthy status

=== Step 4: Kill the app ===

✓ PASS: just kill reports service stopped

=== Step 5: Verify app is stopped ===

✓ PASS: Embeddings service stopped after just kill

=== Step 6: Start Docker container ===

✓ PASS: Docker container started

=== Step 7: Verify Docker container is running ===

  Waiting for service to start (attempt 1/30)...
✓ PASS: Embeddings service is healthy in Docker
✓ PASS: Info endpoint returns version

=== Step 8: Stop Docker container ===

✓ PASS: Docker container stopped

=== Step 9: Verify Docker container is stopped ===

✓ PASS: Embeddings service not responding after Docker stop

=== All Tests Passed! ===

=== Test Summary ===
Passed: 10
Failed: 0
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Port already in use | Previous service still running | Run `just kill` or `docker compose down` |
| Service never becomes healthy | Startup crash or timeout | Check logs with `docker logs` or terminal output |
| Docker container won't start | Image not built | Run `just docker-build` first |
| `just kill` doesn't stop service | PID file stale | Kill process manually: `pkill -f uvicorn` |
| Timeout waiting for health | Service slow to load model | Increase `max_attempts` in `wait_for_healthy` |

### Debugging Commands

```bash
# Check what's using a port
lsof -i :8001

# View Docker container logs
docker logs artwork-matcher-embeddings

# List running containers
docker ps

# Force stop all containers
docker compose down --remove-orphans

# Kill all uvicorn processes
pkill -9 -f uvicorn
```

### Cleanup After Failed Test

If a test fails mid-execution, clean up manually:

```bash
# Stop any local processes
cd services/embeddings
just kill 2>/dev/null || true

# Stop any Docker containers
just docker-down 2>/dev/null || true

# Verify port is free
lsof -i :8001
```

---

## Adapting for New Services

When creating lifecycle tests for a new service:

1. **Copy the template** from an existing service
2. **Update port number** to match the new service
3. **Update service name** in all messages
4. **Adjust timeouts** based on service startup time
   - ML services (embeddings): 30+ seconds
   - Simple services (gateway): 5-10 seconds
5. **Add service-specific checks** if needed (e.g., model loaded, index available)

### Service-Specific Timeout Guidelines

| Service | Local Startup | Docker Startup | Reason |
|---------|---------------|----------------|--------|
| Gateway | 15 attempts | 15 attempts | Fast, no model loading |
| Embeddings | 15 attempts | 30 attempts | DINOv2 model loading |
| Search | 10 attempts | 15 attempts | FAISS index loading |
| Geometric | 10 attempts | 15 attempts | ORB initialization |

---

## Summary

### Key Principles

1. **Test both local and Docker** — Different issues appear in each environment
2. **Verify clean state first** — Ensures tests are reproducible
3. **Use timeouts** — Services take time to start, especially with ML models
4. **Fail fast with context** — Print what went wrong and why
5. **Clean up on failure** — Document how to recover from failed tests

### Checklist for New Lifecycle Tests

- [ ] Script has shebang and documentation header
- [ ] `set -e` to exit on first error
- [ ] Helper functions for colored output
- [ ] Clean state verification (port free)
- [ ] Local start/verify/stop sequence
- [ ] Docker start/verify/stop sequence
- [ ] Appropriate timeouts for service type
- [ ] Summary printed at end
- [ ] Troubleshooting section in this guide updated
