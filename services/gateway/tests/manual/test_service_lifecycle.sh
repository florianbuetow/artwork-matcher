#!/usr/bin/env bash
#
# Manual Test: Service Lifecycle (Local + Docker)
#
# This script tests the complete lifecycle of the gateway service:
# - Starting/stopping locally with just run/kill
# - Starting/stopping with Docker
# - Verifying status at each step
#
# Usage:
#   ./tests/manual/test_service_lifecycle.sh
#
# Prerequisites:
#   - Docker must be running
#   - No services should be running on port 8000
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
    ((++TESTS_PASSED))
}

fail() {
    printf "${RED}✗ FAIL${NC}: %s\n" "$1"
    printf "${RED}Reason${NC}: %s\n" "$2"
    ((++TESTS_FAILED))
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
        ((++attempt))
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
        ((++attempt))
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

if check_service_not_responding 8000; then
    pass "Gateway service not responding on port 8000"
else
    fail "Initial state check" "Port 8000 is already in use. Stop any running services first."
fi

# -----------------------------------------------------------------------------
# Step 2: Start app locally
# -----------------------------------------------------------------------------
print_header "Step 2: Start app locally"

cd services/gateway
just run > /dev/null 2>&1 &
LOCAL_PID=$!
cd ../..

printf "  Started local service (background PID: %s)\n" "$LOCAL_PID"

# -----------------------------------------------------------------------------
# Step 3: Verify app is running
# -----------------------------------------------------------------------------
print_header "Step 3: Verify app is running"

# Gateway is fast to start (no model loading), 15 attempts should be plenty
if wait_for_healthy 8000 15; then
    pass "Gateway service is healthy after local start"
else
    fail "Local service startup" "Service did not become healthy within timeout"
fi

# Verify we get expected response fields
response=$(curl -s "http://localhost:8000/health")
if echo "$response" | grep -q '"status"'; then
    pass "Health endpoint returns status field"
else
    fail "Health response" "Expected 'status' field, got: $response"
fi

# -----------------------------------------------------------------------------
# Step 4: Kill the app
# -----------------------------------------------------------------------------
print_header "Step 4: Kill the app"

cd services/gateway
kill_output=$(just kill 2>&1)
cd ../..

if echo "$kill_output" | grep -q "Service stopped"; then
    pass "just kill reports service stopped"
elif echo "$kill_output" | grep -q "Service is not running"; then
    pass "just kill reports service not running (already stopped)"
else
    fail "Kill command" "Expected 'Service stopped' message, got: $kill_output"
fi

# -----------------------------------------------------------------------------
# Step 5: Verify app is stopped
# -----------------------------------------------------------------------------
print_header "Step 5: Verify app is stopped"

if wait_for_stopped 8000 5; then
    pass "Gateway service stopped after just kill"
else
    fail "Local service shutdown" "Service still responding after kill"
fi

# -----------------------------------------------------------------------------
# Step 6: Build and start Docker container (standalone mode)
# -----------------------------------------------------------------------------
print_header "Step 6: Build and start Docker container"

# Clean up any existing test container
docker rm -f gateway-lifecycle-test > /dev/null 2>&1 || true

# Build the image
cd services/gateway
printf "  Building Docker image...\n"
build_output=$(docker build -t gateway-lifecycle-test . 2>&1)
if [ $? -ne 0 ]; then
    cd ../..
    fail "Docker build" "Failed to build image"
fi
pass "Docker image built successfully"

# Run container standalone (without backend dependencies)
# Note: Using standalone mode because docker-compose requires all backend services
printf "  Starting container (standalone mode)...\n"
docker run -d --name gateway-lifecycle-test -p 8000:8000 \
    -e CONFIG_PATH=/app/config.yaml \
    -e EMBEDDINGS_URL=http://localhost:8001 \
    -e SEARCH_URL=http://localhost:8002 \
    -e GEOMETRIC_URL=http://localhost:8003 \
    gateway-lifecycle-test > /dev/null 2>&1

cd ../..

if docker ps | grep -q "gateway-lifecycle-test"; then
    pass "Docker container started (standalone mode)"
else
    fail "Docker startup" "Container failed to start"
fi

# -----------------------------------------------------------------------------
# Step 7: Verify Docker container is running
# -----------------------------------------------------------------------------
print_header "Step 7: Verify Docker container is running"

# Gateway is fast, 15 attempts should be enough
if wait_for_healthy 8000 15; then
    pass "Gateway service responds in Docker"
else
    fail "Docker service startup" "Service did not respond within timeout"
fi

# Verify info endpoint works
info_response=$(curl -s "http://localhost:8000/info")
if echo "$info_response" | grep -q '"version"'; then
    pass "Info endpoint returns version"
else
    fail "Info response" "Expected version in response, got: $info_response"
fi

# -----------------------------------------------------------------------------
# Step 8: Stop Docker container
# -----------------------------------------------------------------------------
print_header "Step 8: Stop Docker container"

docker stop gateway-lifecycle-test > /dev/null 2>&1
if [ $? -eq 0 ]; then
    pass "Docker container stopped"
else
    fail "Docker shutdown" "Failed to stop container"
fi

# -----------------------------------------------------------------------------
# Step 9: Verify Docker container is stopped and cleanup
# -----------------------------------------------------------------------------
print_header "Step 9: Verify Docker container is stopped"

if wait_for_stopped 8000 10; then
    pass "Gateway service not responding after Docker stop"
else
    fail "Docker service shutdown" "Service still responding after docker stop"
fi

# Cleanup
docker rm gateway-lifecycle-test > /dev/null 2>&1 || true
pass "Docker container removed"

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print_header "All Tests Passed!"
print_summary

exit 0
