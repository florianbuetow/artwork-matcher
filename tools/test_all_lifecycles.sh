#!/usr/bin/env bash
#
# Run all service lifecycle tests
#
# This script runs the lifecycle tests for all services sequentially.
# Each service's lifecycle test verifies:
# - Starting/stopping locally with just run/kill
# - Starting/stopping with Docker
# - Health/info endpoint verification
#
# Usage:
#   ./tools/test_all_lifecycles.sh
#
# Prerequisites:
#   - Docker must be running
#   - No services should be running on ports 8000-8003
#   - Run from the repository root directory
#

set -e  # Exit on first error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Ensure we're in the right directory
if [ ! -f "justfile" ] || [ ! -d "services" ]; then
    printf "${RED}Error:${NC} Must run from repository root directory\n"
    exit 1
fi

printf "\n${BLUE}=== Running All Service Lifecycle Tests ===${NC}\n\n"

# Track results
SERVICES_PASSED=0
SERVICES_FAILED=0
FAILED_SERVICES=""

# List of services with their lifecycle tests
SERVICES=(
    "embeddings:services/embeddings/tests/manual/test_service_lifecycle.sh"
    "search:services/search/tests/manual/test_service_lifecycle.sh"
    "gateway:services/gateway/tests/manual/test_service_lifecycle.sh"
)

for service_entry in "${SERVICES[@]}"; do
    service_name="${service_entry%%:*}"
    test_script="${service_entry##*:}"

    printf "${BLUE}--- Testing %s service ---${NC}\n\n" "$service_name"

    if [ ! -f "$test_script" ]; then
        printf "${RED}✗ SKIP${NC}: %s (test script not found: %s)\n\n" "$service_name" "$test_script"
        continue
    fi

    if "$test_script"; then
        printf "${GREEN}✓ PASS${NC}: %s lifecycle tests passed\n\n" "$service_name"
        ((SERVICES_PASSED++))
    else
        printf "${RED}✗ FAIL${NC}: %s lifecycle tests failed\n\n" "$service_name"
        ((SERVICES_FAILED++))
        FAILED_SERVICES="$FAILED_SERVICES $service_name"
    fi
done

# Print summary
printf "\n${BLUE}=== Lifecycle Test Summary ===${NC}\n"
printf "Services passed: ${GREEN}%d${NC}\n" "$SERVICES_PASSED"
printf "Services failed: ${RED}%d${NC}\n" "$SERVICES_FAILED"

if [ $SERVICES_FAILED -gt 0 ]; then
    printf "Failed services:${RED}%s${NC}\n" "$FAILED_SERVICES"
    exit 1
fi

printf "\n${GREEN}All service lifecycle tests passed!${NC}\n\n"
exit 0
