#!/bin/bash
# Scans all service justfiles and creates a comparison table showing which targets exist in each service
# Usage: ./justfile-diff.sh

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Symbols
CHECK="${GREEN}✓${NC}"
CROSS="${RED}✗${NC}"

# Find all service justfiles relative to script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERVICES_DIR="$PROJECT_ROOT/services"

if [[ ! -d "$SERVICES_DIR" ]]; then
    echo "Error: services directory not found at $SERVICES_DIR" >&2
    exit 1
fi

# Create temp directory for working files
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# Collect all services
SERVICES=""
for dir in "$SERVICES_DIR"/*/; do
    if [[ -f "${dir}justfile" ]]; then
        service_name=$(basename "$dir")
        SERVICES="$SERVICES $service_name"
    fi
done

# Trim leading space and sort
SERVICES=$(echo "$SERVICES" | xargs -n1 | sort | xargs)

if [[ -z "$SERVICES" ]]; then
    echo "Error: No justfiles found in $SERVICES_DIR" >&2
    exit 1
fi

# Convert to array
read -ra SERVICE_ARRAY <<< "$SERVICES"

# Extract targets from a justfile (exclude private targets starting with _)
extract_targets() {
    local justfile="$1"
    grep -E '^[a-zA-Z][a-zA-Z0-9_-]*:' "$justfile" 2>/dev/null | \
        grep -v ':=' | \
        sed 's/:.*$//' | \
        sort -u
}

# Extract targets for each service and save to temp files
for service in "${SERVICE_ARRAY[@]}"; do
    justfile="$SERVICES_DIR/$service/justfile"
    extract_targets "$justfile" > "$TMPDIR/$service.targets"
done

# Collect all unique targets
cat "$TMPDIR"/*.targets | sort -u > "$TMPDIR/all_targets.txt"

# Calculate column widths
TARGET_COL_WIDTH=20
SERVICE_COL_WIDTH=12

# Print header
printf "\n${BLUE}=== Justfile Target Comparison ===${NC}\n\n"

# Print table header
printf "%-*s" "$TARGET_COL_WIDTH" "Target"
for service in "${SERVICE_ARRAY[@]}"; do
    printf " | %-*s" "$SERVICE_COL_WIDTH" "$service"
done
printf "\n"

# Print separator
printf '%*s' "$TARGET_COL_WIDTH" '' | tr ' ' '-'
for service in "${SERVICE_ARRAY[@]}"; do
    printf '%s' "-+-"
    printf '%*s' "$SERVICE_COL_WIDTH" '' | tr ' ' '-'
done
printf "\n"

# Print each target row
while IFS= read -r target; do
    printf "%-*s" "$TARGET_COL_WIDTH" "$target"
    for service in "${SERVICE_ARRAY[@]}"; do
        if grep -qx "$target" "$TMPDIR/$service.targets" 2>/dev/null; then
            printf " | %-*b" "$SERVICE_COL_WIDTH" "$CHECK"
        else
            printf " | %-*b" "$SERVICE_COL_WIDTH" "$CROSS"
        fi
    done
    printf "\n"
done < "$TMPDIR/all_targets.txt"

# Print summary
printf "\n${BLUE}=== Summary ===${NC}\n"
printf "Services: %d\n" "${#SERVICE_ARRAY[@]}"
total_targets=$(wc -l < "$TMPDIR/all_targets.txt" | xargs)
printf "Total unique targets: %s\n" "$total_targets"

# Count targets per service
printf "\nTargets per service:\n"
for service in "${SERVICE_ARRAY[@]}"; do
    count=$(wc -l < "$TMPDIR/$service.targets" | xargs)
    printf "  %-15s: %s/%s\n" "$service" "$count" "$total_targets"
done

# List missing targets per service
printf "\nMissing targets per service:\n"
for service in "${SERVICE_ARRAY[@]}"; do
    missing=$(comm -23 "$TMPDIR/all_targets.txt" "$TMPDIR/$service.targets" 2>/dev/null || true)
    if [[ -n "$missing" ]]; then
        printf "  ${RED}%s${NC}:\n" "$service"
        echo "$missing" | while read -r m; do
            printf "    - %s\n" "$m"
        done
    else
        printf "  ${GREEN}%s${NC}: (none missing)\n" "$service"
    fi
done

printf "\n"
