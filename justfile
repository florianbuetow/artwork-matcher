# Default recipe: show available commands
_default:
    @just --list

# Show help information
help:
    @echo ""
    @clear
    @echo ""
    @printf "\033[0;34m=== Artwork Matcher ===\033[0m\n"
    @echo ""
    @echo "Available commands:"
    @just --list
    @echo ""

# === Docker (All Services) ===

# Start all services with Docker Compose
docker-up:
    @echo ""
    @printf "\033[0;34m=== Starting All Services (Docker) ===\033[0m\n"
    docker compose up -d
    @printf "\033[0;32m✓ Services started\033[0m\n"
    @echo ""

# Start all services in development mode (with hot reload)
docker-up-dev:
    @echo ""
    @printf "\033[0;34m=== Starting All Services (Docker Dev Mode) ===\033[0m\n"
    docker compose -f docker-compose.yml -f docker-compose.dev.yml up
    @echo ""

# Stop all services
docker-down:
    @echo ""
    @printf "\033[0;34m=== Stopping All Services ===\033[0m\n"
    docker compose down
    @printf "\033[0;32m✓ Services stopped\033[0m\n"
    @echo ""

# View Docker logs (optionally for a specific service)
docker-logs service="":
    docker compose logs -f {{service}}

# Build all Docker images
docker-build:
    @echo ""
    @printf "\033[0;34m=== Building All Docker Images ===\033[0m\n"
    docker compose build
    @printf "\033[0;32m✓ Build complete\033[0m\n"
    @echo ""

# Check health status of all services
status:
    #!/usr/bin/env bash
    printf "\n"
    printf "\033[0;34m=== Service Status ===\033[0m\n"
    printf "\n"

    check_service() {
        local name=$1
        local port=$2
        local health_response
        local info_response

        if health_response=$(curl -s --connect-timeout 2 "http://localhost:${port}/health" 2>/dev/null); then
            status=$(echo "$health_response" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
            uptime=$(echo "$health_response" | grep -o '"uptime":"[^"]*"' | cut -d'"' -f4)
            if [ "$status" = "healthy" ]; then
                printf "\033[0;32m✓ %s\033[0m (port %s) - healthy\n" "$name" "$port"
                [ -n "$uptime" ] && printf "  Uptime: %s\n" "$uptime"
                # Fetch additional info
                if info_response=$(curl -s --connect-timeout 2 "http://localhost:${port}/info" 2>/dev/null); then
                    version=$(echo "$info_response" | grep -o '"version":"[^"]*"' | cut -d'"' -f4)
                    model=$(echo "$info_response" | grep -o '"name":"[^"]*"' | head -1 | cut -d'"' -f4)
                    dimension=$(echo "$info_response" | grep -o '"embedding_dimension":[0-9]*' | cut -d':' -f2)
                    [ -n "$version" ] && printf "  Version: %s\n" "$version"
                    [ -n "$model" ] && printf "  Model: %s\n" "$model"
                    [ -n "$dimension" ] && printf "  Embedding dim: %s\n" "$dimension"
                fi
            else
                printf "\033[0;33m⚠ %s\033[0m (port %s) - %s\n" "$name" "$port" "${status:-unknown}"
            fi
        else
            printf "\033[0;31m✗ %s\033[0m (port %s) - not responding\n" "$name" "$port"
        fi
    }

    check_service "Gateway" 8000
    check_service "Embeddings" 8001
    check_service "Search" 8002
    check_service "Geometric" 8003

    printf "\n"

# === Individual Services (Local Development) ===

# Run embeddings service locally
run-embeddings:
    cd services/embeddings && just run

# Run search service locally
run-search:
    cd services/search && just run

# Run geometric service locally
run-geometric:
    cd services/geometric && just run

# Run gateway service locally
run-gateway:
    cd services/gateway && just run

# === Initialize All Services ===

# Initialize all service environments
init-all:
    @echo ""
    @printf "\033[0;34m=== Initializing All Services ===\033[0m\n"
    cd services/embeddings && just init
    cd services/search && just init
    cd services/geometric && just init
    cd services/gateway && just init
    cd tools && just init
    @printf "\033[0;32m✓ All services initialized\033[0m\n"
    @echo ""

# Destroy all virtual environments
destroy-all:
    @echo ""
    @printf "\033[0;34m=== Destroying All Virtual Environments ===\033[0m\n"
    cd services/embeddings && just destroy
    cd services/search && just destroy
    cd services/geometric && just destroy
    cd services/gateway && just destroy
    cd tools && just destroy
    @printf "\033[0;32m✓ All virtual environments removed\033[0m\n"
    @echo ""

# === Testing ===

# Run tests for all services
test-all:
    @echo ""
    @printf "\033[0;34m=== Running All Tests ===\033[0m\n"
    cd services/embeddings && just test
    cd services/search && just test
    cd services/geometric && just test
    cd services/gateway && just test
    @printf "\033[0;32m✓ All tests passed\033[0m\n"
    @echo ""

# Run tests for a specific service
test service:
    cd services/{{service}} && just test

# Run tests for tools scripts
test-tools:
    cd tools && uv run python -m unittest discover -s tests -p "test_*.py" -v

# === CI ===

# Run CI checks for all services (verbose)
ci-all:
    #!/usr/bin/env bash
    set -e
    printf "\n"
    printf "\033[0;34m=== Running CI for All Services ===\033[0m\n"
    printf "\n"
    cd services/embeddings && just ci
    cd services/search && just ci
    cd services/geometric && just ci
    cd services/gateway && just ci
    printf "\n"
    printf "\033[0;32m✓ All CI checks passed\033[0m\n"
    printf "\n"

# Run CI checks for all services (quiet mode)
ci-all-quiet:
    #!/usr/bin/env bash
    set -e
    printf "\033[0;34m=== Running CI for All Services (Quiet Mode) ===\033[0m\n"

    printf "Checking embeddings...\n"
    cd services/embeddings && just ci-quiet

    printf "Checking search...\n"
    cd services/search && just ci-quiet

    printf "Checking geometric...\n"
    cd services/geometric && just ci-quiet

    printf "Checking gateway...\n"
    cd services/gateway && just ci-quiet

    printf "\n"
    printf "\033[0;32m✓ All CI checks passed\033[0m\n"
    printf "\n"

# Run CI for a specific service
ci service:
    cd services/{{service}} && just ci

# === Code Quality (Root Level) ===

# Format all services
format-all:
    cd services/embeddings && just code-format
    cd services/search && just code-format
    cd services/geometric && just code-format
    cd services/gateway && just code-format

# === Data Pipeline ===

# Download diverse batch from Rijksmuseum (10 objects per type, configurable)
download-batch:
    cd tools && just download-batch

# Download Rijksmuseum data with custom options (e.g., just download --limit 100)
download *ARGS:
    cd tools && just download {{ ARGS }}

# Build the FAISS index from object images
build-index:
    cd tools && uv run python build_index.py --objects ../data/evaluation/objects --embeddings-url http://localhost:8001 --search-url http://localhost:8002

# Evaluate accuracy against labels.csv
evaluate:
    cd tools && uv run python evaluate.py --testdata ../data/evaluation --output ../reports/evaluation --gateway-url http://localhost:8000 --k 10 --threshold 0.0

# Run full E2E evaluation (starts Docker, builds index, evaluates)
run-evaluation:
    cd tools && uv run python run_evaluation.py --testdata ../data/evaluation --output ../reports/evaluation --gateway-url http://localhost:8000 --embeddings-url http://localhost:8001 --search-url http://localhost:8002 --geometric-url http://localhost:8003 --k 10 --threshold 0.0
