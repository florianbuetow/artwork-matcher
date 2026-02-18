# Default recipe: show help
_default:
    @just help

# Show available commands
[private]
help:
    @clear
    @echo ""
    @printf "\033[0;34m=== Artwork Matcher ===\033[0m\n"
    @echo ""
    @printf "\033[1;33mLocal Development\033[0m\n"
    @printf "  \033[0;37mjust init-all         \033[0;34m Initialize all service environments\033[0m\n"
    @printf "  \033[0;37mjust start-all        \033[0;34m Start all services in background\033[0m\n"
    @printf "  \033[0;37mjust start-embeddings \033[0;34m Start embeddings service locally\033[0m\n"
    @printf "  \033[0;37mjust start-search     \033[0;34m Start search service locally\033[0m\n"
    @printf "  \033[0;37mjust start-geometric  \033[0;34m Start geometric service locally\033[0m\n"
    @printf "  \033[0;37mjust start-gateway    \033[0;34m Start gateway service locally\033[0m\n"
    @printf "  \033[0;37mjust stop-all         \033[0;34m Stop all locally running services\033[0m\n"
    @printf "  \033[0;37mjust stop-embeddings  \033[0;34m Stop embeddings service\033[0m\n"
    @printf "  \033[0;37mjust stop-search      \033[0;34m Stop search service\033[0m\n"
    @printf "  \033[0;37mjust stop-geometric   \033[0;34m Stop geometric service\033[0m\n"
    @printf "  \033[0;37mjust stop-gateway     \033[0;34m Stop gateway service\033[0m\n"
    @printf "  \033[0;37mjust status           \033[0;34m Check health status of all services\033[0m\n"
    @printf "  \033[0;37mjust destroy-all      \033[0;34m Destroy all virtual environments\033[0m\n"
    @echo ""
    @printf "\033[1;33mDocker\033[0m\n"
    @printf "  \033[0;37mjust docker-up        \033[0;34m Start all services\033[0m\n"
    @printf "  \033[0;37mjust docker-up-dev    \033[0;34m Start all services with hot reload\033[0m\n"
    @printf "  \033[0;37mjust docker-down      \033[0;34m Stop all services\033[0m\n"
    @printf "  \033[0;37mjust docker-logs      \033[0;34m View logs (optionally: just docker-logs <service>)\033[0m\n"
    @printf "  \033[0;37mjust docker-build     \033[0;34m Rebuild all Docker images from scratch\033[0m\n"
    @echo ""
    @printf "\033[1;33mEvaluation\033[0m\n"
    @printf "  \033[0;37mjust download-batch   \033[0;34m Download diverse batch from Rijksmuseum\033[0m\n"
    @printf "  \033[0;37mjust download <args>  \033[0;34m Download with custom options\033[0m\n"
    @printf "  \033[0;37mjust build-index      \033[0;34m Build FAISS index from object images\033[0m\n"
    @printf "  \033[0;37mjust evaluate         \033[0;34m Evaluate accuracy against labels.csv\033[0m\n"
    @printf "  \033[0;37mjust run-evaluation   \033[0;34m Full E2E evaluation pipeline\033[0m\n"
    @echo ""
    @printf "\033[1;33mCI & Code Quality\033[0m\n"
    @printf "  \033[0;37mjust test-all         \033[0;34m Run tests for all services\033[0m\n"
    @printf "  \033[0;37mjust test <service>   \033[0;34m Run tests for a specific service\033[0m\n"
    @printf "  \033[0;37mjust test-tools       \033[0;34m Run tests for tools scripts\033[0m\n"
    @printf "  \033[0;37mjust ci               \033[0;34m Run CI checks (all or: just ci <service>)\033[0m\n"
    @printf "  \033[0;37mjust ci-all           \033[0;34m Run CI checks for all services (verbose)\033[0m\n"
    @printf "  \033[0;37mjust ci-all-quiet     \033[0;34m Run CI checks for all services (quiet)\033[0m\n"
    @printf "  \033[0;37mjust format-all       \033[0;34m Auto-format all services\033[0m\n"
    @echo ""

# --- Local Development ---

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

# Start embeddings service locally
start-embeddings:
    @echo ""
    cd services/embeddings && just run
    @echo ""

# Start search service locally
start-search:
    @echo ""
    cd services/search && just run
    @echo ""

# Start geometric service locally
start-geometric:
    @echo ""
    cd services/geometric && just run
    @echo ""

# Start gateway service locally
start-gateway:
    @echo ""
    cd services/gateway && just run
    @echo ""

# Start all services in background
start-all:
    #!/usr/bin/env bash
    printf "\n"
    printf "\033[0;34m=== Starting All Services (Background) ===\033[0m\n"
    printf "\n"

    cd services/embeddings && uv run uvicorn embeddings_service.app:create_app --factory --host 0.0.0.0 --port 8001 &
    cd services/search && uv run uvicorn search_service.app:create_app --factory --host 0.0.0.0 --port 8002 &
    cd services/geometric && uv run uvicorn geometric_service.app:create_app --factory --host 0.0.0.0 --port 8003 &
    cd services/gateway && uv run uvicorn gateway.app:create_app --factory --host 0.0.0.0 --port 8000 &

    printf "Services starting in background...\n"

    printf "\n"

# Stop embeddings service
stop-embeddings:
    @echo ""
    cd services/embeddings && just kill
    @echo ""

# Stop search service
stop-search:
    @echo ""
    cd services/search && just kill
    @echo ""

# Stop geometric service
stop-geometric:
    @echo ""
    cd services/geometric && just kill
    @echo ""

# Stop gateway service
stop-gateway:
    @echo ""
    cd services/gateway && just kill
    @echo ""

# Stop all locally running services
stop-all:
    @echo ""
    @printf "\033[0;34m=== Stopping All Services (Local) ===\033[0m\n"
    cd services/embeddings && just kill
    cd services/search && just kill
    cd services/geometric && just kill
    cd services/gateway && just kill
    @printf "\033[0;32m✓ All services stopped\033[0m\n"
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
                    version=$(echo "$info_response" | grep -o '"version":"[^"]*"' | head -1 | cut -d'"' -f4)
                    model=$(echo "$info_response" | grep -o '"name":"[^"]*"' | head -1 | cut -d'"' -f4)
                    dimension=$(echo "$info_response" | grep -o '"embedding_dimension":[0-9]*' | head -1 | cut -d':' -f2)
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

# --- Docker ---

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
    @echo ""
    docker compose logs -f {{service}}
    @echo ""

# Build all Docker images (destroys existing images first)
docker-build:
    @echo ""
    @printf "\033[0;34m=== Destroying Existing Docker Images ===\033[0m\n"
    docker compose down --rmi all --volumes 2>/dev/null || true
    @printf "\033[0;34m=== Building All Docker Images ===\033[0m\n"
    docker compose build
    @printf "\033[0;32m✓ Build complete\033[0m\n"
    @echo ""

# --- Evaluation ---

# Download diverse batch from Rijksmuseum (10 objects per type, configurable)
download-batch:
    @echo ""
    cd tools && just download-batch
    @echo ""

# Download Rijksmuseum data with custom options (e.g., just download --limit 100)
download *ARGS:
    @echo ""
    cd tools && just download {{ ARGS }}
    @echo ""

# Build the FAISS index from object images
build-index:
    @echo ""
    cd tools && uv run python build_index.py --objects ../data/evaluation/objects --embeddings-url http://localhost:8001 --search-url http://localhost:8002
    @echo ""

# Evaluate accuracy against labels.csv
evaluate:
    @echo ""
    cd tools && uv run python evaluate.py --testdata ../data/evaluation --output ../reports/evaluation --gateway-url http://localhost:8000 --k 10 --threshold 0.0
    @echo ""

# Run full E2E evaluation (starts Docker, builds index, evaluates)
run-evaluation:
    @echo ""
    cd tools && uv run python run_evaluation.py --testdata ../data/evaluation --output ../reports/evaluation --gateway-url http://localhost:8000 --embeddings-url http://localhost:8001 --search-url http://localhost:8002 --geometric-url http://localhost:8003 --k 10 --threshold 0.0
    @echo ""

# --- CI & Code Quality ---

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
    @echo ""
    cd services/{{service}} && just test
    @echo ""

# Run tests for tools scripts
test-tools:
    @echo ""
    cd tools && uv run python -m unittest discover -s tests -p "test_*.py" -v
    @echo ""

# Run CI (all services by default, or one specific service)
ci service="all":
    #!/usr/bin/env bash
    set -e
    printf "\n"
    if [ "{{service}}" = "all" ]; then
        just ci-all
    else
        cd services/{{service}} && just ci
    fi
    printf "\n"

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
    printf "\n"
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

# Format all services
format-all:
    @echo ""
    cd services/embeddings && just code-format
    cd services/search && just code-format
    cd services/geometric && just code-format
    cd services/gateway && just code-format
    @echo ""
