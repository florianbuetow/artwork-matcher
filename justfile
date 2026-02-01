# Default recipe: show available commands
_default:
    @just --list

# Show help information
help:
    @echo ""
    @clear
    @echo ""
    @echo "\033[0;34m=== Artwork Matcher ===\033[0m"
    @echo ""
    @echo "Available commands:"
    @just --list
    @echo ""

# === Full Stack (Docker) ===

# Start all services with Docker Compose
up:
    @echo ""
    @echo "\033[0;34m=== Starting All Services ===\033[0m"
    docker compose up -d
    @echo "\033[0;32m✓ Services started\033[0m"
    @echo ""

# Start all services in development mode (with hot reload)
up-dev:
    @echo ""
    @echo "\033[0;34m=== Starting All Services (Dev Mode) ===\033[0m"
    docker compose -f docker-compose.yml -f docker-compose.dev.yml up
    @echo ""

# Stop all services
down:
    @echo ""
    @echo "\033[0;34m=== Stopping All Services ===\033[0m"
    docker compose down
    @echo "\033[0;32m✓ Services stopped\033[0m"
    @echo ""

# View logs (optionally for a specific service)
logs service="":
    docker compose logs -f {{service}}

# Build all Docker images
build:
    @echo ""
    @echo "\033[0;34m=== Building All Images ===\033[0m"
    docker compose build
    @echo "\033[0;32m✓ Build complete\033[0m"
    @echo ""

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
    @echo "\033[0;34m=== Initializing All Services ===\033[0m"
    cd services/embeddings && just init
    cd services/search && just init
    cd services/geometric && just init
    cd services/gateway && just init
    cd tools && just init
    @echo "\033[0;32m✓ All services initialized\033[0m"
    @echo ""

# Destroy all virtual environments
destroy-all:
    @echo ""
    @echo "\033[0;34m=== Destroying All Virtual Environments ===\033[0m"
    cd services/embeddings && just destroy
    cd services/search && just destroy
    cd services/geometric && just destroy
    cd services/gateway && just destroy
    cd tools && just destroy
    @echo "\033[0;32m✓ All virtual environments removed\033[0m"
    @echo ""

# === Testing ===

# Run tests for all services
test-all:
    @echo ""
    @echo "\033[0;34m=== Running All Tests ===\033[0m"
    cd services/embeddings && just test
    cd services/search && just test
    cd services/geometric && just test
    cd services/gateway && just test
    @echo "\033[0;32m✓ All tests passed\033[0m"
    @echo ""

# Run tests for a specific service
test service:
    cd services/{{service}} && just test

# === CI ===

# Run CI checks for all services (verbose)
ci-all:
    #!/usr/bin/env bash
    set -e
    echo ""
    echo "\033[0;34m=== Running CI for All Services ===\033[0m"
    echo ""
    cd services/embeddings && just ci
    cd services/search && just ci
    cd services/geometric && just ci
    cd services/gateway && just ci
    echo ""
    echo "\033[0;32m✓ All CI checks passed\033[0m"
    echo ""

# Run CI checks for all services (quiet mode)
ci-all-quiet:
    #!/usr/bin/env bash
    set -e
    echo "\033[0;34m=== Running CI for All Services (Quiet Mode) ===\033[0m"

    echo "Checking embeddings..."
    cd services/embeddings && just ci-quiet

    echo "Checking search..."
    cd services/search && just ci-quiet

    echo "Checking geometric..."
    cd services/geometric && just ci-quiet

    echo "Checking gateway..."
    cd services/gateway && just ci-quiet

    echo ""
    echo "\033[0;32m✓ All CI checks passed\033[0m"
    echo ""

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

# Build the FAISS index from object images
build-index:
    cd tools && just build-index

# Evaluate accuracy against labels.csv
evaluate:
    cd tools && just evaluate
