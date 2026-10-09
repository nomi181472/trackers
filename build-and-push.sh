#!/usr/bin/env bash

set -Eeuo pipefail

# Docker Hub configuration
REGISTRY="docker.io"
BACKEND_IMAGE="docker.io/botonetics/tracker-sim-be:latest"
FRONTEND_IMAGE="docker.io/botonetics/tracker-sim-fe:latest"

# Resolve the project directory from this script's location
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Print status messages
info() {
    printf '\n\033[1;34m[INFO]\033[0m %s\n' "$1"
}

success() {
    printf '\033[1;32m[SUCCESS]\033[0m %s\n' "$1"
}

error() {
    printf '\033[1;31m[ERROR]\033[0m %s\n' "$1" >&2
}

# Stop if any command fails
trap 'error "Command failed on line $LINENO. Stopping."' ERR

# Check Podman installation
if ! command -v podman >/dev/null 2>&1; then
    error "Podman is not installed or is not in PATH."
    exit 1
fi

info "Checking Docker Hub login..."

# Check whether a Docker Hub login already exists
if podman login --get-login "$REGISTRY" >/dev/null 2>&1; then
    CURRENT_USER="$(podman login --get-login "$REGISTRY")"
    success "Already logged in to $REGISTRY as $CURRENT_USER"
else
    info "Not logged in. Please log in to Docker Hub."
    podman login "$REGISTRY"
    success "Docker Hub login completed."
fi

# Verify required Dockerfiles exist
if [[ ! -f "$PROJECT_DIR/backend/Dockerfile" ]]; then
    error "Backend Dockerfile not found: backend/Dockerfile"
    exit 1
fi

if [[ ! -f "$PROJECT_DIR/frontend/Dockerfile" ]]; then
    error "Frontend Dockerfile not found: frontend/Dockerfile"
    exit 1
fi

# Build backend
info "Building backend image..."
podman build \
    -t "$BACKEND_IMAGE" \
    -f "$PROJECT_DIR/backend/Dockerfile" \
    "$PROJECT_DIR/backend"

success "Backend image built."

# Build frontend
info "Building frontend image..."
podman build \
    -t "$FRONTEND_IMAGE" \
    -f "$PROJECT_DIR/frontend/Dockerfile" \
    "$PROJECT_DIR/frontend"

success "Frontend image built."

# Push backend directly through Podman
info "Pushing backend image to Docker Hub..."
podman push "$BACKEND_IMAGE"
success "Backend image pushed."

# Push frontend directly through Podman
info "Pushing frontend image to Docker Hub..."
podman push "$FRONTEND_IMAGE"
success "Frontend image pushed."

printf '\n'
success "Both images have been built and pushed successfully!"
printf '%s\n' \
    "Backend:  $BACKEND_IMAGE" \
    "Frontend: $FRONTEND_IMAGE"
