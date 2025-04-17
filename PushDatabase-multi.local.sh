#!/bin/bash

# Exit on error
set -e

docker system prune -af
docker system prune -af --volumes

# Load environment variables from .secrets file
if [ -f ".secrets" ]; then
    echo "Loading secrets from .secrets file..."
    export $(cat .secrets | xargs)
else
    echo "Error: .secrets file not found!"
    exit 1
fi

# Configuration
DOCKERHUB_USERNAME="datagnomesteam"
REPO_NAME="datagnomesteam/project_database"
DOCKERFILE_PATH="db"
#MEMORY_LIMIT="2G"  # Reduced for Raspberry Pi; uncomment and adjust if needed
SWAP_SIZE="100G"     # Add swap space

docker system prune -af --volumes

# Verify required secrets are loaded
if [ -z "$DOCKERHUB_TOKEN" ] || [ -z "$DATASET_URL" ]; then
    echo "Error: Required secrets not found in .secrets file!"
    echo "Please ensure DOCKERHUB_TOKEN and DATASET_URL are defined."
    exit 1
fi

# Create and enable swap (if needed)
# Remove existing swapfile if it exists
if [ -f /swapfile ]; then
    echo "/swapfile exists. Removing it before creating a new one..."
    sudo swapoff /swapfile || true
    sudo rm /swapfile
fi
# Uncomment these lines to create a new swapfile if required
sudo fallocate -l $SWAP_SIZE /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Install required packages
echo "Installing required packages..."
sudo apt-get update
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    jq \
    gzip \
    tar \
    qemu-user-static \
    aria2 \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-compose-plugin

# Configure Docker memory limits
echo "Configuring Docker memory limits..."
sudo mkdir -p /etc/docker
# Uncomment and adjust the JSON below if you need to configure Docker daemon settings
# echo '{
#   "default-runtime": "runc",
#   "runtimes": {
#     "runc": {
#       "path": "runc"
#     }
#   },
#   "builder": {
#     "gc": {
#       "enabled": true,
#       "defaultKeepStorage": "80GB"
#     }
#   },
#   "experimental": true,
#   "features": {
#     "buildkit": true
#   }
# }' | sudo tee /etc/docker/daemon.json

# Restart Docker to apply changes if necessary
# sudo systemctl restart docker

# Set up Docker Buildx
echo "Setting up Docker Buildx..."
docker buildx create --use

# Log in to Docker Hub
echo "Logging in to Docker Hub..."
echo "$DOCKERHUB_TOKEN" | docker login -u $DOCKERHUB_USERNAME --password-stdin

# Download dataset if needed
if [ ! -f "$DOCKERFILE_PATH/pg.zst" ]; then
    echo "Downloading dataset..."
    aria2c -x16 -s16 -k10M \
        --file-allocation=falloc \
        --max-connection-per-server=16 \
        --min-split-size=10M \
        --max-concurrent-downloads=1 \
        --optimize-concurrent-downloads=true \
        --max-tries=50 \
        --retry-wait=5 \
        --check-certificate=false \
        --allow-overwrite=true \
        --auto-file-renaming=false \
        "$DATASET_URL" \
        -o "$DOCKERFILE_PATH/pg.zst"
fi

echo "Downloading buildx"
docker run --privileged --rm tonistiigi/binfmt --install all

# Build and push the amd64 image first
echo "Building and pushing amd64 image..."
DOCKER_BUILDKIT=1 docker buildx build \
    --platform linux/amd64 \
    --output type=image,compression=zstd,compression-level=19 \
    --push \
    -t "$REPO_NAME:full-amd64" \
    "$DOCKERFILE_PATH"

# Build and push the arm64 image next
echo "Building and pushing arm64 image..."
DOCKER_BUILDKIT=1 docker buildx build \
    --platform linux/arm64 \
    --output type=image,compression=zstd,compression-level=19 \
    --push \
    -t "$REPO_NAME:full-arm64" \
    "$DOCKERFILE_PATH"

# Create and push the multi-arch manifest
echo "Creating multi-arch manifest..."
docker manifest create "$REPO_NAME:full" \
    "$REPO_NAME:full-amd64" \
    "$REPO_NAME:full-arm64"

echo "Pushing multi-arch manifest..."
docker manifest push "$REPO_NAME:full"

# Clean up swap (if swap was enabled)
echo "Cleaning up swap..."
sudo swapoff /swapfile
sudo rm /swapfile

echo "Build and push completed successfully!"
echo "Find your image at: https://hub.docker.com/r/$REPO_NAME"
