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
MEMORY_LIMIT="6G"  # Reduced for Raspberry Pi
SWAP_SIZE="100G"     # Add swap space

docker system prune -af --volumes

# Verify required secrets are loaded
if [ -z "$DOCKERHUB_TOKEN" ] || [ -z "$DATASET_URL" ]; then
    echo "Error: Required secrets not found in .secrets file!"
    echo "Please ensure DOCKERHUB_TOKEN and DATASET_URL are defined."
    exit 1
fi

# Create and enable swap
#echo "Setting up swap space..."
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
#echo '{
#  "default-runtime": "runc",
#  "runtimes": {
#    "runc": {
#      "path": "runc"
#    }
#  },
#  "builder": {
#    "gc": {
#      "enabled": true,
#      "defaultKeepStorage": "80GB"
#    }
#  },
#  "experimental": true,
#  "features": {
#    "buildkit": true
#  }
#}' | sudo tee /etc/docker/daemon.json

# Restart Docker to apply changes
#sudo systemctl restart docker

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

# Build and push multi-arch image with memory limits
echo "Building and pushing multi-arch image..."
DOCKER_BUILDKIT=1 docker buildx build \
    --output type=image,compression=zstd,compression-level=19 \
    --platform linux/amd64,linux/arm64 \
    --push \
    --memory $MEMORY_LIMIT \
    --memory-swap $MEMORY_LIMIT \
    -t "$REPO_NAME:full" \
    "$DOCKERFILE_PATH"

# Clean up swap
echo "Cleaning up swap..."
sudo swapoff /swapfile
sudo rm /swapfile

echo "Build and push completed successfully!"
echo "Find your image at: https://hub.docker.com/r/$REPO_NAME" 