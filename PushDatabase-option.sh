#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (override via env or .secrets file)
# -----------------------------------------------------------------------------
DOCKERHUB_USERNAME="datagnomesteam"
REPO_NAME="datagnomesteam/project_database"
DOCKERFILE_PATH="db"
TAR_AMD64="full-amd64.tar"
TAR_ARM64="full-arm64.tar"
MANIFEST_TAG="full"

# Load secrets if present (e.g. DOCKERHUB_TOKEN, DATASET_URL, etc.)
if [ -f ".secrets" ]; then
  echo "Loading .secrets…"
  # strips comments and exports key=val
  export $(grep -vE '^\s*#' .secrets | xargs)
fi

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

function ensure_qemu() {
  echo ">> Ensuring QEMU emulation is installed for cross-build…"
  sudo su -c 'docker run --privileged --rm tonistiigi/binfmt --install all'
}

function ensure_builder() {
  # create a builder called "multiarch" if not already
  if ! docker buildx inspect multiarch >/dev/null 2>&1; then
    docker buildx create --name multiarch --use
  else
    docker buildx use multiarch
  fi
  docker buildx inspect --bootstrap
}

# -----------------------------------------------------------------------------
# build: compile + save locally
# -----------------------------------------------------------------------------
function build_images() {
  echo "=== BUILD STEP ==="

  ensure_qemu
  ensure_builder

  # (Optional) download dataset if needed—uncomment if you need it
  # if [ ! -f "$DOCKERFILE_PATH/pg.zst" ]; then
  #   aria2c ... -o "$DOCKERFILE_PATH/pg.zst"
  # fi

  echo ">> Building amd64 image and loading into Docker…"
  docker buildx build \
    --platform linux/amd64 \
    --load \
    -t "${REPO_NAME}:amd64" \
    "${DOCKERFILE_PATH}"

  echo ">> Saving amd64 image to ${TAR_AMD64}…"
  docker save "${REPO_NAME}:amd64" -o "${TAR_AMD64}"

  echo ">> Building arm64 image and loading into Docker…"
  docker buildx build \
    --platform linux/arm64 \
    --load \
    -t "${REPO_NAME}:arm64" \
    "${DOCKERFILE_PATH}"

  echo ">> Saving arm64 image to ${TAR_ARM64}…"
  docker save "${REPO_NAME}:arm64" -o "${TAR_ARM64}"

  echo
  echo "✔︎ Images built and saved:"
  echo "    • ${TAR_AMD64}"
  echo "    • ${TAR_ARM64}"
}

# -----------------------------------------------------------------------------
# push: load + push to Docker Hub + manifest
# -----------------------------------------------------------------------------
function push_images() {
  echo "=== PUSH STEP ==="

  # check files exist
  for f in "${TAR_AMD64}" "${TAR_ARM64}"; do
    if [ ! -f "$f" ]; then
      echo "Error: Required file '$f' not found. Run './$0 build' first."
      exit 1
    fi
  done

  echo ">> Loading images from tarballs…"
  docker load -i "${TAR_AMD64}"
  docker load -i "${TAR_ARM64}"

  if [ -z "${DOCKERHUB_TOKEN-}" ]; then
    echo "Error: DOCKERHUB_TOKEN not set in env or .secrets"
    exit 1
  fi

  echo ">> Logging in to Docker Hub as ${DOCKERHUB_USERNAME}…"
  echo "${DOCKERHUB_TOKEN}" | docker login -u "${DOCKERHUB_USERNAME}" --password-stdin

  echo ">> Pushing amd64 tag…"
  docker push "${REPO_NAME}:amd64"

  echo ">> Pushing arm64 tag…"
  docker push "${REPO_NAME}:arm64"

  echo ">> Creating multi‑arch manifest '${MANIFEST_TAG}'…"
  docker manifest create "${REPO_NAME}:${MANIFEST_TAG}" \
    "${REPO_NAME}:amd64" \
    "${REPO_NAME}:arm64"

  echo ">> Pushing multi‑arch manifest…"
  docker manifest push "${REPO_NAME}:${MANIFEST_TAG}"

  echo
  echo "✔︎ All tags pushed to https://hub.docker.com/r/${REPO_NAME}"
}

# -----------------------------------------------------------------------------
# entrypoint
# -----------------------------------------------------------------------------
case "${1:-}" in
  build) build_images ;;
  push)  push_images ;;
  *)
    echo "Usage: $0 {build|push}"
    echo "  build  → build both platforms locally and save as tarballs"
    echo "  push   → load from tarballs, push to Docker Hub, and publish manifest"
    exit 1
    ;;
esac
