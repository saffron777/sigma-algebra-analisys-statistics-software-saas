#!/usr/bin/env bash
# Publica la imagen en GHCR. Requiere Docker en marcha y un PAT con write:packages.
#   ./scripts/publish_docker.sh 1.0.0
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="ghcr.io/saffron777/sigma-algebra-analisys-statistics-software-saas"
TAG="${1:-1.0.0}"

docker login ghcr.io
docker buildx build \
  --platform linux/amd64 \
  -t "${IMAGE}:${TAG}" \
  -t "${IMAGE}:latest" \
  --push \
  .

echo "Publicada ${IMAGE}:${TAG} y :latest"
