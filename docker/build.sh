#!/usr/bin/env bash
# Build the server image on the Jetson itself (it is an arm64 CUDA image;
# cross-building it from a Mac is not worth the trouble).
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${IMAGE:-jetlink:latest}"
exec docker build -f docker/Dockerfile -t "$IMAGE" .
