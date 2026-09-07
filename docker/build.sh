#!/usr/bin/env bash
# build on the Jetson itself: an arm64 CUDA image, not worth cross-building
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${IMAGE:-jetlink:latest}"
exec docker build -f docker/Dockerfile -t "$IMAGE" .
