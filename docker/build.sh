#!/usr/bin/env bash
# Build the server image on the machine that will run it: an arm64 CUDA image
# is not worth cross-building. docker/Dockerfile serves NVIDIA PCs and Jetsons on
# JetPack 7.2 or newer; a Jetson on JetPack 6 (L4T r36) needs
# docker/Dockerfile.jetpack6. install.sh does all of this for you.
#
# IMAGE names the result (default jetlink:latest); DOCKERFILE overrides the choice.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${IMAGE:-jetlink:latest}"
if [ -z "${DOCKERFILE:-}" ]; then
  DOCKERFILE=docker/Dockerfile
  if grep -q '^# R36 ' /etc/nv_tegra_release 2>/dev/null; then
    DOCKERFILE=docker/Dockerfile.jetpack6
  fi
fi
echo "building $IMAGE from $DOCKERFILE"
# host networking: Docker 28 on a JetPack 6 kernel cannot give a build step a
# bridge network (no iptables raw table)
exec docker build --network host -f "$DOCKERFILE" -t "$IMAGE" .
