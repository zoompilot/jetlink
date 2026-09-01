#!/usr/bin/env bash
# Run the server. Defaults to TCP for bench.
#
# In the car pass --transport usb: the Jetson is the USB *host* and the comma is
# the gadget, which is why /dev/bus/usb has to be visible in here. See
# docs/transport.md for why the roles are that way round.
set -euo pipefail
IMAGE="${IMAGE:-jetlink:latest}"
CACHE="${JETLINK_CACHE_HOST:-/mnt/data/jetlink}"
mkdir -p "$CACHE"
exec docker run --rm -it \
  --runtime nvidia \
  --network host \
  --ipc host \
  -v "$CACHE":/mnt/data/jetlink \
  -v /dev:/dev \
  -v /sys:/sys:ro \
  --name jetlink \
  "$IMAGE" "$@"
