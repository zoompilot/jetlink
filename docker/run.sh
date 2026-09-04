#!/usr/bin/env bash
# Run the server. Defaults to TCP for bench.
#
# In the car pass --transport usb: the Jetson is the USB *host* and the comma is
# the gadget, which is why /dev/bus/usb has to be visible in here. See
# docs/transport.md for why the roles are that way round.
#
# /sys/power is mounted read-write on top of the read-only /sys so that
# --sleep-after can write /sys/power/state. On an always-on supply pass
# --transport usb --sleep-after 120; see jetlink/server/sleep.py.
set -euo pipefail
IMAGE="${IMAGE:-jetlink:latest}"
CACHE="${JETLINK_CACHE_HOST:-/mnt/data/jetlink}"
mkdir -p "$CACHE"
exec docker run --rm -it \
  --runtime nvidia \
  --network host \
  --ipc host \
  --device-cgroup-rule "c 189:* rmw" \
  -v "$CACHE":/mnt/data/jetlink \
  -v /dev:/dev \
  -v /sys:/sys:ro \
  -v /sys/power:/sys/power \
  --name jetlink \
  "$IMAGE" "$@"
