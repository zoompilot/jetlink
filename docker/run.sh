#!/usr/bin/env bash
# Run the server. Defaults to TCP for bench.
#
# In the car pass --transport usb: the Jetson is the USB *host* and the comma is
# the gadget, which is why /dev/bus/usb has to be visible in here. See
# docs/transport.md for why the roles are that way round.
#
# /sys/power is mounted read-write on top of the read-only /sys so that
# --sleep-after can write /sys/power/state, and rtc0's real directory so that
# Sleeper can arm its wake backstop. Resolved rather than hardcoded: rtc0 is a
# PMIC RTC on one board and a Tegra one on another. --mount, not -v, because
# the resolved path contains colons (bpmp:i2c) and -v cannot parse those. On an always-on supply
# pass --transport usb --sleep-after 120; see jetlink/server/sleep.py.
#
# Arming the hubs for remote wakeup is NOT done from in here - /sys is
# read-only in the container. It is scripts/99-jetlink-usb-wakeup.rules at
# boot and scripts/jetlink-wake-setup.sh at server start, both on the host.
# Without it the comma presenting the gadget does not wake a sleeping Jetson.
set -euo pipefail
IMAGE="${IMAGE:-jetlink:latest}"
CACHE="${JETLINK_CACHE_HOST:-/mnt/data/jetlink}"
RTC="$(readlink -f /sys/class/rtc/rtc0 2>/dev/null || true)"
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
  ${RTC:+--mount "type=bind,source=$RTC,target=$RTC"} \
  --name jetlink \
  "$IMAGE" "$@"
