#!/usr/bin/env bash
# Run the server. Defaults to TCP for bench; in the car pass --transport usb,
# and --sleep-after 120 on an always-on supply (jetlink/server/sleep.py). The
# Jetson is the USB host, which is why /dev/bus/usb is visible in here.
#
# /sys/power is mounted read-write over the read-only /sys so --sleep-after can
# write /sys/power/state, and rtc0's real directory so Sleeper can arm its wake
# backstop. Resolved rather than hardcoded (PMIC RTC on one board, Tegra on
# another), and --mount not -v because that path contains colons (bpmp:i2c).
#
# Arming the hubs for remote wakeup is not done from in here: /sys is read-only
# in the container. 99-jetlink-usb-wakeup.rules and jetlink-wake-setup.sh do it
# on the host, and without it the gadget does not wake a sleeping Jetson.
#
# For a car install use install.sh instead: it sets all of this up as a service.
set -euo pipefail
IMAGE="${IMAGE:-jetlink:latest}"
CACHE="${JETLINK_CACHE_HOST:-/mnt/data/jetlink}"
RTC="$(readlink -f /sys/class/rtc/rtc0 2>/dev/null || true)"
# the JetPack 6 image predates --gpus on Jetson; everything else wants both
GPU=(--runtime nvidia --gpus all)
if grep -q '^# R36 ' /etc/nv_tegra_release 2>/dev/null; then GPU=(--runtime nvidia); fi
mkdir -p "$CACHE"
exec docker run --rm -it \
  "${GPU[@]}" \
  --network host \
  --ipc host \
  --device-cgroup-rule "c 189:* rmw" \
  -v "$CACHE":/var/cache/jetlink -e JETLINK_CACHE=/var/cache/jetlink \
  -v /dev:/dev \
  -v /sys:/sys:ro \
  -v /sys/power:/sys/power \
  ${RTC:+--mount "type=bind,source=$RTC,target=$RTC"} \
  --name jetlink \
  "$IMAGE" "$@"
