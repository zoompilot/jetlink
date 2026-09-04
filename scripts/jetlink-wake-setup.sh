#!/bin/bash
# Arm every USB hub for remote wakeup, from the host.
#
# The comma is the gadget and hangs off the onboard Realtek hub, so a connect
# on a downstream port has to be signalled up by that hub before the root hub
# or tegra-xusb hear about it. Those ship enabled; the SuperSpeed hub does
# not, and at SuperSpeed it is the one in our path. Without this the Jetson
# never wakes when the comma presents the gadget - measured 2026-09-04, the
# comma's UDC sat at "default" for fifteen minutes and the box took its button.
#
# 99-jetlink-usb-wakeup.rules does this at boot. This runs again from the
# server unit's ExecStartPre, because that rule lives on the host filesystem
# and a re-flash loses it silently, at the cost of a whole drive.
#
# On the host and not in the container on purpose: /sys is mounted read-only
# in there, so the same writes from inside are a no-op.
set -u
armed=0
for dev in /sys/bus/usb/devices/*/; do
  [ -f "$dev/bDeviceClass" ] || continue
  [ "$(cat "$dev/bDeviceClass" 2>/dev/null)" = "09" ] || continue   # 09: hub
  w="$dev/power/wakeup"
  [ -f "$w" ] || continue
  [ "$(cat "$w" 2>/dev/null)" = "disabled" ] || continue
  if echo enabled > "$w" 2>/dev/null; then
    echo "jetlink-wake-setup: armed $(basename "$dev") for remote wakeup"
    armed=$((armed + 1))
  else
    echo "jetlink-wake-setup: could not arm $(basename "$dev") (need root)" >&2
  fi
done
[ "$armed" -eq 0 ] && echo "jetlink-wake-setup: all hubs already armed"
exit 0
