#!/bin/bash
# Arm every USB hub for remote wakeup, from the host.
#
# The gadget hangs off the onboard Realtek hub, which has to signal a connect up
# before the root hub or tegra-xusb hear it. That hub ships with wakeup disabled
# and at SuperSpeed it is the one in our path, so without this the comma
# presenting the gadget never wakes a sleeping Jetson.
#
# 99-jetlink-usb-wakeup.rules does this at boot; this runs again from the server
# unit's ExecStartPre, because a re-flash loses that rule silently.
#
# On the host, not in the container: /sys is read-only in there and the same
# writes are a no-op.
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
