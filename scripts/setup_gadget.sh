#!/usr/bin/env bash
#
# Copyright (c) 2026-, Zeph Leggett.
# This file is part of jetlink and is licensed under the MIT License.
#
# Bring up the jetlink USB gadget on the Jetson. Run as root, before the server.
#
# Deliberately does NOT bind the UDC: a FunctionFS gadget cannot attach to a
# controller until its descriptors have been written, and the server writes
# them when it opens ep0. The server binds afterwards (--gadget/--udc).
#
#   sudo scripts/setup_gadget.sh
#   docker run ... jetlink:latest --transport ffs
#
# Undo with: sudo scripts/setup_gadget.sh --teardown
set -euo pipefail

GADGET=/sys/kernel/config/usb_gadget/jetlink
FFS_MOUNT=${FFS_MOUNT:-/dev/ffs-jetlink}
FFS_NAME=jetlink
# pid.codes test allocation. Get a real PID before distributing this.
VID=${JETLINK_VID:-0x1209}
PID=${JETLINK_PID:-0x0001}

if [[ "${1:-}" == "--teardown" ]]; then
  if [[ -d "$GADGET" ]]; then
    echo "" > "$GADGET/UDC" 2>/dev/null || true
    rm -f "$GADGET/configs/c.1/ffs.$FFS_NAME" 2>/dev/null || true
    rmdir "$GADGET/configs/c.1/strings/0x409" 2>/dev/null || true
    rmdir "$GADGET/configs/c.1" 2>/dev/null || true
    rmdir "$GADGET/functions/ffs.$FFS_NAME" 2>/dev/null || true
    rmdir "$GADGET/strings/0x409" 2>/dev/null || true
    rmdir "$GADGET" 2>/dev/null || true
  fi
  # A plain umount can block, or segfault, on a FunctionFS instance whose
  # userspace owner died with endpoints still open. Lazy-detach instead: it
  # unhooks the mount immediately and lets the kernel finish when it can.
  umount -l "$FFS_MOUNT" 2>/dev/null || umount "$FFS_MOUNT" 2>/dev/null || true
  rmdir "$FFS_MOUNT" 2>/dev/null || true
  echo "jetlink gadget torn down"
  exit 0
fi

[[ $EUID -eq 0 ]] || { echo "run as root" >&2; exit 1; }

modprobe libcomposite 2>/dev/null || true
mountpoint -q /sys/kernel/config || mount -t configfs none /sys/kernel/config

mkdir -p "$GADGET"
cd "$GADGET"

echo "$VID"   > idVendor
echo "$PID"   > idProduct
echo 0x0100   > bcdDevice
echo 0x0320   > bcdUSB            # 3.2: advertise SuperSpeed
echo 0x00     > bDeviceClass      # class is per-interface (vendor specific)

mkdir -p strings/0x409
echo "zoompilot"                              > strings/0x409/manufacturer
echo "jetlink"                                > strings/0x409/product
echo "$(cat /proc/device-tree/serial-number 2>/dev/null | tr -d '\0' || echo 0001)" \
                                              > strings/0x409/serialnumber

mkdir -p configs/c.1/strings/0x409
echo "jetlink inference link" > configs/c.1/strings/0x409/configuration
# Self-powered, and ask for as little as the spec allows. The Jetson runs from
# its own ignition-gated 12 V feed; it must never try to draw from the comma.
echo 0xC0 > configs/c.1/bmAttributes
echo 8    > configs/c.1/MaxPower

mkdir -p "functions/ffs.$FFS_NAME"
ln -sf "$GADGET/functions/ffs.$FFS_NAME" "configs/c.1/ffs.$FFS_NAME" 2>/dev/null || true

mkdir -p "$FFS_MOUNT"
mountpoint -q "$FFS_MOUNT" || mount -t functionfs "$FFS_NAME" "$FFS_MOUNT"

echo "gadget ready at $GADGET"
echo "functionfs mounted at $FFS_MOUNT"
echo "available UDCs: $(ls /sys/class/udc | tr '\n' ' ')"
echo "now start the server; it writes the descriptors and binds the UDC"
