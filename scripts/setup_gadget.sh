#!/usr/bin/env bash
#
# Copyright (c) 2026-, Zeph Leggett.
# This file is part of jetlink and is licensed under the MIT License.
#
# Bring up the jetlink USB gadget on the comma. Run as root, at boot, before
# anything opens the link.
#
# The comma is the USB *device* and the Jetson is the host, so this runs on the
# comma: AGNOS has the gadget drivers built in, while an L4T rootfs usually has
# them stripped. See docs/transport.md.
#
# Deliberately does NOT bind the UDC: a FunctionFS gadget cannot attach to a
# controller until its descriptors have been written, and whoever opens ep0
# writes them and binds afterwards (--gadget/--udc).
#
#   sudo scripts/setup_gadget.sh
#
# Undo with: sudo scripts/setup_gadget.sh --teardown
#
# On failure the reason is left in $STATUS_FILE as well as on stderr, so the
# openpilot side can tell the user why the link is unavailable instead of
# silently staying on the small model.
set -euo pipefail

GADGET=/sys/kernel/config/usb_gadget/jetlink
FFS_MOUNT=${FFS_MOUNT:-/dev/ffs-jetlink}
FFS_NAME=jetlink
CONFIGFS=/sys/kernel/config
# tmpfs on purpose: this is per-boot state and the comma's flash is precious.
STATUS_FILE=${JETLINK_STATUS_FILE:-/dev/shm/jetlink-gadget}
# pid.codes test allocation. Get a real PID before distributing this.
VID=${JETLINK_VID:-0x1209}
PID=${JETLINK_PID:-0x0001}

status() {
  # Best effort: a device with no /dev/shm still gets the stderr line.
  { echo "$1" > "$STATUS_FILE" && chmod 0644 "$STATUS_FILE"; } 2>/dev/null || true
}

fail() {
  echo "jetlink: $1" >&2
  status "error: $1"
  exit 1
}

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
  status "error: gadget torn down"
  echo "jetlink gadget torn down"
  exit 0
fi

[[ $EUID -eq 0 ]] || fail "setup_gadget.sh must run as root"

# Only worth trying where a module tree exists at all. AGNOS builds the gadget
# drivers into the kernel and ships no /lib/modules, so an unconditional
# modprobe here fails on every comma and teaches you nothing - which is exactly
# why the checks below test for the *result* instead of for the module.
if [[ -d "/lib/modules/$(uname -r)" ]]; then
  for m in configfs libcomposite usb_f_fs; do
    modprobe "$m" 2>/dev/null || true
  done
fi

mountpoint -q "$CONFIGFS" || mount -t configfs none "$CONFIGFS" 2>/dev/null || true
mountpoint -q "$CONFIGFS" || fail "no configfs at $CONFIGFS; this kernel cannot configure a USB gadget"

# The directory only appears once the gadget framework is in the kernel. Its
# absence is the one failure a user cannot fix from userspace: it means this
# AGNOS build has no CONFIG_USB_LIBCOMPOSITE, and jetlink cannot work here.
[[ -d "$CONFIGFS/usb_gadget" ]] || fail "kernel has no USB gadget support (CONFIG_USB_LIBCOMPOSITE); jetlink needs an AGNOS build that has it"

# FunctionFS is what carries our two bulk endpoints. Without it the gadget
# would build and then have nothing to attach.
grep -qw functionfs /proc/filesystems || fail "kernel has no FunctionFS (CONFIG_USB_FUNCTIONFS); jetlink cannot present its endpoints"

# A device controller has to exist before anything can be a gadget. A comma has
# exactly one; a machine wired host-only has none.
shopt -s nullglob
udcs=("/sys/class/udc"/*)
shopt -u nullglob
[[ ${#udcs[@]} -gt 0 ]] || fail "no USB device controller in /sys/class/udc; this device cannot act as a USB gadget"

# Refuse to fight another gadget for the controller rather than silently
# unbinding whatever owns it.
for other in "$CONFIGFS"/usb_gadget/*/UDC; do
  if [[ -e "$other" ]]; then
    owner=$(basename "$(dirname "$other")")
    bound=$(cat "$other" 2>/dev/null || true)
    if [[ "$owner" != "jetlink" && -n "$bound" ]]; then
      fail "USB gadget '$owner' already holds the device controller ($bound); tear it down first"
    fi
  fi
done

mkdir -p "$GADGET" || fail "could not create the gadget at $GADGET"
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

# configfs instantiates the function on mkdir, so this is where a kernel that
# reports functionfs but has no usb_f_fs gadget function actually shows up.
mkdir -p "functions/ffs.$FFS_NAME" ||
  fail "kernel has no ffs gadget function (CONFIG_USB_CONFIGFS_F_FS); jetlink cannot present its endpoints"
ln -sf "$GADGET/functions/ffs.$FFS_NAME" "configs/c.1/ffs.$FFS_NAME" 2>/dev/null || true

mkdir -p "$FFS_MOUNT"
# Mount owned by the user openpilot runs as. launch_chffrplus.sh runs as
# `comma`, so a root-only mount means modeld and jetlinkd cannot open the
# endpoints at all - and Path.exists() on them raises PermissionError rather
# than returning False, which hides the problem.
FFS_USER="${JETLINK_USER:-comma}"
if id -u "$FFS_USER" >/dev/null 2>&1; then
  FFS_OPTS="uid=$(id -u "$FFS_USER"),gid=$(id -g "$FFS_USER")"
else
  FFS_OPTS=""
fi
mountpoint -q "$FFS_MOUNT" || mount -t functionfs ${FFS_OPTS:+-o "$FFS_OPTS"} "$FFS_NAME" "$FFS_MOUNT" ||
  fail "could not mount functionfs at $FFS_MOUNT"

# The one check that proves the whole chain: ep0 is what a client opens to write
# descriptors and bind the controller. Everything above can look right and this
# still be missing.
[[ -e "$FFS_MOUNT/ep0" ]] || fail "functionfs mounted at $FFS_MOUNT but has no ep0"

# The client binds the UDC, and it does so as the openpilot user, so hand it
# that one attribute. Everything else in the gadget stays root-owned.
if [ -n "$FFS_OPTS" ]; then
  chown "$FFS_USER" "$GADGET/UDC" 2>/dev/null || true
fi

status ok
echo "gadget ready at $GADGET"
echo "functionfs mounted at $FFS_MOUNT"
echo "available UDCs: $(ls /sys/class/udc | tr '\n' ' ')"
echo "now start the server; it writes the descriptors and binds the UDC"
