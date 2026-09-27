#!/usr/bin/env bash
#
# Copyright (c) 2026-, Zeph Leggett.
# This file is part of jetlink and is licensed under the MIT License.
#
# Present the comma's USB-C port as a USB network adapter, for an iPhone
# running Jetlink on the other end of one cable.
#
# An iPhone cannot use jetlink's own gadget (scripts/setup_gadget.sh): iOS
# gives apps no access to a vendor-specific USB device. It does drive USB
# Ethernet (CDC-NCM and CDC-ECM) itself, so this makes the comma one, gives
# it 192.168.60.1, and the comma's TCP client reaches the phone at
# 192.168.60.2 over the cable.
#
# NOT YET VALIDATED WITH AN IPHONE. It needs a kernel with the NCM or ECM
# gadget function (CONFIG_USB_CONFIGFS_NCM or _ECM) and fails with that
# reason if there is none. With zoompilot's JetlinkEndpoint set and its
# Accelerator Link off, run as root after every boot:
#
#   echo -n 192.168.60.2:5599 > /data/params/d/JetlinkEndpoint
#   sudo bash setup_net_gadget.sh
#
# zoompilot sets up jetlink's own FunctionFS gadget at boot, which holds the
# port. This releases it, and clears the "gadget torn down" error its
# teardown leaves in /dev/shm/jetlink-gadget: zoompilot reads any error
# there as "no link", and would not try the TCP endpoint either.
#
# Plug an iPhone straight into the comma and the comma tries to power it and
# reboots. Go through a hub on the phone and a USB-A to USB-C cable to the
# comma: the A end can only supply power, so the comma never sources it.
#
# then on the iPhone: Settings > Ethernet > (the new adapter) > Configure IP >
# Manual, address 192.168.60.2, subnet mask 255.255.255.0, no router.
#
#   sudo ios/comma/setup_net_gadget.sh --teardown
#
# To find out first whether this comma can do it at all, without changing
# anything it depends on (the check makes a throwaway gadget, tries the
# functions, and removes it; it never binds the controller):
#
#   sudo ios/comma/setup_net_gadget.sh --check
set -euo pipefail

GADGET=/sys/kernel/config/usb_gadget/jetlink-net
CONFIGFS=/sys/kernel/config
STATUS_FILE=${JETLINK_STATUS_FILE:-/dev/shm/jetlink-net-gadget}
# pid.codes test allocations, as the FunctionFS gadget uses 0x0001; get a real
# PID before distributing this
VID=${JETLINK_VID:-0x1209}
PID=${JETLINK_NET_PID:-0x0002}
COMMA_ADDR=${JETLINK_COMMA_ADDR:-192.168.60.1/24}
PHONE_ADDR=${JETLINK_PHONE_ADDR:-192.168.60.2}
# NCM first: it batches packets, which a 460 KB frame benefits from. ECM is
# the older class and the fallback.
FUNCTIONS=${JETLINK_NET_FUNCTIONS:-"ncm ecm"}
# jetlink's own FunctionFS gadget, as zoompilot carries it, and its status file
JETLINK_REPO=${JETLINK_REPO:-/data/openpilot/jetlink_repo}
FFS_STATUS=/dev/shm/jetlink-gadget

status() {
  { echo "$1" > "$STATUS_FILE" && chmod 0644 "$STATUS_FILE"; } 2>/dev/null || true
}

fail() {
  echo "jetlink-net: $1" >&2
  status "error: $1"
  # Leave nothing half-made: a gadget that failed partway still holds a
  # function the next run would trip over.
  trap - ERR
  teardown
  exit 1
}

teardown() {
  if [[ -d "$GADGET" ]]; then
    echo "" > "$GADGET/UDC" 2>/dev/null || true
    for f in "$GADGET"/configs/c.1/*.usb0; do
      [[ -L "$f" ]] && rm -f "$f"
    done
    rmdir "$GADGET/configs/c.1/strings/0x409" 2>/dev/null || true
    rmdir "$GADGET/configs/c.1" 2>/dev/null || true
    for f in "$GADGET"/functions/*.usb0; do
      [[ -d "$f" ]] && rmdir "$f" 2>/dev/null || true
    done
    rmdir "$GADGET/strings/0x409" 2>/dev/null || true
    rmdir "$GADGET" 2>/dev/null || true
  fi
}

check() {
  local ok=1
  echo "kernel: $(uname -r)"
  if mountpoint -q "$CONFIGFS" && [[ -d "$CONFIGFS/usb_gadget" ]]; then
    echo "configfs USB gadgets: yes"
  else
    echo "configfs USB gadgets: NO (no $CONFIGFS/usb_gadget)"
    ok=0
  fi
  shopt -s nullglob
  local udcs=(/sys/class/udc/*)
  shopt -u nullglob
  if [[ ${#udcs[@]} -gt 0 ]]; then
    for u in "${udcs[@]}"; do echo "device controller: $(basename "$u"), state $(cat "$u/state" 2>/dev/null || echo unknown)"; done
  else
    echo "device controller: NONE"
    ok=0
  fi
  for g in "$CONFIGFS"/usb_gadget/*/; do
    [[ -d "$g" ]] || continue
    echo "gadget $(basename "$g"): bound to '$(cat "$g/UDC" 2>/dev/null)'"
  done
  if [[ -r /proc/config.gz ]]; then
    zcat /proc/config.gz | grep -E '^CONFIG_USB_(CONFIGFS_(NCM|ECM|ECM_SUBSET|RNDIS|F_FS)|F_NCM|F_ECM)=' | sed 's/^/kernel option: /' || true
  fi
  for port in /sys/class/typec/port*; do
    [[ -d "$port" ]] || continue
    echo "USB-C $(basename "$port"): data role $(cat "$port/data_role" 2>/dev/null), power role $(cat "$port/power_role" 2>/dev/null)"
  done
  if [[ $ok -eq 1 ]]; then
    local probe="$CONFIGFS/usb_gadget/jetlink-net-probe" found=""
    mkdir -p "$probe" 2>/dev/null || true
    for kind in ncm ecm; do
      if mkdir "$probe/functions/$kind.probe" 2>/dev/null; then
        found="$found $kind"
        rmdir "$probe/functions/$kind.probe" 2>/dev/null || true
      fi
    done
    rmdir "$probe" 2>/dev/null || true
    if [[ -n "$found" ]]; then
      echo "network gadget functions:$found"
      echo "RESULT: this comma can present a USB network adapter"
      return 0
    fi
    echo "network gadget functions: none"
  fi
  echo "RESULT: this comma cannot present a USB network adapter; use USB-C Ethernet adapters instead"
  return 1
}

if [[ "${1:-}" == "--check" ]]; then
  [[ $EUID -eq 0 ]] || { echo "run --check as root (sudo)" >&2; exit 1; }
  check
  exit $?
fi

if [[ "${1:-}" == "--teardown" ]]; then
  teardown
  status "error: gadget torn down"
  echo "jetlink network gadget torn down"
  exit 0
fi

[[ $EUID -eq 0 ]] || fail "setup_net_gadget.sh must run as root"
trap 'fail "line $LINENO: $BASH_COMMAND failed"' ERR

if [[ -d "/lib/modules/$(uname -r)" ]]; then
  for m in configfs libcomposite usb_f_ncm usb_f_ecm; do
    modprobe "$m" 2>/dev/null || true
  done
fi

mountpoint -q "$CONFIGFS" || mount -t configfs none "$CONFIGFS" 2>/dev/null || true
mountpoint -q "$CONFIGFS" || fail "no configfs at $CONFIGFS; this kernel cannot configure a USB gadget"
[[ -d "$CONFIGFS/usb_gadget" ]] || fail "kernel has no USB gadget support (CONFIG_USB_LIBCOMPOSITE)"

shopt -s nullglob
udcs=(/sys/class/udc/*)
shopt -u nullglob
[[ ${#udcs[@]} -gt 0 ]] || fail "no USB device controller in /sys/class/udc; this device cannot act as a USB gadget"
UDC=${JETLINK_UDC:-$(basename "${udcs[0]}")}

# jetlink's own gadget, which zoompilot sets up at boot, is released; any
# other gadget holding the controller is someone else's and left alone.
for other in "$CONFIGFS"/usb_gadget/*/UDC; do
  if [[ -e "$other" ]]; then
    owner=$(basename "$(dirname "$other")")
    bound=$(cat "$other" 2>/dev/null || true)
    if [[ "$owner" == "jetlink" && -n "$bound" ]]; then
      echo "jetlink-net: releasing jetlink's USB gadget (turn Accelerator Link off first)"
      [[ -f "$JETLINK_REPO/scripts/setup_gadget.sh" ]] || fail "jetlink's gadget holds the port, and there is no $JETLINK_REPO/scripts/setup_gadget.sh to tear it down; set JETLINK_REPO"
      bash "$JETLINK_REPO/scripts/setup_gadget.sh" --teardown >/dev/null 2>&1 || true
      [[ -z "$(cat "$other" 2>/dev/null || true)" ]] || fail "jetlink's USB gadget would not let go of the port; turn Accelerator Link off and try again"
    elif [[ "$owner" != "jetlink-net" && -n "$bound" ]]; then
      fail "USB gadget '$owner' already holds the device controller ($bound); tear it down first"
    fi
  fi
done

teardown
mkdir -p "$GADGET" || fail "could not create the gadget at $GADGET"
cd "$GADGET"

echo "$VID"   > idVendor
echo "$PID"   > idProduct
echo 0x0100   > bcdDevice
echo 0x0320   > bcdUSB            # 3.2: advertise SuperSpeed
echo 0x02     > bDeviceClass      # Communications: a CDC network device
echo 0x00     > bDeviceSubClass
echo 0x00     > bDeviceProtocol

# The device tree carries a serial on some commas and not others (AGNOS on a
# 4.9.103 kernel has none); any stable string will do.
serial=$({ tr -d '\0' < /proc/device-tree/serial-number; } 2>/dev/null || cat /etc/machine-id 2>/dev/null || echo 0001)
mkdir -p strings/0x409
echo "zoompilot"              > strings/0x409/manufacturer
echo "jetlink network link"   > strings/0x409/product
echo "$serial"                > strings/0x409/serialnumber

mkdir -p configs/c.1/strings/0x409
echo "jetlink network" > configs/c.1/strings/0x409/configuration
# self-powered: the comma runs from the car, and the phone must not feed it
echo 0xC0 > configs/c.1/bmAttributes
echo 8    > configs/c.1/MaxPower

# Stable, locally administered MAC addresses from the serial, so the phone
# sees the same adapter every drive and keeps its manual address for it.
hash=$(printf '%s' "$serial" | md5sum | cut -c1-10)
mac() { printf '%s:%s:%s:%s:%s:%s' "$1" "${hash:0:2}" "${hash:2:2}" "${hash:4:2}" "${hash:6:2}" "${hash:8:2}"; }

chosen=""
for kind in $FUNCTIONS; do
  if mkdir -p "functions/$kind.usb0" 2>/dev/null; then
    chosen=$kind
    break
  fi
done
[[ -n "$chosen" ]] || fail "kernel has none of the '$FUNCTIONS' gadget functions (CONFIG_USB_CONFIGFS_NCM / _ECM); the comma cannot present a network adapter"
# The comma's end, then the phone's. Qualcomm's 4.9 kernel refuses these
# writes (ENODEV) and picks its own addresses; that works, the phone just may
# see a new adapter after a reboot and need its manual address again.
if ! { echo "$(mac 02)" > "functions/$chosen.usb0/dev_addr" && echo "$(mac 06)" > "functions/$chosen.usb0/host_addr"; } 2>/dev/null; then
  echo "jetlink-net: the kernel chose the adapter's MAC addresses itself" >&2
fi
ln -s "$GADGET/functions/$chosen.usb0" "configs/c.1/$chosen.usb0"

echo "$UDC" > UDC || fail "could not bind the gadget to $UDC"

ifname=$(cat "functions/$chosen.usb0/ifname" 2>/dev/null || echo usb0)
for _ in $(seq 1 50); do
  [[ -e "/sys/class/net/$ifname" ]] && break
  sleep 0.1
done
[[ -e "/sys/class/net/$ifname" ]] || fail "the $chosen function did not create a network interface"
ip addr flush dev "$ifname" 2>/dev/null || true
ip addr add "$COMMA_ADDR" dev "$ifname"
ip link set "$ifname" up

status ok
# jetlink's teardown leaves "error: gadget torn down" behind, which zoompilot
# reads as no link at all. The link is this gadget now.
if grep -qs "torn down" "$FFS_STATUS"; then
  rm -f "$FFS_STATUS"
fi
echo "USB network gadget ($chosen) bound to $UDC as $ifname, comma at $COMMA_ADDR"
echo "on the iPhone: Settings > Ethernet > Configure IP > Manual, $PHONE_ADDR / 255.255.255.0"
echo "on the comma:  echo -n $PHONE_ADDR:5599 > /data/params/d/JetlinkEndpoint"
