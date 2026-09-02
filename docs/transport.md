# Choosing a transport

Bandwidth was never the question. 533 KB/frame at 20 Hz is 85 Mbit/s, against
5 Gbit/s for USB 3 or 1 Gbit/s for ethernet. What decides the design is **what
is actually built into the two kernels**, and on both machines the answer is
surprising. Check this before buying hardware.

## What the comma has (AGNOS, kernel 4.9.103, comma mici)

Read from `/proc/config.gz` on a comma 3X:

| | |
|---|---|
| `CONFIG_USB_NET_CDC_NCM` | **not set** |
| `CONFIG_USB_NET_CDC_ETHER` | **not set** |
| `CONFIG_USB_NET_RNDIS_HOST` | **not set** |
| `CONFIG_USB_NET_CDC_SUBSET` / `ZAURUS` / `QMI_WWAN` | `=y` |
| `CONFIG_USB_RTL8152` | **`=y`** |
| `CONFIG_USB_NET_AX8817X`, `..._AX88179_178A` | **`=y`** |
| `CONFIG_USB_LIBCOMPOSITE`, `CONFIG_USB_CONFIGFS`, `..._CONFIGFS_NCM` | `=y` |
| loadable modules | none at all — monolithic kernel, empty `lsmod` |

Two consequences:

1. **A USB ethernet *gadget* on the Jetson will never enumerate on a comma.**
   There is no host-side NCM/ECM/RNDIS driver and no way to add one without
   rebuilding AGNOS's kernel.
2. **A USB ethernet *adapter* works out of the box.** Realtek RTL8152/8153 and
   ASIX AX88179 are built in.

Raw USB needs no host driver at all: libusb goes through usbfs, which is how
openpilot already talks to the panda and to chestnut.

The USB-C port is a host port (`usb2`, an xHCI USB-3 root hub, nothing attached)
and its VBUS is switchable at
`/sys/kernel/debug/regulator/smb2-vbus/enable`.

## What the Jetson has (Orin Nano, L4T r36.4)

`/proc/config.gz` says `CONFIG_USB_LIBCOMPOSITE=m`, `CONFIG_USB_F_FS=m`,
`CONFIG_USB_CONFIGFS_NCM=y` — but on a **stripped rootfs the modules may not be
installed**. On the unit this was developed against, `/lib/modules` holds 166
modules, essentially all NVIDIA-specific: `tegra-xudc.ko` (the device
controller) is present, `libcomposite.ko` and `usb_f_fs.ko` are not, there are
no apt sources, and `/` is 100% full. Check before planning on USB:

```bash
modinfo libcomposite usb_f_fs      # or: find /lib/modules/$(uname -r) -name 'usb_f_*'
ls /sys/class/udc/                 # the device controller, e.g. 3550000.usb
cat /sys/class/udc/*/maximum_speed # want: super-speed
```

If they are missing, assume they stay missing. On the Orin Nano Super devkit
(L4T r36.4.3, kernel `5.15.148-l4t-r36.4-1012.12+g8dc079d5c8c4`) there is no
`/usr/src`, no `/lib/modules/$(uname -r)/build` symlink, no `linux-headers`
package, and `CONFIG_MODVERSIONS=y`, so an out-of-tree build needs the exact
`Module.symvers` from that kernel build. Getting there means pulling NVIDIA's
`public_sources` for the matching L4T, reproducing the `+g<sha>` vermagic and
building the tree: a few hours, and a custom kernel to maintain afterwards.

This is why the comma is the gadget and the Jetson is the host, and why it
cannot be swapped over. jetlink supports the inversion in software already
(`JetlinkClient.open_usb` on the comma, server `--transport ffs`); the blocker
is only ever the Jetson's kernel.

## The two supported transports

### Do not use the Jetson's Type-C port

It looks like the right port and the device tree agrees: `usb2-0` is `mode=otg`
with a `usb-role-switch` and a `vbus-supply`, and `usb3-1` is its SuperSpeed
lane. It still does not work.

The devkit has no Type-C port controller (no `typec` class, no extcon), so the
CC lines are hardwired Rd. Plug a comma into it and the comma sees a sink,
becomes the DFP and sources VBUS, while the Jetson becomes the device: exactly
backwards, and the Jetson cannot be a gadget for the reason above. Writing
`host` to `/sys/class/usb_role/usb2-0-role-switch/role` flips the data role but
not the CC resistors, so you get two hosts and both ends driving VBUS.

The port also carries VBUS from the board's 5 V rail whether or not anything is
hosting, so a comma plugged into it reports `real_type=USB_DCP`: power with no
data host behind it, which is indistinguishable from a charge-only cable.

Use one of the Type-A ports. They hang off the onboard Realtek hub, so the
gadget appears one hop down at `2-1.2`.


### Direct USB — preferred, and needs no kernel changes on either side

One cable, comma USB-C to a Jetson USB host port. No IP stack, no DHCP, no NCM
aggregation timer, no adapter.

The roles are the reverse of what you would guess, and that is the whole trick:

| | role | why |
|---|---|---|
| **comma** | USB **gadget** (FunctionFS) | AGNOS has `CONFIG_USB_F_FS=y`, `CONFIG_USB_CONFIGFS_F_FS=y` and `CONFIG_USB_LIBCOMPOSITE=y` built in |
| **Jetson** | USB **host** (libusb) | a host needs no driver at all: libusb goes through usbfs |

Putting the gadget on the comma is what sidesteps the stripped L4T rootfs
entirely — the Jetson never needs `libcomposite` or `usb_f_fs`.

Verified on a comma 3X (mici): the gadget configures, the kernel accepts the
FunctionFS descriptors, `ep1`/`ep2` appear, and the UDC (`a600000.dwc3`) binds
and unbinds cleanly.

```bash
sudo scripts/setup_gadget.sh          # on the comma, once per boot
# jetlinkd/modeld then open ep0, write descriptors and bind the UDC
docker/run.sh --transport usb         # on the Jetson: it is the host
```

`setup_gadget.sh` deliberately does not bind the UDC: a FunctionFS gadget cannot
attach to a controller until its descriptors are written, and the client writes
them when it opens `ep0`.

Uses the pid.codes test allocation `1209:0001`. Get a real PID before
distributing this.

Cable: the comma's USB-C to a Jetson USB-A host port (an A-to-C cable makes the
comma the peripheral by CC pull-up, with no role negotiation to get wrong). Note
this is the same port chestnut would use — you get one or the other.

### Ethernet (TCP) — the fallback, and the way to bench

A USB-C→gigabit adapter on the comma (`RTL8152`/`AX88179`, both built into
AGNOS) to the Jetson's 1 GbE. Set the `JetlinkEndpoint` param to
`<jetson-ip>:5599`.

- Useful when you want the USB-C port for something else, or while bringing up.
- 533 KB at 1 Gbit/s is ~4.5 ms of wire time, so expect roughly **31 ms** end to
  end. *(Estimate: the 26.3 ms measurement was over loopback.)*
- Give the link a static subnet of its own. Do not route model traffic over
  shared wifi: measured comma→Jetson over wifi, the same benchmark ran at
  **164 ms mean with 40 ms of jitter**, and every single frame missed the
  50 ms budget. That is the number that rules wifi out.

## Power, and why it is the real risk

- The Jetson needs ~25 W at MAXN_SUPER. USB VBUS from the comma cannot supply
  that: it needs its own ignition-gated 12 V feed.
- Two power domains on one cable, so the gadget config is marked **self-powered**
  (`bmAttributes 0xC0`, `MaxPower 8`) — it must never try to sink from the comma.
- Engine-crank voltage dips are a known killer; chestnut has dedicated
  `supplyFault`/`supplyVoltage` alerts for exactly this. It is worse for a
  Jetson: a GPU re-trains PCIe in milliseconds, a Jetson takes ~30 s to reboot.
- Boot time means the big model is not ready at engagement. openpilot's existing
  shape already covers it (`BIG_MODEL_TIMEOUT = 60`, start on the small model),
  and jetlinkd provisions offroad so the engine is cached before you drive.
