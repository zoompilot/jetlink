# Cables, networking, and power

For initial setup, use the [README](../README.md#quick-start), the [Jetson guide](jetson.md), or [platform setup](platforms.md).
Use USB 3 for the comma connection and wired Ethernet for a bench test.

| Connection | What to use |
| --- | --- |
| Jetson to comma | Jetson USB-A → comma USB-C, with a USB 3 data cable |
| Mac to comma | USB-A hub or dock → comma USB-C, with the same cable |
| Ethernet bench | Wired network; TCP port 5599 on a trusted network |
| Power | Separate supplies for the comma and server; size the Jetson supply for its 25 W mode |

The Jetson devkit's USB-C port does not work for this setup. The comma acts as
the USB device (called a gadget), and the server computer is the USB host.
The compatible fork sets up the comma's USB connection at boot when enabled.

For optional idle sleep, see [always-on supply and suspend](#always-on-supply-and-suspend).
The rest of this page explains the tested hardware, kernel requirements, and
power behavior for custom setups. Hardware observations apply to the versions
listed, not every board or operating-system release.

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
sudo scripts/setup_gadget.sh          # manual integration only: on the comma, once per boot
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

## Power requirements

- The Jetson needs ~25 W at MAXN_SUPER. USB VBUS from the comma cannot supply
  that: it needs its own ignition-gated 12 V feed.
- Two power domains on one cable, so the gadget config is marked **self-powered**
  (`bmAttributes 0xC0`, `MaxPower 8`) — it must never try to sink from the comma.
- Engine-crank voltage dips are a known killer; chestnut has dedicated
  `supplyFault`/`supplyVoltage` alerts for exactly this. It is worse for a
  Jetson: a GPU re-trains PCIe in milliseconds, a Jetson takes ~30 s to reboot.
- Boot time means the big model is not ready at engagement. openpilot's existing
  integration starts on the small model and joins the server in the background.
  jetlinkd prepares the engine while parked so it is cached before you drive.

### Always-on supply, and suspend

An ignition-switched feed reboots the Jetson at every crank: ~65 s from power
to engine ready once dockerd stops waiting for a network, and every shutdown
is an ungraceful one. An always-on feed avoids both, and deep suspend makes it
affordable: the loaded engine, the CUDA context and libusb all survive a
suspend, and a resume is ~6 s to a kernel and ~13 s to a network. Measured on
the bench 2026-09-04: after a resume the live bench joined with a 6 ms build
and ran 31.2 ms mean over 90 s with no reload.

USB is the wake source, and *both* edges wake it: the comma presenting the
gadget and the comma dropping it. So ignition-off, which pulls the gadget,
wakes the Jetson, and the policy has to be a loop, not a command. The server
runs it (`--sleep-after`, `jetlink/server/sleep.py`): awake with no gadget for
120 s means nobody wants us, suspend again. 120 s is longer than the gadget's
re-enumeration at the jetlinkd/modeld handover (45 to 70 s observed), so a
handover never sleeps through. The same rule covers a mid-drive disconnect
longer than that: the next enumeration is a wake.

The comma side does have to let go: a parked comma stays awake for up to 30
hours holding the gadget, and with the gadget held the Jetson never sleeps.
jetlinkd releases it once the engine is ready and a minute has passed since
ignition-off (`DORMANT_HOLD` in the fork), and presents it again only for
work or for the shutdown below. The compatible integration manages this release automatically.

### Powering off with the comma

Sleep is not off. When the comma's own battery policy shuts it down (11.8 V
or 30 hours parked) it asks the Jetson to power off too: `SHUTDOWN_REQ` on
the wire, a flag file on the cache volume from the server, and a host-side
path unit (`scripts/jetlink-poweroff.path`) that runs `systemctl poweroff`.
The script deletes the flag before powering off and ignores one older than
the current boot, so a stale flag cannot loop the box. `touch
/mnt/data/jetlink/poweroff-dry-run` disarms it on a bench. Off stays off on
an always-on feed: fit a low-voltage disconnect that reconnects when the
alternator is running (the devkit auto-powers-on when DC returns), or wire
ignition to the power button pin on the J14 header.

A suspend attempt can fail without saying so: the freezer gives up on a
process that will not freeze (a bench ssh session did it) and the write to
`/sys/power/state` returns `EBUSY` with the box still awake, or a wake edge
lands during the freeze and the write returns cleanly having slept nothing.
The server checks `suspend_stats/success` moved, and backs off 10 s doubling
to 5 min between failed attempts, because each one freezes every process on
the box for the freezer's 20 s timeout.

The container needs `/sys/power` read-write (`run.sh` and the unit mount it
over the read-only `/sys`) and `mem_sleep` has to offer `deep`; L4T r36.4
selects it by default and the server selects it if not. USB wakeup is
enabled by default on the root hubs and the onboard Realtek hub; the comma's
gadget does not advertise remote wakeup and does not need to, the hub port's
connect/disconnect is what wakes the SoC. `tegra-dce` logs a failed resume
(`-22`) every time; it is the display engine and there is no display.

Unmeasured, and it decides whether this is safe to wire permanently: the
suspended draw at the barrel jack. Awake and idle is 6.8 W, about 7 Ah over
a twelve hour park; if suspend lands near 1 W it is a non-issue. Put a meter
on it first.
