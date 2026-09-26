# Cables, networking, and power

For initial setup, see the [README](../README.md#quick-start), [Jetson
guide](jetson.md), or [platform setup](platforms.md).

## USB connection

Use a USB 3 A-to-C data cable. Charge-only cables do not work.

| Server | Connection to the comma's USB-C port |
| --- | --- |
| Jetson | USB-A port on the Jetson |
| Mac | USB-A port on a hub, dock, or USB-C-to-A adapter |
| Linux PC | USB-A port on the PC |

The comma acts as the USB device (also called a gadget). The server computer
acts as the USB host. The compatible comma build configures this connection at
boot when **Accelerator Link** is enabled.

Use the Jetson devkit's USB-A ports. Its USB-C port selects the wrong USB role
for this connection. A direct C-to-C cable on a Mac may also select the wrong
role. The comma's USB-C port cannot serve Jetlink and chestnut at the same time.

## Ethernet (TCP)

Use wired Ethernet for TCP. On the comma, use a USB-C gigabit Ethernet adapter
with a Realtek RTL8152/8153 or ASIX AX88179 chipset. AGNOS includes these
drivers. It does not include the host drivers for NCM, ECM, or RNDIS USB
Ethernet gadgets.

1. Connect the comma and server to a wired network with fixed IP addresses.
2. Start the server with `--transport tcp`.
3. Set the comma's `JetlinkEndpoint` parameter to `<server-ip>:5599`, replacing
   `<server-ip>` with the server's wired-network IP address.

TCP has no client authentication. Use a trusted network. Wi-Fi exceeds the 50 ms
frame budget; use USB 3 or wired Ethernet.

To test a server without a comma, follow [test without a
comma](platforms.md#test-without-a-comma).

## Power requirements

Use separate power supplies for the comma and server. Size the Jetson supply for
its 25 W power mode. The comma's USB port cannot power the Jetson.

The supply must tolerate voltage drops when the engine starts. A voltage drop
can reboot the Jetson and interrupt the link. With ignition-switched power,
allow about 65 to 96 seconds from power-on until the model is ready. The comma
uses its small model during startup. Switching requires a stop with cruise
off, or lateral control off; see [daily use](using-jetlink.md#what-to-expect-when-driving).

### Always-on supply and suspend

An always-on supply allows the Jetson to suspend while parked and keep the
loaded engine in memory. Suspend power consumption is not measured. Measure it
on your installation before leaving the Jetson connected permanently. The
measured awake idle power is 6.8 W.

To enable idle suspend, choose **Always on**, the recommended answer, when the
[installer](jetson.md#2-run-the-installer) asks how the Jetson is powered, or
run `jetlink setup` to change the answer later. The installer arms USB wake on
the Jetson's hubs, grants the container access to `/sys/power`, and starts the
server with `--sleep-after 120`. Suspend requires `deep` support in
`/sys/power/mem_sleep`; the installer checks, and says so when it is missing.

With idle suspend enabled:

1. After ignition turns off, the comma releases the USB connection once the
   engine is ready and at least one minute has passed.
2. The Jetson suspends after 120 seconds without a USB device connection.
3. A USB connection or disconnection wakes the Jetson. If no device connects,
   the server waits another 120 seconds and suspends again.

The comma reconnects when it needs the server. Without `--sleep-after`, the link
stays connected while the comma remains awake after parking.

If suspend fails, the server retries after 10 seconds and doubles the delay
between attempts, up to 5 minutes. Check the server logs if the Jetson stays
awake. The container needs write access to `/sys/power`, and USB wake must be
enabled on the root hubs and onboard hub.

### Powering off with the comma

When the comma shuts down under its battery policy (11.8 V or 30 hours parked),
it asks the Jetson to power off. The installer sets this up when you allow the
comma to shut down the Jetson; it is the host-side
`jetlink-poweroff.path` unit and its service. The server writes a flag in the
models folder; the host service removes the flag and powers off. Flags from
earlier boots are ignored.

For testing, disable this poweroff action by creating the dry-run file:

```bash
touch /mnt/data/jetlink/poweroff-dry-run
```

Remove the file to enable poweroff again:

```bash
rm /mnt/data/jetlink/poweroff-dry-run
```

A powered-off Jetson stays off on an always-on supply. The installation needs a
way to restart it, such as a low-voltage disconnect that restores power when the
alternator runs, or an ignition-controlled connection to the J14 power-button
input. The devkit starts automatically when DC power returns.

## Custom USB integrations

The comma 3X with AGNOS kernel 4.9.103 includes FunctionFS and USB gadget
support. The Jetson host uses libusb and does not need gadget kernel modules.
Reversing these roles requires gadget modules that may be missing from the
Jetson's L4T installation.

For manual integration, run this from the Jetlink checkout on the comma once per
boot:

```bash
sudo scripts/setup_gadget.sh
```

The script creates the gadget configuration. `jetlinkd` opens `ep0`, writes
FunctionFS descriptors, and binds the USB device controller. The setup script
cannot bind the controller before those descriptors exist.

On the Jetson, the installer sets up the server as a service; for a manual run,
from a checkout:

```bash
sudo docker/run.sh --transport usb
```

Jetlink uses the pid.codes test allocation `1209:0001`. Custom distributions
need their own USB product ID.
