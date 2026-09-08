# Testing jetlink in the car

You need a comma 3X or comma 4 on zoompilot, a Jetson Orin Nano Super 8 GB on
JetPack 6.2, a USB 3 A-to-C data cable, and a 9 to 20 V supply for the Jetson's
barrel jack sized for 25 W (ignition-switched is simplest). The Jetson needs a
terminal once. The comma is done on its screen. Allow two hours, mostly waiting,
parked, with both devices online.

**Experimental.** The small model drives until the Jetson joins. A lost link falls
back to the small model on its own; while engaged that is a soft disable. Hands on
the wheel.

## Jetson, once

Paste and wait. The build is mostly a several-GB download.

```bash
sudo apt update && sudo apt install -y git docker.io nvidia-container
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
git clone https://github.com/zoompilot/jetlink.git && cd jetlink
sudo docker/build.sh
```

Then this. Keep the `sed` line unless the Jetson is on an always-on supply and you
want it to sleep between drives.

```bash
sudo systemctl mask systemd-networkd-wait-online.service
sudo nvpmodel -m 2
sudo mkdir -p /mnt/data/jetlink /etc/jetlink
sudo docker image inspect --format 'JETLINK_IMAGE={{.Id}}' jetlink:latest | sudo tee /etc/jetlink/server.env
sudo install -m 755 scripts/jetlink-wake-setup.sh /usr/local/bin/
sudo install -m 644 scripts/99-jetlink-usb-wakeup.rules /etc/udev/rules.d/
sudo install -m 644 scripts/jetlink-server.service /etc/systemd/system/
sudo sed -i 's/ --sleep-after 120//' /etc/systemd/system/jetlink-server.service
sudo udevadm control --reload-rules && sudo systemctl daemon-reload
sudo systemctl enable --now jetlink-server
```

Done. The server starts at every boot and waits for the comma. To watch it:
`sudo journalctl -u jetlink-server -f`.

## comma

1. **Branch.** Settings > Software > Target Branch > `jetson-trt` (under Non-Prebuilt
   Branches). Update, reboot, and wait out the build screen.
2. **Link.** Settings > Models > Accelerator Link, on. Within seconds an Accelerator
   Model row appears set to Cinque Terre. Leave it.
3. **Cable.** Jetson USB-A to comma USB-C. Never the Jetson's USB-C port. The icon
   on the comma's home button pulses while it downloads the model (766 MB), sends it
   to the Jetson and the Jetson builds the engine (about 2 minutes), then turns
   green. Orange means it failed and the alert on the home screen says why. This
   happens once. The icon going back to normal a minute later is the parked release,
   not a fault.

## Drive

- The small model drives from the first second.
- "Big Model Ready" chime once the Jetson is up, about a minute after the crank on
  switched power. The swap waits until you are disengaged: "Big Model Available,
  disengage to switch".
- The path draws much further ahead on the big model.
- Link lost while engaged: soft disable, "Big Model Lost". Re-engage on the small
  model; it rejoins on its own and swaps back at the next disengaged moment.

Send back the dongle ID (Settings > Device), the time of the drive, and what you
saw and heard. To stop, turn the same toggle off. Do not power either device from
the other, and do not pick Lebowski or Be Right Here as the model: they are over the
frame budget on this Jetson.

## If something is off

| You see | It means |
|---|---|
| No Accelerator Link toggle | the branch did not install; Settings > Software shows which branch you are on |
| Toggle on, no Accelerator Model row | the USB gadget did not come up; the offroad alert has the reason |
| Icon never pulses after plugging in | the Jetson does not see the comma: wrong port, charge-only cable, or the server is down |
| Orange icon, download failed | the comma needs Wi-Fi; toggle the link off and on to retry |
| Big model drops out every 30 to 45 s | one side is browning out; separate supplies |
| "Speed Error: nan", no path at all | modeld crashed; send the logs |

Logs, only when asked. On the comma: Settings > Device > Enable SSH, SSH Keys set
to your GitHub username, address on the Wi-Fi row in Settings > Network. Then:

```bash
ssh comma@<comma-ip> 'tar czf - /data/log' > comma-log.tgz          # laptop
sudo journalctl -u jetlink-server -b -1 --no-pager > jetson.log      # Jetson, after the drive
```
