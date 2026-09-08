# Set up Jetlink on a Jetson and comma

This guide gets a Jetson server running, connects it to zoompilot, and explains
what you should see. For another computer, start with [platform setup](platforms.md)
and return to [Connect the comma](#connect-the-comma).

## Before you start

Allow up to two hours for the first setup, mostly downloads and builds. Keep the
car parked and both devices on stable power with internet access.

You need:

- A comma 3X or comma 4 running zoompilot's `jetson-trt` branch.
- A Jetson Orin Nano Super, 8 GB, with JetPack installed. The repository's Docker
  image uses JetPack 6.1 / L4T r36.4 and TensorRT 10.3. Use the host JetPack version
  specified for your paired release; earlier tester notes also used JetPack 6.2.
- A USB 3 A-to-C **data** cable. A charge-only cable will not work.
- A separate regulated Jetson supply sized for its 25 W power mode, and several
  GB of free storage for downloads, the container, and model engines.

Jetlink is experimental. If the link fails while engaged, the comma gives a soft
disable and falls back to the small model. Be ready to take over.

Commands below run in **Terminal on the Jetson**. Paste each block and wait for
it to finish. `sudo` may ask for your Jetson password; typing it shows no characters.
Text in angle brackets, such as `<pinned-jetlink-revision>`, must be replaced.

## Install the Jetson server

Docker runs the server with its model dependencies. On a JetPack installation
with NVIDIA's package sources configured:

```bash
sudo apt update
sudo apt install -y git docker.io nvidia-container
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

If an NVIDIA package or command is missing, check your JetPack installation before
continuing.

Use the Jetlink revision paired with your openpilot build. In the fork's GitHub
file list at that build's commit, `jetlink_repo` points to the required Jetlink
commit. Copy that commit ID into the checkout command. See [release compatibility](releasing.md#compatibility).

```bash
git clone https://github.com/zoompilot/jetlink.git
cd jetlink
git checkout <pinned-jetlink-revision>
sudo docker/build.sh
sudo docker/run.sh --transport usb
```

The build downloads several GB. When the server says it is waiting for a gadget,
it is ready for the comma. Leave this terminal open. “Gadget” means the comma's
USB connection.

## Connect the comma

Use the comma's screen for these steps. No comma terminal is needed.

1. Open **Settings > Software > Target Branch > Non-Prebuilt Branches** and select
   **jetson-trt**. Update, reboot, and wait for the build screen to finish.
2. Open **Settings > Models** and turn on **Accelerator Link**. The **Accelerator
   Model** row should appear within seconds. Keep the default model for your first
   setup. Avoid Lebowski and Be Right Here for initial Jetson testing: larger
   models have little or no margin within the 50 ms frame budget.
3. Connect **Jetson USB-A → comma USB-C** with the server running.
4. Wait while parked and online. The home-button icon pulses as the comma downloads
   the model, sends it to the Jetson, and prepares the engine. A typical 766 MB
   model takes about 3 minutes to build after downloading; larger models take longer.

| Icon or message | What it means |
| --- | --- |
| Pulsing icon | Downloading, transferring, building, or loading the model; keep waiting |
| Green icon | Model preparation is complete |
| Orange icon | Preparation failed; read the alert on the home screen |
| Icon returns to normal about a minute after parked preparation | The comma released the idle connection; this is expected |

Jetson engines stay cached for later starts. You can select another model in
**Accelerator Model** while parked; wait for its preparation to finish too.

## Start the Jetson server at boot

After the connection works, press **Ctrl-C** in the Jetson terminal to stop the
foreground server. Run this from the `jetlink` folder:

```bash
sudo install -d /mnt/data/jetlink /etc/jetlink
sudo docker image inspect --format 'JETLINK_IMAGE={{.Id}}' jetlink:latest \
  | sudo tee /etc/jetlink/server.env >/dev/null
sudo chmod 644 /etc/jetlink/server.env
sudo install -m 755 scripts/jetlink-wake-setup.sh /usr/local/bin/
sudo install -m 644 scripts/99-jetlink-usb-wakeup.rules /etc/udev/rules.d/
sudo install -m 644 scripts/jetlink-server.service /etc/systemd/system/
sudo sed -i 's/ --sleep-after 120//' /etc/systemd/system/jetlink-server.service
sudo udevadm control --reload-rules
sudo systemctl daemon-reload
sudo systemctl enable --now jetlink-server
sudo systemctl status jetlink-server
```

Look for `active (running)`. Press **q** to leave the status screen. The service
uses the exact image you built and locks Jetson clocks. Select the appropriate
25 W power mode for your board in Jetson's power settings.

The `sed` command disables idle suspend. For an always-on supply with tested USB
wake, follow [power management](transport.md#always-on-supply-and-suspend) before
omitting that command. Neither device should power the other.

## What to expect when driving

- The small model runs while the Jetson starts. On switched power, readiness may
  take about a minute after starting the car.
- A **Big Model Ready** chime announces readiness. If controls are active, the
  comma says **Big Model Available, disengage to switch**.
- The displayed path reaches farther ahead when the large model is running.
- **Big Model Lost** while engaged means a soft disable. Take over and follow the
  alerts. The small model is the fallback; Jetlink retries and can switch back
  after controls are disengaged.

To stop using Jetlink, turn off **Settings > Models > Accelerator Link**. This
disables the integration; it leaves the Jetson service and cached models installed.
To stop the Jetson service too, run `sudo systemctl disable --now jetlink-server`.

## Troubleshooting

| Problem | What to do |
| --- | --- |
| No Accelerator Link toggle | Check the installed branch in Settings > Software |
| Toggle is on, but no Accelerator Model row | Read the home-screen setup alert; the comma's USB setup may have failed |
| Server keeps waiting, or icon never pulses | Check the server is running, use Jetson USB-A, and try another USB 3 data cable |
| Orange icon or failed download | Read the alert, check comma internet access, then toggle Accelerator Link off and on to retry |
| Engine build fails | Check free space in `/mnt/data/jetlink`, the paired source revision, and JetPack/TensorRT compatibility |
| Model stays unavailable | Wait for preparation to finish and check both devices use the paired release |
| Model repeatedly drops out | Check separate supplies and voltage dips, cable, cooling, and server logs |
| Frame time exceeds 50 ms | Check USB 3 speed, Jetson power mode, clocks, cooling, and model choice |
| Jetson fails to wake | Check [USB wake setup](transport.md#always-on-supply-and-suspend) |
| “Speed Error: nan” or no path | Stop the test and collect logs for diagnosis |

For live Jetson logs (Ctrl-C stops viewing them):

```bash
sudo journalctl -u jetlink-server -b -f
```

When reporting a problem, include your platform, fork and Jetlink revisions,
selected model, time of the test, exact alert, and what you saw or heard. For a
drive investigation, share the dongle ID from **Settings > Device** with the maintainer.

If asked for comma logs, enable SSH in **Settings > Device**, set SSH Keys to your
GitHub username, and find the comma's IP address in **Settings > Network**. From
a laptop with the matching SSH key, replace `<comma-ip>` and run:

```bash
ssh comma@<comma-ip> 'tar czf - /data/log' > comma-log.tgz
```

On the Jetson, save this boot's logs with:

```bash
sudo journalctl -u jetlink-server -b --no-pager > jetson.log
```

Use `-b -1` instead of `-b` if the test was on the previous boot and those logs
are retained. For updates or rollback, keep the [client and server paired](releasing.md).
