# Jetson setup

This gets the server running on a Jetson Orin Nano Super (8 GB) and starting at
boot. Set up the comma first with the three steps in the
[README](../README.md#quick-start).

Allow up to two hours the first time, mostly downloads and builds. Keep the
Jetson on a supply sized for its 25 W power mode with internet access, and
several GB free on `/mnt/data`.

Run the commands below in a terminal on the Jetson. `sudo` asks for your Jetson
password; typing it shows nothing.

## Install and run

Docker runs the server with its dependencies. JetPack 6.1 (L4T r36.4,
TensorRT 10.3) is what the image is built against.

```bash
sudo apt update
sudo apt install -y git docker.io nvidia-container
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

git clone https://github.com/zoompilot/jetlink.git
cd jetlink
sudo docker/build.sh
sudo docker/run.sh --transport usb
```

The build downloads several GB. When the server prints that it is waiting for a
gadget, it is ready for the comma. Plug **Jetson USB-A → comma USB-C** and wait
for the green icon as described in the README. The default model takes about
3 minutes to build; the engine is cached for later starts.

## Start at boot

Once the connection works, press **Ctrl-C** to stop the foreground server, then
from the `jetlink` folder:

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

Look for `active (running)` and press **q**. The service runs the exact image
you built and pins the Jetson's clocks. Select the 25 W power mode in the
Jetson's power menu.

The `sed` line turns off idle suspend, which is the right default for an
ignition-switched supply. If the Jetson has always-on power, read
[always-on supply and suspend](transport.md#always-on-supply-and-suspend) first
and skip that line.

On switched power, the large model is ready about a minute after the car
starts; the small model drives until then.

To stop the service: `sudo systemctl disable --now jetlink-server`.

## Choosing a model

You can pick another model under **Settings > Models > Accelerator Model**
while parked, then wait for it to prepare. Stick with 766 MB models on the
Jetson. Lebowski (1.7 GB) runs at 46 ms against a 50 ms frame budget, which
leaves little margin. See [measured performance](status.md#measured-performance).

## Troubleshooting

| Problem | What to do |
| --- | --- |
| Server keeps waiting, or icon never pulses | Check the server is running, use a Jetson USB-A port, try another USB 3 data cable |
| Engine build fails | Check free space in `/mnt/data/jetlink` and JetPack/TensorRT version |
| Model repeatedly drops out | Check separate supplies and voltage dips, cable, cooling, and server logs |
| Frame time exceeds 50 ms | Check USB 3 speed, Jetson power mode, cooling, and model choice |
| Jetson fails to wake | See [USB wake setup](transport.md#always-on-supply-and-suspend) |
| Server refuses the comma after an update | Update both sides together, see [updates](releasing.md) |
| “Speed Error: nan” or no path | Stop the test and collect logs |

Live server logs (Ctrl-C stops viewing):

```bash
sudo journalctl -u jetlink-server -b -f
```

### Reporting a problem

Include your platform, Jetlink commit, selected model, time of the test, the
exact alert, and what you saw or heard. For a drive investigation, share the
dongle ID from **Settings > Device**.

Save this boot's Jetson log (use `-b -1` for the previous boot):

```bash
sudo journalctl -u jetlink-server -b --no-pager > jetson.log
```

For comma logs, enable SSH in **Settings > Device** with your GitHub username
as the SSH key, find the comma's IP in **Settings > Network**, then from a
laptop with that key:

```bash
ssh comma@<comma-ip> 'tar czf - /data/log' > comma-log.tgz
```
