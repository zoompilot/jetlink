# Jetlink

Run openpilot's large driving models on a computer plugged into your comma.
The comma keeps the cameras and vehicle control. It sends prepared camera
images over USB, the other computer runs the model, and predictions come back
20 times per second.

Jetlink is experimental. It needs a zoompilot build with Jetlink built in. The
comma side lives on the [zoompilot `jetson-trt` branch](https://github.com/zoompilot/zoompilot/tree/jetson-trt).
The small model you picked in sunnypilot keeps driving whenever the link is down. If the link
drops while engaged, the comma soft-disables and tells you to take over. See
[status and known limitations](docs/status.md).

<p align="center">
  <img src="docs/images/mac-status.webp" width="49%" alt="Jetlink for Mac: server status and model loading progress">
  <img src="docs/images/mac-models.webp" width="49%" alt="Jetlink for Mac: available models and download status">
</p>

## Quick start

Choose your computer: **[Mac](#mac)** · **[Jetson](#jetson)** · **[CUDA laptop](#cuda-laptop)**.
Then follow the shared [comma setup](#comma-setup-all-platforms).

You need a **comma 3X or comma 4**, a **USB 3 A-to-C data cable**, and
**separate power for both devices**. Charge-only cables will not work.
Keep the comma online and stay parked for the first setup.

### Mac

For **Apple silicon, macOS 15 or later**. 16 GB of memory is recommended.
Mac is bench-tested; Jetson is the tested in-car setup.

1. Download the **Mac ZIP** from [Releases](https://github.com/zoompilot/jetlink/releases).
2. Double-click the ZIP to unzip it, then drag **Jetlink.app** to **Applications**.
3. Open **Jetlink**. The server starts automatically; **Waiting for comma**
   means it is ready to connect. Keep your Mac powered and awake.
4. Follow [comma setup](#comma-setup-all-platforms) below.

If macOS blocks an unsigned build, follow the release notes or the
[Mac install guide](docs/macos-app.md#if-the-build-is-not-signed).
No Python or Homebrew is needed for the app.

<details>
<summary>Developers: run from source</summary>

With Homebrew installed, run in Terminal:

```bash
brew install python libusb
git clone https://github.com/zoompilot/jetlink.git
cd jetlink
scripts/run-mac.sh
```

The script installs its dependencies on first run. To build or work on the
GUI, see [macOS development](macos/README.md).

</details>

See the [Mac guide](docs/macos-app.md) for model downloads, settings, and logs.

### Jetson

For **Jetson Orin Nano Super (8 GB) with JetPack 6.2**. Use a power supply
sized for 25 W mode and allow several GB of free space on `/mnt/data`.

Run these commands in a terminal on the Jetson:

```bash
sudo apt update
sudo apt install -y git docker.io nvidia-container
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

git clone https://github.com/zoompilot/jetlink.git
cd jetlink
```

For a **release install**, copy the Jetson image reference from the
[release notes](https://github.com/zoompilot/jetlink/releases). Set `IMAGE` to
that reference, replacing `VERSION` below with its version:

```bash
IMAGE=ghcr.io/zoompilot/jetlink:VERSION-jetson
sudo docker pull "$IMAGE"
sudo env IMAGE="$IMAGE" docker/run.sh --transport usb
```

If the release has no Jetson image, or you want to **build from source**, run
these commands instead from the `jetlink` folder:

```bash
sudo docker/build.sh
sudo docker/run.sh --transport usb
```

Leave the terminal open and follow [comma setup](#comma-setup-all-platforms).
Once it works, use the [Jetson guide](docs/jetson.md#start-at-boot) to start
Jetlink automatically at boot. For a release image, use your image reference
in place of `jetlink:latest` in that guide's image-inspection command.

### CUDA laptop

For a **Linux laptop with an NVIDIA GPU**, a working NVIDIA driver, and
**Python 3.10 or later**. This setup is hardware-tested.

On Ubuntu or Debian, run in a terminal:

```bash
sudo apt update
sudo apt install -y git python3-venv libusb-1.0-0
git clone https://github.com/zoompilot/jetlink.git
cd jetlink
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[trt,usb,nvml]"
sudo install -m 644 scripts/99-jetlink-host.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
jetlink-server --backend trt --transport usb
```

Leave the terminal open, keep the laptop powered and awake, then follow
[comma setup](#comma-setup-all-platforms). If the comma was already plugged
in, unplug and reconnect it so the USB permissions take effect.

Next time, open a terminal in `jetlink` and run:

```bash
source .venv/bin/activate
jetlink-server --backend trt --transport usb
```

For **Docker** or **Windows with WSL2**, see the
[platform guide](docs/platforms.md#docker-nvidia-laptops-and-desktops).
Windows USB setup is not validated; start with the guide's TCP bench test.

## Comma setup (all platforms)

Do this once, whichever computer you chose.

1. **Install the Jetlink branch.** On a comma already running zoompilot, open
   **Settings > Software > Target Branch > Non-Prebuilt Branches** and select
   **jetson-trt**. Let it update, reboot, and finish building. Coming from
   another fork? Start with [zoompilot's installation instructions](https://github.com/zoompilot/zoompilot/tree/jetson-trt).
2. **Enable Jetlink.** Open **Settings > Models** and turn on
   **Accelerator Link**. Leave **Big Model** on the default for your first run.
3. **Connect USB.** With Jetlink running on your computer, connect its
   **USB-A port to the comma's USB-C port** using a USB 3 data cable.
   On a Mac, use a USB-A port on a hub, dock, or USB-C-to-A adapter.
   On Jetson, use a USB-A port, not its USB-C port. A plain C-to-C cable may
   not connect reliably.
4. **Wait for green.** Stay parked with the comma online. Its home-button
   icon pulses while the model downloads, transfers, and prepares, then
   turns green when ready. No manual model download or SSH setup is needed.

The default model takes about **3 minutes to prepare on Jetson**. On Mac,
preparation takes about **10 seconds**, with later loads around **2 seconds**;
download time is extra. The screenshots above show an earlier app build.

| Icon | Meaning |
| --- | --- |
| Pulsing | Downloading, transferring, or preparing the model. Keep waiting. |
| Green | Parked: ready. Driving: the large model is driving. |
| Green, dimmed | Driving: ready and waiting for a chance to switch. Stop with cruise off, or turn lateral control off. |
| Orange | Preparation failed. Read the alert on the home screen. |
| Back to normal a minute later | Only with `--sleep-after`: the comma let the link go so the computer can sleep. Otherwise it stays connected the whole time you are parked. |

## What to expect when driving

- The small model drives while the server starts. On a computer that powers up
  with the car, the large model is prepared 65 to 96 seconds later.
- It takes over only when nothing is steering: **at a stop with cruise off, or
  with lateral control off**. Until then the icon is dimmed and the comma says
  **Big Model Available** at every stop. Disengaging alone is not enough on a
  car with lateral control always on.
- A **Big Model Ready** chime means it has taken over.
- Picking a new model needs the comma online once, while parked, to download
  it. After that it is prepared wherever you are: drive off in the middle and
  the small model drives, the panel counts the preparation down, and the large
  model joins at the first chance to switch.
- **Big Model Lost** while engaged is a soft disable. Take over. The small model
  drives, and Jetlink reconnects and switches back at the next chance.

To stop using Jetlink, turn off **Settings > Models > Accelerator Link**.

## If something is wrong

| Problem | Try |
| --- | --- |
| No Accelerator Link toggle | Check the branch in Settings > Software. |
| Toggle is on, nothing happens | Read the setup alert on the home screen. |
| Big Model list is empty | Connect the comma to the internet and use Refresh Model List. |
| Server keeps waiting, icon never pulses | Check the server is running, use a USB-A port, try another USB 3 data cable. |
| Orange icon | Read the alert, check the comma's internet, then toggle Accelerator Link off and on. |
| Model drops out repeatedly | Check the cable, separate power supplies, and cooling. |

<details>
<summary>Mac: an example of a server error</summary>

The Status screen shows the failure and recent output. Open **Logs** for more
detail, then check the [Mac troubleshooting guide](docs/macos-app.md#troubleshooting).

![Jetlink for Mac showing a server failure and diagnostic output](docs/images/mac-error.webp)

</details>

The [Jetson guide](docs/jetson.md#troubleshooting) has more, including how to
collect logs when reporting a problem.

## More

- [Jetlink for Mac, the app](docs/macos-app.md)
- [Jetson setup, boot service, troubleshooting, logs](docs/jetson.md)
- [Mac, Linux, Windows, Docker, and testing without a comma](docs/platforms.md)
- [Models, the model CLI, and the control channel](docs/models.md)
- [Status, known limitations, and measured performance](docs/status.md)
- [Updates and rollback](docs/releasing.md)
- [Cables, networking, and power](docs/transport.md)
- [Backends and measurements, for developers](docs/backends.md)

## License

[MIT](LICENSE).
