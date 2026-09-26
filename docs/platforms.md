# Platform setup

Run Jetlink on a Mac, Linux PC, or Windows PC. Mac and Linux NVIDIA systems have
hardware test results. Windows WSL2 is untested. Set up the comma with the steps
in the [README](../README.md#quick-start).

On Linux the [installer](#linux-nvidia-gpu) needs nothing else. For the
terminal and Docker setups below, clone the repository first:

```bash
git clone https://github.com/zoompilot/jetlink.git
cd jetlink
```

## Mac (Apple silicon)

Use the Mac app for setup without terminal commands. Download it from
[Releases](https://github.com/zoompilot/jetlink/releases), open it, and leave it
running. It includes Python and does not require Homebrew. The [Mac
guide](macos-app.md) covers installing it, preparing a model ahead of a drive,
and its settings.

### From a terminal

Install Python 3.10 or later and libusb with Homebrew:

```bash
brew install python libusb
scripts/run-mac.sh
```

The first run creates a Python environment and installs dependencies. The server
then serves the comma over USB using CoreML, with the model's vision layers on
the Neural Engine and the rest on the GPU. Plug the comma into a
**USB-A port on a hub or dock** with an A-to-C data cable, or use a USB-C-to-A
adapter. Going through USB-A makes the Mac take the host role reliably; a plain
C-to-C cable may not.

Runtime and storage:

- CoreML takes **about 20 seconds** to prepare the model the first time, and
  from under a second to about 10 seconds to load it again every time the server restarts.
- The script holds the Mac awake on AC power. On battery, keep the lid open.
- Models and prepared engines live in `models_cache/` in the checkout,
  about 3 GB per model with CoreML: a 766 MB download plus a 2.1 GB engine.
  Set `JETLINK_CACHE` to move them.

Options:

```bash
# Serve a test client over TCP instead of the comma (see Test without a comma)
JETLINK_TRANSPORT=tcp scripts/run-mac.sh

# The GPU only, if another app keeps the Neural Engine busy: 44 ms a frame
scripts/run-mac.sh --device coreml

# tinygrad on Metal: 66 ms a frame on an M1 Pro
JETLINK_BACKEND=tinygrad scripts/run-mac.sh

# Prepare a model ahead of time, then exit
scripts/run-mac.sh --build /path/to/big_driving_supercombo.onnx
```

Measured on an M1 Pro: the default runs 29 to 32 ms per frame against a 50 ms
budget, the GPU alone 44 ms. Details in [backends and measurements](backends.md#mac-measured).

## Linux (NVIDIA GPU)

For a PC or laptop with a GeForce RTX 20 series or newer GPU, on Ubuntu or
Debian. Run the installer:

```bash
curl -fsSL https://raw.githubusercontent.com/zoompilot/jetlink/main/install.sh | bash
```

It checks the NVIDIA driver (580 or newer, which CUDA 13 needs) and on Ubuntu
offers to install it, in which case restart and run the installer again. It
then installs Docker and the NVIDIA Container Toolkit if they are missing, gets
the Jetlink server, and asks whether to start it with the computer. Afterwards
`jetlink status`, `jetlink logs`, `jetlink update` and `jetlink uninstall` look
after it; see [everyday use](jetson.md#everyday-use), which is the same on a PC.

Plug the comma into a USB-A port, and keep the computer powered and awake while
driving: sleep drops the link.

### Without Docker

For development. You need a working NVIDIA driver and Python 3.10 or later.

```bash
sudo apt update
sudo apt install -y python3-venv libusb-1.0-0
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[trt,usb,nvml]"
sudo install -m 644 scripts/99-jetlink-host.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
jetlink-server --backend trt --transport usb
```

The udev rule grants USB access without root; unplug and replug the comma after
installing it. In a new terminal, run `source .venv/bin/activate` before using
`jetlink-server` again.

Use `jetlink-models` to download and prepare a model on the server before
connecting the comma. See [models and the model CLI](models.md).

## Windows (NVIDIA GPU)

Use **Ubuntu in WSL2** and follow the Linux steps inside it, or use
[Docker](#docker-nvidia-laptops-and-desktops). Start with a [TCP
test](#test-without-a-comma). USB from WSL2 needs `usbipd-win` to attach the
comma to Ubuntu and is not a validated path.

## Docker (NVIDIA laptops and desktops)

The installer uses these images; this section is for running them yourself,
for example on Windows with WSL2. The image includes Python, CUDA 13, TensorRT,
and USB support. The host needs the NVIDIA driver, 580 or newer.

| Image | For |
| --- | --- |
| `ghcr.io/zoompilot/jetlink:VERSION-cuda` | NVIDIA PCs (x86-64) and Jetsons on JetPack 7.2 or newer: one tag, and Docker pulls the right architecture |
| `ghcr.io/zoompilot/jetlink:VERSION-jetpack6` | Jetsons on JetPack 6 (also tagged `-jetson`) |
| `ghcr.io/zoompilot/jetlink:edge-cuda`, `edge-jetpack6` | the newest `main`, what the installer uses |

**Enable GPU access.** On Linux, install Docker Engine and the [NVIDIA Container
Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
then:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

On Windows, install Docker Desktop with the WSL 2 backend and follow [Docker's
GPU guide](https://docs.docker.com/desktop/features/gpu/). Run the remaining
commands in your Ubuntu WSL terminal. Verify GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:13.2.1-base-ubuntu24.04 nvidia-smi
```

**Pull or build.** Pull a release image, replacing `VERSION` with the release
version, such as `0.4.0`, or use `edge-cuda`:

```bash
docker pull ghcr.io/zoompilot/jetlink:VERSION-cuda
docker tag ghcr.io/zoompilot/jetlink:VERSION-cuda jetlink:cuda
```

To build it yourself instead, from the checkout (`docker/Dockerfile.jetpack6`
on a JetPack 6 Jetson):

```bash
docker build -f docker/Dockerfile -t jetlink:cuda .
```

**Run it.**

```bash
docker volume create jetlink-cache
docker run --rm -it --gpus all --name jetlink-cuda \
  -p 127.0.0.1:5599:5599 \
  -v jetlink-cache:/var/cache/jetlink \
  jetlink:cuda
```

This serves TCP on port 5599 for a [test](#test-without-a-comma). For USB on
native Linux, run instead:

```bash
docker run --rm -it --gpus all --name jetlink-cuda \
  --device-cgroup-rule 'c 189:* rmw' \
  --mount type=bind,source=/dev/bus/usb,target=/dev/bus/usb \
  -v jetlink-cache:/var/cache/jetlink \
  jetlink:cuda --transport usb
```

Logs: `docker logs -f jetlink-cuda`. If the container name is in use, run
`docker stop jetlink-cuda` first. Laptop sleep disconnects the link. Keep the
laptop powered and awake.

## CPU only

For checking the protocol and model loading without a GPU. It will not keep up
with driving.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[ort]"
jetlink-server --backend ort --device cpu --transport tcp
```

## Test without a comma

You need a large driving-model ONNX file. With a TCP server running (on an
installed Jetson or PC: `jetlink stop`, then `sudo docker/run.sh --transport
tcp` from a checkout), run these commands from the
checkout in a second terminal. Replace `/path/to/big_model.onnx` with your model
file path:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e . onnx
python3 scripts/bench_link.py --host 127.0.0.1 --onnx /path/to/big_model.onnx --rate 20
```

The benchmark uploads the model, waits for it to build, and reports round-trip
latency and how many frames exceeded the 50 ms budget at 20 Hz. For a server on
another machine, use its wired-network IP. TCP has no authentication, so use a
trusted network. Wi-Fi does not meet the frame budget.

With the Docker image, the same test runs without installing Python. Put the
model in a `models` folder inside the checkout and run:

```bash
docker run --rm -it --network container:jetlink-cuda \
  --mount "type=bind,source=$(pwd)/models,target=/models,readonly" \
  --entrypoint python jetlink:cuda \
  scripts/bench_link.py --host 127.0.0.1 --onnx /models/big_model.onnx --rate 20
```

## Troubleshooting

| Problem | Check |
| --- | --- |
| `python3` too old or not found | Install Python 3.10+ and reopen the terminal |
| `jetlink-server` not found | Run `source .venv/bin/activate` from the project folder |
| Backend missing | `jetlink-server --list-backends`; check that platform's dependencies and GPU driver |
| USB library error | Install native libusb as well as the Python package |
| USB permission error on Linux | Install the udev rule, then replug the comma |
| TCP connection refused | Start the server with `--transport tcp`. Check the IP address and allow port 5599 through the firewall. |
| GPU not found in Docker | Redo the GPU access setup and rerun the `nvidia-smi` check; the driver must be 580 or newer |
| Mac looks stuck loading | CoreML prepares in about 20 seconds and loads in up to 10 on an M1 Pro. If loading takes minutes, remove the prepared engine and prepare it again. Check the server output for errors. |
| Link drops when laptop sleeps | Keep it awake, powered, and open |

Desktop caches use `JETLINK_CACHE` if set, otherwise `~/.cache/jetlink`, or
`~/Library/Caches/jetlink` on a Mac. The Mac script sets it to `models_cache/`.
For comma-side alerts, see the [README](../README.md#if-something-is-wrong).
