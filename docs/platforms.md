# Platform setup

Run Jetlink on a Mac, Linux PC, or Windows PC. Mac and Linux NVIDIA systems have
hardware test results. Windows WSL2 is untested. Set up the comma with the steps
in the [README](../README.md#quick-start).

For terminal and Docker setup, clone the repository first:

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
then serves the comma over USB using CoreML on the GPU. Plug the comma into a
**USB-A port on a hub or dock** with an A-to-C data cable, or use a USB-C-to-A
adapter. Going through USB-A makes the Mac take the host role reliably; a plain
C-to-C cable may not.

Runtime and storage:

- CoreML takes **about 10 seconds** to prepare the model the first time, and
  about 2 seconds to load it again every time the server restarts.
- The script holds the Mac awake on AC power. On battery, keep the lid open.
- Models and prepared engines live in `models_cache/` in the checkout,
  about 3 GB per model with CoreML: a 766 MB download plus a 2.3 GB engine.
  Set `JETLINK_CACHE` to move them.

Options:

```bash
# Serve a test client over TCP instead of the comma (see Test without a comma)
JETLINK_TRANSPORT=tcp scripts/run-mac.sh

# tinygrad on Metal: 66 ms a frame on an M1 Pro, against CoreML's 43
JETLINK_BACKEND=tinygrad scripts/run-mac.sh

# Prepare a model ahead of time, then exit
scripts/run-mac.sh --build /path/to/big_driving_supercombo.onnx
```

Measured on an M1 Pro: CoreML on the GPU runs 43 ms per frame against a 50 ms
budget. Details in [backends and measurements](backends.md#mac-measured).

## Linux (NVIDIA GPU)

Use [Docker](#docker-nvidia-laptops-and-desktops) or the native install below.
You need a working NVIDIA driver.

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

Plug the comma into a USB-A port. The udev rule grants USB access without root;
unplug and replug the comma after installing it. In a new terminal, run `source
.venv/bin/activate` before using `jetlink-server` again.

Use `jetlink-models` to download and prepare a model on the server before
connecting the comma. See [models and the model CLI](models.md).

## Windows (NVIDIA GPU)

Use **Ubuntu in WSL2** and follow the Linux steps inside it, or use
[Docker](#docker-nvidia-laptops-and-desktops). Start with a [TCP
test](#test-without-a-comma). USB from WSL2 needs `usbipd-win` to attach the
comma to Ubuntu and is not a validated path.

## Docker (NVIDIA laptops and desktops)

For an x86-64 machine with an NVIDIA GPU, on Linux or Windows with WSL2. The
image includes Python, CUDA, TensorRT, and USB support. You still need the
NVIDIA driver on the host.

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
docker run --rm --gpus all nvidia/cuda:12.9.1-runtime-ubuntu24.04 nvidia-smi
```

**Pull or build.** Pull a release image. Replace `VERSION` with the release
version, such as `0.3.0`:

```bash
docker pull ghcr.io/zoompilot/jetlink:VERSION-cuda
docker tag ghcr.io/zoompilot/jetlink:VERSION-cuda jetlink:cuda
```

Use `-cuda` for an NVIDIA PC and `-jetson` for a Jetson. To build it yourself
instead, from the checkout:

```bash
docker build -f docker/Dockerfile.cuda -t jetlink:cuda .
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

You need a large driving-model ONNX file. With a TCP server running (on a
Jetson: `sudo docker/run.sh --transport tcp`), run these commands from the
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
| GPU not found in Docker | Redo the GPU access setup and rerun the `nvidia-smi` check |
| Mac looks stuck loading | CoreML prepares in about 10 seconds and loads in about 2 on an M1 Pro. If loading takes minutes, remove the prepared engine and prepare it again. Check the server output for errors. |
| Link drops when laptop sleeps | Keep it awake, powered, and open |

Desktop caches use `JETLINK_CACHE` if set, otherwise `~/.cache/jetlink`, or
`~/Library/Caches/jetlink` on a Mac. The Mac script sets it to `models_cache/`.
For comma-side alerts, see the [README](../README.md#if-something-is-wrong).
