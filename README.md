# jetlink

Run openpilot's large driving models on an NVIDIA Jetson over USB.

The comma handles cameras, image warping, model parsing, and vehicle control.
The Jetson runs inference with TensorRT FP16 and returns model outputs at 20 Hz.
It has no CAN access.

```text
comma                              Jetson
cameras → image warp ─── USB ────→ history buffers → TensorRT
controls ← model parser ←──────── model outputs
```

## Features

- Direct USB 3 transport; wired Ethernet for bench use or an alternate connection.
- On-device engine builds with progress reporting and a persistent model cache.
- GPU temperature, power, utilization, and inference timing telemetry.
- Automatic reconnect, with the loaded engine retained between sessions.
- Optional suspend while idle and USB wake, preserving the loaded engine.

With a compatible openpilot integration, model selection and preparation happen
in the UI. The local small model runs while the Jetson starts. Switching to the
large model waits until controls are disengaged. A link failure returns to the
small model and retries the connection; failure while engaged triggers a soft
disable.

**Status:** experimental. A compatible openpilot build is required for vehicle
integration; installing this repository alone does not add UI or model switching.
See [validation status](docs/release-readiness.md) for remaining qualification work.

## Requirements

| Component | Tested configuration |
| --- | --- |
| Device | comma 3X or comma 4 with a build that includes the Jetlink integration |
| Inference server | NVIDIA Jetson Orin Nano Super, 8 GB |
| Software | JetPack 6.1 / L4T r36.4, TensorRT 10.3; Docker with NVIDIA runtime |
| Connection | USB 3 Type-A to Type-C data cable |
| Power | Separate regulated supply for the Jetson, sized for its 25 W power mode |
| Storage | Several GB free for the container, ONNX models, and cached engines |

Connect **Jetson USB-A → comma USB-C**. The Jetson is the USB host; the comma is
the USB gadget. The Orin Nano devkit's USB-C port does not support this connection.
Check [transport and power requirements](docs/transport.md) for other hardware.

## Setup

Perform initial setup while parked, with stable power and internet access for
the container and model downloads. For a step-by-step walkthrough on zoompilot,
from a fresh Jetson to the first drive, see [the tester guide](docs/tester-setup.md).

### 1. Build and start the Jetson server

Use the Jetlink revision pinned by your compatible openpilot build on both ends.
On the Jetson:

```bash
git clone https://github.com/zoompilot/jetlink.git
cd jetlink
git checkout <pinned-jetlink-revision>
sudo docker/build.sh
sudo docker/run.sh --transport usb
```

The server waits for the comma to present its USB gadget. Leave it running for
the next step.

### 2. Enable Jetlink on the comma

Connect the USB cable. In your build's model settings, enable Jetlink and select
a large model. Keep the device offroad until download, engine build, and warmup
finish and the UI reports ready. The integration sets up USB at boot and manages
the connection.

The first engine build takes roughly 3 to 5 minutes on the tested Jetson, depending
on the model. Engines persist in `/mnt/data/jetlink/engines` on the Jetson and are
reused on later starts. Changing the model, TensorRT version, or GPU architecture
requires a matching engine.

### 3. Run at boot (optional)

After verifying the connection, stop the foreground server with Ctrl-C. From
the repository on the Jetson:

```bash
sudo install -d /etc/jetlink
sudo docker image inspect --format 'JETLINK_IMAGE={{.Id}}' jetlink:latest \
  | sudo tee /etc/jetlink/server.env >/dev/null
sudo chmod 644 /etc/jetlink/server.env
sudo install -m 755 scripts/jetlink-wake-setup.sh /usr/local/bin/
sudo install -m 644 scripts/99-jetlink-usb-wakeup.rules /etc/udev/rules.d/
sudo install -m 644 scripts/jetlink-server.service /etc/systemd/system/
sudo udevadm control --reload-rules
sudo systemctl daemon-reload
sudo systemctl enable --now jetlink-server
```

The service pins the built image by ID, locks Jetson clocks, and suspends after
120 seconds without a USB gadget. Suspend assumes an always-on supply with USB
wake configured. For a setup without suspend, remove `--sleep-after 120` from
the installed unit before starting it. See [power management](docs/transport.md#always-on-supply-and-suspend)
for wake behavior and limitations.

## Bench test over Ethernet

A Linux machine with Python 3.10+ can test inference without the driving-software
integration. Connect it to the Jetson over wired Ethernet.

On the Jetson, run this instead of the USB server:

```bash
sudo docker/run.sh --transport tcp
```

On the client, from a checkout of this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e . onnx
python3 scripts/bench_link.py --host <jetson-ip> --onnx /path/to/big_model.onnx
```

Use a compatible large driving-model ONNX file. The benchmark uploads it if
needed, builds the engine, then reports round-trip latency and frames over the
50 ms budget. TCP listens on port `5599`; use a trusted, dedicated network.
Wi-Fi did not meet the frame budget in testing.

## Performance

Recorded bench results on Orin Nano Super 8 GB, TensorRT 10.3 FP16, over USB 3:

| Model | GPU inference | Full modeld mean / max | First engine build |
| --- | ---: | ---: | ---: |
| BMRLNAP, 766 MB | 19.8 ms | 31.0 / 32.7 ms | 166 s |
| TGC v2, 766 MB | ~20 ms | 31.1 / 33.5 ms | 166 s |
| Lebowski, 1757 MB | 36.2 ms | 46.3 / 49.5 ms | 290 s |

Full modeld timings include local image processing, transport, inference, and
output parsing during recorded-segment replay. The frame budget is 50 ms;
Lebowski leaves little margin. These short bench runs do not establish sustained
performance under heat or load. See [latency analysis](docs/drive-2026-09-05-latency.md)
and [bench validation](docs/validation-2026-09-07.md).

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Server keeps waiting for a gadget | Enable Jetlink, use the Jetson's USB-A port, and check the data cable. On the comma, read `/dev/shm/jetlink-gadget` for setup errors. |
| Model stays unavailable | Confirm preparation finished and both devices use the paired Jetlink revision. Check server logs for build or protocol errors. |
| Engine build fails | Check free space under `/mnt/data/jetlink` and the JetPack/TensorRT versions. |
| Latency exceeds 50 ms | Check USB SuperSpeed negotiation, Jetson power mode, clocks, and cooling. |
| Jetson fails to wake | Check host USB hub wake configuration in the [power guide](docs/transport.md#always-on-supply-and-suspend). |

For the installed service:

```bash
sudo systemctl status jetlink-server
sudo journalctl -u jetlink-server -b -f
```

Update or roll back the client and server together. See [paired releases and
rollback](docs/releasing.md) for the procedure.

## License

[MIT](LICENSE).
