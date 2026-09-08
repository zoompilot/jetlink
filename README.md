# Jetlink

Run openpilot's large driving models on a separate computer connected to your
comma. Jetson is the tested in-car setup; Mac, Linux, and Windows options are
available for experiments and bench testing.

The comma still handles the cameras and vehicle control. Jetlink sends prepared
camera images to the other computer, which runs the model and sends predictions
back 20 times per second. The other computer has no CAN access.

**Experimental:** you need an openpilot build with Jetlink integration to use it
with your comma. Installing this repository alone does not add model selection
or switching. See [validation status](docs/status.md).

## Start here

| Your computer | Setup guide | What to expect |
| --- | --- | --- |
| NVIDIA Jetson Orin Nano Super, 8 GB | [Jetson and comma setup](docs/tester-setup.md) | Tested on the car; first setup needs a terminal on the Jetson |
| Mac with Apple silicon | [Mac setup](docs/platforms.md#mac-apple-silicon) | Bench-tested on M1 Pro; CoreML takes about 9 minutes to load a model each session |
| Linux PC with an NVIDIA GPU | [Linux setup](docs/platforms.md#linux-nvidia-gpu) | Implemented, not yet tested on hardware |
| Windows PC with an NVIDIA GPU | [Windows setup](docs/platforms.md#windows-nvidia-gpu) | Experimental through WSL2; start with a TCP bench test |
| Other computer / CPU only | [CPU setup](docs/platforms.md#cpu-only) | For functional testing; no real-time performance claim |

For a car setup, you also need a comma 3X or comma 4 with a compatible build,
a USB 3 **A-to-C data cable**, and separate power for both devices. Connect the
computer's **USB-A port to the comma's USB-C port**. On a Mac, use a USB-A hub
or dock. On the Jetson devkit, use USB-A, not its USB-C port.

The setup follows three steps:

1. Install and start the server using your platform guide above.
2. On a compatible zoompilot build, enable **Settings > Models > Accelerator Link**.
3. Connect the cable and wait for model preparation to finish while parked.
   A green icon means it is ready. The [comma setup steps](docs/tester-setup.md#connect-the-comma)
   explain the branch, model selector, and status icons.

For NVIDIA laptops and desktops, [Docker setup](docs/platforms.md#docker-cuda-laptops-and-desktops)
includes the server dependencies and works with Linux or Windows/WSL2.
You can also [test a model without a comma](docs/platforms.md#test-without-a-comma).

## Features

- Direct USB 3 connection, or wired Ethernet for bench testing and alternate setups.
- Model downloads and preparation from the comma UI with a compatible integration.
- Engine builds with progress reporting and a persistent model cache.
- GPU temperature, power, utilization, and inference timing where the host supports them.
- Automatic reconnect that keeps the loaded engine between connections.
- Optional Jetson idle suspend and USB wake that preserve the loaded engine.
- TensorRT on NVIDIA GPUs, CoreML or tinygrad on Apple silicon, and onnxruntime fallback.

With the compatible integration, the local small model runs while the server
starts. The large model takes over only when controls are disengaged. If the link
fails, the comma falls back to the small model and retries. A failure while
engaged triggers a soft disable; follow the comma's alerts and take over.

## How it works

```text
comma                                  Server computer
cameras → image preparation ── USB ───→ history buffers → model inference
controls ← model parser ←───────────── predictions
```

The comma prepares camera images using its calibration. The server keeps the
model's input history and runs inference (the model calculation). The comma
parses the results and uses them for driving control. USB and TCP use the same
Jetlink protocol and client; the server reports its runtime when it connects.

An engine is a model prepared for a particular GPU and runtime. Jetson uses
TensorRT FP16 engines, stored in `/mnt/data/jetlink/engines` and reused on later
starts. The first build takes roughly 3 to 5 minutes on the tested Jetson.
Changing the model, runtime version, or GPU may require another build. Mac
CoreML also keeps cached files, but measured session loads still took about
9 minutes. See [platform details](docs/platforms.md#backend-reference).

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
performance under heat or load. See [status and known limitations](docs/status.md).

## Help and further reading

- [Setup, status icons, and troubleshooting](docs/tester-setup.md)
- [Platform setup and benchmarks](docs/platforms.md)
- [Cables, networking, and power](docs/transport.md)
- [Updates and rollback](docs/releasing.md)
- [Validation status and remaining work](docs/status.md)

## License

[MIT](LICENSE).
