# Backends and performance

Jetlink supports TensorRT, tinygrad, and ONNX Runtime. For installation, see
[platform setup](platforms.md).

## Runtime comparison

| Backend | Devices | Prepared files | Requirements |
| --- | --- | --- | --- |
| `trt` | NVIDIA CUDA | `.plan` | TensorRT 10.3 on JetPack 6, or TensorRT 11.x from PyPI |
| `tinygrad` | METAL, CUDA, NV, AMD, CPU | `.pkl` | The pinned tinygrad source version below |
| `ort` | CoreML, CUDA, CPU | `.ortcache/` | ONNX Runtime 1.22+ |

`--backend auto` selects TensorRT if available, then CoreML on macOS, then
tinygrad, then ONNX Runtime with an available device. Prepared files are cached
separately for each runtime version and device.

## Platform matrix

| Platform | Backend | USB | Telemetry | Sleep support |
| --- | --- | --- | --- | --- |
| Jetson Orin | TensorRT | USB-A host with libusb | Tegra sensors | Suspend and poweroff |
| Linux with NVIDIA GPU | TensorRT; tinygrad or ONNX Runtime as alternatives | libusb with `scripts/99-jetlink-host.rules` | NVML | `--sleep-after` requires `/sys/power`; USB wake depends on hardware |
| Windows with NVIDIA GPU | TensorRT in WSL2 | Requires `usbipd-win`; untested | NVML | None |
| macOS with Apple silicon | ONNX Runtime with CoreML on the GPU | USB-A hub, dock, or adapter with libusb | Not available | `scripts/run-mac.sh` prevents idle sleep on AC power |

See [status and limitations](status.md) for platform testing coverage.

## Mac, measured

These results use a 16 GB M1 Pro, macOS 25.5, the 766 MB Cinque Terre model,
tinygrad 0.14.0 at `e837e367aac9`, and ONNX Runtime 1.29.0. Performance on other
Macs may differ.

The frame budget is 50 ms at 20 frames per second (20 Hz). CoreML on the GPU is
the default because it meets this budget on the M1 Pro. tinygrad exceeds the
budget. The Neural Engine option (`--device ane`) has more frames over budget at
20 Hz, even though it is faster with no pause between requests.

| | tinygrad METAL | CoreML, GPU (`--device coreml`, default) | CoreML, all compute units (`--device ane`) |
| --- | ---: | ---: | ---: |
| round trip at 20 Hz through the server, mean / p99 / max | 66.2 / 67.6 / 67.9 ms | 43.3 / 44.4 / 44.5 ms | 44.6 / 58.8 / 68.9 ms |
| frames over the 50 ms budget, of 390 | 390 | 0 | 69 |
| round trip with no pause between requests through the server, mean / p99 | 66.2 / 67.6 ms | 39.9 ms server side | 32.6 / 38.7 ms |
| parity gate, worst column | 0.99954 pass | 0.99957 pass | 0.99957 pass |
| parity, mean error on `plan` / `lead_prob` | 0.0060 / 0.0156 | 0.0046 / 0.0150 | 0.0057 / 0.0138 |
| build / load in a fresh process | 13 s / 1.1 s | 8.2 s / 2.0 s | not measured |
| artifact on disk | 777 MB | 2.3 GB | not measured |
| peak memory use while building | 0.6 GB | 3.0 GB | not measured |
| peak memory use while loading | not measured | 2.5 GB | not measured |

Memory figures include the server and its worker process, sampled once per
second. Build and load times for the Neural Engine option are not measured with
the current model preparation code.

The mean is the average frame time. The p99 is the time at or below which 99% of
frames complete. The maximum is the slowest frame.

### How to measure

`scripts/verify_parity.py` compares 32 frames against ONNX Runtime on the CPU,
using the model's hidden-state feedback. Every output slice and column must have
a correlation of at least 0.999 to pass.

`scripts/bench_link.py --rate 20` measures round-trip latency through the server
over TCP loopback. Use the 20 Hz results when assessing the driving frame
budget. See [test without a comma](platforms.md#test-without-a-comma).

### Keeping the Mac GPU responsive between frames

CoreML's GPU path enables a small Metal keep-alive workload while inference
requests are arriving. On an M2 Pro, the gaps in a 20 Hz stream allowed GPU
clocks to fall even though continuous inference met the 50 ms deadline. The
machine reported nominal thermal pressure. A similar intermittent GPU workload
problem and a small-workload workaround are described in
[Anukari's development report](https://anukari.com/blog/devlog/apple-performance-progress).

On an M2 Pro with ONNX Runtime 1.29.0 and model `09d080f36965bb2a`, five-minute
TCP loopback runs at 20 Hz on 2026-09-21 measured:

| | Original run | With keep-alive |
| --- | ---: | ---: |
| mean round trip | 44.13 ms | 35.41 ms |
| p99 round trip | 64.66 ms | 38.62 ms |
| maximum round trip | 83.57 ms | 70.20 ms |
| frames exceeding 50 ms | 1,119 / 5,990 (18.68%) | 3 / 5,990 (0.05%) |

Each run excludes ten warm-up frames. With keep-alive, every 30-second
window had a mean below 35.6 ms and p99 below 39 ms. Three isolated deadline
misses remained; this is a desktop TCP measurement, not USB end-to-end validation.
A subsequent 90-second control run with the helper disabled missed 498 of
1,790 deadlines (27.82%), with a 69.10 ms p99. The first 32 recurrent frames
produced bit-for-bit identical outputs with the helper enabled and disabled.

The helper uses a separate 128-byte buffer, with one finite command in flight
at a time on its own thread. It does not change model inputs, hidden state,
precision, or CoreML compute units. It stops submitting work after one second
without an inference request, on inference errors, or when the worker exits.
CPU and Neural Engine sessions do not start it. If Metal initialization or a
helper command fails, inference continues without the helper and logs a warning.

This trades additional GPU activity and power consumption for lower latency;
it does not change thermal limits or force a GPU clock setting. To disable it
for comparison, launch the server with `JETLINK_METAL_KEEPALIVE=0`:

```bash
JETLINK_METAL_KEEPALIVE=0 JETLINK_TRANSPORT=tcp \
  scripts/run-mac.sh --backend ort --device coreml --host 127.0.0.1
```

Use the same model and a sustained paced benchmark (`--rate 20 --n 6000`).
A continuous benchmark (`--rate 0`) alone does not establish that the server
can meet deadlines with pauses between frames. Measure on the target machine;
results depend on hardware and competing workloads.

### Model preparation

CoreML stores weights in a binary file. A prepared GPU model uses about 2.3 GB,
takes about 8 seconds to build, and loads in about 2 seconds on the M1 Pro. If a
cached model takes minutes to load, remove its prepared engine and prepare it
again.

For the Neural Engine option, Jetlink normalizes negative Gather indices and
runs the policy's LayerNormalization operations in fp32 for accuracy. The trunk
uses fp16.

## Runtime requirements

- tinygrad calls run on one dedicated thread because its JIT and Metal state
  require the same thread for loading and inference.
- ONNX Runtime sessions run in a worker process so model preparation does not
  block server connections, progress updates, or pings. Frame inputs and
  outputs use shared memory.
- Jetlink disables ONNX Runtime telemetry to avoid a macOS shutdown crash.
- Install tinygrad from the pinned source commit. The PyPI 0.14.0 package lacks
  the `org.tinygrad` ONNX domain required by exported models:

```bash
pip install --no-deps "tinygrad @ git+https://github.com/sunnypilot/tinygrad@e837e367aac9e1a66e689f4f32ce20ca9367df13"
```

## Hardware limitations

Native Windows USB requires WinUSB and is not validated. Use WSL2 for initial
TCP testing. On a Mac, use a USB-A port on a hub, dock, or adapter to ensure
that the Mac acts as the USB host.

Keep laptops powered and awake. Sustained GPU use can cause thermal throttling;
check frame times during use. NVIDIA systems can report GPU telemetry through
NVML with `pip install "jetlink[nvml]"`.
