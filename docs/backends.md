# Backends and performance

Jetlink supports TensorRT, tinygrad, and ONNX Runtime. For installation, see
[platform setup](platforms.md).

## Runtime comparison

| Backend | Devices | Prepared files | Requirements |
| --- | --- | --- | --- |
| `trt` | NVIDIA CUDA | `.plan` | TensorRT 10.3 on JetPack 6, 10.16 on JetPack 7.2, or 11.x from PyPI on a PC |
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
| macOS with Apple silicon | ONNX Runtime with CoreML on the Neural Engine and GPU | USB-A hub, dock, or adapter with libusb | Not available | `scripts/run-mac.sh` prevents idle sleep on AC power |

See [status and limitations](status.md) for platform testing coverage.

## Mac, measured

These results use a 16 GB M1 Pro, macOS 26.5, ONNX Runtime 1.29.0, tinygrad
0.14.0 at `e837e367aac9`, and the 766 MB Cinque Terre V3 (`404a18cfd86d2963`)
and V2 (`09d080f36965bb2a`) models. Performance on other Macs may differ.

The frame budget is 50 ms at 20 frames per second (20 Hz). On Apple silicon the
default runs the model's vision layers on the Neural Engine and the rest on the
GPU. `--device coreml` runs everything on the GPU. It is slower, but use it if
another app keeps the Neural Engine busy: the default assumes Jetlink is the
only thing using it. tinygrad exceeds the budget.

| | Default: Neural Engine and GPU | GPU only (`--device coreml`) | tinygrad METAL |
| --- | ---: | ---: | ---: |
| V3 round trip at 20 Hz through the server, mean / p99 / max | 31.6 / 37.0 / 54.4 ms | 43.7 / 47.5 / 58.7 ms | not measured |
| V2 round trip at 20 Hz through the server, mean / p99 / max | 28.6 / 32.3 / 45.0 ms | 43.7 / 47.9 / 65.8 ms | not measured |
| frames over the 50 ms budget | V3 1 of 900, V2 0 of 1,200 | V3 7 of 900, V2 6 of 900 | 390 of 390 |
| parity gate, worst column (V3 / V2) | 0.99957 / 0.99957 pass | 0.99957 pass, earlier model | 0.99954 pass |
| build / load in a fresh process | about 20 s / 0.8 to 11 s | about 10 s / 1.8 to 4.7 s | 13 s / 1.1 s |
| artifact on disk | 2.1 GB | 2.3 GB | 777 MB |

The tinygrad column is an earlier measurement of Cinque Terre at 20 Hz: 66.2 ms
mean, 67.6 ms p99, against 43.3 ms and 44.4 ms for the GPU in the same run. The
default and GPU columns were measured on 2026-09-25 in 300-frame blocks, one
server at a time, alternating between the options and the variants below so that
each saw the same conditions. The first table is the shipped code, measured last
and with the Mac busier than during the comparisons below, which is why its GPU
p99 is higher than their 44.7 ms. A load of the default takes under a second
when the same model was the last one loaded, and 5 to 11 s after another, while
macOS prepares the Neural Engine's part again.

The mean is the average frame time. The p99 is the time at or below which 99% of
frames complete. The maximum is the slowest frame.

### How the default runs

The default (`--device ane`, which `auto` picks on Apple silicon) runs the
convolutional trunk, which reads the camera frames, on the Neural Engine in
about 20 ms, where the GPU takes 31 ms, and the rest on the GPU. How it gets
there depends on the model:

- A model that keeps its own history, such as Cinque Terre V3, runs as two
  CoreML sessions: the trunk on the Neural Engine, then everything after it,
  including the policy and its history, on the GPU. The two halves exchange
  32 KB per frame. One session that lets CoreML choose among all compute
  units measured 115 ms on V3, because the Neural Engine cannot run its
  stateful policy efficiently.
- A model whose history the server keeps, such as V1 and V2, runs as one
  session that lets CoreML choose among all compute units. Its policy's
  LayerNormalizations run in fp32, which the Neural Engine cannot do, so
  CoreML keeps the policy on the GPU. One CPU core is kept busy while frames
  arrive, for CoreML's own work in each frame.

Mean / p99 in ms at 20 Hz, from the interleaved runs:

| | V3 | V2 |
| --- | ---: | ---: |
| two sessions, trunk on the Neural Engine | **32.2 / 36.6** | 29.7 / 33.5 |
| one session, every compute unit, one core kept busy | 114.8 / 123.2 | **28.6 / 31.5** |
| GPU only | 43.1 / 44.7 | 43.6 / 46.2 |

The two-session split keeps the Neural Engine's fp16 LayerNormalization out
of the layers after the trunk: with them on the Neural Engine,
`road_transform` fell to a correlation of 0.9988 over 32 frames and failed
the parity gate.

Both layouts rewrite two Expand operations CoreML will not take as the
equivalent Tiles, so the policy stays one CoreML program instead of two with a
CPU step between them (worth 3.7 ms mean and 9 ms p99 on V3), ask CoreML for its
FastPrediction specialization, and run the Metal keep-alive described below for
their GPU work (without it the split measured 46.4 ms mean and 53.5 ms p99).

The busy CPU core helps only the one-session layout: on V2 it took that session
from 29.2 ms mean and 32.3 ms p99 to 28.7 ms and 31.9 ms, over 1,200 frames
each. With the split it changed neither the mean nor the p99, so the split does
not spin a core. Spinning only in the 10 ms before each expected frame spun a
tenth as much and matched the mean, but one of its three blocks reached a 38 ms
p99, so the core spins for as long as frames arrive. The core stops spinning
within a second of the last frame.

Other apps can use the Neural Engine too, and the default slows down when they
do: with another process running a model on it back to back, the split measured
52.5 ms mean and 65 ms p99 where the GPU option measured 43.5 ms. Switch to
`--device coreml`, or **CoreML on the GPU** in the Mac app, if that happens.

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
It runs whenever a session uses the GPU, including the GPU half of the default;
CPU sessions do not start it. If Metal initialization or a
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

For the default, Jetlink normalizes negative Gather indices, which the Neural
Engine mishandles, rewrites the two Expands as Tiles, and either splits the model
after the trunk or runs its policy's LayerNormalizations in fp32, as above. It
takes about 20 seconds to build. An engine prepared for the Neural Engine by an earlier
version is rebuilt automatically.

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
