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
| macOS with Apple silicon | ONNX Runtime with CoreML on the GPU | USB-A hub, dock, or adapter with libusb | Not available | `scripts/run-mac.sh` prevents idle sleep on AC power |

See [status and limitations](status.md) for platform testing coverage.

## Mac, measured

These results use a 16 GB M1 Pro, macOS 25.5, the 766 MB Cinque Terre model,
tinygrad 0.14.0 at `e837e367aac9`, and ONNX Runtime 1.29.0. Performance on other
Macs may differ.

The frame budget is 50 ms at 20 frames per second (20 Hz). CoreML on the GPU is
the default because it meets this budget on the M1 Pro. tinygrad exceeds the
budget. The Neural Engine option (`--device ane`) is about 7 ms faster at 20 Hz,
but it depends on nothing else using the Neural Engine; see
[the Neural Engine option](#the-neural-engine-option).

| | tinygrad METAL | CoreML, GPU (`--device coreml`, default) | CoreML, Neural Engine and GPU (`--device ane`) |
| --- | ---: | ---: | ---: |
| round trip at 20 Hz through the server, mean / p99 / max | 66.2 / 67.6 / 67.9 ms | 43.3 / 44.4 / 44.5 ms | 36.5 / 40.6 / 42.1 ms |
| frames over the 50 ms budget | 390 of 390 | 0 of 390 | 0 of 1,740 |
| round trip with no pause between requests through the server, mean / p99 | 66.2 / 67.6 ms | 39.9 ms server side | not measured |
| parity gate, worst column | 0.99954 pass | 0.99957 pass | 0.99957 pass |
| parity, mean error on `plan` / `lead_prob` | 0.0060 / 0.0156 | 0.0046 / 0.0150 | 0.0057 / 0.0140 |
| build / load in a fresh process | 13 s / 1.1 s | 8.2 s / 2.0 s | 16 s / 0.6 s |
| artifact on disk | 777 MB | 2.3 GB | 2.1 GB |
| peak memory use while building | 0.6 GB | 3.0 GB | not measured |
| peak memory use while loading | not measured | 2.5 GB | 1.6 GB, worker only |

Memory figures include the server and its worker process, sampled once per
second, except where noted. The Neural Engine column is the 766 MB Cinque
Terre V3 model, measured 2026-09-25 in six 300-frame blocks alternating with
the GPU option, which measured 43.2 to 43.4 ms mean and p99 44.0 to 44.1 in
the same run. Cinque Terre V1 measured 35.3 to 35.7 ms against 41.8 to 42.0.

### The Neural Engine option

`--device ane` runs the model as two CoreML sessions. The convolutional
trunk, which reads the camera frames, runs on the Neural Engine in about
20 ms, where the GPU takes 31 ms. Everything after it, including the heads,
the policy and a stateful model's history, runs on the GPU in about 12 ms.
The two halves exchange 32 KB per frame.

The split is placed where it is for accuracy and speed:

- The Neural Engine computes LayerNormalization in fp16, which is not precise
  enough for the layers after the trunk. With them on the Neural Engine,
  `road_transform` fell to a correlation of 0.9988 over 32 frames and failed
  the parity gate.
- The Neural Engine cannot run the stateful policy efficiently: it took
  83 ms there.
- A single session that lets CoreML choose among all compute units measured
  45 ms for Cinque Terre V1, with 55 of 300 frames over budget, and 107 ms
  for V3.

The option depends on the Metal keep-alive described below. Without it the
same split measured 46.4 ms mean and 53.5 ms p99.

It is not the default because other processes share the Neural Engine.
While another process ran a model on the Neural Engine back to back, the
split measured 52.5 ms mean and 65 ms p99, with 28% of frames over budget;
the GPU option measured 43.5 ms and 45 ms. With the other process busy half
the time, the split averaged 34.5 ms but one block reached a 49 ms p99.
Photos' media analysis can use the Neural Engine on an idle Mac. With all CPU
cores saturated, both options missed the budget (49 to 51 ms mean, 60 to
65 ms p99).

The mean is the average frame time. The p99 is the time at or below which 99% of
frames complete. The maximum is the slowest frame.

### The Neural Engine split

`--device ane` runs the vision trunk on the Neural Engine and the policy on
the GPU: the policy's LayerNormalizations run in fp32, which the Neural Engine
cannot do. At 20 Hz each unit is idle for most of every 50 ms and pays for it
on the next frame, which is what made this option miss the budget. It now
keeps both awake while frames arrive: the Metal keep-alive (below) for the
GPU's half, and one CPU core kept busy for CoreML's share of each prediction.
It also rewrites two Expands CoreML will not take as the equivalent Tiles,
which kept the model from splitting into two CoreML programs with a CPU step
between them, and asks for Apple's FastPrediction specialization.

On the same M1 Pro with model `09d080f36965bb2a`, 2026-09-25,
`bench_link.py --rate 20` over TCP loopback in one session, 1,190 frames after
a warm-up run:

| | CoreML, GPU (`--device coreml`, default) | `--device ane` before | `--device ane` now |
| --- | ---: | ---: | ---: |
| round trip at 20 Hz, mean / p99 / max | 46.9 / 49.5 / 50.7 ms | 44.0 / 55.9 / 59.9 ms | 27.7 / 29.6 / 47.4 ms |
| frames over the 50 ms budget | 6 (0.5%) | 137 (11.5%) | 0 |
| server side: model / queues | 45.2 / 1.1 ms | 41.4 / 1.7 ms | 26.1 / 1.1 ms |
| build in a fresh process | | 12.7 s | 16.0 s |

The parity gate passes with the new preparation: worst column 0.99957, mean
error on `plan` / `lead_prob` 0.0054 / 0.0141.

What each part is worth, measured on the model alone through one onnxruntime
session paced at 20 Hz, 1,200 frames each:

| `--device ane` | mean / p99 | frames over 50 ms |
| --- | ---: | ---: |
| new preparation, no helpers | 44.3 / 55.4 ms | 73 |
| with the Metal keep-alive | 33.8 / 41.6 ms | 0 |
| with the Metal keep-alive and the CPU keep-warm | 27.3 / 28.7 ms | 0 |
| both helpers, without the Tile rewrite | 29.5 / 31.1 ms | 0 |

Set `JETLINK_CPU_KEEPWARM=0` or `JETLINK_METAL_KEEPALIVE=0` to turn either
helper off. Both stop within a second of the last frame. The keep-warm is a
separate process, so a busy loop never holds the server worker's GIL; it
blocks in `select()` while no frames arrive and exits with the worker.

An artifact built with the earlier preparation is refused on load and rebuilt
from the ONNX, as an artifact from before the weight rewrite is.

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
It runs whenever a session uses the GPU, including the GPU half of the Neural
Engine option; CPU sessions do not start it. If Metal initialization or a
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

For the Neural Engine option, Jetlink normalizes negative Gather indices,
which the Neural Engine mishandles, and splits the model after the trunk. An
engine prepared by an earlier version as a single session is rebuilt
automatically.

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
