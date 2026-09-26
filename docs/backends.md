# Backends and performance

A backend prepares and runs the model on your hardware. Keep the default for
normal use: TensorRT on NVIDIA, or ONNX Runtime with CoreML on Apple silicon.
For installation, see [platform setup](platforms.md).

Use this page to compare runtimes and check their requirements. Detailed Mac
measurements are in the [performance reference](mac-performance.md).

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
| Windows with NVIDIA GPU | TensorRT in WSL2 | Requires `usbipd-win` | NVML | None |
| macOS with Apple silicon | ONNX Runtime with CoreML on the Neural Engine and GPU | USB-A hub, dock, or adapter with libusb | Not available | `scripts/run-mac.sh` prevents idle sleep on AC power |

See [performance and operating limits](status.md) for timing and power considerations.

<a id="how-the-default-runs"></a>
<a id="how-to-measure"></a>
<a id="keeping-the-mac-gpu-responsive-between-frames"></a>
<a id="model-preparation"></a>

## Mac, measured

On a 16 GB M1 Pro, the default Neural Engine/GPU backend averaged 31 to 32 ms
per frame in paced 20 Hz tests. GPU-only averaged 44 ms. tinygrad averaged
66 ms and missed every 50 ms deadline in its test.

These are bench measurements. Some CoreML runs still had individual frames
over the deadline; averages alone do not establish driving reliability.
See the [full measurements and test conditions](mac-performance.md).

| If you need to... | Read |
| --- | --- |
| Compare latency, preparation time, and disk use | [Measured results](mac-performance.md) |
| Understand the Neural Engine/GPU split | [How the default runs](mac-performance.md#how-the-default-runs) |
| Reproduce the tests | [How to measure](mac-performance.md#how-to-measure) |
| Investigate intermittent GPU latency | [GPU keep-alive](mac-performance.md#keeping-the-mac-gpu-responsive-between-frames) |
| Understand CoreML engine preparation | [Model preparation](mac-performance.md#model-preparation) |

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

Native Windows USB requires WinUSB. Use WSL2 for the
[Windows setup](platforms.md#windows-nvidia-gpu). On a Mac, use a USB-A port on a hub, dock, or adapter to ensure
that the Mac acts as the USB host.

Keep laptops powered and awake. Sustained GPU use can cause thermal throttling;
check frame times during use. NVIDIA systems can report GPU telemetry through
NVML with `pip install "jetlink[nvml]"`.
