# Backends and performance

A backend prepares and runs the model. Every platform runs the same Swift
server: TensorRT on a Jetson or NVIDIA PC, ONNX Runtime everywhere else. Setup:
[setup guides](README.md#set-up). Mac detail:
[performance reference](mac-performance.md).

## Runtime comparison

| Backend | Where | Prepared files | Runtime |
| --- | --- | --- | --- |
| `trt` | Jetson, NVIDIA PC | `.plan` | TensorRT 10 on a Jetson (10.16 on JetPack 7.2, 10.3 on 6.2), JetPack's own; 11.3.0.99 on a PC, the installer's copy in `/opt/jetlink/tensorrt` (`--tensorrt-libs`). Loaded at run time |
| `ort` | Mac, iPhone and iPad (CoreML); Android (QNN); Linux without a GPU (CPU) | `.ortcache/` | ONNX Runtime 1.29.0 |

- `--backend auto`: TensorRT if it loads, else ONNX Runtime. The installed
  service asks for `trt`, so a Jetson or PC without TensorRT fails loudly
  instead of serving from the CPU.
- Prepared files are cached per runtime version and device. A TensorRT plan
  that no longer loads (after a TensorRT update) is prepared again, once.
- Options: [the server command](installation-reference.md#the-server-command).
  Jetson measurements: [performance](status.md#measured-performance).

`--device` picks the ONNX Runtime profile (`OrtProfile`), where it runs the
model. On a Mac:

| `--device` | Layout |
| --- | --- |
| `ane` (default) | vision trunk on the Neural Engine, the rest on the GPU |
| `ane-both` | the `ane` cut with the policy on the Neural Engine too, nothing on the GPU. Experimental: the policy's LayerNormalizations run in fp16 (see [how the default runs](mac-performance.md#how-the-default-runs)), so check a model's parity on it before driving. In the iPhone app on an iPad Air (M1) with Light Speed 2: 25.1 ms median and 26.5 worst over 3,600 frames at 20 Hz (`ane`: 32.2 and 44.6). It passes `scripts/verify_parity.py` against the CPU, hidden state included, on its 32 frames and on 300 real night frames, the policy's mean errors 1.5 to 8 times `ane`'s. |
| `coreml` | everything on the GPU |
| `ane-whole` | one CoreML program, every compute unit allowed (policy LayerNormalization inputs scaled by 1/8 so their fp16 squares do not overflow; heads after the trunk in fp32). For comparing against the default, not daily use; see [how the default runs](mac-performance.md#how-the-default-runs). |
| `cpu` | ONNX Runtime's CPU provider, for tests |

The iPhone app's **Processor** offers **Automatic** (the default), `ane-both`
(**Neural Engine Only**), `ane` (**Neural Engine + GPU**: 14 ms a frame on an
iPhone 18 Pro) and `coreml` (**GPU Only**); `cpu` in the Simulator. Automatic
runs a model on `ane-both` unless [Compare Processors](iphone-app.md#compare-processors)
has picked another for it. The Mac app's **Backend** offers `ane` (the
default), `ane-both` and `coreml`.

The Android app's **Processor**, on the same preparation:

| Device | Layout |
| --- | --- |
| `htp` (NPU + GPU, default) | vision trunk on the NPU in fp16, the rest on the GPU, as `ane` splits it; the NPU part compiled once into onnxruntime's EP context |
| `htp-whole` (NPU) | the whole graph on the NPU, prepared as `ane-whole`; fp16 throughout, heads included |
| `gpu` | everything on the Adreno GPU |
| `cpu` | onnxruntime's CPU provider, for the emulator |

None of the Android layouts has run on a Snapdragon yet.

## Platform matrix

| Platform | Backend | USB | Telemetry | Sleep |
| --- | --- | --- | --- | --- |
| Jetson Orin | TensorRT | USB-A host through usbfs | Tegra sensors | Suspend, USB wake and poweroff |
| Linux with NVIDIA GPU | TensorRT 11.3 | usbfs; `scripts/99-jetlink-host.rules` without root | NVML | `--sleep-after` needs `/sys/power`; USB wake depends on hardware |
| Windows with NVIDIA GPU (untested) | TensorRT 11.3 in WSL2 | `usbipd-win` | NVML | None |
| macOS with Apple silicon | ONNX Runtime with CoreML on the Neural Engine and GPU | IOKit; a USB 3 USB-C cable | Not available | The app prevents idle sleep on power |
| iPhone and iPad | ONNX Runtime with CoreML | TCP over the comma's USB network interface | Not available | Keep the app on screen |
| Android with Snapdragon | ONNX Runtime with QNN on the NPU and GPU | USB host through a hub; usbfs on the app's descriptor | Not available | A foreground service keeps it serving |

On Linux the server turns off USB 3 link power management on the comma's port:
[custom USB integrations](installation-reference.md#custom-usb-integrations).

Timing and power: [performance and operating limits](status.md).

<a id="how-the-default-runs"></a>
<a id="how-to-measure"></a>
<a id="keeping-the-mac-gpu-responsive-between-frames"></a>
<a id="model-preparation"></a>

## Mac, measured

16 GB M1 Pro, paced 20 Hz: the default Neural Engine/GPU split about 30 ms a
frame, GPU-only 41 to 44 ms. Bench numbers only: some CoreML runs still had
single frames over the deadline, so averages do not establish driving
reliability. [Full measurements and test conditions](mac-performance.md).

| If you need to... | Read |
| --- | --- |
| Compare latency, preparation time, and disk use | [Measured results](mac-performance.md) |
| Understand the Neural Engine/GPU split | [How the default runs](mac-performance.md#how-the-default-runs) |
| Reproduce the tests | [How to measure](mac-performance.md#how-to-measure) |
| Investigate intermittent GPU latency | [GPU keep-alive](mac-performance.md#keeping-the-mac-gpu-responsive-between-frames) |
| Understand CoreML engine preparation | [Model preparation](mac-performance.md#model-preparation) |

## Hardware limitations

- Native Windows is unsupported; use WSL2 ([Windows setup](platforms.md#windows-nvidia-gpu)).
- Mac: a USB 3 USB-C cable; the comma holds its port as the device, so the
  Mac is the USB host.
- Keep laptops powered and awake. Sustained GPU use can throttle thermally;
  check frame times.
