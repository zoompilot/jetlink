# Backends and measurements

Developer reference for the inference backends, the platform matrix, and the
measurements behind the defaults. For setup, see [platform setup](platforms.md).

The details below preserve the runtime comparison and measurements recorded on
September 7, 2026. Expectations are not hardware validation.

## Runtime comparison

| | `trt` | `tinygrad` | `ort` |
| --- | --- | --- | --- |
| Runtime | TensorRT 10.3 (JetPack 6) or 11.x (PyPI) | tinygrad 0.11+ | onnxruntime 1.22+ |
| Devices | CUDA | METAL, CUDA, NV, AMD, CPU | coreml, cuda, cpu |
| Artifact | `.plan` | `.pkl`, the captured JIT with its weights | `.ortcache/`, the prepared ONNX, a manifest, onnxruntime's cache |
| ONNX surgery | uint8 images to fp16, `org.tinygrad` op stripped | none: tinygrad runs the export as it is | as TensorRT, plus negative Gather indices normalised; for `ane`, the policy's LayerNormalizations in fp32 |
| Build, big model | 166 s on an Orin Nano | 13 s on an M1 Pro, *measured* | 9 to 10 min on an M1 Pro (CoreML), *measured* |
| Load | 6 to 25 s | 1 s, *measured* | the same 9 to 10 min: CoreML compiles per session, in a worker while the server keeps answering |
| Status | validated on the car (Jetson); 11.x untested | parity passed, over budget on an M1 Pro | parity passed, under budget on the GPU |

`--backend auto` takes TensorRT where it imports, then CoreML on a Mac, then
tinygrad, then onnxruntime on whatever it has. Every backend keys its
artifacts by its own runtime version and device, so a machine with two
installed keeps two per model and each sees only its own. TensorRT's key is
byte for byte what it was before there were backends: a Jetson's existing
plans load without a rebuild.

## Platform matrix

| | Jetson Orin | Linux, NVIDIA GPU | Windows, NVIDIA GPU | macOS, Apple silicon |
| --- | --- | --- | --- | --- |
| Backend | `trt`, unchanged | `trt` from `pip install "jetlink[trt]"`; `tinygrad` or `ort` as fallbacks | `trt`: wheels exist for `win_amd64`, but start in WSL2 | `ort` (CoreML on the GPU) by default; `--device ane` or `--backend tinygrad` by name |
| USB link | libusb host, today | libusb host plus `scripts/99-jetlink-host.rules` | WSL2 with `usbipd-win`; native needs WinUSB, see risks | libusb host, no driver; a USB-A port on a hub and the same A-to-C cable |
| TCP link | yes | yes | yes | yes |
| Telemetry | Tegra sysfs | NVML (`pip install "jetlink[nvml]"`) | NVML | none: the sensors need privileges, and the comma is told nothing rather than zeros |
| Sleep, poweroff | yes | `--sleep-after` works where `/sys/power` does; nothing wakes a laptop on a USB edge | no | no; `scripts/run-mac.sh` holds `caffeinate` |
| Install | Docker image, unchanged | `pip install -e ".[trt,usb,nvml]"` | pip inside WSL2 | `scripts/run-mac.sh`, which makes the venv |
| Status | validated on the car | expected to work; not yet run | untested | measured below |

## Mac, measured

M1 Pro, 16 GB, macOS 25.5, Cinque Terre (766 MB). tinygrad 0.14.0 at
`e837e367aac9`, onnxruntime 1.29.0. Parity is `scripts/verify_parity.py`'s
gate: 32 frames through the queues with hidden-state feedback, against the
onnxruntime CPU provider on the unmodified graph (`org.tinygrad` op stripped),
every slice and column at or above 0.999. Round trips are `bench_link.py`
over TCP loopback through the real server at 20 Hz.

| | tinygrad METAL | CoreML, GPU (`--device coreml`, default) | CoreML, every unit (`--device ane`) |
| --- | ---: | ---: | ---: |
| round trip at 20 Hz through the server, mean / p99 / max | 66.2 / 67.6 / 67.9 ms | 43.3 / 44.4 / 44.5 ms | 44.6 / 58.8 / 68.9 ms |
| frames over the 50 ms budget, of 390 | 390 | 0 | 69 |
| round trip back to back through the server, mean / p99 | 66.2 / 67.6 ms | 39.9 ms server side | 32.6 / 38.7 ms |
| parity gate, worst column | 0.99954 pass | 0.99957 pass | 0.99957 pass |
| parity, mean error on `plan` / `lead_prob` | 0.0060 / 0.0156 | 0.0046 / 0.0150 | 0.0057 / 0.0138 |
| build / load in a fresh process | 13 s / 1.1 s | 8.2 s / 2.0 s | not re-measured |
| artifact on disk | 777 MB | 2.3 GB | not re-measured |
| peak RSS while building | 0.6 GB | 3.0 GB | 9.1 GB |
| peak RSS while loading | not measured | 2.5 GB | not measured |

The CoreML build and load were 524 s and 527 s until 2026-09-10, when the
weights stopped travelling through the compiled model as text. onnxruntime's
MatMulAddFusion emits `Gemm` with `transB=0`, and the CoreML EP's Gemm builder
transposes that weight on the host and writes it into `model.mil` as hex float
literals at 6.06 bytes per fp16 value: the trunk's MIL was 4.109 GB against a
47 MB `weight.bin`, and coremlc wrote all of it on a build and parsed all of
it on every load. `onnx_patch.gemm_with_transposed_weight` does the same
fusion first with the weight transposed and `transB=1`, which the builder
passes through as a TensorProto, so it lands in the weight file. Measured on
the same machine, back to back, before and after:

| | before | after |
| --- | ---: | ---: |
| build | 526.4 s | 8.2 s |
| load, warm cache | 464.6 s | 2.0 s |
| artifact on disk | 5.91 GB | 2.30 GB |
| peak RSS while loading | 9.59 GB | 2.52 GB |
| trunk `model.mil` | 4,109,111,037 B | 1,339,974 B |
| trunk `weights/weight.bin` | 47,246,080 B | 717,700,480 B |
| trunk BLOBFILE consts / fp16 immediates | 241 / 74 | 315 / 0 |
| parity gate, worst column | 0.999619 pass | 0.999619 pass |

The arithmetic is untouched: the MIL op is `linear` either way, the parity
columns agree digit for digit, and the server-side GPU time was 41.28 ms
before against 41.19 to 41.28 ms over three runs after. Round trips that
session were 45.8 ms mean before and 45.6 to 45.8 ms after, with 1 frame of
390 over budget before and 0 to 4 after; the machine was under more memory
pressure than when the 43.3 ms row above was taken, so read those as a
before-and-after pair rather than against the table.

An engine prepared before this change still loads in minutes. The artifact is
still valid and the cache key still finds it, so nothing forces a rebuild;
preparing the model again is what makes it fast.

Read it this way:

- **tinygrad is correct and over budget on this machine.** 66 ms against a
  50 ms frame, every frame. The kernels account for 32 ms of it and the rest
  is per-kernel launch overhead across 393 launches,
  which a faster GPU only partly removes. A newer Mac is expected under budget
  and has to be measured, not assumed.
- **CoreML on the GPU is correct, under budget, and the Mac default.** 43 ms
  round trip with a p99 of 44, no frame over budget in 390. It used to cost
  nine minutes of compile every time a process created the session, which is
  what the compiled model carrying its weights as text cost; with the weights
  in the weight file a build is 8 s and a load 2 s. The compile still runs in
  a worker process while the server keeps answering, so a first prepare never
  blocks the comma. `--backend tinygrad` trades a 43 ms frame for a 66 ms one.
- **The Neural Engine is correct now, and faster only back to back.** As
  exported it was 25 ms and wrong (whole-output correlation 0.91 to 0.97).
  A sub-model bisect found one node: `Gather(add_53, -1)`, the last-token
  select after the temporal transformer, comes back as garbage on the
  Neural Engine and is exact with the index written as 287. With that
  rewritten (`onnx_patch.normalize_gather_indices`) the frame is 28 ms and
  the gate fails by one column: the Neural Engine's fp16 LayerNormalization
  overflows on the residual stream, and the policy half comes out seven
  times less precise than on the GPU. Running the policy's 41
  LayerNormalizations in fp32 (`onnx_patch.layernorm_in_fp32`, which CoreML
  then places off the Neural Engine) restores the GPU's precision exactly
  and makes the policy faster; doing the same to the trunk's 41 costs a
  compute-unit switch each and took the frame to 70 ms, so the trunk stays
  in fp16. Result: gate passed through the server, 32.6 ms round trip back
  to back. But at 20 Hz, with the units idle between frames, the same
  session is 44.6 ms with a p99 of 59 and 69 frames of 390 over budget:
  every CoreML unit pays a cost on the first request after an idle gap (a
  127 MB sub-model went from 3 ms to 11 on the Neural Engine and 6 to 14 on
  the GPU with any gap over 5 ms, and a keep-warm model in the gap did not
  help), and the Neural Engine pays more of it. A trunk-on-the-Neural-Engine,
  policy-on-the-GPU split of two sessions was measured too and pays it
  twice, 45 ms at 20 Hz. So `ane` is the opt-in: faster on a server that is
  never idle, worse at the car's cadence on an M1 Pro, and the thing to
  measure first on a faster Mac, with `bench_link.py --rate 20`.

Two things the runtimes made the server do differently:
- **tinygrad is one thread.** A JIT unpickled on one thread crashes when run
  from another, and a thread that has used Metal crashes as it exits, in
  objc's autorelease-pool drain. The backend routes every tinygrad call
  through one daemon thread that never exits (`backends/tinygrad/owner.py`);
  the crash report that found it is quoted there.
- **onnxruntime is one process.** It holds the GIL while it creates a
  session, measured with a ticker thread that ran three times across twenty
  session creations and a CoreML build that logged nothing for ten minutes.
  In the server that would stop the pings, the progress and the accept for
  the whole compile, so the sessions live in a spawned worker process with
  the frame's inputs and outputs in shared memory (`backends/ort/worker.py`).
  The cost is a message each way per frame.
- **onnxruntime phones home, and crashes doing it.** The macOS wheel ships
  Microsoft's telemetry SDK, which uploads over HTTP from its own thread; a
  response arriving as the process exits is dispatched through a mutex that
  is already gone, and the process aborts (one test run in three, crash
  report in `backends/ort/__init__.py`). The backend calls
  `disable_telemetry_events()` in every process that imports onnxruntime.
- **tinygrad must come from git**, at or after commit
  `e837e367aac9e1a66e689f4f32ce20ca9367df13` of
  <https://github.com/sunnypilot/tinygrad>; the PyPI 0.14.0 wheel has no
  `org.tinygrad` ONNX domain and cannot load the exported models. Install it
  with:

```bash
pip install --no-deps "tinygrad @ git+https://github.com/sunnypilot/tinygrad@e837e367aac9e1a66e689f4f32ce20ca9367df13"
```

## Risks and decisions

- **Windows native USB.** libusb on Windows needs WinUSB bound to the device.
  Zadig does that by hand for development; for users the gadget has to carry
  Microsoft OS descriptors so Windows binds WinUSB itself, which is a
  FunctionFS descriptor change on the comma, hence a fork release and a
  validation drive. WSL2 with `usbipd-win` avoids all of that and is where
  Windows starts.
- **Mac Type-C to comma Type-C.** Both ends are dual-role ports and the
  comma's is a host normally. An A-to-C cable from a USB-A port on a hub or
  dock fixes the roles the way the Jetson's USB-A port does. Test C-to-C
  afterwards, not first.
- **Host sleep.** A laptop lid or a Mac's idle sleep kills the link. The
  server refuses `--sleep-after` where there is no `/sys/power`, and
  `run-mac.sh` holds `caffeinate`. Nothing wakes a Mac on a USB edge the way
  the Jetson's hub does.
- **Thermal.** Laptops throttle under a sustained 20 Hz load. NVML telemetry
  makes that visible on the comma; nothing else changes.
- **What stays the same.** The protocol, the queues, the client, the fork's
  patch, the Jetson image and its validated numbers. The hello gains
  `backend` and `runtime_version`; `trt_version` is still sent by the
  TensorRT backend, and only by it.
