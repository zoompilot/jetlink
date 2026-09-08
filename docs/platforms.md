# Serving from something other than a Jetson

The server runs wherever one of its backends does: TensorRT on an NVIDIA GPU,
tinygrad on Apple silicon (or any GPU tinygrad drives), onnxruntime as a
fallback and for CoreML. The comma side does not change; it talks to the
server over USB or TCP and learns what it is talking to from the hello.

Written 2026-09-07 against the measurements below. Numbers marked *measured*
were taken that day; everything else is an expectation to be replaced by one.

## Backends

| | `trt` | `tinygrad` | `ort` |
| --- | --- | --- | --- |
| Runtime | TensorRT 10.3 (JetPack 6) or 11.x (PyPI) | tinygrad 0.11+ | onnxruntime 1.22+ |
| Devices | CUDA | METAL, CUDA, NV, AMD, CPU | coreml, cuda, cpu |
| Artifact | `.plan` | `.pkl`, the captured JIT with its weights | `.ortcache/`, the patched ONNX and onnxruntime's cache |
| ONNX surgery | uint8 images to fp16, `org.tinygrad` op stripped | none: tinygrad runs the export as it is | same as TensorRT |
| Build, big model | 166 s on an Orin Nano | 13 s on an M1 Pro, *measured* | 10 min on an M1 Pro (CoreML), *measured* |
| Load | 6 to 25 s | 1 s, *measured* | the same 10 min: CoreML compiles per session, see below |
| Status | validated on the car (Jetson); 11.x untested | parity passed, over budget on an M1 Pro | parity passed on the GPU; the Neural Engine is wrong |

`--backend auto` takes TensorRT where it imports, then tinygrad, then
onnxruntime. Every backend keys its artifacts by its own runtime version and
device, so a machine with two installed keeps two per model and each sees only
its own. TensorRT's key is byte for byte what it was before there were
backends: a Jetson's existing plans load without a rebuild.

## Platform matrix

| | Jetson Orin | Linux, NVIDIA GPU | Windows, NVIDIA GPU | macOS, Apple silicon |
| --- | --- | --- | --- | --- |
| Backend | `trt`, unchanged | `trt` from `pip install "jetlink[trt]"`; `tinygrad` or `ort` as fallbacks | `trt`: wheels exist for `win_amd64`, but start in WSL2 | `tinygrad` on METAL; `ort` for CoreML |
| USB link | libusb host, today | libusb host plus `scripts/99-jetlink-host.rules` | WSL2 with `usbipd-win`; native needs WinUSB, see risks | libusb host, no driver; a USB-A port on a hub and the same A-to-C cable |
| TCP link | yes | yes | yes | yes |
| Telemetry | Tegra sysfs | NVML (`pip install "jetlink[nvml]"`) | NVML | none: the sensors need privileges, and the comma is told nothing rather than zeros |
| Sleep, poweroff | yes | `--sleep-after` works where `/sys/power` does; nothing wakes a laptop on a USB edge | no | no; `scripts/run-mac.sh` holds `caffeinate` |
| Install | Docker image, unchanged | `pip install -e ".[trt,usb,nvml]"` | pip inside WSL2 | `scripts/run-mac.sh`, which makes the venv |
| Status | validated on the car | expected to work; not yet run | untested | measured below |

## Running it

Linux with an NVIDIA GPU:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[trt,usb,nvml]"
sudo install -m 644 scripts/99-jetlink-host.rules /etc/udev/rules.d/ && sudo udevadm control --reload
jetlink-server --transport tcp            # bench from another machine
jetlink-server --transport usb            # the comma on a USB port
```

The cache goes to `JETLINK_CACHE`, else `~/.cache/jetlink`. `pip` brings
TensorRT 11.x, which removed weakly typed networks and the FP16 builder flag:
the build asks for a strongly typed network there and precision follows the
ONNX, which is fp16 end to end, so the engine is the same and only the calls
differ (`backends/trt/build.py`). That path has not been run on hardware yet;
the Jetson's TensorRT 10.3 path is untouched.

macOS:

```bash
scripts/run-mac.sh --build /path/to/big_driving_supercombo.onnx   # once per model
scripts/run-mac.sh                                                  # tinygrad on Metal, TCP
JETLINK_TRANSPORT=usb scripts/run-mac.sh                            # the comma on a USB-A port
JETLINK_BACKEND=ort scripts/run-mac.sh                              # CoreML, see below
```

Any machine, to see what would be chosen:

```bash
jetlink-server --list-backends
```

## Mac, measured

M1 Pro, 16 GB, macOS 25.5, Cinque Terre (766 MB). tinygrad 0.14.0 at
`e837e367aac9`, onnxruntime 1.29.0. The parity numbers are
`scripts/verify_parity.py` over TCP loopback against the onnxruntime CPU
provider on the unmodified graph (`org.tinygrad` op stripped), 32 frames,
every slice and column gated at 0.999.

| | tinygrad METAL | onnxruntime CoreML, GPU | onnxruntime CoreML, ALL (Neural Engine) |
| --- | ---: | ---: | ---: |
| frame, server side, mean | 66.3 ms | 38.9 ms | 25.1 ms |
| round trip over TCP loopback at 20 Hz, mean / p99 / max | 67.9 / 82.2 / 89.9 ms | | |
| parity, worst slice correlation | 0.999935 (lead_prob) | 0.999998 whole output, 1.00000 per slice | **0.91 to 0.97: wrong** |
| build | 13 s | 590 s | 668 s |
| load in a fresh process | 1.1 s | the same 590 s | 1158 s with the cache directory present |
| peak RSS while building | 0.6 GB | 7.9 GB | 8.0 GB |

What that means:

- **tinygrad is correct and over budget on this machine.** 66 ms against a
  50 ms frame, every frame. The kernels account for 32 ms of it and the rest
  is per-kernel launch overhead across 393 launches (`docs/multi-platform-plan.md`),
  which a faster GPU only partly removes. A newer Mac is expected under budget
  and has to be measured, not assumed.
- **CoreML on the GPU is correct and under budget.** 39 ms a frame, with a
  p99 of 40 ms. It costs ten minutes of compile every time a process creates
  the session, and onnxruntime's `ModelCacheDirectory` did not shorten the
  second session. So it is the faster frame and the slower start: a server
  that restarts is unusable for ten minutes, while tinygrad's is back in one
  second. tinygrad is the default and CoreML is `--backend ort` for a server
  that stays up.
- **CoreML with the Neural Engine is fast and wrong.** With `MLComputeUnits`
  at `ALL` the output correlates at 0.91 to 0.97 with the CPU reference, with
  errors above 100 in places, and CoreML logs "ANE model load has failed for
  on-device compiled macho". The backend pins `CPUAndGPU`. If a later macOS
  or onnxruntime fixes the Neural Engine path, the 25 ms frame is there to be
  re-measured, and parity is the gate.

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
  the whole compile, so the session lives in a spawned worker process with
  the frame's inputs and outputs in shared memory (`backends/ort/worker.py`).
  The cost is a message each way per frame.

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
