# Set up Jetlink on your platform

Choose the computer that will run the model. For the tested Jetson setup, use
[the Jetson guide](tester-setup.md). These desktop paths are for experiments and
bench testing: Mac has measured results below; Linux NVIDIA and Windows have not
yet been tested on hardware.

A **backend** is the software that runs the model: TensorRT for NVIDIA, CoreML
through onnxruntime for Mac, or tinygrad as an alternative. The comma uses the
same Jetlink client with each backend.

## Before you start

You need Git, Python 3.10 or later, internet access, and several GB of free disk
space. Mac CoreML testing used a 16 GB M1 Pro and about 5.5 GB per cached model.
Open Terminal on Mac or Linux, or an Ubuntu terminal inside WSL2 on Windows.

Download the project first:

```bash
git clone https://github.com/zoompilot/jetlink.git
cd jetlink
```

If connecting to a comma, check out the Jetlink revision pinned by your compatible
fork before installing dependencies:

```bash
git checkout <pinned-jetlink-revision>
```

Replace the placeholder with the commit linked by `jetlink_repo` in your fork's
GitHub file list at the installed build's commit. Use the same revision for a
separate benchmark client. See [paired releases](releasing.md#compatibility).

## Mac (Apple silicon)

Install Python 3.10+ and Git first. For USB, you also need the native `libusb`
library; if you use Homebrew, run `brew install libusb`.

From the `jetlink` folder, start the server:

```bash
scripts/run-mac.sh
```

The script creates a Python environment and installs dependencies on its first
run. It then serves over TCP with CoreML on the GPU by default. Keep it running
and use [Test without a comma](#test-without-a-comma) in a second terminal.

For a comma USB connection, stop the TCP server with **Ctrl-C**, then run:

```bash
JETLINK_TRANSPORT=usb scripts/run-mac.sh
```

Use a USB-A port on a hub or dock and an A-to-C data cable to the comma. Follow
[Connect the comma](tester-setup.md#connect-the-comma). A waiting-for-gadget message
is normal until the comma connects.

CoreML took **about 9 minutes to prepare or load a model for each new session**
on the tested Mac, including after restarting the server. Keep the server running
to avoid repeating that wait. The script uses `caffeinate` to prevent idle system
sleep while on AC power; keep the Mac powered and its lid open.

Optional commands, run one at a time:

```bash
# Prepare a model file, then exit. CoreML still needs a long load when serving.
scripts/run-mac.sh --build /path/to/big_driving_supercombo.onnx

# Include the Neural Engine. Benchmark at 20 Hz before using this option.
scripts/run-mac.sh --device ane

# Use tinygrad on Metal instead of CoreML.
JETLINK_BACKEND=tinygrad scripts/run-mac.sh
```

Replace model paths with a compatible large driving-model ONNX file. On M1 Pro,
CoreML GPU averaged 43 ms per round trip; tinygrad averaged 66 ms, above the
50 ms frame budget. Neural Engine testing missed some deadlines at 20 Hz despite
faster back-to-back results. See [measurements](#mac-measured).

## Linux (NVIDIA GPU)

You need a working NVIDIA driver compatible with the installed CUDA/TensorRT
runtime. On Ubuntu or Debian, install the Python environment and USB library:

```bash
sudo apt update
sudo apt install -y python3-venv libusb-1.0-0
```

From the `jetlink` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[trt,usb,nvml]"
jetlink-server --list-backends
jetlink-server --backend trt --transport tcp
```

The server should report its backend and listen on port 5599. Keep it running
for [a bench test](#test-without-a-comma). TensorRT is installed from PyPI; this
PC setup has not yet been run on hardware.

For USB, stop the TCP server with **Ctrl-C**, install the device permission rule,
then unplug and reconnect the comma:

```bash
sudo install -m 644 scripts/99-jetlink-host.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
jetlink-server --backend trt --transport usb
```

Follow [Connect the comma](tester-setup.md#connect-the-comma). In a new terminal,
run `source .venv/bin/activate` from the project folder before using `jetlink-server`.

## Windows (NVIDIA GPU)

Use **Ubuntu in WSL2** for this experimental path. Set up WSL2 and NVIDIA GPU
access first, then run the checkout and [Linux setup](#linux-nvidia-gpu) commands
inside the Ubuntu terminal. Start with TCP and run the benchmark in a second
Ubuntu terminal using `--host 127.0.0.1`.

Windows USB is not a validated quick-start path. WSL2 needs `usbipd-win` to attach
the comma's USB device to Ubuntu. Native Windows needs WinUSB driver binding;
there is no automatic driver setup in this project. See [platform limitations](#risks-and-decisions)
before attempting USB integration.

## CPU only

Use this path to test the protocol and model loading without a supported GPU.
It does not establish that the computer can keep up with driving.

From the project folder on Mac, Linux, or Windows WSL2:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[ort]"
jetlink-server --backend ort --device cpu --transport tcp
```

## Test without a comma

You need a compatible large driving-model ONNX file, supplied separately. The
comma integration normally downloads the selected model for you; this standalone
benchmark needs a local file. Replace `/path/to/big_model.onnx` below with its path.

Keep your platform's TCP server running. On a Jetson, start it with
`sudo docker/run.sh --transport tcp` instead of USB. In a second terminal on the
same computer, or another computer with a checkout of the same Jetlink revision:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e . onnx
python3 scripts/bench_link.py --host 127.0.0.1 --onnx /path/to/big_model.onnx --rate 20
```

Use `127.0.0.1` when client and server run on the same computer. For separate
computers, replace it with the server's wired-network IP address. TCP uses port
**5599** and has no client authentication: use a trusted network. Wi-Fi missed
the frame budget in testing.

The benchmark uploads the model if needed, waits for preparation, and reports
round-trip latency and frames over **50 ms**, the budget at 20 frames per second.
A passing short test does not establish sustained in-car performance.

## Cache and troubleshooting

Desktop caches use `JETLINK_CACHE` if set, otherwise `~/.cache/jetlink`. Each
backend keeps artifacts for its runtime version and device. The Jetson container
uses `/mnt/data/jetlink`.

| Problem | Check |
| --- | --- |
| `python3` or `git` not found | Install Python 3.10+ or Git and reopen the terminal |
| `jetlink-server` not found | Activate `.venv` from the project folder |
| Backend missing | Run `jetlink-server --list-backends`; check that platform's dependencies and GPU driver |
| USB library error | Install native libusb as well as the Python `usb` extra |
| USB permission error on Linux | Install the udev rule above, then reconnect the device |
| TCP connection refused | Check the server is running with `--transport tcp`, the address, and firewall access to port 5599 |
| Mac appears stuck loading | Allow about 9 minutes for CoreML; check server output for errors |
| Link drops when laptop sleeps | Keep it awake, powered, and open |

Stop a foreground server with **Ctrl-C**. For comma alerts and reporting a problem,
see [troubleshooting](tester-setup.md#troubleshooting).

## Backend reference

The details below preserve the runtime comparison and measurements recorded on
September 7, 2026. Expectations are not hardware validation.

### Runtime comparison

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

### Platform matrix

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
| build / load in a fresh process | 13 s / 1.1 s | 524 s / 527 s | 621 s / 635 s |
| artifact on disk | 777 MB | 5.5 GB | 5.5 GB |
| peak RSS while building | 0.6 GB | 7.9 GB | 9.1 GB |

Read it this way:

- **tinygrad is correct and over budget on this machine.** 66 ms against a
  50 ms frame, every frame. The kernels account for 32 ms of it and the rest
  is per-kernel launch overhead across 393 launches,
  which a faster GPU only partly removes. A newer Mac is expected under budget
  and has to be measured, not assumed.
- **CoreML on the GPU is correct, under budget, and the Mac default.** 43 ms
  round trip with a p99 of 44, no frame over budget in 390. It costs nine
  minutes of compile every time a process creates the session, and
  onnxruntime's `ModelCacheDirectory` did not shorten a second session; the
  compile runs in a worker process while the server keeps answering (526
  pings during a 527 s load, worst 11 ms), so the comma sees a long "loading
  engine" and the small model drives, the same wait and the same fallback as
  a Jetson rebuilding a plan. `--backend tinygrad` trades that for a
  one-second start and a 66 ms frame.
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
