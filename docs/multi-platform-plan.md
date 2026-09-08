# Running the server on any accelerator

Jetson today; CUDA laptops and Apple silicon Macs next. Written 2026-09-07
against jetlink `5da0415`, and implemented the same day on the
`jetlink-anywhere` branch: phases 1 to 3 are in the tree, phase 0 (a Linux
NVIDIA laptop) and 4 (USB off the Jetson) are still measurements to make.
`docs/platforms.md` is the reference for what exists; this document is the
reasoning that led there, and the measurements that changed it.

Numbers marked *measured* were taken on the day; everything else is an
expectation to be replaced by a measurement.

## The short version

- **Most of the server is already generic.** The wire protocol, framing, both
  host-side transports, the history queues, the session state machine, the
  engine cache and the client know nothing about TensorRT or a Jetson. What
  does is about 650 lines behind two function calls in `session.py`
  (`_load_engine`, `_build`) plus three lines in `on_hello`.
- **A Linux laptop with an NVIDIA GPU probably works today over TCP** with
  `pip install tensorrt-cu12 cuda-python onnx libusb1` and one environment
  variable for the cache path. Try that before refactoring anything; it is
  the cheapest data point in this plan.
- **The refactor is one seam, not a rewrite:** a `Backend` that builds and
  loads an artifact, an `Engine` that runs a frame. TensorRT becomes the
  first implementation with its on-disk keys unchanged, so the Jetson's
  existing plans keep loading. tinygrad becomes the second and covers Metal,
  and any GPU tinygrad supports.
- **tinygrad on Metal is not under budget yet.** Measured on an M1 Pro: the
  766 MB model replays in 65 ms against the 50 ms frame budget, and the
  kernels themselves account for 32 ms of it. The Mac is a development bench
  on day one and a drive-capable accelerator only after a tuning pass; the
  seam is what makes trying a second Mac backend cheap if tinygrad's own
  levers are not enough.
- Windows goes through WSL2 first. Native Windows USB needs a change on the
  comma's gadget descriptors, which is a fork change and a validation drive.

## What is coupled to what

Counted from the tree, so a later session can re-count rather than guess.

| Concern | Where | Lines | Portable? |
| --- | --- | ---: | --- |
| TensorRT execution, CUDA runtime | `server/engine.py`, `server/cudart.py` | 309 | NVIDIA only. Moves under `backends/trt/`. |
| TensorRT build, progress monitor, workspace sizing, device tag | half of `server/builder.py` | ~180 | NVIDIA only. Moves under `backends/trt/`. |
| Engine cache: entries, prune, sweep, last-loaded, timing cache path | other half of `server/builder.py` | ~170 | Generic once the key comes from the backend. Becomes `server/cache.py`. |
| ONNX surgery for TensorRT's parser | `onnx_patch.py` | 161 | TensorRT only. tinygrad takes uint8 inputs and runs its own `org.tinygrad` ops (verified in the fork's tinygrad, `nn/onnx.py:543`). |
| Session, engine host, hot path | `server/session.py` | 565 | Generic except `on_hello` (imports tensorrt for a version string) and the two seams above. |
| Entrypoint | `server/main.py` | 211 | Generic except `--build` calling `build_engine` directly. |
| Tegra telemetry | `server/telemetry.py` | 158 | Jetson sysfs, fails open to zeros elsewhere. Needs a source per platform. |
| Suspend, poweroff | `server/sleep.py`, `server/power.py` | 315 | Jetson host lifecycle, already opt-in by flag. Stays as is. |
| Gadget presence | `transport/usbbulk.py` `present()` | 10 | Linux sysfs. libusb enumeration everywhere else. |
| Free memory for the build workspace | `builder.available_bytes` | 15 | `/proc/meminfo`. Falls back to a flat 4 GB elsewhere, which is fine on a 16 GB laptop. |
| Default cache path | `DEFAULT_CACHE = /mnt/data/jetlink` | 1 | Jetson. Env override exists. |
| Packaging | `docker/`, `scripts/*.service`, udev rules | | Jetson image and host units. Laptops and Macs get pip extras. |

The fork imports only `jetlink.client` (and through it the transports, spec
and queues). Nothing in the fork imports `jetlink.server`, so the server can
be reshaped without touching the upstream patch or the submodule contract.

## The architecture

```
jetlink/server/
  main.py           --backend auto|trt|tinygrad, --device; the rest as today
  session.py        EngineHost(cache, backend, telemetry); Session unchanged
  cache.py          EngineCache, CacheEntry, prune, sweep, last-loaded (out of builder.py)
  telemetry.py      CachedTelemetry, plus pick_source(): tegra | nvml | none
  sleep.py          unchanged, Jetson only, opt-in
  power.py          unchanged, Jetson only, opt-in
  platform.py       is_jetson(), usb_present(vid, pid), available_bytes(), default_cache_dir()
  backends/
    __init__.py     available(), select(name): lazy imports, never at module scope
    base.py         the Backend and Engine protocols below
    trt/            engine.py, cudart.py, build.py: today's code, moved, keys unchanged
    tinygrad/       engine.py, build.py
```

### The seam

```python
class Backend(Protocol):
  name: str                 # 'trt' | 'tinygrad'
  suffix: str               # '.plan' | '.pkl'
  def describe(self) -> dict            # {'backend', 'runtime_version', 'device'} for hello
  def key(self, sha256: str) -> str     # trt: '<sha16>.trt10.3.0.Orin-sm87', byte for byte as today
                                        # tinygrad: '<sha16>.tg0.13.0.METAL-AppleM1Pro'
  def build(self, onnx: Path, out: Path, report: ProgressFn, cache: EngineCache) -> None
  def load(self, artifact: Path) -> Engine      # raises ArtifactInvalid on an unloadable file

class Engine(Protocol):
  inputs: dict[str, IO]     # IO(shape, dtype); outputs likewise
  outputs: dict[str, IO]
  def host_input(self, name: str) -> np.ndarray   # writable staging array, engine dtype
  def run(self) -> dict[str, np.ndarray]          # views valid until the next run()
  def warm(self) -> str                           # CUDA graph capture or JIT capture; a log line
  last_gpu_us: int
  def close(self) -> None
```

This is the shape `session.py` already assumes. `_infer` writes through
`host_input` and calls `run`; `_warm` calls `capture_graph` today and
`warm()` tomorrow; `_check_shapes` reads `input_shapes` and
`output_shapes`. `tests/test_session.py`'s `FakeEngine` implements exactly
this and nothing else, which is the evidence the seam is in the right place.

The cache keeps its layout: `engines/<key><suffix>` with a `<key>.json`
sidecar carrying the spec, `models/<sha16>.onnx`. The sidecar gains a
`backend` field. `inventory()` and `entry()` already go through `key()`, so
on a machine with both backends installed each sees only its own artifacts.
`last-loaded.json` records the backend so a restart preloads the right one.

`on_hello` reports `backend`, `runtime_version` and `device`, and keeps
`trt_version` for one release. The comma reads `device` and `trt_version`
only to log them (`backend.py:248`, `jetlinkd.py:310`) and `cached_models`
for the picker, so this is additive JSON under protocol 2, no bump.

`ArtifactInvalid` from `load()` is treated as "artifact missing": the host
deletes it and rebuilds if the ONNX is on disk, else answers `need_upload`.
That is what makes a pickled JIT from an older tinygrad self-healing rather
than a support ticket, and it costs TensorRT nothing.

### The tinygrad backend

Follows `selfdrive/modeld/compile_modeld.py` in the fork, which is the same
job on the comma's GPU.

- **build**: `OnnxRunner(onnx)`, wrapped in `TinyJit(prune=True)` around
  `outputs.cast('float32')`. Three calls with random inputs: compile,
  capture, replay, checking the replay reproduces the capture as
  `compile_jit` does. Pickle the captured JIT to `<key>.pkl`; weights go with
  it, so the artifact is model-sized like a plan. Progress is phase timing,
  since tinygrad has no progress monitor: parse, capture, pickle.
- **no ONNX patching**: tinygrad takes the uint8 image inputs as exported and
  runs `Contiguous` natively.
- **host_input dtype**: declare the image inputs as float16, which is what
  `PolicyQueues` gathers at memcpy speed, and cast to uint8 on the device
  inside the jitted function. Exact for 0..255, one line in the backend,
  nothing in the queues.
- **run**: `Tensor(host_array)` per input, JIT, `.numpy()` into a persistent
  output array. On Metal that is unified memory; on CUDA a 2 MB pageable copy.
- **device**: `--device METAL|CUDA|NV|AMD|CPU` sets tinygrad's `DEV` before
  import; `auto` takes `Device.DEFAULT`. CPU is accepted with a warning,
  because it will not make the budget.
- **key**: `importlib.metadata.version('tinygrad')` (0.13.0 in the sunnypilot
  venv), else the git sha of a source checkout, else `src`, plus any kernel
  search flag (`BEAM`) because it changes the generated kernels. A key
  mismatch rebuilds; an unpickle failure rebuilds via `ArtifactInvalid`.

### Platform helpers

- `usb_present()`: sysfs on Linux (today's code), libusb enumeration
  elsewhere. libusb initialises on this Mac with no driver (measured, 1.0.29
  via Homebrew); vendor-class interfaces need no kext on macOS.
- `available_bytes()`: `/proc/meminfo` on Linux, `sysctl hw.memsize` minus
  wired on macOS, `GlobalMemoryStatusEx` on Windows, or 0 which means "the
  flat cap". Only TensorRT reads it.
- `default_cache_dir()`: `/mnt/data/jetlink` if it exists, else the user
  cache directory. The Jetson image sets `JETLINK_CACHE` anyway.
- `--sleep-after` refuses with a message off a Jetson: the host OS owns
  sleep there, and a laptop lid is not a USB edge.
- Telemetry: `nvidia-ml-py` for desktop NVIDIA (temperature, power, clocks,
  utilisation map straight onto today's dict), nothing on macOS without
  privileges, so a `none` source that returns `{}` and lets the UI show
  nothing rather than zeros.

## Platform matrix

| | Jetson Orin | Linux, NVIDIA laptop | Windows, NVIDIA laptop | macOS, Apple silicon |
| --- | --- | --- | --- | --- |
| Backend | trt, unchanged | trt via `tensorrt-cu12` wheels; tinygrad CUDA as fallback | trt: wheels exist for `win_amd64`, but go through WSL2 first | tinygrad METAL |
| USB link | libusb host, today | libusb host plus a udev rule for `1209:0001` | WSL2 with `usbipd-win` passthrough; native needs WinUSB, see risks | libusb host; USB-A port on a hub or dock and the same A-to-C cable |
| TCP link | yes | yes | yes | yes, USB-C Ethernet adapter on either end |
| Telemetry | Tegra sysfs | NVML | NVML | none |
| Sleep, poweroff | yes | no | no | no; `caffeinate -s` while serving |
| Packaging | Docker image, unchanged | `pip install "jetlink[trt]"` or an x86 image from `nvcr.io/nvidia/tensorrt` | pip inside WSL2 | `pip install "jetlink[tinygrad]"` plus `brew install libusb` |
| Status | validated on the car | expected to work now over TCP; USB untested | untested | Metal measured 2026-09-07, over budget, see below |

PyPI facts checked 2026-09-07: `tensorrt-cu12-bindings` 11.2.1.2 ships
`manylinux_2_28_x86_64` and `win_amd64` wheels for CPython 3.8 to 3.14;
`cuda-bindings` 13.3.1 ships Linux x86_64, Linux aarch64 and Windows;
`libusb1` 3.4.0 is pure Python with a bundled DLL on Windows. The Jetson
stays on JetPack's TensorRT 10.3; plans are per version already, so a laptop
on 11.x builds its own and nothing is shared or broken.

## Phases

### 0. Try the CUDA laptop as it is

Half a day, no code. On a Linux laptop with an NVIDIA GPU:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e . "numpy<2" onnx tensorrt-cu12 "cuda-python>=12.6" libusb1
JETLINK_CACHE=$HOME/jetlink-cache python3 -m jetlink.server.main --transport tcp
# from any machine with the ONNX
python3 scripts/bench_link.py --host <laptop> --onnx big_driving_supercombo.onnx --n 1200
```

Known deltas to expect, none of them design problems: the `cuda-python<13`
pin in `pyproject.toml` (relax after the test; `cudart.py` already imports
both layouts), the cache path (the env var), telemetry reading as zeros, and
TensorRT 11 being untested where 10.3 is validated (the API surface used,
`Builder`, `OnnxParser`, `IProgressMonitor`, timing cache,
`execute_async_v3`, is stable across 10 and 11). Then USB: a udev rule
granting the VID:PID, `--transport usb`, and the comma's jetlinkd presenting
the gadget. Pass condition: `bench_link` mean and p99 recorded, and a
`verify_parity` capture at or above 0.999 per slice.

### 1. The seam, with no behaviour change

One to two days. Move `engine.py`, `cudart.py` and the build half of
`builder.py` under `backends/trt/`; extract `cache.py`; add `backends/base.py`
and `select()`; rename `capture_graph` to `warm` behind the protocol; add the
hello fields; add `platform.py`. Tests: a `FakeBackend` replaces
`install_stubs()` for the session tests, and `fake_trt.py` survives only for
the TensorRT backend's own unit tests. Keep shim modules at the old import
paths for `verify_engine.py` and anything else outside the tree for one
release.

Pass condition: the suite is green with no hardware; on the Jetson the same
plan files load without a rebuild because the key did not move; `bench_link`
and `verify_parity` reproduce the numbers in `README.md`.

Do not change what the Jetson builds while doing this. No new TensorRT flags,
no workspace changes: the validated numbers are against today's artifact.

### 2. The tinygrad backend

Two to three days. `backends/tinygrad/`, `--backend tinygrad --device METAL`,
the pickle artifact and its sidecar, progress, `verify_engine.py --backend`
replaying a capture through it. Parity first, then speed: `verify_parity`
over TCP loopback on the Mac against onnxruntime (`onnxruntime` 1.29 has
macOS arm64 wheels; the CPU provider is enough for 32 frames), and against a
comma capture over the LAN when one is available.

Pass condition: correlation at or above 0.999 per output slice, pooled over
32 frames. Speed is the next section's problem, not this phase's gate.

### 3. Packaging and platform polish

One to two days. Extras in `pyproject.toml`: `trt = [tensorrt-cu12,
cuda-python, onnx]`, `tinygrad = [tinygrad]`, `nvml = [nvidia-ml-py]`. The
`jetlink-server` console script is already declared. A `scripts/run-mac.sh`
that makes the venv, runs under `caffeinate -s` and passes
`--backend tinygrad`. A `scripts/99-jetlink-host.rules` for Linux hosts.
Optionally `docker/Dockerfile.x86` from an `nvcr.io/nvidia/tensorrt` base for
people who want a container on a laptop. `docs/platforms.md` and a platform
table in the README.

### 4. USB host off the Jetson

Linux laptop first, then the Mac, with the comma's gadget: `bench_link --ffs`
from the comma at 20 Hz, then the fork's live bench with the laptop as the
server. Windows via WSL2 and `usbipd-win`, measured; USB/IP adds a hop per
transfer and the jitter is unknown until measured. Native Windows only if
that number is bad.

### 5. Optional: the whole loop on one Mac

Not needed for the three platforms, high value for development: the fork's
modeld replay on the Mac talking to a Metal server over loopback would let
the joining state, the fallback and the UI be developed without a car or a
Jetson. Today the fork's `modeld/SConscript` compiles the small model with
`DEV=CPU` on Darwin, so this needs a `DEV=METAL` build on that side first.

## Mac performance, measured

M1 Pro, 16 GB, 16-core GPU, macOS 25.5. Cinque Terre (766 MB, the default in
`models.json`). tinygrad 0.13.0 from the sunnypilot venv (pin `e837e367a`,
2026-09-01), `DEV=METAL`, default flags. Script:
`scratchpad/metal_feasibility.py` from the 2026-09-07 session; it does what
the backend's per-frame path would do, fresh numpy inputs in, float32 out.

| | ms |
| --- | ---: |
| OnnxRunner construct | 900 |
| first call (compile) | 8 400 |
| second call (JIT capture) | 2 800 |
| per frame with host copies, mean | 75.4 |
| per frame with host copies, p99 / max | 79.2 / 80.4 |
| replay only, inputs resident, mean | 65.4 |
| replay only, p99 / max | 67.9 / 68.1 |
| non-JIT run: 393 kernels, summed kernel time | 31.9 |
| largest single kernel | 0.82 |
| ten largest kernels together | 5.3 |
| peak RSS | 601 MB |

Read it this way. The kernels do 32 ms of work and the graph replay takes 65
ms, so half of every frame is per-kernel scheduling in tinygrad's Metal
graph path, about 85 us across 393 launches, not arithmetic. The bandwidth
floor is nowhere near: 766 MB of weights over roughly 200 GB/s is 4 ms. So
the hardware is not the limit, the kernel count and the launch path are.

Levers, in the order to try them:

1. `BEAM=2`, tinygrad's per-kernel schedule search. Measured: 62 ms, see
   below. Not the lever. Would go in the cache key if ever used.
2. Fewer kernels: newer tinygrad, scheduler flags. The comma runs this same
   graph through the same scheduler on QCOM, so openpilot has an interest in
   this too.
3. A newer Mac. M3 and M4 Pro and Max GPUs are two to four times an M1 Pro.
4. A second Mac backend behind the same seam. onnxruntime's CoreML provider
   is being measured on the same model during the session (it needs
   `strip_tinygrad_ops` first: one `Contiguous` node); MLX is the other
   candidate. Either is a ~200 line backend once the seam exists, which is
   the point of building the seam before tuning.

For comparison the Jetson does the same model in 19.8 ms on the GPU and 31 ms
end to end in modeld, and a laptop RTX 4060 has three times the Jetson's
memory bandwidth, so TensorRT there should sit well under the Jetson. Measure
it in phase 0.

### Tuning results

Filled in as the runs finish.

| Configuration | replay mean | p99 | notes |
| --- | ---: | ---: | --- |
| tinygrad METAL, default | 65.4 | 67.9 | above; through the finished backend over TCP loopback: 66.2 mean, 67.6 p99, parity passed on every slice |
| tinygrad METAL, `BEAM=2` | 61.8 | 63.7 | 404 s to compile; with host copies 65.4 mean, max 159.9. Five percent, so kernel search is not the lever. |
| onnxruntime CoreML, `MLComputeUnits=ALL`, as exported | 25.1 | 30.3 | **wrong**: whole-output correlation 0.91 to 0.97 against the CPU provider; 668 s session |
| onnxruntime CoreML, `CPUAndGPU` | 38.9 | 40.5 | correct: 0.999998 whole output, 1.00000 per slice; 590 s session, and a second session with onnxruntime's cache directory present took 1158 s |
| onnxruntime CoreML, `ALL`, negative Gather index normalised | 28.2 | 30.7 | the "wrong" above was one node, `Gather(add_53, -1)`: exact with the index written as 287. Fixed, every slice is above 0.9996 and the parity gate fails by one column (road_transform std[3] at 0.9988); the policy half is seven times less precise on the Neural Engine and no faster than on the GPU |
| onnxruntime CoreML, trunk `ALL` + policy `CPUAndGPU`, two sessions | 31.1 (standalone harness) | 38.7 | passes the gate with the GPU path's precision, but 45 ms through the server at 20 Hz: each session pays an after-idle cost while the other runs |
| onnxruntime CoreML, `ALL`, Gather fixed, all 85 LayerNorms in fp32 | 70.2 | 106.9 | passes the gate; every fp32 LayerNorm in the trunk is a compute-unit switch |
| onnxruntime CoreML, `ALL`, Gather fixed, the policy's 41 LayerNorms in fp32 | 31.5 | 40.0 | passes the gate with the GPU path's precision; through the server 32.6 back to back and 44.6 mean, p99 58.8 at 20 Hz. Ships as `--device ane` |

`BEAM=2` moving the replay from 65 to 62 ms is the confirmation: the kernels
were already cheap, and searching their schedules cannot remove launches.
Lever 1 is crossed off; levers 2 and 4 are the ones left with leverage.

Lever 4 was then built and measured (`backends/ort`), and it is the Mac
default: CoreML on the GPU, correct by the parity gate and under budget at
43 ms round trip, at the cost of nine minutes of compile every time a
process creates its session, cache or no cache. The Neural Engine was made
correct too (a sub-model bisect found one `Gather` with a negative index,
then the policy's LayerNormalizations in fp32) and is 31 ms back to back,
but at the car's 20 Hz it pays an after-idle cost that puts it level with
the GPU on the mean and behind on the tail, so it ships as `--device ane`
for a faster Mac to measure. The reasoning and the numbers are in
`docs/platforms.md`; the tooling that found the node is
`scratchpad/submodel.py` from the 2026-09-08 session.

## Risks and decisions

- **Windows native USB.** libusb on Windows needs WinUSB bound to the
  device. Zadig does that by hand for development; for users the gadget has
  to carry Microsoft OS descriptors so Windows binds WinUSB itself, which is
  a FunctionFS descriptor change in `ffs.py` and `setup_gadget.sh` on the
  comma, hence a fork release and a validation drive. WSL2 with `usbipd-win`
  avoids all of that and is where Windows starts.
- **Mac Type-C to comma Type-C.** Both ends are dual-role ports and the
  comma's port is a host normally. An A-to-C cable from a USB-A port on a hub
  or dock fixes the roles by CC pull-up the way the Jetson's USB-A port does.
  Test C-to-C afterwards, not first.
- **Host sleep.** A laptop lid or a Mac's idle sleep kills the link. The
  server refuses `--sleep-after` off a Jetson and `run-mac.sh` holds
  `caffeinate`. Nothing wakes a Mac on a USB edge the way the Jetson's hub
  does.
- **Two backends on one machine.** A laptop can have TensorRT and tinygrad
  CUDA installed. `--backend auto` picks trt when it imports and a CUDA
  device exists, tinygrad otherwise; keys and inventories are per backend so
  neither sees the other's artifacts.
- **Thermal.** Laptops throttle under a sustained 20 Hz load. NVML telemetry
  makes that visible on the comma's screen; nothing else changes.
- **What stays the same.** The protocol, the queues, the client, the fork's
  patch, the Jetson image and its validated numbers. The refactor's own pass
  condition is that the Jetson cannot tell it happened.
