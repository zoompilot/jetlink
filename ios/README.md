# Jetlink for iPhone

Runs the large (chestnut) driving models on an iPhone's Neural Engine and
serves them to a comma over a wired link. Same protocol as the Jetson and Mac
servers, over TCP, so the comma side is zoompilot's `JetlinkEndpoint` mode.

**Status: experimental.** Queued models (V1, V2) only: stateful V3 graphs
(`new_img`, `state_img_q`) aren't supported yet, so the app reads catalog v25,
which has none. On an iPhone 17 Pro (A19 Pro) the 766 MB model runs
at 17.7 ms mean, 21.5 ms p99 a frame at 20 Hz, and passes the parity gate.
Over one USB cable, zoompilot's parked live bench ran every frame on the phone
(35.1 ms median, 39.1 ms p99). Driven once, for about 30 minutes: the phone
slowed as it heated.

## What you need

- An iPhone with USB-C and 8 GB of memory (iPhone 15 Pro or a later Pro:
  USB 3 and the strongest Neural Engines), iOS 17 or later.
- A Mac with Xcode 26 to build and install it.
- A comma on zoompilot's `jetson-trt` with `comma/zoompilot-tcp-provisioning.patch`
  ([below](#the-zoompilot-patch)).
- A wired link; Wi-Fi misses the 50 ms budget.
- Power for the phone: its port carries the link, so MagSafe or a hub with
  power pass-through.

## Install with a free Apple account

1. iPhone: Settings > Privacy & Security > Developer Mode on, and restart. The
   switch appears once the phone has been plugged into a Mac with Xcode open.
2. Xcode > Settings > Accounts: add your Apple ID (a "Personal Team"). For its
   ten-character team ID, pick that team once under the Jetlink target's
   Signing & Capabilities, run
   `grep -m1 DEVELOPMENT_TEAM ios/Jetlink.xcodeproj/project.pbxproj`, then
   `git checkout ios/Jetlink.xcodeproj`.
3. Put your team and a bundle ID of your own in `ios/Config/Local.xcconfig`
   (git-ignored), not in Signing & Capabilities, which writes them into the
   project:

   ```
   DEVELOPMENT_TEAM = ABCDE12345
   PRODUCT_BUNDLE_IDENTIFIER = com.yourname.jetlink
   ```

4. `open ios/Jetlink.xcodeproj`, pick the phone, Run. Run builds Release. The
   first build downloads onnxruntime 1.29.0 (64 MB); Xcode may ask to install
   the iOS platform (Settings > Components).
5. iPhone: Settings > General > VPN & Device Management > your Apple ID > Trust.

A free account's install stops opening after **7 days**; Run again to renew
(models and settings are kept). The app uses the Increased Memory Limit
capability, which a free team can sign: the model and its CoreML compile need
more than the default. The Status tab shows the memory left.

## Using it

1. Open Jetlink: the server starts on port 5599 and Status lists the phone's
   addresses. Keep it on screen; iOS suspends background apps. It keeps the
   screen on while serving.
2. Models: download the model the comma drives (its Big Model pick) on Wi-Fi,
   and Prepare. The comma can upload it instead, but that takes minutes.
3. Status > Benchmark > **Run for 1 minute**: the model at 20 Hz. Aim for p99
   ≤ 35 ms and nothing over 50 ms; the rest is for the cable. **Run for 10
   minutes** as in the car (charging, mounted) to see it slow as it warms.
4. **Accuracy**: the Benchmark screen gives a `scripts/verify_parity.py`
   command for a Mac on the same Wi-Fi. It must end with OK.

## Connecting to the comma

iOS gives apps no access to a vendor USB device, so the phone can't use
Jetlink's usual libusb link. It does drive USB network adapters, so the phone
is a TCP server and the comma connects to it through `JetlinkEndpoint`.

### The link

**One USB cable, through a hub:** iPhone > USB-C hub > USB-A to USB-C cable >
comma. The comma presents itself as a USB network adapter (CDC-NCM,
`comma/setup_net_gadget.sh`) at `192.168.60.1`; the phone is `192.168.60.2`.
AGNOS kernel 4.9.103 has NCM built in (`CONFIG_USB_CONFIGFS_NCM=y`).

A direct C-to-C cable reboots the comma. The two negotiate power and the comma
ends up supplying the phone. Its charger logged "Weak charger detected" and
"Reverse boost detected", then the log stopped at a Type-C change: a power
loss, not a crash. A USB-A port only supplies power, so through the hub the
comma never does. The link negotiates USB 3 (`super-speed`).

**Ethernet:** a USB-C Ethernet adapter on the comma (Realtek RTL8152/8153 or
ASIX AX88179, drivers in AGNOS), a hub with Ethernet on the phone, a cable. Set
the addresses by hand on another subnet, so the comma doesn't also present its
network adapter: comma `192.168.61.1/24`, endpoint `192.168.61.2:5599`.

### Switching between a Mac and the iPhone

With the patch, on the comma, parked: Settings > Models > **Accelerator on
iPhone**.
- **On:** points the link at `192.168.60.2:5599`, and the gadget owner
  presents the network adapter.
- **Off:** removes the adapter and presents Jetlink's gadget for a Mac or
  Jetson.

The owner also does this at boot, and never while the link is in use. The
first time, set the phone's address: Settings > Ethernet > (adapter) >
Configure IP > Manual, `192.168.60.2`, `255.255.255.0`, no router.

### The zoompilot patch

`comma/zoompilot-tcp-provisioning.patch` (for `jetson-trt` at `bcb49d7`)
fixes TCP mode, which was written for a Jetson on Ethernet during bring-up:

- **The parked provisioning run never started.** The owner waited for a
  gadget TCP mode never presents. That run records the engine ready, drives
  the home icon, and compiles the camera warp; without the warp, modeld never
  takes the large model.
- **Every endpoint borrow waited out an 8 s timeout.**
- **modeld's join never finished.** It waited for a USB host
  (`gadget.wait_for_host`), which never comes over TCP: 45 s, then it started
  over.
- **Nothing swapped the USB port between Jetlink's gadget and the iPhone's
  adapter.** The patch adds the toggle above, and `net_gadget.sh` (a copy of
  `setup_net_gadget.sh`) for the owner to run.

zoompilot's tests with it: 121 of its backend, gadget, owner, lending and
jetlinkd tests pass, 21 of them new. Three warp tests need openpilot's
hardware module and fail the same way without the patch. The panels aren't
tested (they need a raylib window).

To try it on a comma, with updates off so the updater doesn't reset it:

```bash
cd /data/openpilot && git apply /data/zoompilot-tcp-provisioning.patch
echo -n 1 > /data/params/d/DisableUpdates
```

Without the patch, bring the adapter up by hand after each boot
(`setup_net_gadget.sh`). The script releases Jetlink's FunctionFS gadget and
clears the "error: gadget torn down" its teardown leaves in
`/dev/shm/jetlink-gadget`, which zoompilot would otherwise read as "no link",
TCP included.

```bash
sudo bash setup_net_gadget.sh --check      # changes nothing; says whether NCM exists
sudo bash setup_net_gadget.sh              # the adapter, as 192.168.60.1
sudo bash setup_net_gadget.sh --teardown
```

### Measuring the link

Parked, with the link up:
- The Benchmark screen's "Over the link, from the comma" command runs
  `scripts/bench_link.py` on the comma (zoompilot carries this repo as
  `jetlink_repo`). Turn Accelerator Link off first, so the comma's client
  isn't holding the server.
- zoompilot's `tools/jetlink_live_bench.sh 180`: real cameras and modeld,
  controls off, three minutes.

## Neural Engine, GPU, or Automatic

Settings > Run models on:
- **Automatic** (default). On the first Prepare it builds for the Neural
  Engine and the GPU and times each at 20 Hz. It keeps the Neural Engine if
  its p99 is ≤ 35 ms (same work, far less power), else the GPU if it makes
  35 ms, else the faster, and deletes the other. "Measure again" on Models
  repeats it.
- **Neural Engine**: the whole model on the Neural Engine
  ([below](#how-it-differs-from-the-python-server)), FastPrediction, and a CPU
  core kept busy while frames arrive (Settings > Keep the CPU ready between
  frames).
- **GPU**: CoreML on the GPU with the Metal keep-alive. 154 ms p99 on an
  iPhone 17 Pro, too slow to drive.

## How it works

`JetlinkKit` is the server in Swift, for iOS and macOS (`jetlink-swift` is
the Mac command):

| File | Ports |
| --- | --- |
| `WireProtocol.swift`, `Transport.swift` | `jetlink/protocol.py`, `transport/base.py`, `transport/tcp.py` |
| `Session.swift`, `EngineHost.swift`, `Server.swift` | `server/session.py`, `_serve` in `server/main.py` |
| `Queues.swift`, `Convert.swift` | `jetlink/queues.py` |
| `ModelSpec.swift`, `Onnx.swift`, `Pickle.swift` | `jetlink/spec.py`, `jetlink/onnx_meta.py` |
| `OnnxPrepare.swift`, `Proto.swift` | `jetlink/onnx_patch.py`, `_prepared_model`, `scripts/ane_passes.py` |
| `OrtBackend.swift`, `OrtEngine.swift`, `MetalKeepAlive.swift` | `server/backends/ort/` |
| `CPUKeepWarm.swift` | the phone's own (the Mac's `ane` no longer uses one) |
| `EngineCache.swift`, `Registry.swift` | `server/cache.py`, `jetlink/registry/` |
| `Benchmark.swift` | the in-app benchmark |

onnxruntime is Microsoft's 1.29.0 iOS/macOS build, the version the Mac backend
is measured with, through a small C shim (`COrtShim`). The comma uploads raw
ONNX, so the phone prepares it: `OnnxPrepare` maps the file, copies untouched
weights through, and transposes Gemm weights one at a time as it writes, so
the model is never in memory twice.

### How it differs from the Python server

- **Neural Engine build.** The Mac's `ane` runs everything after the vision
  trunk on the GPU. That's fast on a Mac, but slow on a phone, whose GPU is
  far weaker than its Neural Engine. The phone keeps the whole model on the
  Neural Engine instead:
  - Expand → Tile (`jetlink.onnx_patch.expand_to_tile`, as every CoreML build).
  - Each policy LayerNorm's input divided by 8 in fp16. LayerNorm is
    unchanged up to epsilon, and the inputs reach 1,189, whose square
    overflows fp16. 99.5% of the estimated cost lands on the Neural Engine.
- **Vision heads in fp32.** The 24 nodes (4 MB of weights) between the
  vision trunk and the output are cast to fp32, so CoreML puts them on the GPU
  or CPU. In fp16 on the phone's Neural Engine, `road_transform` failed the
  parity gate (worst column 0.9989, `lane_lines_prob` mean error 0.080). With
  them it passes (0.99954; 0.009). The Mac's trunk split keeps them off the
  Neural Engine for the same reason.
- **A CPU core kept busy** while frames arrive, for the all-Neural-Engine
  build: p99 54 → 36 ms on an M1 Pro. Keeping the Neural Engine busy didn't
  help. (The Mac's trunk split doesn't need it.)
- **vImage conversions.** About 800,000 camera bytes a frame to fp16: 0.07 ms,
  bit-identical, against 0.2 ms (optimized) or 63 ms (Debug) as loops.
- **No worker process** (Swift has no GIL). The session is built on the build
  thread and run on the serving thread, with I/O bound to fixed buffers once.
- **Only CoreML's compiled model is kept** after a build: 1.4 GB instead of
  2.2 GB, with identical outputs.
- **A comma shutdown request is answered `ok: false`** (a comma can't power a
  phone off). No sensor telemetry, as on the Mac.
- **No shape inference fallback.** The Neural Engine passes refuse a model
  missing a shape they need; the Gemm rewrite leaves that MatMul, which only
  loads slower. The driving models record every shape.

## Verified

16 GB M1 Pro, macOS 26.5, model `09d080f36965bb2a`:
- **Preparation**: Swift and Python give the same protobuf, field for field,
  for the GPU, Neural Engine and CPU paths (`scripts/check_prepare.py`).
  They also match on synthetic models covering each branch
  (`scripts/prepare_variants.py`), and every build computes what the CPU build
  does.
- **Queues and conversions**: bit-identical to the Python, over runs that wrap
  every ring, with resets, NaN, overflow and subnormals.
- **End to end**: unmodified `scripts/bench_link.py` and
  `scripts/verify_parity.py` against `jetlink-swift serve` (upload, build,
  load, serve):

  | | GPU | Neural Engine |
  | --- | ---: | ---: |
  | worst column correlation | 0.99957 | 0.99956 |
  | mean error, `plan` / `lead_prob` | 0.0046 / 0.0150 | 0.0073 / 0.0346 |
  | round trip at 20 Hz, mean / p99 | 48.0 / 60.2 ms | 32.5 / 36.9 ms |
  | frames over 50 ms | 9.7% (Python server, same Mac) | 0 of 1,190 |

  The Neural Engine build has less parity margin (`lead_prob` error 0.0346
  against 0.0149), from the policy's LayerNorms in fp16.
- **Restart**: a restarted server preloads the last engine and serves a client
  that sends only the model's identity, as modeld does.
- `swift test`: 23 tests, including the whole protocol over TCP against the
  Python reference.

iPhone 17 Pro:
- **In-app benchmark**, 1,201 frames at 20 Hz, Release, CPU keep-warm:
  17.7 / 21.5 / 33.9 ms mean / p99 / max. That's 17.5 ms model and 0.2 ms
  queues, none over 35 ms, nominal temperature. Measured just before the fp32
  heads.
- **Parity** from a Mac over Wi-Fi: passes (worst column 0.99954).
- **Link**, one cable through a hub, `bench_link.py` from the comma, 1,190
  frames: 36.6 / 46.5 / 60.7 ms mean / p99 / max, 2 over 50 ms. About 19 ms of
  that is the link, although it runs at USB 3; the overhead isn't tracked down.
- **zoompilot's live bench**, parked, 180 s: 3,431 frames, all on the phone,
  35.1 / 39.1 / 111 ms p50 / p99 / max, 1 over 50 ms, none dropped.

## Not yet known

- **Heat.** On a ~30-minute drive the phone slowed as it warmed; a mount with
  cooling is untested. Run the 10-minute benchmark as in the car.
- **The phone's address after a comma reboot.** The comma's kernel picks its
  own MAC addresses, so the phone may see a new adapter and need
  `192.168.60.2` again.
- **Memory.** The Mac server peaks at 3.0 GB while building; the 1.7 GB
  Lebowski model may not fit an 8 GB phone.
- **Driving.** Link drops, and interruptions by a call or notification.

## Developing

```bash
make -C ios test      # JetlinkKit's tests
make -C ios check     # Swift preparation against the Python (needs the repo's .venv with onnx)
make -C ios device    # build for a phone, unsigned
make -C ios server    # ios/JetlinkKit/.build/release/jetlink-swift
make -C ios project   # regenerate Jetlink.xcodeproj from project.yml (xcodegen)

ios/JetlinkKit/.build/release/jetlink-swift serve --device ane --port 5599
python3 scripts/bench_link.py --host 127.0.0.1 --onnx big.onnx --rate 20
ios/JetlinkKit/.build/release/jetlink-swift build big.onnx --device ane --bench 60
python3 ios/scripts/check_prepare.py --onnx big.onnx --swift ios/JetlinkKit/.build/release/jetlink-swift
```

`scripts/queue_fixtures.py` and `scripts/server_fixtures.py` regenerate the
test fixtures from the Python package.
