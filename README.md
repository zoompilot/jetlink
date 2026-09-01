# jetlink

Run openpilot's large ("chestnut") driving models on an attached NVIDIA Jetson.

openpilot's big model needs an accelerator. comma's answer is [chestnut][] — an
AMD PCIe GPU behind an ASM2464PD USB→PCIe bridge, driven from the comma's own
CPU by tinygrad. If you have a Jetson instead, it cannot be an eGPU: Tegra's GPU
is fused into the SoC and exposes no PCIe endpoint for another host to drive.

So jetlink does the other thing. The Jetson becomes an **inference server**. The
comma keeps the cameras, the warp, the parser, controlsd, the panda and CAN; the
Jetson runs the policy under TensorRT and hands back 18452 floats. It is a pure
function — it holds no control state and never touches the bus.

To the driver it behaves like an eGPU: pick a large model in the UI, it
downloads, it gets built, and the car drives on it.

## Does it actually fit?

Measured on an Orin Nano Super 8 GB, JetPack 6.1, TensorRT 10.3 FP16:

| | |
|---|---|
| Frame budget (`MODEL_RUN_FREQ = 20`) | **50 ms** |
| Model, GPU (CUDA graph) | 19.8 ms |
| History buffers, CPU | 1.4 ms |
| Transport, over the USB cable | 4.7 ms |
| **Round trip, comma to Jetson and back** | **26.0 ms** |
| p99 / max | 27.3 / 28.3 ms |
| Jitter (p99−p50) | **1.4 ms** |
| Frames over budget, 290 sampled | **0** |
| Reading `warped` off the comma's GPU | 4.4 ms |
| **On-car total** | **~32.5 ms** |

Measured comma-to-Jetson over the real USB 3 link, not a loopback: a comma 3X
as the FunctionFS gadget, an Orin Nano as the libusb host, SuperSpeed. Of the
4.7 ms transport, 2.9 ms is the 459 KB request and the rest is the 74 KB reply.
Over TCP loopback the same benchmark runs at 24.2 ms, so the cable costs about
4 ms.

That last row matters and is easy to miss: the benchmark hands the client a
numpy array, but on the car the warped frame has to be read back from the
comma's QCOM GPU first, and that runs at ~90 MB/s because it is write-combined
memory. It is a floor, not an inefficiency — upstream's chestnut path pays a
similar read to push `warped` to the AMD device.

The 1.4 ms of history-buffer work used to be 3.5 ms. Two thirds of it was one
`uint8 → float16` cast: numpy has no vectorised float16 *store* loop on
aarch64, so it converts at ~5.2 ns/element. The source is `uint8`, which has
only 256 possible values, so `queues.py` converts through a lookup table
instead — same bits, memcpy speed, no new dependency.

Numerics: **corr 0.999994** against an onnxruntime CPU reference.
Per frame the link carries 459 KB up and 74 KB down — 85 Mbit/s at 20 Hz.

## How it splits

The seam is `run_policy` in openpilot's `modeld`.

```
comma 3X                                    Jetson
────────────────────────────────────────    ──────────────────────────────
camerad ─► VisionIPC
          warp  (QCOM GPU, stays local)
            │ warped (2,6,128,256) u8  393 KB
            │ packed scalars + prev_feat 66 KB
            ╰──────────── link ───────────►  img_q / big_img_q
                                             feat_q / desire_q
                                             TensorRT FP16 engine
            ◄─────────── link ────────────╯  outputs (18452) f32   74 KB
          Parser ─► modelV2 ─► controlsd ─► panda ─► CAN
```

`warp` stays on the comma because its input is a 2 MB camera buffer already in
GPU memory and its output is the 393 KB the link has to carry anyway. The four
history queues live on the Jetson because shipping them per frame would cost
~10 MB instead of ~0.5 MB. That makes the queue logic the one piece of openpilot
that jetlink reimplements, so `tests/test_queues.py` checks it frame by frame
against the real tinygrad functions.

## Transport

Two are implemented behind one interface. **Check `docs/transport.md` before
buying anything** — which one you can use is decided by what is in your kernels,
not by preference.

| | |
|---|---|
| **Direct USB (raw bulk)** | Preferred. One cable, no IP stack, lowest jitter. The comma is the USB **gadget** (FunctionFS) and the Jetson is the **host** (libusb) — that way round neither side needs a kernel change: AGNOS has `CONFIG_USB_F_FS=y` built in, and a libusb host needs no driver at all. |
| **Ethernet (TCP)** | Fallback and bench path. A USB-C→gigabit adapter on the comma (`RTL8152`/`AX88179`, both built into AGNOS) to the Jetson's 1 GbE. |

USB *ethernet gadgets* are not an option: AGNOS has no host-side CDC-NCM,
CDC-ECM or RNDIS driver. And wifi is not an option either — measured
comma→Jetson it runs at 164 ms with 40 ms of jitter, missing every frame.

## Layout

```
jetlink/
  protocol.py          wire format: 32-byte header + payload
  spec.py              every size on the wire, derived from the model's ONNX
  queues.py            the history buffers, in numpy
  onnx_meta.py         model metadata, via tinygrad's or onnx's parser
  onnx_patch.py        uint8 -> fp16 graph surgery TensorRT needs
  client.py            the comma side
  transport/           tcp.py | usbbulk.py (host) | ffs.py (gadget)
  server/              engine.py, builder.py, session.py, telemetry.py
docker/                the Jetson image (l4t-jetpack r36.4.0)
scripts/               gadget setup, engine verify, link benchmark
```

## Quick start

On the Jetson:

```bash
docker/build.sh
docker/run.sh --transport tcp                       # serve
docker/run.sh --build /path/to/big_model.onnx       # or prebuild an engine
```

From the comma or a dev machine:

```bash
python3 scripts/bench_link.py --host <jetson> --onnx /path/to/big_model.onnx
```

The engine is built on the Jetson and cached under `/mnt/data/jetlink/engines`,
keyed by model hash, TensorRT version and GPU architecture — a plan is portable
across none of the three.

## Using it with openpilot

See [`docs/openpilot-integration.md`](docs/openpilot-integration.md). The patch
to upstream is 37 lines across three files; everything else is new modules.

## Licence

MIT. See [LICENSE](LICENSE).

[chestnut]: https://github.com/commaai/openpilot
