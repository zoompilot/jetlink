# 80. CoreML: load fast when cached, and report real progress

Workstream H, decided 2026-09-09 after the first end-to-end runs. Two asks
from the user: "if a model is cached already we want to get the user on road
and big model loaded as fast as possible", and "actual progress output for
gui when building coreml; we must have this in cli and all platforms also".

## What was found (measured on the M1 Pro, 16 GB, onnxruntime 1.29.0)

1. **The compile cache was never hit on the first load.** The metadata key
   went in as `CACHE_KEY`; onnxruntime reads `COREML_CACHE_KEY`
   (`coreml_provider_factory.h`) and otherwise keys the cache on a hash of the
   model *path*. The build compiles under a temp dir, then moves the artifact,
   so the first load compiled everything again: the "18 min first prepare"
   was 9 min compile + 8.5 min recompile, and the artifact carried two 4.8 GB
   compiles. Fixed in `jetlink/server/backends/ort/__init__.py` (commit
   `a064a07`): the right key, and `repair_coreml_cache` on load renames the
   final-path compile to the key and deletes the stale one. **Not yet
   verified end to end** (a fresh build followed by a load must create one
   cache directory named by the key and the load must not compile).

2. **The compiled program is a 3.8 GB text file, and every load parses it.**
   `compiled_model.mlmodelc/model.mil` for the trunk partition is 3.8 GB;
   `weights/weight.bin` is 45 MB. 335 million fp16 values (670 MB of weights)
   are immediate values in the MIL text, all named `linear_val_NNN_t`. Cause:
   onnxruntime's graph optimizer fuses `MatMul + Add` into `Gemm` with
   `transB=0`, and the CoreML EP's Gemm builder (`gemm_op_builder.cc`) then
   transposes the weight on the host and adds it through
   `AddConstant(span)`, which is always an immediate value; only initializers
   that reach the builder as `TensorProto` go to the weight file
   (`ShouldWriteInitializerToWeightsFile`, 10 elements or more). `MatMul`
   alone lowers to `matmul` with B used as is (weight file); `Gemm` with
   `transB=1` uses B as is too. The ONNX has 132 MatMul (116 with an
   initializer B, 357 million elements) and 37 Gemm.
   A load with the cache hit took 148 s on 2026-09-09 23:07; 9 min earlier
   in the evening under swap pressure.

3. Progress today: `build` is elapsed time against `EXPECTED_COREML_SECONDS`
   (660 s) and `load` is frac 0 with an elapsed-time message. tinygrad reports
   real phases (`backends/tinygrad/build.py`), TensorRT uses
   `IProgressMonitor` (`backends/trt/build.py`). The CLI (`jetlink-server
   --build`, which `jetlink-models prepare` calls) logs one line per 2 % from
   the same `report(stage, frac, msg)`.

## Goals, in order

1. **A cached model loads as fast as this machine allows.** Target: the
   server reports `engine ready` within 30 s of `preloading the engine loaded
   last` for the big model on CoreML, cache warm. Get there by keeping the
   transposed weights out of the MIL text. Candidates, try in this order and
   measure each:
   a. Session option `session.disable_specified_optimizers` =
      `MatMulAddFusion` (and any other fusion that produces a host-transposed
      weight; check the MIL for immediates after each attempt). MatMul stays
      `matmul` + `add`; CoreML may fuse them itself.
   b. Rewrite the ONNX in `_prepared_model`: `MatMul(x, W) [+ Add(b)]` with a
      2-D `x` becomes `Gemm(x, W^T, b, transB=1)` with the transposed
      initializer, so the builder takes the TensorProto path. Rank-3 inputs
      need a Reshape pair or stay MatMul.
   c. Anything else that ends with `weight.bin` holding the weights.
   The frame-time gate must still pass: `scripts/verify_parity.py` at or
   above 0.999 every column, and `scripts/bench_link.py --rate 20` over TCP
   loopback through the real server at 20 Hz with mean at or under 45 ms and
   zero frames over 50 ms of 390, as in `docs/backends.md#mac-measured`.
   Record load time, build time, artifact size, peak RSS, frame times before
   and after.
2. **Verify the cache key fix** end to end (item 1 above) and that
   `repair_coreml_cache` turns an old artifact into a working one without a
   rebuild. Write a test for the repair that uses a fake cache layout (no
   CoreML in CI).
3. **Real progress for CoreML, through the existing `report(stage, frac,
   msg)`**, so the app, the comma and the CLI all get it with no protocol
   change:
   - stage `convert`: onnxruntime writing the MLProgram into the cache
     directory (`Data/com.microsoft.OnnxRuntime/model.mlmodel` and
     `weights/weight.bin` growing). frac = bytes written / bytes of the ONNX
     initializers, msg like "converting for CoreML, 412 MB of 766 MB".
   - stage `compile`: `compiled_model.mlmodelc` appearing and growing. With
     weights in the blob the MIL is small; measure what grows and against
     what total it can honestly be reported. If nothing on disk grows for
     the whole stage, frac stays at the elapsed estimate but the msg says
     what is happening and how far along the last measured compile of this
     size was (sidecar records `compile_seconds`).
   - stage `load`: session creation from a warm cache. If goal 1 lands this
     is seconds and needs only the message. If not, report the worker's
     resident size against the last load's recorded peak (the sidecar
     records `load_rss_bytes` and `load_seconds` after every load).
   - The parent's ticker thread does the watching (the worker holds the GIL
     inside `InferenceSession`); a poll every 2 s is enough. Bytes on disk
     come from `os.scandir`; the worker's RSS from `ps -o rss=` on macOS and
     `/proc/<pid>/status` on Linux, nothing on Windows.
   - The CLI prints the same lines it prints today; every stage ends with a
     100 % line and the elapsed time. `jetlink-models prepare` shows the
     download bar, then the build lines.
   - `macos/Jetlink/Views/Components/ProgressRow.swift` `stageName` gains
     `convert` ("Converting for CoreML") and `compile` ("Compiling"). The
     toolbar and the menu bar use `StatusBadge.summary`, unchanged.
   - `EXPECTED_COREML_SECONDS` and the "about 18 minutes" and "9 minutes"
     strings in `SettingsView.backendCaption`, `docs/macos-app.md`,
     `docs/backends.md` and `plans/macos-app/30-swiftui-views.md` become the
     new measurements.

## Constraints

- Memory: a CoreML build peaks near 8 GB and this Mac has 16 GB with a lot of
  swap in use. Before any CoreML build or load, make sure no Jetlink app
  server is running (`pgrep -f jetlink.server.main`); if one is, wait for it
  to go away (poll every 30 s, up to 30 min). Never kill it. Build into a
  scratch cache under the worktree's `build/`, not the app's cache.
- `docs/backends.md#mac-measured` is the method for every number.
- Python 3.10 and 3.12 on Linux and 3.14 on the Mac must stay green:
  `ruff check jetlink tests` and `pytest -q` (312 tests).
- No edits to `plans/macos-app/*.md`; report deviations and numbers in the
  final message and the orchestrator records them in `01-contracts.md`.

## Acceptance

- One cache directory named by the key after build + load; no recompile.
- Load time, artifact size, frame time table (before / after), and where the
  weights live now (`grep -c BLOBFILE` and the immediate count in the MIL).
- `jetlink-server --build` on CoreML prints progress that moves with the
  work, and the app's Status view shows the same stages.
