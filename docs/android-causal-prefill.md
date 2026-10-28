# Experimental Android causal prefill

This build-time opt-in overlaps Cinque Terre V3's current-image vision work on
HTP with its historical-token work on the QNN GPU. HTP decode waits for both,
then a small CPU graph assembles the original outputs and next queues. Current
queries still attend to all 288 original keys and values; no previous-frame
answer replaces current-frame computation.

The scheduler and provider discovery are generic. The graph split is deliberately
limited to the exact CTV3 export below. Other models select the existing whole-NPU
route **before preparation**; the sidecar records that route and Settings reports
NPU. A missing or altered CTV3 bundle, unsupported operator/provider, allocation
failure or inference error fails explicitly. Such errors do not trigger a silent
CPU fallback or a different graph.

There is no SM8750, manufacturer, HTP architecture or ORT-version whitelist.
The normal QNN backend discovers hardware using the runtime/DSP libraries the
build ships. This does not establish numerical correctness or speed on every
Snapdragon. Run original-model accuracy checks and sustained tests on each
device/runtime/firmware before relying on it. Existing Automatic and processor
defaults remain unchanged.

## Generate the source-only bundle

Use the normal Android/Swift build prerequisites and the original catalog ONNX:

```
SHA-256: 404a18cfd86d29637d20c697dfde245bb47c666ae016730ab674c65f4d1e1aa4
Size: 766354845 bytes
```

```bash
python3 -m venv .venv-prefill
.venv-prefill/bin/python -m pip install -r scripts/android/requirements-causal-prefill.txt
swift build --package-path JetlinkKit --product jetlink-prefill-base
BASE_TOOL="$(swift build --package-path JetlinkKit --show-bin-path)/jetlink-prefill-base"
.venv-prefill/bin/python scripts/android/compile_causal_prefill.py \
  --model /path/to/original.onnx --base-tool "$BASE_TOOL" \
  --out /path/to/NEW-causal-prefill-v1
```

The compiler checks the full source, prepared-base, four-part and manifest
identities, and rejects Python `-O`. It preserves original learned weights,
precision, output slices and recurrent contract. The legacy metadata cache-key
string in the base helper is retained solely to reproduce the historical graph
bytes; it is not a hardware selector. Model weights and compiled contexts are
not distributed in the source tree.

## Build and stage

```bash
cd android
./gradlew :app:assembleRelease :app:testDebugUnitTest \
  -Pjetlink.causalPrefill=true -Pjetlink.qnnSharedIO=true
```

Ordinary builds hide and reject the experimental choice. The experimental
profile always disables CPU EP fallback in its HTP and GPU sessions, including
HTP context compilation and reload. Shared I/O is optional: only actual HTP
allocators label tensors `QnnHtpShared`; GPU and CPU bindings use host metadata.
The engine retains allocations, sessions and bindings in the required order.

The historical Xiaomi experiment required disabling advanced HTP activation
fusion to pass its original accuracy gates. To reproduce that numerical path,
build the optional adapter and **one complete matching SDK runtime** as described
in [Android QNN policy](android-qnn-npu.md), adding
`-Pjetlink.qnnNoActivationFusion=true -Pjetlink.qnnJniDir=/path/to/NEW-qnn-jni`
to the command above. No fixed SDK binary hashes or SoC settings are imposed.
Actual SDK library bytes identify its separate cache namespace. The generic
stock-runtime path still needs independent numerical qualification.

Stage only generated `vision.onnx`, `history.onnx`, `decode.onnx`, `join.onnx`
and `manifest.json` into the app's external files directory, using the project's
normal signing/install workflow. For an unlocked development device:

```bash
BUNDLE=/sdcard/Android/data/io.zoompilot.jetlink.android/files/causal-prefill-v1
adb shell mkdir -p "$BUNDLE"
adb push /path/to/NEW-causal-prefill-v1/. "$BUNDLE/"
adb shell chmod 755 "$BUNDLE"
adb shell chmod 644 "$BUNDLE"/*.onnx "$BUNDLE/manifest.json"
```

Keep the original CTV3 model available in Jetlink and select **NPU + GPU
(experimental)**. The app verifies source/manifest/part hashes before copying;
reload verifies the exact session contract and retained GPU/assembly graphs.
Contexts compile locally for the device. Do not transfer another phone's
compiled contexts; rebuild and recheck accuracy after firmware/runtime changes.

## Validation and evidence

`SessionDAG` derives barriers from typed graph I/O and rejects cycles/multiple
writers. All branches join before an error escapes. Recurrent parity advances
only after a successful frame. Focused tests exercise reordered sessions,
serial/parallel recurrent equivalence, reset, worker failure, generic provider
options, model routing and invalid bundle/cache metadata.

A historical Xiaomi 15 prototype measured **27.2 ms mean / 29.5 ms P99** across
12,001 synthetic frames at 20 Hz, with original 128-frame accuracy checks before
and after. Its exact four graphs measured **45.2 / 47.2 ms serial** versus
**26.8 / 29.0 ms parallel** in shorter sequential runs. These are historical
prototype measurements, not measurements of this portable port or another phone.
[Immutable prototype evidence](https://github.com/joe-sama/jetlink/blob/e681e507b5e83384f3316da1d25b4cce823e5ea9/docs/evidence/sm8750-causal-prefill/README.md)
retains the logs, screenshot, identities and thermal limitations.

Use the existing `scripts/verify_parity.py` capture/reference/compare flow with
the original model, 128 identical inputs and unchanged gates, including recurrent
resets. Compare the same graphs serially and in parallel, then run a sustained
benchmark and cold reload. Hardware occupancy, a universal speedup and vehicle
readiness are not established by the host tests or provider-call concurrency.
