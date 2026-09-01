# Integrating with openpilot

Design rule: **minimise the diff against upstream.** Everything jetlink-specific
is a new module; the patch to openpilot's own files is 37 lines across three,
small enough to survive an upstream sync and to lift into another fork.

## The patch

`selfdrive/modeld/modeld.py` — an import and three edits:

```python
from openpilot.sunnypilot.jetlink import hook as jetlink

JETLINK = jetlink.available()
CHESTNUT = JETLINK or (chestnut_present() and chestnut_compiled())

m = (jetlink.make_model_state(w, h) if JETLINK else ModelState(w, h, True))

chestnut_state = (jetlink.make_state_publisher(pm, model)
                  if getattr(model, 'is_jetlink', False)
                  else ChestnutState(pm, model.chestnut))
```

`selfdrive/modeld/helpers.py` — `chestnut_present()` also returns True for an
attached Jetson. That single line is what makes the model manager offer the
large-model bundles, so downloading and selecting one behaves exactly as it does
with a real chestnut.

`system/manager/process_config.py` — one line adding `jetlinkd` (offroad only).

## The new modules (in the fork, `sunnypilot/jetlink/`)

| | |
|---|---|
| `helpers.py` | where the model is, whether the link is up, progress params |
| `model_state.py` | a `ModelState` whose policy is a link round trip |
| `state.py` | publishes the Jetson's health as `chestnutState` |
| `hook.py` | the three functions modeld calls |
| `jetlinkd.py` | offroad: uploads the model, builds the engine, caches it |

## Why provisioning is a separate daemon

Uploading 766 MB and building a TensorRT engine takes ~3 minutes. modeld's
`BIG_MODEL_TIMEOUT` is 60 s, and upstream's fallback to the small model is
one-way for the rest of the drive. So jetlinkd does the slow work **offroad**,
caches the plan on the Jetson, and records readiness in a param. Onroad, modeld
finds an engine already built and only loads it (~1 s).

It also keeps exactly one process on the link at a time: jetlinkd offroad,
modeld onroad.

## Reusing chestnut's surfaces

Nothing in cereal, the UI or the alerts changed. The Jetson sets
`ChestnutActive`/`ChestnutLoading` and publishes `chestnutState`, with Tegra
sysfs mapped onto chestnut's fields (`tempC` ← tj-thermal, `powerDrawW` ←
INA3221 VDD_IN, `pcieLtssm` ← `0x78` when the link is up, so existing "link
down" logic keeps working). The mapping is in `jetlink/server/telemetry.py`.

Health arrives piggybacked on the previous inference response, so publishing it
costs no round trip: at 20 Hz there is no gap in which to run a separate
request without racing a frame.

## Failure behaviour

Every link error raises. modeld already wraps the model call in `try/except`,
sets `ChestnutActive=False` and swaps to the already-warmed small model, so a
dead link inherits that path for free. The per-frame deadline is 35 ms — a stall
is worse than the small model.

The server additionally checks the output is finite and reports `NOT_FINITE`
rather than returning it, matching openpilot's own guard on big-model output.

Note upstream's fallback is one-way: once it drops to the small model it stays
there for the drive. jetlink does not change that.

## Params

| param | meaning |
|---|---|
| `JetlinkEnabled` | user toggle; absent means "auto: on if a link is present" |
| `JetlinkEndpoint` | `host:port` to use TCP instead of USB |
| `JetlinkEngineReady` | sha256 of the model the Jetson has an engine for |
| `JetlinkSpec` | cached model spec, so modeld never re-reads a 766 MB ONNX |
| `JetlinkProgress` | `{stage, frac, msg}` while provisioning |

## Safety framing

The comma stays the sole authority over the car: cameras, calibration, warp,
`controlsd`, panda, CAN. The Jetson is a pure function — warped frames and
context in, 18452 floats out. It holds no control state and never touches CAN.
Keep it that way and the second box stays out of the safety argument.
