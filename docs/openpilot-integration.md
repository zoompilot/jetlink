# Integrating with openpilot

Design rule: **minimise the diff against upstream.** Everything jetlink-specific
is a new module, and the patch to openpilot's own files is a few dozen lines,
small enough to survive an upstream sync and to lift into another fork.

jetlink is not wired into modeld directly. It is one backend behind an
accelerator layer, so core openpilot never mentions it by name.

## The accelerator layer

openpilot already had an accelerator abstraction with one implementation and no
name: `ChestnutState` is pure AMD telemetry yet publishes the generic
`chestnutState` message, and `chestnut_present()` was being asked to mean "some
accelerator is attached". Naming the axis is what keeps this backend out of
upstream's files.

`sources/` and `runners/` were already taken in the fork (`source` = which model
catalog, `runner` = snpe/tinygrad/stock), so the layer is `accelerators/`.

```
openpilot/sunnypilot/accelerators/
  base.py       the Accelerator protocol, and Daemon
  __init__.py   discovery, per-question guards, the progress param
  chestnut.py   comma's AMD board (ChestnutState moved here, unchanged)
  jetlink/      this project's backend
  setup.sh      runs each backend's setup.sh at boot
```

A backend answers seven questions: `present()`, `ready()`,
`unavailable_reason()`, `prepare()`, `make_model_state()`,
`make_health_publisher()`, `daemon()`. Backends are imported on the first
`backends()` call; `ImportError` is skipped silently (a fork that does not ship
one), anything else is logged. Every call goes through `_ask()`, so a broken
backend cannot take down hardwared or the UI.

`Daemon` is a description, not a `PythonProcess`, so a backend never imports
manager, which imports backends.

## The patch

`selfdrive/modeld/modeld.py` — an import and three edits, and it comes out **75
lines shorter** because `ChestnutState` moved to `accelerators/chestnut.py`:

```python
from openpilot.sunnypilot import accelerators

# Whatever runs the large model here: comma's chestnut board, an attached
# Jetson, nothing. See sunnypilot/accelerators/.
accel = accelerators.active()
CHESTNUT = accel is not None
if CHESTNUT:
  accel.prepare()

m = accel.make_model_state(vipc_client_main.width, vipc_client_main.height)

chestnut_state = accel.make_health_publisher(pm, model) if CHESTNUT else None
```

`common/hardware/usb.py` — `deviceState.chestnutPresent` is true for any
accelerator, not just a board with comma's USB ids. That field, not
`modeld.helpers.chestnut_present()`, is what the model manager, the UI and
selfdrived gate on, so this one line is what makes the large-model bundles
appear. The import is function-local because `modeld.helpers` imports this
module and a top-level import would close the cycle.

`selfdrive/modeld/helpers.py` — `chestnut_present()` stays strictly "is chestnut
hardware attached". Do not widen it: SConscript and modeld_v2 use it to decide
whether to build and load a tinygrad pkl against the AMD GPU, which another
accelerator does not have.

`system/manager/process_config.py` — one line, and no mention of jetlink:

```python
*[PythonProcess(d.name, d.module, and_(only_offroad, d.should_run)) for d in accelerators.daemons()],
```

`system/hardware/hardwared.py` — surfaces `accelerators.unavailable_reason()` as
`Offroad_AcceleratorUnavailable`, so a kernel without the gadget drivers is not
silently absent.

`launch_chffrplus.sh` — 31 lines of gadget setup became two:

```sh
# accelerator backends: USB gadgets, symlinks, anything needing root at boot
./openpilot/sunnypilot/accelerators/setup.sh
```

which iterates `*/setup.sh` and never fails the launch.

The UI (`ui_state.py`, `model_info.py`, the mici layouts) is backend-agnostic and
contains zero jetlink references.

## The modules (in the fork, `sunnypilot/accelerators/jetlink/`)

| | |
|---|---|
| `backend.py` | the only class core openpilot touches |
| `helpers.py` | where the model is, whether a Jetson is attached, gadget state |
| `model_state.py` | a `ModelState` whose policy is a link round trip |
| `state.py` | publishes the Jetson's health as `chestnutState` |
| `spec_cache.py` | the parsed model spec, cached in a param |
| `lfs.py` | fetching a model out of comma's git-lfs history |
| `models.json` | the large models known to run here |
| `jetlinkd.py` | offroad: fetches the model, builds the engine, caches it |
| `warp_cache.py` | the comma-side warp JIT: capture, load, warm |
| `compile_warp.py` | its CLI, run by `accelerators/SConscript` at build time |
| `setup.sh` | boot-time gadget setup, called by `accelerators/setup.sh` |

Hold the seam at `backend.py`. Anything only `jetlinkd` needs belongs in
`helpers` or `spec_cache`, never in the backend.

## Why provisioning is a separate daemon

Uploading 766 MB and building a TensorRT engine takes ~3 minutes. modeld's
`BIG_MODEL_TIMEOUT` is 60 s, and upstream's fallback to the small model is
one-way for the rest of the drive. So jetlinkd does the slow work **offroad**,
caches the plan on the Jetson, and records readiness in a param. Onroad, modeld
finds an engine already built and only loads it (~1 s).

It also keeps exactly one process on the link at a time: jetlinkd offroad,
modeld onroad.

### The parked car

Holding the gadget is the daemon's first job, but not for the whole park. A
Jetson on an always-on supply sleeps when it has had no gadget for 120 s
(`jetlink/server/sleep.py`) and wakes on the next USB edge, so once the
engine is confirmed ready and `DORMANT_HOLD` (60 s from ignition-off, which
is when manager starts the daemon) has passed, jetlinkd releases the gadget
on purpose. That is the disconnect that lets the Jetson sleep, about three
minutes after the car is parked. It writes `/dev/shm/jetlink-dormant` first
so `present()` keeps answering true and the offroad alert stays quiet: the
Jetson is there, only unreachable until something presents the gadget again.

It presents it again when there is work the link can do: the readiness
param cleared (modeld found the engine gone), the selected model changed, or
hardwared asking for the Jetson to be powered off. At ignition, modeld's own
bind is the wake; measured on the bench, a server answers about 8 s after
the bind, inside modeld's 45 s connect timeout.

### Taking the Jetson down with the comma

The comma has a battery policy and the Jetson does not: hardwared shuts the
device down below 11.8 V or after 30 hours parked, and a Jetson asleep on an
always-on feed keeps drawing. So hardwared calls `accelerators.shutdown()`
just before it sets `DoShutdown`. The protocol method is optional, comma's
board dies with the device and has none. jetlink's backend cannot touch the
link from hardwared, jetlinkd owns it, so it leaves a request in
`/dev/shm/jetlink-shutdown` and waits up to 25 s. jetlinkd presents the
gadget if it was dormant (which wakes the Jetson), sends `SHUTDOWN_REQ`, and
removes the request; the server drops a flag file on its cache volume and a
host-side path unit runs the poweroff (`scripts/jetlink-poweroff.*`).

Off is off. On an always-on feed nothing but the power button or a DC cycle
brings it back, so pair this with a low-voltage disconnect that reconnects
when the alternator is running, or with ignition wired to the button header.

## Reusing chestnut's surfaces

Nothing in cereal, the UI or the alerts changed. modeld sets
`ChestnutLoading`/`ChestnutActive` for whichever accelerator is active, except
that a model state which is still bringing its accelerator up (`loading` is
true on the object `make_model_state` returned) owns both for the drive: for a
Jetson the load is never over, because it can join, leave and join again.
`ChestnutLoading` is true while the small model proxies and false while the
large one runs, so selfdrived's "Big Model Ready" is the swap and nothing
else; `ChestnutActive` is absent while proxying, true at the swap and false at
a demote, which gets a chestnut's soft disable so the driver hears that the
plan changed under them.

The swap lands on a disengaged frame or on a standstill. Disengaged alone was
not enough: a driver who engages at the ramp and lifts off at their exit gives
the join nowhere to land, and "the Jetson was ready the whole time and never got
used" is what that shape of drive produced. At a standstill the plan is not
turning a wheel or asking for acceleration, so the step between two models that
disagree by ~195 m of planned path lands on nothing. A drive that is neither -
engaged from the driveway to the destination without ever stopping - still runs
small, on purpose. selfdrived only makes "Big Model Loading" a NO_ENTRY
while nothing publishes `modelV2`, so a Jetson that takes a whole drive to
arrive never keeps the driver off the small model. That used to be a 60 s
`LOADING_TIMEOUT` in the joining state, and the edge read as ready to
selfdrived and as "unavailable" to the UI while the join was still trying.
This backend publishes `chestnutState`, with Tegra sysfs mapped onto chestnut's fields (`tempC` ← tj-thermal, `powerDrawW` ←
INA3221 VDD_IN, `pcieLtssm` ← `0x78` when the link is up, so existing "link
down" logic keeps working). The Jetson reports neutral field names; the openpilot
schema is known only to `state.py`, and the sysfs side only to
`jetlink/server/telemetry.py`.

Renaming the cereal fields was considered and rejected: it breaks log
compatibility with every existing tool. On the wire, "chestnut" means "whatever
accelerator is active".

Health arrives piggybacked on the previous inference response, so publishing it
costs no round trip: at 20 Hz there is no gap in which to run a separate
request without racing a frame.

## Failure behaviour

Every link error raises. modeld already wraps the model call in `try/except`,
sets `ChestnutActive=False` and swaps to the already-warmed small model, so a
dead link inherits that path for free. There is no per-frame deadline: a frame
blocks the way a chestnut frame does, a long one is a dropped camera frame that
modeld counts, and only a stall past `client.FRAME_TIMEOUT` (3 s, chestnut's
HCQ wait) is a failure. On the comma that timeout is only enforceable because
the FunctionFS read runs on a thread; the kernel ignores `O_NONBLOCK` once the
host has enabled the endpoint.

`make_health_publisher()` takes `getattr(model, 'client', None)`, because when
the large model failed to load there is no client to report on. The publisher
then sends an invalid message, which is what chestnut does in the same state.

The server additionally checks the output is finite and reports `NOT_FINITE`
rather than returning it, matching openpilot's own guard on big-model output.

Upstream's own fallback is one-way: modeld sets `model = small_model` in the
frame loop's `except` and stays there for the drive. jetlink does change that.
`JoiningModelState.run` catches the failure before modeld sees it, demotes to
the small model, and rejoins after a backoff, swapping back at the next open
window - so a Jetson that reboots, a nudged cable or a transport desync costs
some frames rather than the rest of the drive. modeld's one-way path is still
underneath, for anything the joining state does not catch.

The backoff doubles from `REJOIN_DELAY` to `REJOIN_DELAY_MAX` per consecutive
failure and resets after a join that held for `STABLE_SECONDS`, because a link
that dies on its first frame every time is the expensive shape: each cycle is a
swap frame, a demote frame, a soft disable and a "Big Model Ready" chime, and
at a flat delay that repeats for the drive. A Jetson that reboots once is still
picked up within a minute.

A link that is ready has to wait for the frame loop to find a window, and on a
drive with no stop and no disengagement that is the whole drive. `_keep_alive`
pings it every `KEEPALIVE_PERIOD` while it waits, so a Jetson that reboots in
that window is noticed there rather than at the swap, where finding out costs a
build on a dead link, a demote and the backoff, all on modeld's thread.

What that does not cover is `chestnutPresent`: selfdrived soft-disables on it
dropping while the big model is active, so the driver is still disengaged once
per dropout even though the model layer recovers on its own. The alert that
comes with it no longer says "restart the car to retry", which was true for a
board bolted to the comma and false for this: the rejoin is already running.

The UI asks "loading" before it asks "present", which it did not used to. A
Jetson rebooting mid-drive is not attached for a minute, and reading that as
DISCONNECTED - rendered "unavailable" - while the join loop is actively getting
it back is what sends a driver looking for a way to force a reconnect. There is
nothing to force.

## Getting the package onto the comma

`jetlink` should be a submodule at the openpilot repo root, alongside
`tinygrad_repo` and `opendbc_repo`. That is how the fork already carries its
other Python dependencies, and it puts `jetlink` on the interpreter path with
no install step, no writes to a read-only rootfs, and no fight with the updater.

```bash
git submodule add <url> jetlink
```

Until that repo exists, `scripts/deploy_to_comma.sh <user@host>` rsyncs it to
`/data/jetlink_repo` and configures the gadget. Set `DisableUpdates=1` while
testing that way: the updater does `fetch` + `reset --hard` + `clean` and will
delete untracked files.

`accelerators/jetlink/setup.sh` looks for `$BASEDIR/jetlink_repo` first and
falls back to `/data/jetlink_repo`, symlinks it onto the interpreter path, and
runs `sudo -n bash $REPO/scripts/setup_gadget.sh`. Once the submodule exists the
fallback becomes dead code and should go.

## Params

| param | kind | meaning |
|---|---|---|
| `AcceleratorProgress` | CLEAR_ON_MANAGER_START, JSON | `{stage, frac, msg}` while provisioning; generic, not jetlink-specific |
| `Offroad_AcceleratorUnavailable` | CLEAR_ON_MANAGER_START, JSON | the offroad alert |
| `JetlinkEnabled` | PERSISTENT, BOOL | user toggle; absent means "auto: on if a link is present" |
| `JetlinkEndpoint` | PERSISTENT, STRING | `host:port` to use TCP instead of USB |
| `JetlinkModel` | PERSISTENT, STRING | which entry in `models.json` to run |
| `JetlinkEngineReady` | PERSISTENT, STRING | sha256 of the model the Jetson has an engine for |
| `JetlinkSpec` | PERSISTENT, JSON | cached model spec, so modeld never re-reads a 766 MB ONNX |

`JetlinkEngineReady` and `JetlinkSpec` are deliberately not cleared on manager
start: readiness has to survive a reboot, or every ignition cycle rebuilds a
three minute engine.

New keys are compiled into `libparams_c` from `params_keys.h`, so adding one
needs a scons rebuild. `helpers._get()` swallows `UnknownKeyName`, because
several of these are read from hardwared and the UI's param thread and a raise
there takes down a process that has nothing to do with jetlink.

## Safety framing

The comma stays the sole authority over the car: cameras, calibration, warp,
`controlsd`, panda, CAN. The Jetson is a pure function — warped frames and
context in, 18452 floats out. It holds no control state and never touches CAN.
Keep it that way and the second box stays out of the safety argument.
