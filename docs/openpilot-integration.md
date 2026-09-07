# Integrating with openpilot

Design rule: **minimise the diff against upstream, and never touch a chestnut
code path.** Everything jetlink-specific is a new package under
`openpilot/sunnypilot/`; the patch to openpilot's own files is 99 insertions
and 3 deletions across 12 files, small enough to survive an upstream sync and
to lift into another fork.

The shape a maintainer should be able to check in one pass: chestnut untouched,
jetlink added.

## The accelerator module

`openpilot/sunnypilot/accelerators/` is a module of functions with exactly one
implementation behind them. It is not a registry: there is no `Accelerator`
protocol, no backend discovery, no `active()` and no `catalog()`.

```
openpilot/sunnypilot/accelerators/
  __init__.py   the functions core openpilot calls, plus Daemon and the progress param
  setup.sh      runs each backend's setup.sh at boot
  SConscript    builds the comma-side warp
  jetlink/      this project's backend
```

comma's chestnut board is not one of these. `ChestnutState` stays in
`selfdrive/modeld/modeld.py`, the PCIe-link wait stays in `modeld.main()`, and
hardwared, the UI and the model manager keep their own chestnut code at
upstream's lines. They ask this package only when no board is fitted.
Selection is one expression, and there is nothing between its two arms:

```python
if chestnut_present():                                 # native, upstream's own block
elif accelerators.ready() and accelerators.prepare():  # jetlink
```

A registry with two backends answered the same question two ways: `active()`
meant "the first ready one", `catalog()` meant "the first present one". A
chestnut fitted but not compiled, next to a provisioned Jetson, gave a chestnut
model catalog and a jetlink modeld, and every consumer needed a compensating
check for it. Making chestnut native removes the state rather than handling it.

`sources/` and `runners/` were already taken in the fork (`source` = which model
catalog, `runner` = snpe/tinygrad/stock), so the axis is `accelerators/`.

### What the module exposes

```
present()                are we attached, or dormant and known to be there. USB-independent.
ready()                  params only: enabled, no gadget error, spec == selected == built
unavailable_reason()     the offroad alert text, None unless the user opted in
prepare()                modeld only: tinygrad device init, and a last veto
make_model_state()       the joining state, small model now and Jetson later
make_status_publisher()  modeld's after_enqueue hook
uses_stock_runner()      run stock modeld whatever bundle is stored
model_choices() / select_model() / active_model_name()
daemons()                jetlinkd, offroad, gated on enabled()
shutdown(reason, timeout)  bounded in the module, not trusted to the backend
progress() / report_progress() / clear_progress()
```

`present()`, `ready()`, `progress()` and `uses_stock_runner()` are polled by the
UI at 5 Hz and cost a param read each. Anything that can block belongs in
`prepare()`, which only modeld calls.

The `jetlink` client package can be absent (a fork that does not ship it, a
submodule never fetched). Every function that needs it imports it when called
and answers its negative default when it is not there, logging once. That is
what the registry's swallowed `ImportError` used to do, without the registry.

`Daemon` is a description, not a `PythonProcess`, so this package never imports
manager, which imports it. manager owns the onroad/offroad gating.

## What the integration guarantees

Behaviour that has to survive any refactor. Each row names the code that
implements it, in `openpilot/sunnypilot/accelerators/jetlink/`.

| Situation | Behaviour |
|---|---|
| Provisioned Jetson boots after the comma | `make_model_state` returns at once with the small model driving; `_join_loop` connects in the background (`backend.py`, `joining.py`) |
| Promotion to the large model | only with fresh (0.25 s) valid `selfdriveState`, `carState` and `carControl`, and `enabled`, `latActive`, `longActive` all false. Standstill is tracked and is not a condition (`joining.py`) |
| First big frame | readiness and progress are cleared only after the first successful inference; availability is a separate signal (`joining.py`) |
| Failure after promotion | demote to small, reset small's queues in place through the captured JIT, re-run the same frame on small, back off `min(5 * 2**(n-1), 60)` s with the counter reset by a 60 s stable join (`fallback.py`, `joining.py`) |
| Runtime deadline | `INFERENCE_TIMEOUT = 0.5` s per frame once joined; the client's 3 s `FRAME_TIMEOUT` is for `hello` and `ensure_engine` only (`backend.py`) |
| Ready link with no window to land in | keepalive ping every 10 s; a failed ping schedules a rejoin rather than discovering it at the swap (`joining.py`) |
| Realtime inheritance | `warp_cache.init_device()` runs before `config_realtime_process`; every jetlink thread drops realtime first; the FunctionFS reader re-pins to FIFO 51 off the frame loop's core (`backend.py`, `joining.py`, `jetlink/transport/ffs.py`) |
| Endpoint ownership | jetlinkd offroad, modeld onroad, manager's `only_offroad` gate; retired links are closed on the join loop so an unbind never blocks a frame (`joining.py`) |
| Provisioning | registry identity, no hashing on a parked car, `JetlinkEngineReady`/`JetlinkSpec` survive a reboot and are re-asked once per attach (`jetlinkd.py`) |
| Dormancy and wake | `DORMANT_HOLD = 60` s from process start, `/dev/shm/jetlink-dormant` written before the link closes so presence never blinks, CC pin for presence while dormant (`jetlinkd.py`, `helpers.py`) |
| Shutdown | hardwared asks before `DoShutdown`; 20 s wake plus 5 s request, 25 s bound (`backend.py`, `jetlinkd.py`) |
| Warp | a scons target, presence-only cache, `prepare()` refuses without it (`accelerators/SConscript`, `warp_cache.py`) |

## The patch

### `selfdrive/modeld/modeld.py`, five sites

Every one of them outside the `if CHESTNUT:` block, which is byte-identical to
upstream along with `ChestnutState`, the poller wait, `HCQDEV_WAIT_TIMEOUT_MS`
and the frame loop's `except`.

```python
from openpilot.sunnypilot import accelerators

# 1. select, before config_realtime_process
JETLINK = not CHESTNUT and accelerators.ready() and accelerators.prepare()

# 2. load, beside the chestnut block
elif JETLINK:
  small_model = ModelState(vipc_client_main.width, vipc_client_main.height, False)
  model = accelerators.make_model_state(vipc_client_main.width, vipc_client_main.height, small_model)

# 3. the status hook, same signature as ChestnutState.send
if JETLINK:
  chestnut_state = accelerators.make_status_publisher(pm, model)

# 4. the frame loop's except: the joining state owns its own demotion
if JETLINK:
  raise

# 5. the additive SP fields
mdv2sp_send.modelDataV2SP.bigModelAvailable = getattr(model, 'big_model_available', False)
mdv2sp_send.modelDataV2SP.acceleratorState = getattr(model, 'big_model_state', 'none')
mdv2sp_send.modelDataV2SP.acceleratorName = 'jetlink' if JETLINK else ''
```

Site 1 has to be where it is: `prepare()` calls `warp_cache.init_device()`, and
a thread created after `config_realtime_process(7, 54)` inherits SCHED_FIFO 54
and the core-7 pin. tinygrad brings the GPU up on the first kernel run and that
init spawns a libusb event thread, which is exactly such a thread.

Site 2 has no loader thread and no `BIG_MODEL_TIMEOUT`. A Jetson boots on the
car's ignition rail, so its boot *starts* when the comma goes onroad and takes
45 to 100 s; a 60 s one-shot load is a race it can only lose. The joining state
returns immediately with the small model in the driving seat and swaps later.
That also means the warp is loaded and warmed on modeld's own thread while it
is loading models, before the frame loop exists, so it costs no dropped frame.

Site 4 matters more than it looks. Without it, a small-model fault while the
Jetson is running would fall into upstream's handler, which writes
`ChestnutActive=False` and swaps `model = small_model`, orphaning the joining
state's threads and its open link.

`modelV2.big = model.chestnut` is untouched; `JoiningModelState.chestnut`
proxies whichever model is active.

### `selfdrive/selfdrived/selfdrived.py`, an import and one call

Upstream's big-model block stays verbatim, including the 5 s settling window
and `bigModelFailed` with its "Restart the car to retry" text, which never
fires for jetlink because jetlink writes no `ChestnutActive`. One call goes in
after it:

```python
self.accelerator_events.update(self.sm, self.enabled, self.events, self.events_sp)
```

`sunnypilot/selfdrive/selfdrived/accelerator_events.py` reads messages only,
never a param:

- **`bigModelAvailable`**, once, on `modelDataV2SP.bigModelAvailable` rising
  while `modelV2.big` is false. Informational, it does not disengage. A
  keepalive ping does not rearm it, and a missing or invalid model message can
  neither announce nor rearm it.
- **`bigModelLoading`** NO_ENTRY only while `acceleratorState == joining` *and*
  `modelV2` is not alive. A join that lands onto a model the small side is
  already publishing never keeps the driver out, which is what a whole-drive
  join would otherwise do.
- **`bigModelLinkLost`** SOFT_DISABLE plus a permanent alert on `modelV2.big`
  falling while engaged. The plan just went from ~200 m to ~5 m under the
  driver. Disengaged, the small model simply carries on and nothing is said.
  The edge is tracked only while `acceleratorState != none`, so a chestnut fall
  raises the native `bigModelFailed` and never both.

The "Big Model Ready" chime is backend-neutral: it fires on `modelV2.big`
rising while `modelV2` is alive and valid, which is the only signal that a big
frame was published. Before that fix it fired on a `ChestnutLoading` falling
edge, which modeld also produces after a *failed* load, so a failure chimed
"Big Model Failed" and then "Big Model Ready" with the small model driving.

### `selfdrive/ui/ui_state.py`, seven lines

Upstream's `_update_chestnut_state` and its latched `chestnut_compiled()` are
untouched. Two additions:

```python
def _update_chestnut_state(self) -> None:
  if self.accelerator_view is not None:
    self.chestnut_state = self._accelerator_state()
    return
  ...

# the comma is the gadget for an off-board accelerator and enumerates nothing
self.usb_unknown = not (self.accelerator_view is not None or
                        any(is_chestnut_usb_id(...) for d in get_usb_state()))
```

The view is built in `sunnypilot/ui_state.py` on the same 5 Hz params pass as
the chestnut params, from `present()`, `ready()`, `progress()`,
`uses_stock_runner()` and `modelDataV2SP.acceleratorState`, and only when
`deviceState.chestnutPresent` is false. A chestnut user never reaches a line of
ours.

Its rules: offroad, the progress stage decides, so a provisioning accelerator
is LOADING and only a `failed` stage is FAILED. Onroad, a live `modelV2.big` is
ACTIVE first, then `not present` is DISCONNECTED, then `joining` or `retrying`
is LOADING. Asking "loading" before "present" is deliberate: a Jetson rebooting
mid-drive is not attached for a minute, and rendering that as "unavailable"
while the join loop is actively getting it back sends a driver looking for a
reconnect button that should not exist.

`usb_unknown` needs the guard because upstream's `ui: show usb connection`
(#38745) looks for a chestnut USB id among the devices the comma enumerated as
a host, and the comma is our gadget. It enumerates nothing. An accelerator
recognised after the 10 s grace period also clears "unknown", next to where the
view is built in `sunnypilot/ui_state.py`: the Jetson's port has VBUS up from
power-on but does not configure the gadget until its kernel is up, well after
the one-shot decision was made.

### `system/hardware/hardwared.py`, two calls

```python
accelerator_error = accelerators.unavailable_reason()
set_offroad_alert_if_changed("Offroad_AcceleratorUnavailable", accelerator_error is not None,
                             extra_text=accelerator_error)
...
accelerators.shutdown(f"comma shutting down, offroad since {off_ts}", timeout=25.0)
```

Upstream's `chestnut_valid` line keeps its own meaning; there is no "is this
really a chestnut" filter any more, because `chestnutState` is comma's board
again. `shutdown()` enforces its bound in the module, on a thread with a join,
because `deviceState` is not published while it runs. With the link disabled it
is one param read.

### `system/manager/process_config.py`, one line

```python
*[PythonProcess(d.name, d.module, and_(only_offroad, d.should_run)) for d in accelerators.daemons()],
```

No mention of jetlink.

### `common/hardware/usb.py` and `selfdrive/modeld/helpers.py`: unchanged

`deviceState.chestnutPresent` used to be widened to mean "any accelerator". It
is not any more. `chestnut_present()` stays strictly "is chestnut hardware
attached": SConscript and modeld_v2 use it to decide whether to build and load
a tinygrad pkl against the AMD GPU, which a Jetson does not have.

### `launch_chffrplus.sh`, three lines

```sh
ln -sfn jetlink_repo/jetlink jetlink
...
./openpilot/sunnypilot/accelerators/setup.sh
```

`setup.sh` iterates `*/setup.sh` and never fails the launch: a backend that
cannot come up says so through `unavailable_reason()`, which reaches the user
as an offroad alert.

## Opting in

`JetlinkEnabled == True` is the only enable. There is no "absent means auto".

The auto rule was self-fulfilling. On AGNOS with the package installed,
`setup_gadget.sh` created `/dev/ffs-jetlink/ep0` at boot, so `link_configured()`
was true, so `enabled()` was true, so `uses_stock_runner()` routed manager away
from whatever small-model bundle the user had chosen. Installing the package
enabled the feature and changed which modeld ran, with nobody asking.

So `accelerators/jetlink/setup.sh` reads the param through
`openpilot.common.params` before it touches the gadget and exits 0 unless it is
true: no gadget, no sysctls, no override. A params library not built yet reads
as off.

`uses_stock_runner()` is `JetlinkEnabled is True and JetlinkModel is set`.
Configuration, deliberately not `ready()` and not `link_configured()`: a Jetson
that boots late must not change which modeld manager runs in the middle of a
drive, but a device that is merely capable must not take the override either.

Under the override manager runs stock modeld, which loads the default small
model and never reads the stored qcom bundle. `models.helpers.effective_small_bundle()`
returns None in that case, and both models panels use it, so the UI names the
model that is actually running rather than the one that is stored.

## The VM sysctls

The gadget read shares the kernel with every writer on the device, and a
reclaim storm under recording lands straight on the FunctionFS `readv`. Three
system-wide values fix it:

```
vm.dirty_bytes             16 MB
vm.dirty_background_bytes   8 MB
vm.min_free_kbytes        128 MB
```

They belong to `jetlinkd`, not to boot. It applies them when the link is
enabled and puts them back on the way out, including the "disabled, releasing
the link" branch. The stock values are read once into
`/dev/shm/jetlink-sysctl-prev` before the first change and never overwritten,
so a second run cannot record our own values as stock. A SIGKILL skips the
restore and leaves them until reboot; the record survives in tmpfs, so the next
run still knows what to put back.

## The modules (`sunnypilot/accelerators/jetlink/`)

| | |
|---|---|
| `backend.py` | everything the module API calls, and nothing else |
| `helpers.py` | where the model is, whether a Jetson is attached, gadget state |
| `joining.py` | the small model now, the Jetson swapped in later |
| `model_state.py` | a `ModelState` whose policy is a link round trip |
| `fallback.py` | resetting the small model's queues in place at a demote |
| `jetlinkd.py` | offroad: fetches the model, builds the engine, owns the sysctls |
| `status.py` | modeld's per-frame status hook |
| `spec_cache.py` | the parsed model spec, cached in a param |
| `warp_cache.py` | the comma-side warp JIT: capture, load, warm, device init |
| `compile_warp.py` | its CLI, run by `accelerators/SConscript` at build time |
| `lfs.py` | fetching a model out of comma's git-lfs history |
| `models.json` | the large models known to run here |
| `setup.sh` | boot-time gadget setup, gated on the param |

Hold the seam at `backend.py`. Anything only `jetlinkd` needs belongs in
`helpers` or `spec_cache`, never in the backend.

## Why provisioning is a separate daemon

Uploading 766 MB and building a TensorRT engine takes ~3 minutes. modeld's own
big-model load is a 60 s one-shot, and upstream's fallback to the small model
is one-way for the rest of the drive. So jetlinkd does the slow work
**offroad**, caches the plan on the Jetson, and records readiness in a param.
Onroad, modeld finds an engine already built and only loads it.

It also keeps exactly one process on the link at a time: jetlinkd offroad,
modeld onroad, enforced by manager's `only_offroad` gate.

### The parked car

Holding the gadget is the daemon's first job, but not for the whole park. A
Jetson on an always-on supply sleeps when it has had no gadget for 120 s
(`jetlink/server/sleep.py`) and wakes on the next USB edge, so once the engine
is confirmed ready and `DORMANT_HOLD` (60 s from process start, which is when
manager starts the daemon) has passed, jetlinkd releases the gadget on purpose.
That is the disconnect that lets the Jetson sleep, about three minutes after
the car is parked. It writes `/dev/shm/jetlink-dormant` first so `present()`
keeps answering true and the offroad alert stays quiet: the Jetson is there,
only unreachable until something presents the gadget again.

It presents it again when there is work the link can do: the readiness param
cleared (modeld found the engine gone), the selected model changed, or
hardwared asking for the Jetson to be powered off. At ignition, modeld's own
bind is the wake; measured on the bench, a server answers about 8 s after the
bind.

### Taking the Jetson down with the comma

The comma has a battery policy and the Jetson does not: hardwared shuts the
device down below 11.8 V or after 30 hours parked, and a Jetson asleep on an
always-on feed keeps drawing. So hardwared calls `accelerators.shutdown()` just
before it sets `DoShutdown`. The backend cannot touch the link from hardwared,
because jetlinkd owns it offroad and a Jetson that is asleep has to be woken by
presenting the gadget, so it leaves a request in `/dev/shm/jetlink-shutdown`
and waits. jetlinkd presents the gadget if it was dormant (which wakes the
Jetson), sends `SHUTDOWN_REQ`, and removes the request; the server drops a flag
file on its cache volume and a host-side path unit runs the poweroff
(`scripts/jetlink-poweroff.*`). The 25 s bound is enforced in
`accelerators/__init__.py`, on a thread with a join, so no backend can hold
hardwared longer than it says.

Off is off. On an always-on feed nothing but the power button or a DC cycle
brings it back, so pair this with a low-voltage disconnect that reconnects when
the alternator is running, or with ignition wired to the button header.

## Runtime state on the wire

`deviceState.chestnutPresent` and `chestnutState` mean comma's board and
nothing else. jetlink publishes neither. It used to publish `chestnutState`
with Tegra sysfs mapped onto comma's fields and a synthetic `pcieLtssm = 0x78`
so that existing "link down" logic kept working, and that cost a compensating
"is it really a chestnut" check everywhere the field was read.

What modeld publishes instead, all additive `ModelDataV2SP` fields that default
to zero for older logs, for chestnut and for every other startup-only runner:

```
bigModelAvailable @3 :Bool          connected and waiting for disengagement
acceleratorState  @4 :AcceleratorState   none | joining | running | retrying | unavailable
acceleratorName   @5 :Text          "jetlink"
```

plus two `OnroadEventSP` variants, `bigModelAvailable` and `bigModelLinkLost`.
`validate_sp_cereal_upstream.py` checks union discriminants, so additive SP
fields are safe; a new top-level `Event` variant would not be ours to allocate.

Offroad progress stays in the `AcceleratorProgress` param, because its writer
is a daemon in another process.

Telemetry (temperature, power, GPU load, clock, supply) has no message yet: it
needs a `customReserved` slot that only the maintainers can assign. Until then
`status.py` logs it to swaglog as a `jetlinkTelemetry` event at 1 Hz. The hook
still has to exist even though it publishes nothing, because passing a callback
is what makes the client ask for telemetry: it is piggybacked on the previous
inference response (`want_state`), so it costs no round trip and cannot delay a
frame.

The joining state writes no params at all. `ChestnutLoading` and
`ChestnutActive` describe a load that happens once and is then over; a Jetson's
never is, and a link that joined, left and rejoined was writing an alert cycle
each time through keys upstream's own code also reads.

## Failure behaviour

Every link error raises. `JoiningModelState.run` catches it before modeld sees
it, demotes to the small model, resets the small model's queues in place
through the captured JIT, re-runs the same frame on it, and rejoins after a
backoff, swapping back at the next open window. So a Jetson that reboots, a
nudged cable or a transport desync costs some frames rather than the rest of
the drive. Upstream's one-way fallback is still underneath for anything the
joining state does not catch, which is why `if JETLINK: raise` in modeld's
handler is a small-model fault and nothing else.

The per-frame deadline once joined is `backend.INFERENCE_TIMEOUT`, 0.5 s, ten
frame periods. A frame is not deadlined for being slow: `infer_end` blocks the
way modeld blocks on a chestnut, and a long frame is a dropped camera frame
that modeld counts and tolerates. Only a stall that long is a failure. The
client's 3 s `FRAME_TIMEOUT` covers `hello` and `ensure_engine` on the join
thread, where loading a plan out of the cache is 13 to 25 s of legitimate work.
On the comma any of this is only enforceable because the FunctionFS read runs
on a thread: the kernel ignores `O_NONBLOCK` once the host has enabled the
endpoint.

The backoff doubles from `REJOIN_DELAY` to `REJOIN_DELAY_MAX` per consecutive
failure and resets after a join that held for `STABLE_SECONDS`, because a link
that dies on its first frame every time is the expensive shape: each cycle is a
swap frame, a demote frame, a soft disable and a chime, and at a flat delay it
repeats for the drive. A Jetson that reboots once is still picked up within a
minute.

A link that is ready has to wait for the frame loop to find a window, and on a
drive with no stop and no disengagement that is the whole drive. `_keep_alive`
pings it every `KEEPALIVE_PERIOD` while it waits, so a Jetson that reboots in
that window is noticed there rather than at the swap, where finding out costs a
build on a dead link, a demote and the backoff, all on modeld's thread.

The swap requires fresh, fully disengaged controls, including MADS lateral and
longitudinal activity. Standstill alone does not permit it: longitudinal
control can still hold the brake or request a restart. A driver who stays
engaged keeps the model they have.

The server additionally checks the output is finite and reports `NOT_FINITE`
rather than returning it, matching openpilot's own guard on big-model output.

## Getting the package onto the comma

`jetlink` is a git submodule at `jetlink_repo`, alongside `tinygrad_repo` and
`opendbc_repo`, and `launch_chffrplus.sh` symlinks it onto the interpreter
path. No install step, no writes to a read-only rootfs, no fight with the
updater, and the submodule sha is the pin. `release.json` and the
`verify_release.py` call it used to carry are gone with it; the Jetson image
records its own ID (`docs/releasing.md`).

A submodule registered but never fetched leaves the reason in
`/dev/shm/jetlink-gadget`, which is what the offroad alert reads, rather than
the feature being mysteriously absent.

For bench work, `scripts/deploy_to_comma.sh <user@host>` rsyncs the package and
configures the gadget. Set `DisableUpdates=1` while doing that: the updater
does `fetch` + `reset --hard` + `clean` and deletes untracked files.

## Params

| param | kind | meaning |
|---|---|---|
| `AcceleratorProgress` | CLEAR_ON_MANAGER_START, JSON | `{stage, frac, msg}` while provisioning; generic, not jetlink-specific |
| `Offroad_AcceleratorUnavailable` | CLEAR_ON_MANAGER_START, JSON | the offroad alert |
| `JetlinkEnabled` | PERSISTENT \| BACKUP, BOOL | the only enable; True and nothing else |
| `JetlinkEndpoint` | PERSISTENT \| BACKUP, STRING | `host:port` to use TCP instead of USB |
| `JetlinkModel` | PERSISTENT \| BACKUP, STRING | which entry in `models.json` to run |
| `JetlinkEngineReady` | PERSISTENT, STRING | sha256 of the model the Jetson has an engine for |
| `JetlinkSpec` | PERSISTENT, JSON | cached model spec, so modeld never re-reads a 766 MB ONNX |
| `JetlinkCachedModels` | PERSISTENT, JSON | oids the Jetson has plans for, so the picker can mark them |

The last four are deliberately not cleared on manager start: readiness has to
survive a reboot, or every ignition cycle rebuilds a three minute engine.
Neither `JetlinkEngineReady` nor `JetlinkSpec` is trusted blindly: jetlinkd
re-asks the server once per attach, and `backend._open_link` clears
`JetlinkEngineReady` when the server answers `EngineMissing`, so the next
parked period re-provisions instead of every drive failing at connect.

New keys are compiled into `libparams_c` from `params_keys.h`, so adding one
needs a scons rebuild. `helpers._get()` swallows `UnknownKeyName`, because
several of these are read from hardwared and the UI's param thread and a raise
there takes down a process that has nothing to do with jetlink.

## Safety framing

The comma stays the sole authority over the car: cameras, calibration, warp,
`controlsd`, panda, CAN. The Jetson is a pure function: warped frames and
context in, 18452 floats out. It holds no control state and never touches CAN.
Keep it that way and the second box stays out of the safety argument.
