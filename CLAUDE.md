# jetlink

Runs openpilot's large driving model on an attached Jetson. `README.md` covers what
it is and why the split works; `docs/transport.md` covers why the USB roles are
what they are; `docs/openpilot-integration.md` covers the integration design.

This file is the operational layer: the things that are not in the code, and the
ones that have already cost a session to rediscover.

## September 5 reliability review

Read `docs/drive-2026-09-05-latency.md`, `docs/release-readiness.md`, and
`docs/releasing.md` before deployment. They supersede older readiness claims
below: the latest offline changes are not hardware-qualified or installed.
Protocol 2 requires paired client/server updates; the fork now pins client
source and the service requires an immutable image ID. Upgrades require fully
disengaged controls, not merely standstill. USB power-role startup and the full
parking/power lifecycle remain release blockers.

## September 6 upstreaming

The fork was reshaped for two PRs against sunnypilot. `docs/upstreaming-plan.md`
is what was decided and why, `docs/upstreaming-worklog.md` is what landed and
what was decided while doing it. The openpilot side below describes the result;
anything in an older doc that says registry, `base.py`, `chestnut.py`,
`active()` or `catalog()` predates it.

## The openpilot side

Lives in the fork (`sunnypilot-jetson-trt` worktree), not here. jetlink is a plain
Python package the fork imports; it knows nothing about openpilot.

### One module, one implementation

```
openpilot/sunnypilot/accelerators/
  __init__.py      the functions core openpilot calls, plus Daemon and the progress param
  setup.sh         runs each backend's setup.sh at boot
  SConscript       builds the comma-side warp
  jetlink/         this project's backend
    backend.py     the only module the package's functions call into
    helpers.py     where the model is, whether a Jetson is attached, gadget state
    joining.py     the small model now, the Jetson swapped in later
    model_state.py what modeld drives per frame once joined
    fallback.py    resetting the small model's queues in place at a demote
    jetlinkd.py    the offroad daemon, and the VM sysctls
    warp_cache.py  the comma-side warp JIT: capture, load, warm, device init
    compile_warp.py  its CLI, invoked by accelerators/SConscript
    status.py      modeld's per-frame status hook
    spec_cache.py  the model spec, cached in a param
    lfs.py         fetching a model out of comma's LFS
    models.json    the model registry
```

There is no registry. `base.py` and `chestnut.py` are gone, and with them the
`Accelerator` protocol, `_BACKENDS`, `backends()`, `_ask()`, `active()`,
`catalog()` and `name`. `accelerators/__init__.py` is a module of plain
functions with exactly one implementation behind them.

comma's chestnut board is not one of them. `ChestnutState` is in `modeld.py`
where upstream put it, the registry's move out of it having been reverted, and
hardwared, the UI and the model manager keep upstream's chestnut code at
upstream's lines. They ask this package only when no board is fitted. Selection
is one expression:

```python
if chestnut_present():                              # native, upstream's own block
elif accelerators.ready() and accelerators.prepare():  # jetlink
```

A registry with two backends answered the same question two ways: `active()`
meant "first ready", `catalog()` meant "first present", so a fitted but
uncompiled chestnut next to a provisioned Jetson gave a chestnut model catalog
and a jetlink modeld. With chestnut native that state cannot be constructed.
The other half of the registry, `_ask()` swallowing every backend exception,
is gone too: with the package absent the expensive functions answer their
negative default and log once, which is the only failure mode left.

`sources/` and `runners/` were both taken (`source` = which model catalog,
`runner` = snpe/tinygrad/stock), so the axis got called `accelerators`.

Keep the seam at `backend.py`. Anything only `jetlinkd` needs goes in `helpers`
or `spec_cache`, never in the backend. That is what keeps the upstream patch to
a hundred lines and liftable by another fork.

### Installing it is not enabling it

`JetlinkEnabled == True` is the only enable. There is no "absent means auto"
rule any more, and `link_configured()` is not an enable either.

The auto rule was self-fulfilling: on AGNOS with the package installed,
`setup_gadget.sh` created `/dev/ffs-jetlink/ep0` at boot, so `link_configured()`
was true, so `enabled()` was true, so `uses_stock_runner()` quietly routed
manager away from whatever small-model bundle the user had picked. Installing
the package enabled the feature and changed which modeld ran, with nobody
asking for either.

So `jetlink/setup.sh` now reads the param through `openpilot.common.params`
before it touches the gadget, and exits 0 when it is not true. With it off
there is no gadget, jetlinkd applies no sysctls, and manager takes no runner
override. A params library that is not built yet reads as off (on
AGNOS the updater builds in the staging overlay, so it is normally there by
the time the launcher runs). `helpers.enabled()` is `bool(_get(P_ENABLED))`
and nothing else.

### Upstream files touched

A hundred lines across a dozen files, none of them inside a chestnut code path.
`modeld.py` no longer gets *smaller*, because `ChestnutState` no longer moves:
that was the registry's doing and it is exactly the kind of churn that makes a
fork hard to sync. Small and dull is the whole reason another fork can lift
this. Count it, do not guess it:

```bash
git diff --numstat develop..HEAD -- . ':!openpilot/sunnypilot' ':!tools' ':!*/tests/*' \
  ':!openpilot/selfdrive/ui/sunnypilot' ':!openpilot/selfdrive/ui/mici'
```

Measured 2026-09-06, 12 files, 99 insertions, 3 deletions:

```
.gitignore                                    +1     the warp pkl
.gitmodules                                   +3     jetlink at jetlink_repo
jetlink_repo                                  +1     the submodule pin
launch_chffrplus.sh                           +3     the symlink and accelerators/setup.sh
openpilot/cereal/custom.capnp                +20     additive ModelDataV2SP fields, two events
openpilot/common/params_keys.h               +12     the params below
openpilot/selfdrive/modeld/modeld.py     +21  -1     five sites, chestnut untouched
openpilot/selfdrive/selfdrived/alerts_offroad.json  +4
openpilot/selfdrive/selfdrived/selfdrived.py +11 -1  the chime fix, an import and one call
openpilot/selfdrive/ui/ui_state.py        +7  -1     the view short-circuit and usb_unknown
openpilot/system/hardware/hardwared.py       +12     the offroad alert and the shutdown call
openpilot/system/manager/process_config.py    +4     daemons()
```

`openpilot/common/hardware/usb.py`, `openpilot/selfdrive/modeld/helpers.py` and
`openpilot/selfdrive/selfdrived/events.py` are byte-identical to `develop`.
`usb.py` used to widen `deviceState.chestnutPresent` to mean any accelerator;
it does not any more (see below).

modeld's five sites, all outside the `if CHESTNUT:` block:

1. `JETLINK = not CHESTNUT and accelerators.ready() and accelerators.prepare()`,
   before `config_realtime_process`. It has to be there: `prepare()` runs
   `warp_cache.init_device()`, and a thread created after modeld goes realtime
   inherits FIFO 54 on core 7.
2. an `elif JETLINK:` beside the chestnut load, which builds the small model
   and hands it to `make_model_state`. No loader thread and no
   `BIG_MODEL_TIMEOUT`: the joining state returns at once with the small model
   driving.
3. `chestnut_state = accelerators.make_status_publisher(pm, model)`, the same
   `after_enqueue` callback the chestnut path passes.
4. `if JETLINK: raise` at the top of the frame loop's `except`. The joining
   state owns its own demotion, so a small-model fault is fatal here as it is
   on stock. Without it upstream's handler would write `ChestnutActive=False`
   and orphan the joining state's threads and its open link.
5. the three `modelDataV2SP` writes, from `big_model_available`,
   `big_model_state` and the constant `'jetlink'`.

`modelV2.big = model.chestnut` is untouched; `JoiningModelState.chestnut`
proxies it.

### The warp is a build product, and scons builds it

jetlink runs `warp` on the comma and `run_policy` on the Jetson. Upstream used
to ship the warp as its own JIT inside the small model's pkl, so borrowing it
cost nothing. `commaai/openpilot#38684` fused the two into a single `run_model`
graph with no seam, and `WARP_INPUTS`, `POLICY_INPUTS`, `make_warp_input_queues`
and the `WARP_DEV`/`QUEUE_DEV` split went with it.

`make_warp` still exists and still builds the same closure, so the wire format
did not move. It just has to be JIT-compiled somewhere, and that somewhere is
not modeld: the compile would land inside the model load on the one GPU, every
ignition.

So it is a scons target, `openpilot/sunnypilot/accelerators/SConscript`, wired
in through the one-line `sunnypilot/SConscript` dispatcher and gated on
`arch == comma_arm64` and the jetlink package being checked out. That is what
upstream does with `dm_warp_*.pkl` a few lines away in `modeld/SConscript`, and
it is the whole point: `launch_chffrplus.sh` runs `setup.sh` and then
`build.py`, so an update that moves tinygrad has a rebuilt warp before manager
starts, never mind before ignition. The whole `tinygrad_repo` glob is in the
dependency list, so a submodule bump rebuilds it.

The target is
`openpilot/sunnypilot/accelerators/jetlink/models/warp_<camWxcamH>_<modelWxmodelH>_tinygrad.pkl`,
covered by the repo-wide `*.pkl` ignore. In the tree on purpose: scons has to
write it, and the updater's `reset --hard` + `clean` deletes it exactly as it
deletes upstream's pkls, with the build that follows putting it back. It used
to live under `Paths.comma_home()`, which on AGNOS resolves under `/home`, an
overlay whose upper layer is in `/rwtmp`, a tmpfs. A warp cached there is gone
at the next boot, and with the car and the comma powering up together jetlinkd
then loses the ~9 s compile race to ignition on every cold boot. That was the
"no warp compiled yet, staying on the small model" of the 2026-09-04 drive.

There is no staleness key any more, and `_build_key`, `_tinygrad_pin` and the
json sidecar are gone with it. `warp_cache.is_cached` asks whether the file
exists and nothing else, because scons owns invalidation and a stale warp
cannot outlive a build that succeeded. What presence cannot catch - a pickle
written by an incompatible tinygrad - fails to unpickle, and `load_warp` turns
that into the small model, which is where it was always going to end up.

`jetlinkd.build_warp` survives only to build a warp that is missing outright,
which in practice means a prebuilt image made without the target. It checks
`is_cached` before reporting anything, so on a device that ran a build it does
nothing at all; reporting first used to flash "compiling the camera warp"
through the UI on every start.

No tinygrad flags on the compile, unlike the `dm_warp` target next door.
jetlinkd built this with a bare environment and that is the artifact the parity
and timing numbers were measured against; adding `DEV`/`IMAGE`/`FLOAT16` would
change the generated kernels and put those behind a fresh bench run. Moving the
build was this change; changing what it builds is a separate one.

`backend.prepare()` still checks for the warp and stays on the small model
without one, rather than opening the link and waiting out `CONNECT_TIMEOUT`
before failing on something local. On a device that completed a build it should
never fire.

### What core openpilot asks

Every function in `accelerators/__init__.py` is a thin call into `backend.py`.
`present()`, `ready()`, `progress()` and `uses_stock_runner()` are polled by
the UI at 5 Hz and must stay cheap.

```
present()             a Jetson is attached, or dormant and known to be there. USB-independent,
                      because the comma is the gadget and enumerates nothing.
ready()               params only: enabled, no gadget error, and the cached spec, the
                      selected model and the Jetson's built engine all agree. What the
                      UI calls "compiled". modeld and the UI both ask.
unavailable_reason()  the offroad alert text, and None unless the user opted in.
prepare()             modeld only, and the last thing before it goes realtime: brings
                      tinygrad's device up, refuses without a compiled warp.
make_model_state()    the joining state, which is the small model with a Jetson
                      arriving underneath it.
make_status_publisher()  modeld's after_enqueue hook.
uses_stock_runner()   should manager run stock modeld whatever bundle is stored.
model_choices() / select_model() / active_model_name()   the models panel.
daemons()             jetlinkd, offroad, gated on enabled().
shutdown()            hardwared, before DoShutdown. The 25 s bound is enforced here,
                      on a thread with a join, because deviceState stops publishing
                      for as long as it takes.
progress() / report_progress() / clear_progress()   the provisioning param.
```

`prepare()` returns a bool and may veto; `ready()` has to stay cheap because
the UI polls it, so anything that blocks belongs in `prepare()`, which only
modeld calls.

Two import cycles are avoided on purpose, do not undo them:

- `Daemon` is a description, not a `PythonProcess`, so this package never
  imports `process_config`, which imports it. manager owns the onroad gating.
- the `jetlink` client package is imported inside the functions that need it,
  never at module scope. hardwared, the UI and the model manager all call in
  here; none of them should pay for it, and a device without the package has
  to answer rather than raise.

### Runtime state is in modelDataV2SP, not chestnutState

`deviceState.chestnutPresent` and `chestnutState` mean comma's board and
nothing else again. jetlink publishes neither. It used to widen
`chestnutPresent` in `usb.py` and publish `chestnutState` with Tegra sysfs
mapped onto comma's fields and a synthetic `pcieLtssm = 0x78`, which cost a
compensating "is it really a chestnut" check in hardwared and in every catalog
decision in the model manager.

What modeld publishes instead. `modelV2.big` is upstream's own field and means
what it always meant; the three `modelDataV2SP` fields are additive and default
to zero for every older log, for chestnut and for every other startup-only
runner:

```
modelV2.big                              a big frame was published
modelDataV2SP.bigModelAvailable          connected, waiting for disengagement
modelDataV2SP.acceleratorState           none | joining | running | retrying | unavailable
modelDataV2SP.acceleratorName            "jetlink"
```

Offroad progress still goes in the `AcceleratorProgress` param, because its
writer is a daemon in another process.

Telemetry has nowhere to go on the wire yet: a new top-level message needs a
`customReserved` slot that only the maintainers can assign. Until then
`status.py` logs it to swaglog as a `jetlinkTelemetry` event at 1 Hz, which is
enough to read temperature, power and GPU load off a drive. The hook still has
to exist even though it publishes nothing: passing a callback is what makes the
client ask for telemetry, piggybacked on the previous inference response
(`want_state`), so it costs no round trip and cannot delay a frame.

selfdrived keeps upstream's native big-model block verbatim, including the 5 s
settling window, and gains one call into
`sunnypilot/selfdrive/selfdrived/accelerator_events.py`. That adapter reads
messages only, never a param: `bigModelAvailable` on the field rising while
`modelV2.big` is false, `bigModelLoading` NO_ENTRY only while
`acceleratorState == joining` and `modelV2` is not alive (a late join never
keeps the driver out), and on `modelV2.big` falling while engaged a latched
loss: the native `bigModelFailed` and the sunnypilot `bigModelLinkLost` on
every tick until the driver disengages. Both, because the main state machine
consumes native events only, and `bigModelFailed` is comma's own "big model
gone, small model driving" soft disable; `bigModelLinkLost` is what MADS reads
and what carries the "reconnecting if it comes back" guidance. Latched,
because `state.py` cancels a soft disable the tick its event disappears, so
the one-tick edge this used to be never disabled anything. The fall is only
counted while `acceleratorState != none`, so a chestnut fall raises the native
`bigModelFailed` once and the adapter says nothing.

The joining state writes no params at all now. `ChestnutLoading` and
`ChestnutActive` describe a load that happens once and is then over; ours never
is, and a Jetson that joins, leaves and rejoins was writing an alert cycle each
time through keys upstream's own code also reads.

### The UI identifies a chestnut by USB id; we are the gadget

Upstream's `ui: show usb connection` (#38745) decides `usb_unknown` by looking
for a chestnut USB id among the devices the comma enumerated as a host. The
comma is jetlink's gadget and enumerates nothing, so that check shows the
generic USB icon instead of the accelerator icon. Expect this again: anything
upstream adds that recognises the board by USB id needs the same guard.

`ui_state.py` keeps upstream's `_update_chestnut_state` and its latched
`chestnut_compiled()` exactly as they are, and gains two things: a
short-circuit at the top for when `accelerator_view` is not None, and the
`usb_unknown` guard. The view itself is built in `sunnypilot/ui_state.py` on
the same 5 Hz params pass as the chestnut params, and only when
`deviceState.chestnutPresent` is false, so a chestnut user never reaches a line
of ours. Offroad it reads the progress stage (a provisioning accelerator is
LOADING, only a `failed` stage is FAILED); onroad `modelV2.big` wins first,
then `present`, then a `joining` or `retrying` state as LOADING. An accelerator
recognised after the 10 s grace period still clears "unknown".

The models panel exists in both layouts, mici and not, with the link toggle and
the model picker. `effective_small_bundle()` is what either one names as the
small model: under the override manager runs stock modeld, which loads the
default small model and never reads the stored qcom bundle, so naming that
bundle would be a lie.

### jetlinkd owns the VM sysctls

Three system-wide values, because the gadget read shares the kernel with every
writer on the device:

```
vm.dirty_bytes             16 MB
vm.dirty_background_bytes   8 MB
vm.min_free_kbytes        128 MB
```

They used to be applied at boot with no param gate and no way back. Now
`jetlinkd` applies them when the link is enabled and puts them back only in
the "disabled, releasing the link" branch, never on exit. The settings are
for the drive and the daemon is not: manager stops it at ignition, which is
exactly when the contention they were measured against starts, so a restore
in `run()`'s `finally` (which is what it was) handed modeld stock values on
every drive. A reboot resets them; disable and SIGKILL are the two ways they
change while the device is up. Apply is idempotent on every start, and the
stock values are read once into `/dev/shm/jetlink-sysctl-prev` before the
first change and never overwritten, so a later run records our own values as
stock under no circumstances; the record survives in tmpfs so a disable after
any number of restarts still knows what to put back.

The record also carries `vm.dirty_ratio` and `vm.dirty_background_ratio`,
because stock AGNOS runs the dirty limits in ratio mode and both `*_bytes`
keys read 0, and 0 cannot be written back. Measured on the comma (kernel
4.9): after our apply, `sudo sysctl -w vm.dirty_bytes=0` returns 0 and the
value stays 16777216, and a direct `/proc/sys` write does the same, because
`proc_doulongvec_minmax` silently skips a value below `dirty_bytes_min` (two
pages). Writing the ratio key is what zeroes the bytes key, so a recorded 0
restores by writing the recorded ratio key instead; a non-zero recorded value
restores by the bytes key as before, and `min_free_kbytes` is unaffected.

64/32 was the earlier set, from `docs/drive-2026-09-05-ba-latency.md`. Under a
500 MB memory hog plus five CPU workers, 64/32 let one 104 ms frame through and
128/16 did not, which is why the newer set wins.

### Packaging: a submodule and one symlink

`jetlink` is a git submodule at `jetlink_repo`, and `launch_chffrplus.sh` has
one more `ln -sfn jetlink_repo/jetlink jetlink` next to the tinygrad line. That
is the whole install: on the interpreter path, no install step, no writes to a
read-only rootfs, and the submodule sha is the pin.

`release.json` and the `verify_release.py` call are gone from the fork with it.
A pinned submodule sha already says which client the fork expects, and the
Jetson image records its own ID (`docs/releasing.md`). A submodule registered
but never fetched leaves the reason in `/dev/shm/jetlink-gadget`, which is what
the offroad alert reads, rather than the feature being mysteriously absent.

For bench work off the submodule, `scripts/deploy_to_comma.sh <user@host>`
still rsyncs the package and configures the gadget. Set `DisableUpdates=1`
while doing that: the updater does `fetch` + `reset --hard` + `clean` and
deletes untracked files.

### Params

```
AcceleratorProgress            CLEAR_ON_MANAGER_START, JSON   provisioning progress for the UI
Offroad_AcceleratorUnavailable CLEAR_ON_MANAGER_START, JSON   the offroad alert
JetlinkEnabled                 PERSISTENT|BACKUP, BOOL        True and only True enables the
                                                              feature; the "accelerator link"
                                                              toggle in the models panels writes it
JetlinkEndpoint                PERSISTENT|BACKUP, STRING      "host:port" forces TCP instead of USB
JetlinkModel                   PERSISTENT|BACKUP, STRING      name from models.json
JetlinkEngineReady             PERSISTENT, STRING             sha256 the Jetson has built
JetlinkSpec                    PERSISTENT, JSON               the parsed model spec
JetlinkCachedModels            PERSISTENT, JSON               oids the Jetson has plans for, from
                                                              hello; the picker marks them "cached"
```

`JetlinkEngineReady`, `JetlinkSpec` and `JetlinkCachedModels` are deliberately
not `CLEAR_ON_MANAGER_START`: readiness has to survive a reboot or every
ignition cycle rebuilds a three minute engine. None is trusted blindly:
jetlinkd re-asks the server once per attach (the Jetson's cache can be pruned,
re-flashed or swapped under the param), and `backend._open_link` clears
`JetlinkEngineReady` when the server answers `EngineMissing`, so the next
parked period re-provisions instead of every drive failing at connect.

`JetlinkSpec` is what the *server* sent back: the Jetson is the only side that
parses the ONNX. The comma hashes the file once (cached against path, mtime and
size) and sends sha256 and size; shapes and output slices come back with
`ENGINE_RESP`. That is what lets a stock device run a model tinygrad refuses to
parse (the `org.tinygrad` domain).

New keys are compiled into `libparams_c` from `params_keys.h`, so adding one needs
a scons rebuild. `helpers._get()` swallows `UnknownKeyName` because a device on an
older params library would otherwise take down hardwared.

### Who owns the link

`jetlinkd` offroad, `modeld` onroad. manager's `only_offroad` gate enforces it.
Something has to hold FunctionFS `ep0` open or the comma never enumerates at all,
which is why `jetlinkd` runs even with nothing to do.

At the handover the gadget briefly unbinds and the Jetson re-enumerates. Both ends
handle it, and re-enumeration has been observed at 45 to 70 seconds, against
`backend.CONNECT_TIMEOUT = 45.0` for one attempt. That is no longer a deadline
on the drive: the attempt fails, the join loop backs off and tries again for as
long as the drive lasts, and modeld runs the small model in the meantime. It is
still worth watching, because a handover that costs 70 s costs the first minute
of every drive.

A `hello()` sent while the Jetson is still booting blocks inside the
gadget `writev` until the server process starts reading, whatever timeout was
passed: FunctionFS writes cannot time out, and the UDC says "configured" the
whole time because the host enumerated us long before the server came up.
Measured 90 s on the car. It is benign (the exchange completes the moment the
server reads) and it is why jetlinkd sits through manager's SIGINT and eats
the SIGKILL 5 s later at an ignition that lands mid-boot.

The swap to the large model happens on modeld's frame thread and has to be
cheap: measured on a comma, unpickling the warp JIT is 0.3 s and its *first
call* 1.9 s, which as one frame is ~26 dropped camera frames, and modeld's
drop filter (cap 10, tau 10 s) then reads 4.74% for 16 s. That was
`modeldLagging` after every join on the 2026-09-04 drive.
`backend.make_model_state` therefore loads and warms the warp in the joining
state's constructor, which modeld runs where it loads a model: on the main
thread, before the frame loop exists, so the cost lands in "models loaded in
N s" where nothing can be dropped. The swap itself then sends no warmup frame;
the first real frame carries the reset. There is no loader thread on this
path - that belongs to the chestnut block, whose 60 s `BIG_MODEL_TIMEOUT` a
Jetson would lose anyway.

The engine does not reload at the handover. `server/session.py`'s `EngineHost`
owns the one loaded engine and the one build in flight for the life of the
process; a `Session` is a view onto it. Before that, every reconnect freed the
engine and the next connect paid 13 to 25 s to deserialize it, once per rejoin.
A client that reconnects during a build attaches to it. Only one engine
is ever resident: a build or a load of a different model unloads the current
one first.

### Frame semantics: what chestnut does

There is no deadline a frame can miss by being slow. `infer_end` blocks for the
frame the way modeld blocks on a chestnut; a frame past 50 ms is a dropped
camera frame, which modeld counts and tolerates. Only a stall past
`backend.INFERENCE_TIMEOUT` (0.5 s, ten frame periods) is a failure, and then
it is a `LinkError` and the joining state demotes to the small model and
starts rejoining. The client's 3 s `FRAME_TIMEOUT` is what `hello` and
`ensure_engine` get, on the join thread, where an engine load out of the plan
cache is 13 to 25 s of legitimate work; it is not what a driving frame gets.
A dead server used to hold the frame thread for those 3 s, four frame periods
past the point the answer could still be useful. An even earlier design had a
35 ms deadline with a non-latching timeout; modeld's `except Exception` made
every one of those a permanent fallback anyway.

Any timeout on the comma only works because reads run on a thread.
FunctionFS ignores `O_NONBLOCK` once the host has enabled the endpoint: a
synchronous read waits for the USB request to complete. Measured on the car
before the fix: a `ping(timeout=0.5)` against a SIGSTOPped server returned
after 14.4 s, when the server was resumed. After it: `LinkTimeout` at 0.501 s,
the stale reply discarded on resume, the stream still in sync. Writes stay on
the caller's thread; a host that is not reading is a dead link either way.

### Signals and FunctionFS

A signal that lands while a thread is blocked in a FunctionFS `writev` is not
a retry. `ffs_epfile_io` dequeues the request, which stops the transfer
wherever it is, and returns `EINTR`; Python's `os.writev` then re-issues the
whole call, so the host receives the start of the message twice and the stream
is lost from there. modeld is exactly the process this happens to: msgq wakes
its subscriber threads with `SIGUSR2` for every camera frame, ~160 a second.
`strace -e trace=writev -e signal=all` on a live modeld showed four of them
inside `writev` in 120 s, one per link failure that run. `FfsTransport` masks
signals for the duration of a write and for the life of its reader thread.

That one bug wore several faces on the car, all at ~1 in 400 frames: bad magic
on the Jetson (the replayed prefix read as a header), `NOT_READY` (a fresh
server session answering the next frame), `NOT_FINITE` (a request assembled
from two frames' bytes overflowing fp16 in the queues), a duplicate request
with the same seq, and a frame timeout. It never showed on the bench with
`bench_link`, `verify_parity`, the replay tool or a scripted reconnect loop,
because none of them run next to camerad. Only the live bench below did.

### Never touch an endpoint file before the host has enabled it

This one costs the gadget until the comma is rebooted, and it is silent.

`ffs_epfile_io` does not fail on an endpoint no host has enabled. It sleeps:
`wait_event_interruptible(ffs->wait, (ep = epfile->ep))`, and only a signal
wakes it. Unbinding the UDC does not - unbind completes a request the hardware
already holds, and an endpoint that was never enabled has no request. So a
read on `ep1` before a host arrives never returns.

A thread stuck there cannot have its fd closed. The syscall holds the `struct
file`, so `ffs_epfile_release` never runs, `ffs->opened` stays above zero, and
`ffs_ep0_open` answers `EBUSY` from then on - to this process, to `jetlinkd`,
and to the next drive's `modeld`. `close()` returns cleanly and `/proc/PID/fd`
shows nothing; only the live thread gives it away. Measured on the car
2026-09-04: one open and close of a gadget no Jetson ever attached to, and
every open after it failed with

```
[Errno 16] Device or resource busy: '/dev/ffs-jetlink/ep0'
```

for the life of the device. This is the "I had to restart the comma to get
jetlink established" of that drive: any ignition without the Jetson up poisoned
the mount, and the retry loop, `jetlinkd` and the next drive all bounced off it.

So `FfsTransport.__init__` opens `ep0`, writes the descriptors and binds, and
nothing else. `_ensure_epfiles()` opens `ep1`/`ep2` and starts the reader on
the first read or write, and only once the UDC reads `configured` - the same
edge that sets `epfile->ep`. A host that never comes leaves `ep0` as the only
fd, and closing that is clean and instant. `_read_loop` re-checks before every
`readv` for the same reason: the host can drop our configuration between two
reads, and the read after that would sleep forever.

If you add a code path that opens an endpoint file, it goes through
`_ensure_epfiles`. Verify with the loop below, which fails on the first
reopen if this regresses:

```python
for i in range(4):
  c = helpers.connect(); print(i, 'opened'); c.close()
```

The endpoints answer `ENODEV` both before a host has configured the gadget
and after it disconnects. `FfsTransport` gives the first a 10 s grace period
(`EP_READY_TIMEOUT`: the server still has to open the device and claim the
interface after enumeration). It used to give the second the same, so the
mid-drive USB disconnect on the 2026-09-04 drive was 10 s and 201 frames of
retrying a link the UDC already reported as "not attached" before modeld fell
back. Now, once a host has talked to us, an endpoint error with the UDC state
anything but "configured" ends the link at once.

Three things in the transport came out of that session and stay for defence
in depth: the host reads exactly what the current message still needs and
never across its end, the gadget pads every message to a 16 KB burst so
nothing it sends ends on a short packet, and the server drops a request whose
seq it has already answered. Each was measured not to be the cause on its own.
The server also drains a desynced gadget rather than reopening it: reopening
took a packet per session at 100 Hz, the comma's frame timeout never fired,
and modeld sat inside one `writev` for the rest of the run.

### Is it actually running

`modelV2.big` is the signal. `JetlinkModelState` sets it; the small model does
not. `chestnutState` is no longer one of ours: it is comma's board again, and
on a Jetson device it is neither published nor valid. What to read instead:

```
modelV2.big                              true once a frame came back over the link
modelDataV2SP.acceleratorState           "running"
modelDataV2SP.acceleratorName            "jetlink"
modelDataV2SP.bigModelAvailable          true while connected and waiting for a window
swaglog                                  "jetlink: Orin-sm87 trt 10.3.0, engine ..."
swaglog, 1 Hz                            jetlinkTelemetry, with tempC / powerDrawW / gpuUsagePercent
```

`deviceState.chestnutPresent` stays false throughout, by design. A plan that
stops at ~5 m instead of ~200 m is the other tell, and that is what a demote to
the small model looks like from the outside.

Before blaming the link, check which modeld is even running. jetlink lives in
stock `modeld`, and manager runs that only while `get_active_model_runner()` is
`stock`. With `JetlinkEnabled` true, `uses_stock_runner()` is true and
`get_active_bundle()` returns None, so the answer is `stock` whatever bundle
is stored: that is the override, and it is why the models panel names the
default small model rather than the stored one. `JetlinkModel` is not part of
the gate: it defaults through `selected_model()`, so an enabled device with no
model set still provisions and reports `ready()`, and gating on it too left
that device on `modeld_tinygrad` under a custom small bundle (measured on the
comma). With the toggle off the stored bundle decides again, and every
bundle the sunnypilot model manager offers has `runner = tinygrad`, so a custom
model moves manager to `modeld_tinygrad` (`sunnypilot/modeld_v2`), which knows
nothing about jetlink: the Jetson provisions, `ready()` is true, the UI says
compiled, and `modelV2.big` is never set because the process that would set it
is not running. `ModelRunnerTypeCache` caches the answer, so clear it whenever
either param changes by hand; `select_model()` and the link toggle already do.

## Backends: the server off the Jetson

`docs/platforms.md` is the reference. The shape:

```
jetlink/server/
  backends/base.py      Backend and Engine protocols, ArtifactInvalid
  backends/trt/         TensorRT: engine.py, cudart.py, build.py, moved from server/, keys unchanged
  backends/tinygrad/    build.py (OnnxRunner + TinyJit, pickled), engine.py, owner.py
  backends/ort/         onnxruntime: CoreML on a Mac, CUDA or CPU elsewhere
  cache.py              EngineCache, out of builder.py; the key is <sha16>.<backend tag>
  platform.py           is_jetson, default_cache_dir, available_bytes, gpu_name
  builder.py, engine.py, cudart.py   shims at the old import paths, one release
```

What the Jetson must never notice: the TensorRT tag is `trt<version>.<device>`
byte for byte, so `<oid16>.trt10.3.0.Orin-sm87.plan` loads without a rebuild;
the timing cache is `timing.trt10.3.0.Orin-sm87.cache` as before; no build flag
changed on TensorRT 10. `tests/test_trt_backend.py` pins the key. TensorRT 11
(PyPI) dropped weak typing and `BuilderFlag.FP16`, so the build asks for a
strongly typed network there and precision follows the fp16 ONNX; untested on
hardware.

The hello gains `backend`, `runtime_version` and keeps `trt_version` only from
the TensorRT backend. The fork logs it and nothing else reads it.

`ArtifactInvalid` from `Backend.load` means the file is wrong (a pickle from
another tinygrad, an empty CoreML cache): the host deletes it and rebuilds
from the ONNX if it has one, else answers `need_upload`. TensorRT never raises
it: a plan that will not deserialize is a memory or device fault, and 770 MB
is not deleted for that.

### tinygrad on Metal is one thread

A JIT unpickled on one thread segfaults when replayed from another, and a
thread that has used Metal segfaults in objc's autorelease-pool drain as it
exits (crash report: `AutoreleasePoolPage::releaseUntil` under
`_pthread_exit`). The server loads on a job thread that exits and runs frames
on the session thread, so every tinygrad call goes through the daemon thread
in `backends/tinygrad/owner.py`, which never exits. Do not run tinygrad from
anywhere else in the server process, including a test.

### onnxruntime is one process

`InferenceSession()` holds the GIL for its whole duration (a 1 ms ticker
thread ran three times across twenty session creations; a CoreML build logged
no progress in ten minutes). In-process that freezes the request loop, the
pings and the accept for as long as CoreML compiles, so the session lives in
a spawned worker (`backends/ort/worker.py`) with the frame's inputs and
outputs in shared memory. Keep it that way; do not "simplify" it back into
the server process.

### The Neural Engine: one broken operator, one imprecise one, and idle

Measured 2026-09-08 on the M1 Pro. The whole model with every CoreML unit
allowed (`MLComputeUnits=ALL`) was 25 ms and wrong, correlation 0.91 to 0.97.
Bisected with sub-models cut at node indices, CPU-probed intermediates fed
in, each piece compiling in seconds (`scratchpad/submodel.py` from that
session is the tool): one node, `Gather(add_53, -1, axis=1)`, the last-token
select after the temporal transformer, comes back as garbage on the Neural
Engine and is exact with the index written as 287.
`onnx_patch.normalize_gather_indices` rewrites every negative constant
Gather index; value-preserving; the onnxruntime backend applies it always.

With that fixed: 28 ms, gate failed by one column. The Neural Engine's fp16
LayerNormalization overflows on the residual stream (values in the
hundreds); the policy half was seven times less precise than on the GPU and
no faster (10.7 vs 10.6 ms). `onnx_patch.layernorm_in_fp32` on the policy's
41 LayerNormalizations (by dataflow, `vision_nodes`) restores the GPU's
precision exactly and makes the policy faster; on the trunk's 41 as well it
costs a unit switch each and the frame went to 70 ms, so only the policy.
Gate passed, 31.5 ms back to back.

Then the cadence: at 20 Hz every CoreML unit pays a cost on the first
request after an idle gap (a 127 MB sub-model: 3 to 11 ms on the Neural
Engine, 6 to 14 on the GPU, from a 5 ms gap up; a keep-warm model in the gap
did nothing), and through the server the Neural Engine session is 44.6 ms
with a p99 of 59 at 20 Hz against the GPU session's 43.3 with a p99 of 44,
while back to back it is 32.6 against 39.9. A trunk-on-ANE, policy-on-GPU
split of two sessions pays the idle cost twice (45 ms). So `--device coreml`
(GPU) is the default and `--device ane` is the opt-in, correct by the gate,
for a Mac where the idle cost measures differently. Do not blame precision
first if a new export goes wrong there: bisect.

### onnxruntime is one process

`InferenceSession()` holds the GIL for its whole duration (a 1 ms ticker
thread ran three times across twenty session creations; a CoreML build logged
no progress in ten minutes). In-process that freezes the request loop, the
pings and the accept for as long as CoreML compiles, so the session lives in
a spawned worker (`backends/ort/worker.py`) with the frame's inputs and
outputs in shared memory. Keep it that way; do not "simplify" it back into
the server process.

### The Neural Engine: one broken operator, then a split

Measured 2026-09-08 on the M1 Pro. The whole model on the Neural Engine
(`MLComputeUnits=ALL`) was 25 ms and wrong, correlation 0.91 to 0.97. Bisected
with sub-models (`onnx.utils`-style cuts, seconds to compile each) down to one
node: `Gather(add_53, -1, axis=1)`, the last-token select after the temporal
transformer, comes back as garbage on the Neural Engine and is exact with the
index written as 287. `onnx_patch.normalize_gather_indices` rewrites every
negative constant Gather index; it is value-preserving and the onnxruntime
backend applies it. With that fixed the Neural Engine is 28 ms and fails the
parity gate by one column (road_transform std[3] at 0.9988): the policy half
is no faster there than on the GPU (10.7 vs 10.6 ms) and seven times less
precise, while the vision trunk is 16.6 ms there against ~28 on the GPU and
precise. So `onnx_patch.split_vision` cuts the graph by dataflow where the
image-only trunk meets `features_buffer`, and the onnxruntime backend runs
two sessions in its worker: trunk with `ALL`, policy with `CPUAndGPU`. Gate
passed on every slice and column. Bisect the same way if a new export goes
wrong there: `scratchpad/submodel.py` from the 2026-09-08 session is the tool
(cut the graph at node indices, feed CPU-probed intermediates, compare).

### onnxruntime's telemetry aborts the test suite

`recursive_mutex lock failed: Invalid argument` at interpreter exit, one run
in three, is onnxruntime's telemetry thread (`Microsoft::Applications::Events`
in the macOS wheel) handling an HTTP response after static destruction. The
backend and the tests call `disable_telemetry_events()` right after the
import; keep it that way in anything else that imports onnxruntime.

### Mac numbers, M1 Pro

CoreML on the GPU (`--device coreml`, the default): 43 ms round trip at
20 Hz, p99 44, none of 390 frames over budget, gate passed; nine to ten
minutes to create the session in every process (in the worker, pings
answered meanwhile), onnxruntime's model cache did not shorten it; 5.5 GB per
artifact. `--device ane`: gate passed, 32.6 ms back to back, 44.6 at 20 Hz
with a p99 of 59.
tinygrad METAL: 66 ms a frame, over budget on that machine, parity passed,
13 s build, 1 s load; `--backend tinygrad` is the switch.

## The cable

### The comma is the gadget and the Jetson is the host, and it cannot be reversed

Decided by what the two kernels have, not by which side is the client. AGNOS has
`CONFIG_USB_F_FS` and libcomposite built in. The Jetson has the UDC driver
(`tegra-xudc.ko`, loaded) but **not the gadget stack**: `/proc/config.gz` says
`CONFIG_USB_LIBCOMPOSITE=m` and `CONFIG_USB_F_FS=m`, yet
`/lib/modules/$(uname -r)/kernel/drivers/usb/gadget/` contains only `udc/`.

There are no headers, no `/usr/src`, no `build` symlink, and `CONFIG_MODVERSIONS=y`,
so building them out of tree needs the exact `Module.symvers` from that kernel
build. jetlink supports the inversion in software already (`JetlinkClient.open_usb`
on the comma, server `--transport ffs`); the blocker is purely the kernel.

### Do not use the Jetson's USB-C port

Tempting, because it looks right and the device tree agrees: `usb2-0` is `mode=otg`
with a `usb-role-switch` and a `vbus-supply`, and `usb3-1` is its SuperSpeed lane.

It does not work. The Orin Nano devkit has no Type-C port controller (no `typec`
class, no extcon), so its CC lines are hardwired Rd. Plug the comma into it and the
comma becomes the DFP and sources VBUS, the Jetson becomes the device, and the
Jetson cannot be a gadget (above). Writing `host` to the role switch flips the data
role but not the CC resistors, so you get two hosts and both ends driving VBUS.

**Use a USB-A port on the Jetson.** They hang off the onboard Realtek hub, so the
gadget appears one hop down at `2-1.2`.

### Power the comma from its own supply

If the comma draws power through the link, it browns out under load and the link
drops. That shows up as `jetlinkd` tearing down and reopening every 30 to 45 s
(`RETRY_BACKOFF`, after a failed provision), which on the Jetson reads as
connect/disconnect for hours.

A comma reporting `real_type=USB_DCP` on that port is seeing power with no data
host behind it: either a charge-only cable, or a port that is not acting as a host.

### Checking the link

```bash
# comma: both must be right
cat /sys/class/udc/a600000.dwc3/state          # configured
cat /sys/class/udc/a600000.dwc3/current_speed  # super-speed
cat /dev/shm/jetlink-gadget                    # ok, or "error: <reason>"

# jetson
lsusb | grep 1209:0001
cat /sys/bus/usb/devices/2-1.2/speed           # 5000
```

`speed=5000` or `super-speed` is the only proof the cable and both ports are USB 3.
480 means you are on a USB 2 path and the transport cost roughly doubles.

### No FunctionFS preflight in setup_gadget.sh

Deliberate, and it is commented in the script. The kernel registers the
`functionfs` filesystem when the first `ffs.*` function is instantiated and
deregisters it with the last, so `/proc/filesystems` lists it only while some
other gadget is already using it. A check before the `mkdir functions/ffs.jetlink`
can never pass on a cold boot. Proven on the car:

```
before mkdir functions/ffs.probe : ABSENT
after                            : nodev functionfs
after rmdir                      : ABSENT
```

AGNOS has no `/lib/modules`, so the conditional modprobe above it is skipped too.
The `mkdir`, `mount` and `ep0` checks downstream test the same capability at the
point it is actually used.

## The model registry

`models.json` in the fork indexes large models by their git-lfs oid in comma's
history. openpilot overwrites one file, so older models exist only as LFS objects
no manifest lists. Every bundle the model manager offers is a tinygrad pkl for the
QCOM GPU, which TensorRT cannot parse, so this history is the only public source.

### Adding a model

Recover the oid from a commit without checking anything out:

```bash
gh api repos/commaai/openpilot/commits/<sha> --jq '.files[].filename'
gh api "repos/commaai/openpilot/contents/openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx?ref=<sha>" --jq .sha
gh api repos/commaai/openpilot/git/blobs/<blob sha> --jq .content | tr -d '\n' | base64 -d
```

The last prints the LFS pointer with `oid` and `size`. Both go in `models.json`
verbatim; `size` is what `shipped_model_path()` checks the local file against.

comma's LFS is on GitLab, not GitHub, and serves unauthenticated:

```bash
curl -s -X POST https://gitlab.com/commaai/openpilot-lfs.git/info/lfs/objects/batch \
  -H 'Accept: application/vnd.git-lfs+json' -H 'Content-Type: application/vnd.git-lfs+json' \
  -d '{"operation":"download","transfers":["basic"],"objects":[{"oid":"<oid>","size":<size>}]}'
```

Probe that before adding an entry. Set `built_seconds` from the Jetson's
`/mnt/data/jetlink/engines/<oid16>.*.json` after the first build; it exists so the
UI can say how long a first provision takes.

If the Jetson already has the ONNX, copy it over the LAN instead of pulling it
from LFS, then verify the size and sha match the registry entry exactly.

Provisioning needs no other step: land the new `models.json` on the device, put
the ONNX at `Paths.model_root()/<oid16>.onnx`, and write `JetlinkModel`. A
dormant jetlinkd sees `has_work()`, presents the gadget, and the bind wakes a
sleeping Jetson. Measured 2026-09-04 for BMRLNAPv6 (766 MB): the Mac's LAN copy
to the comma 55 s, then param write to `JetlinkEngineReady` 3 min, of which the
resume, the hash, the USB upload and the uint8 patch are ~15 s and the TensorRT
build is 163. Staging the ONNX on the Jetson by hand saves nothing; the upload
is not the slow part.

### org.tinygrad ops

comma's exports sometimes carry nodes in the `org.tinygrad` domain. TensorRT's
parser rejects any op in a domain it does not know, so the build fails outright.

`onnx_patch.strip_tinygrad_ops()` bypasses the ones listed in `PASSTHROUGH_OPS`
and drops the dangling opset import. It **refuses** anything not on that list, or
carrying attributes: silently dropping an op that did something would change what
the car sees. If a new one appears, prove it is a no-op before adding it, then
prove the result with `verify_parity.py` against the *unmodified* ONNX.

Seen so far: `Contiguous`, one node, in the 2026-09-01 model. The 2026-08-31 model
has none, so this comes and goes with how a model was exported.

`strip_tinygrad_ops` has one subtle case: a passthrough node feeding a graph
output. The output keeps its name (that is what `output_slices` addresses) and
the producer takes it over. An earlier version renamed both ends and left the
output with no producer; the checker catches it and there is a test.

### The spec comes from the server

AGNOS ships tinygrad and not `onnx`, and tinygrad rejects the `org.tinygrad`
domain outright, so a stock device cannot parse every export. It no longer has
to: the server parses the ONNX at build time (or on first load of a plan whose
sidecar predates this), records the spec in the plan's sidecar json, and
returns it in every ready `ENGINE_RESP`. `onnx_meta` is server-side only now.

## Timing

Frame budget is 50 ms (`MODEL_RUN_FREQ = 20`). Being under it is necessary and not
sufficient. Three separate limits:

1. **Frame drops, the hard one.** `selfdrived.py`: `frameDropPerc > 1` raises
   `modeldLagging`, which is `ET.SOFT_DISABLE` and disengages. Drops come from
   `modeld.py` counting vipc frames that advanced while a run was in flight, so
   they appear when a run overruns 50 ms.
2. **comma's own regression gate is 28 ms mean.** `model_replay.py`:
   `EXEC_TIMINGS = [("modelV2", 0.05, 0.028)]`, asserted in CI. Probably
   calibrated for the small model on tici, so treat it as a reference point, not
   a verdict; we have no chestnut to compare against.
3. **Latency compensation is a constant and does not know about us.**
   `modeld.py`: `frame_delay = DT_MDL`, commented "current_time - timestamp_eof is
   50ms on average". It is never derived from `modelExecutionTime`, which is only
   ever copied into messages for logging. A pipeline slower than the one that
   constant was tuned against is uncompensated by the difference.

Making `frame_delay` reflect measured latency would help chestnut too, and is the
right upstream-shaped fix. Not done.

### Where the comma-side frame time goes, and the two upstream fixes worth making

Measured 2026-09-04 on the car, warp output 393,216 bytes:

```
call_warp (enqueue)    1.19 ms
device.synchronize     0.49 ms      <- the GPU is barely the cost
warped.data()          2.36 ms      <- 165 MB/s
```

The GPU is done in half a millisecond. `.data()` is a copy out of a
write-combined mapping, which is fast for the GPU to write and slow for anyone
to read; a plain CPU read of the same memory is 1.49 ms, so ~0.9 ms on top of
that is tinygrad building the result. Nothing on the comma reads those bytes -
they go straight to `writev`.

**Zero-copy was tried and reverted.** `Buffer.as_memoryview(allow_zero_copy=True)`
returns a view in 0.045 ms with byte-identical contents, and on the live bench
`data` fell 2.6 -> 0.5 ms. But `send` rose 3.6 -> 5.0 ms: the gadget write pays
the write-combined read itself rather than DMAing past it. Net was 31.3 -> 30.7
ms, 0.6 ms, and all of it was tinygrad's overhead rather than the read. Not
worth reaching through `warped.uop.base.realized` - private structure a
submodule bump rearranges - in the one code path with a history of silent
corruption. The arithmetic to keep in mind is that **someone pays the
write-combined read**; moving it is not removing it.

Two fixes that would remove it, both upstream rather than here:

1. **tinygrad: let `Tensor.data()` pass `zero_copy` through.** `Buffer.as_memoryview`
   already takes `allow_zero_copy`; `Tensor.data()` simply does not forward it, so
   the only way to reach it is private API. A one-argument change upstream makes
   this a public call. Worth ~0.6 ms here on its own.
2. **tinygrad: stop mapping the output write-combined, or offer a cached
   alternative.** This is the real one: it removes the 1.5 ms instead of moving
   it, and it helps every tinygrad user on a QCOM device, not just this fork.
   `QCOMAllocator.default_buffer_spec` is where the mapping is chosen.

### The libusb thread that inherited modeld's realtime priority: found and fixed

Threads created after `config_realtime_process(7, 54)` inherit SCHED_FIFO 54
*and* the core-7 pin, which is the documented way to lose frames here. Sampled
on a live bench, every Python thread was SCHED_OTHER on 0-7 as intended, and
one was not:

```
121565  python3        SCHED_FIFO  54   cpu 7     <- the frame loop
121614  libusb_event   SCHED_FIFO  54   cpu 7     <- same core, same priority
```

The creator is tinygrad. It brings the GPU up on the first kernel run, not at
import and not at `load_warp`, and that init spawns the libusb event thread -
the comma's GPU is reached over USB. Whichever of modeld's threads ran the
first tensor op created it, and after `config_realtime_process` that is FIFO 54
on core 7.

The fix is `warp_cache.init_device()`: one `Tensor([0.0]).realize()`, called
from `backend.prepare()`, which modeld calls a few lines *before*
`config_realtime_process`. The device comes up either way, moments later; doing
it there is the whole difference between that thread being SCHED_OTHER on every
core and SCHED_FIFO 54 on modeld's. Failure is not worth refusing the
accelerator over, so it is caught and logged.

It never cost a frame while it went unexplained: over 10 s the thread ran
**0.0 ms across 0 timeslices**, because libusb had nothing to service on a
comma that is the gadget, and the frame loop waited 0.6 ms total over the same
window. Anything that starts using libusb in that process would have found it.
The rule generalises: make the context before modeld goes realtime, never
re-nice a thread afterwards. `joining._background_priority` is the same rule
for the threads we create ourselves, and `ffs.py` for the reader.

The same trap one layer up, found by the next live-bench thread dump: three
more FIFO 54 threads on core 7, `Thread-1 (_handle_workers)`,
`Thread-2 (_handle_tasks)` and `Thread-3 (_handle_results)`. Those are
`multiprocessing.pool`'s handler threads, and the owner is tinygrad's compile
pool (`tinygrad/engine/worker.py`, `get_worker_pool()`, gated on `PARALLEL
!= 0` and not a daemon process), created on the first kernel compile. On the
jetlink path that is the warp JIT's first call inside `make_model_state`,
after `config_realtime_process`; the same dump with the link off shows only
the main thread realtime, because the stock small-model path never compiles
there. So `init_device()` calls `get_worker_pool()` right after the device
comes up, in `prepare()`, before modeld goes realtime: the handler threads
and the worker processes then inherit SCHED_OTHER on every core. It is in
its own try/except that logs and continues, because an older tinygrad
without the module or `PARALLEL=0` is not a reason to refuse the accelerator.

The library's own threads do not rely on the caller getting that right. The
write guard (`transport/watchdog.py`) and the close helper in `ffs.py` call
`transport.priority.background_thread()` first thing and drop to SCHED_OTHER 0
on every core themselves, so a realtime creator cannot lend them its priority.
A live bench had found `jetlink-write-guard` at FIFO 54 on core 7: it waits on
a Condition, and a wake at equal FIFO priority takes the frame loop's core
until it blocks again.

### Measured, Orin Nano Super 8 GB, TensorRT 10.3 FP16, SuperSpeed

modeld execution time is end to end with cores pinned as onroad.

| | BMRLNAP 766 MB | TGC v2 766 MB | Lebowski 1757 MB |
|---|---|---|---|
| GPU, server-side | 19.76 ms | ~20 ms | 36.17 ms |
| link round trip, mean | 26.08 | - | 41.52 |
| link p99 / max | 27.67 / 30.35 | - | 44.81 / 48.06 |
| transport overhead | 4.77 | - | 4.07 |
| modeld exec, mean | 30.98 | 31.10 | 46.28 |
| modeld exec, max | 32.66 | 33.48 | 49.48 |
| headroom vs 50 ms | 17.3 | 16.5 | **0.5** |
| engine build | 165.7 s | 166.3 s | 289.5 s |

Lebowski runs and is numerically correct, but 49.48 ms against a 50 ms budget is
coincidence, not margin, and the GPU is already at its 1020 MHz ceiling at 83%
duty with no boost left. Keep it off the car.

**Re-measured end to end 2026-09-04 evening, and it is faster than the row
above**: 45.1 ms mean on the live bench, per-frame warp 1.7 / data 0.5 / send
4.7 / reply 37.8, one join, no failures, 2870 big frames. BMRLNAPv6 on the same
rig the same evening was 30.7. The gain over the older row is `jetson_clocks`
being pinned, MAXN_SUPER, and the Jetson's vendor services stopped. Two things
that number is not: the bench reports `exec_last100`, a rolling window, so the
**global** worst frame across the run is not in it, and the worst frame is what
a 50 ms budget is decided by; and 150 s is not a sustained thermal run, which
has still never been done for any model. `trtexec` puts Lebowski's GPU compute
at 34.07 ms against a 22.3 ms roofline floor (1.754 GB of weights over 78.5
GB/s of achievable bandwidth, so 66% of achievable), which is why no build
option moves it: opt level 5 gave 34.10, and an INT8 dynamic-range probe 33.83
with invalid numerics. There is no DLA on this part - `num_DLA_cores = 0`, that
is AGX Orin - so cross that off.

TGC v2 and BMRLNAP are within noise because they are the same graph: same 809
standard nodes, same exporter, TGC v2 plus one layout hint.

Not measured, and both matter: sustained thermal behaviour over a real drive, and
jitter with camerad and the rest of the stack competing for cores.

## Bench tools

Run them in this order. Each one covers what the previous cannot.

```bash
# 1. is the link fast enough. p50/p90/p99, jitter, frames over budget.
#    run from the comma, over the cable. The model is named by identity; the
#    server sends the spec back.
python3 scripts/bench_link.py --ffs --wait-host 60 --sha256 <oid> --nbytes <size> --n 1200 --rate 20

# 2. are the numbers right. TensorRT vs onnxruntime on the unmodified ONNX,
#    per output slice and per column. This is what validates any graph surgery.
#    The queues are NOT independent here (both sides use jetlink.queues);
#    tests/test_queues.py against tinygrad on the comma is the queue check.
python3 scripts/verify_parity.py capture   --sha256 <oid> --nbytes <size> --dir out --ffs   # on the comma, 32 frames; never fewer than 16
python3 scripts/verify_parity.py reference --onnx model.onnx --dir out
python3 scripts/verify_parity.py compare   --dir out

# 3. does modeld work. real segment, real warp, real modelV2 parsing.
#    lives in the fork, not here.
tools/jetlink_replay.py --segment /data/media/0/realdata/<seg> --frames 60
#    240 frames is the ceiling: the decoded frames are held in RAM and the fork
#    of modeld fails with ENOMEM above that. --dump out.npz writes per-frame
#    outputs; run it again with the toggle off for the small model on the same
#    frames and the two files are the open-loop smoothness comparison. The
#    replayed selfdriveState is forced disengaged so the swap can happen.
```

```bash
# 4. the whole thing, live. camerad and the real modeld on the bench, with a
#    fake disengaged selfdriveState so the joining state swaps. This is the
#    only one that reproduced the FunctionFS signal bug; run it for a few
#    minutes before any car test. Lives in the fork.
tools/jetlink_live_bench.sh 180
```

`jetlinkd` owns the link offroad, so stop it first or all four time out waiting
for a gadget that is already held.

The replay tool never delivered `selfdriveState`, so the joining state assumed
engaged forever and never swapped. Handled in the tool now; if it reports every
frame on the small model, check that first. It used to need the warp cache
imported before `process_replay` set `OPENPILOT_PREFIX`, because the prefix
moved `Paths.comma_home()` out from under it - that hack is gone with the cache
into the source tree.

`--spec spec.json` still works everywhere as an override; dump one from the
device param with `json.dumps(Params().get("JetlinkSpec"))`.

Correlation, not absolute tolerance, is the bar in `verify_parity`, because a
wrong head, a transposed column or a stale queue moves correlation and an
absolute tolerance waves them through. `MIN_CORR = 0.999`. What it compares is
two float16 implementations: the ONNX is float16 end to end, output tensor
included, so onnxruntime is not an FP32 truth, and the two disagree by roughly
0.005 to 0.03 absolute per head value (2026-09-05, Cinque Terre). Healthy whole
slices sit at 0.9999 and above per frame.

Columns and slices are gated on all frames pooled. Per frame, lead_prob is three
logits and every pose, euler and road_transform column is one value: the old
per-frame column gate read 1.0 on those (so they were never checked) and 0.9976
on flat plan columns (rounding noise against rounding noise), which is the
2026-09-05 "failure". The link had nothing to do with it: two captures were
bit-identical, and the plan replayed on the Jetson equalled the bytes the comma
received on every frame. Pooling needs frames: the one-value-per-frame columns
read 0.95 over 4 frames and 0.9993 over 16, so capture at least 16; the default
is 32. The synthetic inputs are bounded for that reason. The generator used to
ramp `action_t` by 0.05 s a frame, and past frame 20 the plan ran backwards at
-74 m with metre-sized disagreement between the two float16 implementations.

Before blaming the link for any numeric mismatch, run
`verify_engine.py --capture <dir>` inside the container: it replays a capture
through the server's own queues and CUDA graph and demands the bytes the comma
received, bit for bit. Identical bytes end the transport conversation; the
remainder is inference precision and belongs to the model, not the cable.

## Working on the comma

AGNOS traps, each of which has cost real time.

**Cores 4-7 are offline while parked.** `/sys/devices/system/cpu/online` reads
`0-3` offroad and `0-7` at ignition. modeld opens with
`config_realtime_process(7, 54)`, so on a bench it dies of `EINVAL` on its first
statement and leaves a zombie, with the parent blocked forever on messages that
never come. Bring them up the way hardwared does:

```python
from openpilot.common.hardware import HARDWARE
HARDWARE.set_power_save(False)
```

**process_replay gives the process a private `PARAMS_ROOT`.** Without carrying
`JetlinkEngineReady` and `JetlinkSpec` in via `custom_params`, modeld sees no
cached spec, decides no accelerator is ready and quietly runs the small model. The
run then passes every check except `modelV2.big`.

**`/tmp` is a 150 MB tmpfs.** pip fails with `No space left on device` while
`/data` has gigabytes free. Use `TMPDIR=/data/tmp`.

**No ffmpeg and no `av`.** For anything that decodes route video:

```bash
TMPDIR=/data/tmp /usr/local/venv/bin/pip install --target /data/replaydeps av
rm -rf /data/replaydeps/numpy /data/replaydeps/numpy.libs /data/replaydeps/numpy-*.dist-info
```

Delete the numpy it drags in, or it shadows the venv's pinned one on `PYTHONPATH`.

**Do not `pkill -f` over ssh.** The pattern matches your own ssh command string and
kills the session. Get the pid first, or anchor the pattern:

```bash
pgrep -f "^/usr/local/venv/bin/python3 -m openpilot.sunnypilot.accelerators.jetlink.jetlinkd$"
```

**Detach the subshell, not just the python.** `ssh comma 'cd x && setsid nohup
python ... > log & disown'` forks a subshell for the `&&` list whose stdout is
still the ssh channel, so the ssh does not return until the python exits, and
anything you time from your side is off by the daemon's lifetime. Redirect the
whole thing:

```bash
ssh comma@... 'nohup bash -c "cd /data/openpilot && exec env PYTHONPATH=/data/openpilot \
  /usr/local/venv/bin/python3 -m openpilot.sunnypilot.accelerators.jetlink.jetlinkd" \
  >> /tmp/jetlinkd.log 2>&1 < /dev/null & disown'
```

**Running the tests on the comma.** The venv has no pytest; put one under
`/data` and run both suites from the fork root. `test_queues.py` only runs here
(it needs tinygrad), and `test_onnx_patch.py` only runs off the device (it
needs `onnx`):

```bash
TMPDIR=/data/tmp /usr/local/venv/bin/pip install --target /data/tmp/pytest_deps pytest
cd /data/openpilot && PYTHONPATH=/data/openpilot:/data/jetlink_repo:/data/tmp/pytest_deps \
  /usr/local/venv/bin/python3 -m pytest -q -p no:cacheprovider /data/jetlink_repo/tests \
  openpilot/sunnypilot/accelerators openpilot/sunnypilot/models/tests/test_manager_download.py \
  openpilot/sunnypilot/modeld_v2/tests
```

These tests write to whatever params directory they find. On a PC that is
nobody's car; on the comma it is the live one, and several of them exercise
code whose job is to clear a param - `jetlinkd` dropping `JetlinkEngineReady`
when it finds the Jetson's cache gone. A suite run on 2026-09-04 cleared it and
left the device unable to use the accelerator at all: `backend.ready()` is
params-only by design, so `modeld` built no joining state, loaded the small
model and said nothing. `accelerators/conftest.py` now sets `OPENPILOT_PREFIX`
for that whole subtree; keep any new test directory under it, or under its own
conftest doing the same.

`modeld_v2/tests` is in that list because those tests import
`modeld_v2/modeld.py` unstubbed, so they are the only thing that type-checks the
seam. Leaving them out shipped a `ChestnutState` import still pointing at
`selfdrive/modeld/modeld.py` after the registry moved the class to
`accelerators/chestnut.py`. It lives back in `modeld.py` now and stays there, so
that particular move cannot recur, but the failure mode can:
`modeld_tinygrad` died at import on every ignition with a custom model selected,
and the car could not engage. Nothing recorded it. The ImportError beat sentry's
handler, so there was no crash log, no swaglog and no journal line, only
manager's `modeld_tinygrad is dead with 1` and a `Speed Error: nan m/s` on the
screen from `posenetInvalid` formatting `vEgo` against a `modelV2` that never
arrived. Anything that reads "no model at all, big or small" starts there:
run `openpilot/sunnypilot/modeld_v2/modeld.py` by hand and read the traceback.

**manager never respawns a process that exited on its own.** After stopping
`jetlinkd` by hand it stays down until manager restarts. Restarting
`comma.service` can land on the factory reset screen if the touchscreen reads taps
at boot, so prefer starting it detached during a bench session.

## Working on the Jetson

The server runs in a Docker container named `jetlink`. `/opt/jetlink` is **baked
into the image**; only `/mnt/data/jetlink`, `/dev/bus/usb` and `/sys` are mounts.
So a code change means:

```bash
rsync -rc --exclude __pycache__ jetlink/ monarch@<jetson>:/tmp/jetlink_pkg/
sudo docker cp /tmp/jetlink_pkg/. jetlink:/opt/jetlink/jetlink/
sudo docker restart jetlink
sudo docker logs --tail 40 jetlink
```

That survives a container restart, not an image rebuild. Fold anything you keep
back into `docker/Dockerfile`. A restart drops the loaded engine; the next
connect reloads it from the plan cache (6 s for a 766 MB plan, more for
Lebowski) and jetlinkd does that offroad, so modeld never pays it unless the
server restarted mid-drive.

The system python outside the container has neither tensorrt nor onnx. Run
`verify_engine.py` and anything else touching an engine with `docker exec`.

Engine cache is `/mnt/data/jetlink/{engines,models}`. Plans are keyed by TensorRT
version and GPU arch (`<oid16>.trt10.3.0.Orin-sm87.plan`) and are not portable
across either. The sidecar json records `build_seconds`, which is where
`models.json` gets its number, and `spec`, which is what the comma receives.
`prune()` keeps the six newest plans, one per registry entry, so an A/B among
them never rebuilds; the ONNX files are never pruned either.

The build workspace is sized from `MemAvailable` alone. Swap does not count:
the GPU's allocations are pinned system RAM on Tegra and cannot page, and this
Jetson's 25 GB of swap used to hand the builder the old flat 4 GB.

**The clock is unset until NTP.** There is no usable RTC across boots and no
network in the car, so `systemd-timesyncd` never runs on a drive and every
journal line is stamped from a cold-start guess. On 2026-09-04 boot -1 claimed
`17:01` to `17:08` for a boot that actually happened just before its
`Initial clock synchronization to Fri 2026-09-04 18:35:12` line, which is the
only real anchor in that boot. So `journalctl --since/--until` against comma
time finds nothing, and grepping for that sync line is the first thing to do
before trusting any Jetson timestamp. Aligning on USB connect/disconnect
events is the fallback, and for the first seconds of a boot it is the only
option, because enumeration happens before anything could set a clock.

`systemd-networkd-wait-online.service` is masked on the Jetson, on purpose.
docker.service is ordered after `network-online.target`, and in the car there
is no network, so dockerd waited out the 120 s timeout before the container
could start: measured on the 2026-09-04 drive, containerd at uptime 55 s,
dockerd at 175 s, engine ready at 181 s. Masked, the server is up about a
minute after Jetson power-on. A re-flash brings it back; check with
`systemctl is-enabled systemd-networkd-wait-online.service`.

`waiting for a jetlink gadget at 1209:0001` in the log means the comma is not
presenting. That is a comma-side or cable problem, not a server one.

The container is started by hand on the bench Jetson, not by the unit, so a
`docker rm -f` and `docker run` (needed to change mounts or arguments) has to
carry the same flags: `--restart unless-stopped --runtime nvidia
--device-cgroup-rule "c 189:* rmw"`, the three mounts in `run.sh`, and
`--transport usb --sleep-after 120`. A fresh container is the *image's* code:
`docker cp` again after recreating it.

### Suspend

With `--sleep-after`, no gadget for that long deep-suspends the Jetson, and
any USB edge wakes it with the engine still loaded (`server/sleep.py`,
`docs/transport.md`). On the bench that means: kill jetlinkd on the comma and
two minutes later the Jetson is asleep and ssh is gone, and so is the LAN: a
Jetson that answers nothing on the network with a MAC still in `arp -an` is
asleep, not broken. Starting jetlinkd again wakes it in ~6 s, but a dormant
jetlinkd releases the gadget again a minute later, so to keep the Jetson
reachable while working on it hold the gadget from the comma with a small
client that opens ffs, sends hello and pings every 10 s. Kill that with
SIGTERM: a background job from a non-interactive shell ignores SIGINT.

The freezer refuses to freeze an ssh session's processes sometimes, so a bench
attempt can fail with `EBUSY` where the car would not; the server logs
`suspend failed` and retries with backoff. The proof it slept is
`/sys/power/suspend_stats/success` moving, and the server logs `resumed after
N s asleep`. `sudo rtcwake -m no -s 600` before a bench test arms a safety
alarm in case the wake path breaks.

jetlinkd releases the gadget a minute after it has nothing to do (the fork's
`DORMANT_HOLD`), so on the bench a freshly started jetlinkd puts the Jetson
to sleep about three minutes later without anyone killing anything. It keeps
`present()` true through `/dev/shm/jetlink-dormant`; if the accelerator reads
as absent while parked (the UI icon, `accelerators.present()`), check that
marker and the pid in it first. `chestnutPresent` is not the signal any more.

**The USB wake needs the hub armed, and the hub ships disarmed.** The comma is
the gadget and hangs off the onboard Realtek hub, so a connect on a downstream
port has to be signalled up by that hub before the root hub or tegra-xusb ever
hear about it:

```
2-1   0bda:0489  speed=10000  wakeup=disabled   <- SuperSpeed hub, our path
1-2   0bda:5489  speed=480    wakeup=enabled
usb2  1d6b:0003  speed=10000  wakeup=enabled
3610000.usb (tegra-xusb)      wakeup=enabled
```

The controller and both root hubs are enabled out of the box, which is why
this looks fine at a glance. `2-1` is not, and at SuperSpeed `2-1` is the hub
we are behind. Measured 2026-09-04 with it disabled: jetlinkd released the
gadget, the Jetson slept, and presenting it again produced a bus reset with no
SET_ADDRESS - the comma's UDC sat at `default` through four connect cycles and
fifteen minutes, no LAN, no ssh, and a wake-on-LAN magic packet did nothing
either though the NIC answered ARP throughout. It took the button. With `2-1`
enabled, the same test: `suspend_stats/success` 0 to 1, 46 s asleep, and the
box resumed **4 s** after the bind.

This is also why the wake looked like it worked before: which hub carries us
depends on the negotiated speed, and at 480 Mbps it is `1-2`, which happens to
ship enabled. A cable or port that drops you to USB 2 hides the bug.

Both places that arm it are on the **host**, because `/sys` is mounted
read-only in the container and the same write from in there is a no-op:

```bash
sudo install -m 644 scripts/99-jetlink-usb-wakeup.rules /etc/udev/rules.d/
sudo install -m 755 scripts/jetlink-wake-setup.sh /usr/local/bin/
sudo udevadm control --reload && sudo udevadm trigger --subsystem-match=usb --action=add
```

The rule arms every hub at boot; the script does it again from
`jetlink-server.service`'s `ExecStartPre`, because the rule lives on the host
filesystem and a re-flash loses it exactly as it loses the masked networkd
unit - silently, and at the cost of a drive. `Sleeper._check_usb_wakeup` still
tries before each suspend and logs an error naming the rule when it cannot,
so a box that nothing can wake says so in the server log rather than just
never coming back.

`Sleeper` also arms an RTC alarm (`WAKE_BACKSTOP`, 30 min) before each sleep
and clears it after, so a wake path that fails some other way costs one period
rather than the whole park. It is set against
`/sys/class/rtc/rtc0/since_epoch`, not the wall clock, because this box boots
with an unset clock - see below. That needs rtc0's real directory bind-mounted
read-write, which `run.sh` and the unit resolve at runtime: it is a PMIC RTC
here (`nvvrs-pseq-rtc`) and a Tegra one elsewhere. With `--mount`, not `-v` -
the resolved path is `/sys/devices/platform/bpmp/bpmp:i2c/...` and `-v` reads
those colons as field separators. Without the mount the alarm is a warning in
the log and the USB edge is the only way back.

Changing mounts means `docker rm -f` and a fresh `docker run`; a `docker
restart` keeps the old ones and looks like the change did nothing.

If a Jetson is unreachable on both LAN and USB, check `arp -an` for its MAC
before assuming it is powered off: a NIC answering ARP with no ping means
suspended, not dead. `wakediag`-style checks worth keeping: `power/wakeup` on
every `/sys/bus/usb/devices/*`, and `/sys/power/suspend_stats/success` moving.

### Poweroff

`SHUTDOWN_REQ` makes the server write `/mnt/data/jetlink/poweroff`, and
`jetlink-poweroff.path` on the host acts on it (install lines are in the
unit file). The bench Jetson has `poweroff-dry-run` touched in the cache
dir, so the request only logs `jetlink-poweroff: dry run` to the journal;
remove that file to arm it, and know that a powered-off devkit needs the
button or a DC cycle. Read the unit's own journal
(`journalctl -u jetlink-poweroff.service`): `logger` does not reach journald
on this L4T image, so the script speaks through its stdout. The comma-side trigger is
`accelerators.shutdown("reason")`, which hardwared calls before DoShutdown.

The rootfs is 3.7 GB and was found full on 2026-09-04: a 2.1 GB inactive
`/swapfile` (the real swap is on `/mnt/data`) plus a 380 MB journal. The
swapfile is gone and `SystemMaxUse=150M` is in journald.conf; `df -h /`
before installing anything.

## Conventions

- Terse commit messages, upstream style, no em dashes. Never add AI or session
  attribution to commits or PRs.
- Files here get the jetlink MIT header (see any existing file). The fork's files
  get the zoompilot one.
- Comments explain why, not what. The reason a thing is the way it is usually took
  a session to find and will not be rediscovered from the code.
- `ruff check` before committing. This repo allows implicitly concatenated strings
  across lines; the fork does not, so code moving between them needs reformatting.
- Tests must run with no hardware. `tests/fake_trt.py` stubs tensorrt and cuda so
  the protocol and session logic import off a Jetson.
