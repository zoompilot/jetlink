# jetlink

Runs openpilot's large driving model on an attached Jetson. `README.md` covers what
it is and why the split works; `docs/transport.md` covers why the USB roles are
what they are; `docs/openpilot-integration.md` covers the integration design.

This file is the operational layer: the things that are not in the code, and the
ones that have already cost a session to rediscover.

## The openpilot side

Lives in the fork (`sunnypilot-jetson-trt` worktree), not here. jetlink is a plain
Python package the fork imports; it knows nothing about openpilot.

### Three layers, and why

```
openpilot/sunnypilot/accelerators/
  base.py          the Accelerator protocol and the Daemon description
  __init__.py      discovery, guards, the progress param
  chestnut.py      comma's AMD board (ChestnutState moved here from modeld)
  jetlink/         this project's backend
    backend.py     the only class core openpilot touches
    helpers.py     where the model is, whether a Jetson is attached
    jetlinkd.py    the offroad daemon
    model_state.py what modeld drives per frame
    warp_cache.py  the comma-side warp JIT: capture, load, warm
    compile_warp.py  its CLI, invoked by accelerators/SConscript
    state.py       publishes chestnutState
    spec_cache.py  the model spec, cached in a param
    lfs.py         fetching a model out of comma's LFS
    models.json    the model registry
```

openpilot already had an accelerator abstraction with one implementation and no
name. `sources/` and `runners/` were both taken (`source` = which model catalog,
`runner` = snpe/tinygrad/stock), so the axis got called `accelerators`.

Keep the seam at `backend.py`. Anything only `jetlinkd` needs goes in `helpers`
or `spec_cache`, never in the backend. That is what keeps the upstream patch to
a few dozen lines and liftable by another fork.

Backends are imported lazily on the first `backends()` call. `ImportError` is
swallowed (a fork that does not ship this backend), anything else is logged.
Every question goes through `_ask()`, so a broken backend cannot take down
hardwared or the UI.

### Upstream files touched

A couple of hundred lines across a dozen or so files, and `modeld.py` gets
*smaller* because `ChestnutState` left it. Keep it that way: the patch being
small and dull is the whole reason another fork can lift this. Count it, do not
guess it:

```bash
git diff --numstat $(git merge-base HEAD zoom/develop)..HEAD -- . ':!tools' ':!release/ci' \
  ':!openpilot/sunnypilot/accelerators' ':!*/tests/*'
```

```
launch_chffrplus.sh                  +2 (calls accelerators/setup.sh)
openpilot/common/hardware/usb.py     deviceState.chestnutPresent
openpilot/common/params_keys.h       the params below
openpilot/selfdrive/modeld/modeld.py accel = accelerators.active(); small model first
openpilot/selfdrive/ui/...           backend-agnostic, zero jetlink references
openpilot/sunnypilot/models/...      the catalog comes from accelerators.catalog()
openpilot/system/hardware/hardwared.py    the offroad alert
openpilot/system/manager/process_config.py  builds daemons from accelerators.daemons()
```

modeld loads the small model on the main thread *before* starting the big
model's loader thread, and hands it to `make_model_state(cam_w, cam_h, small)`.
That used to be so the backend could borrow its warp; upstream has since fused
warp and policy into one `run_model` JIT, so there is no warp in the pkl to
borrow and `small` is only a geometry cross-check now. A big model that
finishes loading after `BIG_MODEL_TIMEOUT` is closed, not kept.

`prepare()` returns a bool and may veto. `active()` has to stay cheap because
the UI polls it, so the checks that block - upstream waits a few deviceState
ticks for a chestnutState saying the board's PCIe link is actually trained -
live in `prepare()`, which only modeld calls.

### The warp is a build product, and scons builds it

jetlink runs `warp` on the comma and `run_policy` on the Jetson. Upstream used
to ship the warp as its own JIT inside the small model's pkl, so borrowing it
cost nothing. `commaai/openpilot#38684` fused the two into a single `run_model`
graph with no seam, and `WARP_INPUTS`, `POLICY_INPUTS`, `make_warp_input_queues`
and the `WARP_DEV`/`QUEUE_DEV` split went with it.

`make_warp` still exists and still builds the same closure, so the wire format
did not move. It just has to be JIT-compiled somewhere, and that somewhere is
not modeld: the compile would land on the loader thread inside the 60 s
`BIG_MODEL_TIMEOUT`, on the one GPU the main thread is already using, every
ignition.

So it is a scons target, `openpilot/sunnypilot/accelerators/SConscript`, wired
in through the one-line `sunnypilot/SConscript` dispatcher and gated on
`arch == comma_arm64` and the jetlink package being installed. That is what
upstream does with `dm_warp_*.pkl` a few lines away in `modeld/SConscript`, and
it is the whole point: `launch_chffrplus.sh` runs `setup.sh` (line 83, which
makes the `jetlink` symlink) and then `build.py` (line 96), so an update that
moves tinygrad has a rebuilt warp before manager starts, never mind before
ignition. The whole `tinygrad_repo` glob is in the dependency list, so a
submodule bump rebuilds it.

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

`present()` drives `chestnutPresent` and holds for 5 s after the UDC last read
"configured" (selfdrived soft-disables on it dropping). `ready()` is what the
UI calls "compiled". `catalog()` says which model-manager catalog the attached
accelerator draws from: "chestnut" for comma's board, None for jetlink, whose
models come from `models.json`. Every bundle in the chestnut catalog is a
tinygrad pkl for the comma's own GPU, so a Jetson device must never see one
become active: it would route modeld to `modeld_tinygrad` on the small model.

Two import cycles are already avoided on purpose, do not undo them:

- `common/hardware/usb.py` imports `accelerators` inside the function, because
  `modeld.helpers` imports `usb.py`.
- `chestnut.py` imports `tinygrad.device` inside `power_limit` and `send`, not at
  module scope. hardwared, the UI and the model manager all ask whether a board is
  fitted; none of them should pay for tinygrad to answer.
- Backends never import `process_config`. `Daemon` is a description, and manager
  owns the onroad gating.

### The UI identifies a chestnut by USB id; we are the gadget

Upstream's `ui: show usb connection` (#38745) decides `usb_unknown` by looking
for a chestnut USB id among the devices the comma enumerated as a host. The
comma is jetlink's gadget and enumerates nothing, so that check showed the
generic USB icon instead of the accelerator icon. `ui_state.py` now also
accepts `deviceState.chestnutPresent`. Expect this again: anything upstream
adds that recognises the board by USB id needs the same `chestnut_present`
guard. Offroad, an accelerator that is provisioning reads as LOADING from the
progress param, not UNCOMPILED, and only a `failed` stage reads as FAILED.

### Cereal keeps comma's names

`deviceState.chestnutPresent` and `chestnutState` are unchanged. Renaming cereal
fields breaks log compatibility, so on the wire "chestnut" means "whatever
accelerator is active". `state.py` is the only file that knows openpilot's schema.

### Params

```
AcceleratorProgress            CLEAR_ON_MANAGER_START, JSON   provisioning progress for the UI
Offroad_AcceleratorUnavailable CLEAR_ON_MANAGER_START, JSON   the offroad alert
JetlinkEnabled                 PERSISTENT|BACKUP, BOOL        absent means "auto"; the
                                                              "accelerator link" toggle in
                                                              the mici models panel writes it
JetlinkEndpoint                PERSISTENT|BACKUP, STRING      "host:port" forces TCP instead of USB
JetlinkModel                   PERSISTENT|BACKUP, STRING      name from models.json
JetlinkEngineReady             PERSISTENT, STRING             sha256 the Jetson has built
JetlinkSpec                    PERSISTENT, JSON               the parsed model spec
```

`JetlinkEngineReady` and `JetlinkSpec` are deliberately not
`CLEAR_ON_MANAGER_START`: readiness has to survive a reboot or every ignition
cycle rebuilds a three minute engine. Neither is trusted blindly: jetlinkd
re-asks the server once per attach (the Jetson's cache can be pruned,
re-flashed or swapped under the param), and modeld clears
`JetlinkEngineReady` if the server answers `need_upload`, so the next parked
period re-provisions instead of every drive failing at connect.

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
handle it, but re-enumeration has been observed at 45 to 70 seconds, against
`backend.CONNECT_TIMEOUT = 45.0` and modeld's 60 s big-model timeout. That margin
is thin and has not been proven onroad.

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
state's constructor, on the loader thread while modeld's main thread is
blocked in `loader.join`, and the swap sends no warmup frame; the first real
frame carries the reset.

The engine does not reload at the handover. `server/session.py`'s `EngineHost`
owns the one loaded engine and the one build in flight for the life of the
process; a `Session` is a view onto it. Before that, every reconnect freed the
engine and the next connect paid 13 to 25 s to deserialize it, out of the same
60 s. A client that reconnects during a build attaches to it. Only one engine
is ever resident: a build or a load of a different model unloads the current
one first.

### Frame semantics: what chestnut does

There is no per-frame deadline. `infer_end` blocks for the frame the way modeld
blocks on a chestnut; a frame past 50 ms is a dropped camera frame, which
modeld counts and tolerates. Only a stall past `client.FRAME_TIMEOUT` (3 s, the
analogue of chestnut's `HCQDEV_WAIT_TIMEOUT_MS`) is a failure, and then it is
a `LinkError`: the link is done and modeld's one-way fallback to the small
model is the right place to be. An earlier design had a 35 ms deadline with a
non-latching timeout; modeld's `except Exception` made every one of those a
permanent fallback anyway.

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
with the same seq, and a 3 s frame timeout. It never showed on the bench with
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

`modelV2.big` is the signal. `JetlinkModelState` sets it; the small-model fallback
does not, and the fallback is one-way and silent. Also:

```
deviceState.chestnutPresent    true
chestnutState                  valid, with live tempC / powerDrawW / gpuUsagePercent
                               and pcieLtssm == 0x78
swaglog                        "jetlink: Orin-sm87 trt 10.3.0, engine ..."
```

A plan that stops at ~5 m instead of ~200 m is the other tell. That is what a
silent fallback looked like when it happened.

Before blaming the link, check which modeld is even running. jetlink lives in
stock `modeld`, and manager runs that only while `get_active_model_runner()` is
`stock`, which means no bundle in `ModelManager_ActiveBundle`. Every bundle the
sunnypilot model manager offers has `runner = tinygrad`, so picking any custom
model in the UI moves manager to `modeld_tinygrad` (`sunnypilot/modeld_v2`),
which knows nothing about jetlink: the Jetson provisions, `ready()` is true, the
UI says compiled, and `modelV2.big` is never set because the process that would
set it is not running. `ModelRunnerTypeCache` caches the answer, so clear both
params together. This is the same hazard as the chestnut catalog above, one slot
over.

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
python3 scripts/verify_parity.py capture   --sha256 <oid> --nbytes <size> --dir out --ffs --n 4   # on the comma
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

Correlation, not absolute tolerance, is the bar in `verify_parity`: FP16 against
FP32 on a 40 layer network never matches exactly, but correlation moves the moment
a head is wired up wrong, transposed or fed a stale queue. `MIN_CORR = 0.999`.
Healthy runs sit at 0.99997 and above on every slice.

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

`modeld_v2/tests` is in that list because the accelerators work moves code out
from under sunnypilot's own model runner, and those tests import
`modeld_v2/modeld.py` unstubbed, so they are the only thing that type-checks the
seam. Leaving them out shipped a `ChestnutState` import still pointing at
`selfdrive/modeld/modeld.py` after it moved to `accelerators/chestnut.py`:
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
two minutes later the Jetson is asleep and ssh is gone. Starting jetlinkd
again wakes it in ~6 s. The freezer refuses to freeze an ssh session's
processes sometimes, so a bench attempt can fail with `EBUSY` where the car
would not; the server logs `suspend failed` and retries with backoff. The
proof it slept is `/sys/power/suspend_stats/success` moving, and the server
logs `resumed after N s asleep`. `sudo rtcwake -m no -s 600` before a bench
test arms a safety alarm in case the wake path breaks.

jetlinkd releases the gadget a minute after it has nothing to do (the fork's
`DORMANT_HOLD`), so on the bench a freshly started jetlinkd puts the Jetson
to sleep about three minutes later without anyone killing anything. It keeps
`present()` true through `/dev/shm/jetlink-dormant`; if chestnutPresent
drops while parked, check that marker and the pid in it first.

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
