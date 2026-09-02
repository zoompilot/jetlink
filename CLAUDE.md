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
The jetlink backend borrows its warp instead of loading the same pkl again from
a thread the main thread is racing on the same tinygrad device. A big model
that finishes loading after `BIG_MODEL_TIMEOUT` is closed, not kept.

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
```

`jetlinkd` owns the link offroad, so stop it first or all three time out waiting
for a gadget that is already held.

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
  openpilot/sunnypilot/accelerators openpilot/sunnypilot/models/tests/test_manager_download.py
```

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
`prune()` keeps the two newest plans and the registry has five entries, so
switching among three models rebuilds; the ONNX files are never pruned.

The build workspace is sized from `MemAvailable` alone. Swap does not count:
the GPU's allocations are pinned system RAM on Tegra and cannot page, and this
Jetson's 25 GB of swap used to hand the builder the old flat 4 GB.

`waiting for a jetlink gadget at 1209:0001` in the log means the comma is not
presenting. That is a comma-side or cable problem, not a server one.

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
