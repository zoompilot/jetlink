# jetlink

Runs openpilot's large driving model on an attached Jetson. `README.md` covers what
it is and why the split works; `docs/transport.md` covers why the USB roles are
what they are; `docs/openpilot-integration.md` covers the integration design.

This file is the operational layer: the things that are not in the code, and the
ones that have already cost a session to rediscover.

**`docs/openpilot-integration.md` is stale on paths.** It says the fork modules
live in `sunnypilot/jetlink/`. They moved to `sunnypilot/accelerators/jetlink/`
when the accelerator layer landed. Everything else in it still holds.

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

A few dozen lines across eight files, and `modeld.py` gets *smaller* by 75 of them
because `ChestnutState` left it. Keep it that way: the patch being small and dull
is the whole reason another fork can lift this.

```
launch_chffrplus.sh                  31 lines -> 2 (calls accelerators/setup.sh)
openpilot/common/hardware/usb.py     deviceState.chestnutPresent
openpilot/common/params_keys.h       the params below
openpilot/selfdrive/modeld/modeld.py accel = accelerators.active()
openpilot/selfdrive/ui/...           backend-agnostic, zero jetlink references
openpilot/sunnypilot/models/helpers.py
openpilot/system/hardware/hardwared.py    the offroad alert
openpilot/system/manager/process_config.py  builds daemons from accelerators.daemons()
```

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
JetlinkEnabled                 PERSISTENT|BACKUP, BOOL        absent means "auto"
JetlinkEndpoint                PERSISTENT|BACKUP, STRING      "host:port" forces TCP instead of USB
JetlinkModel                   PERSISTENT|BACKUP, STRING      name from models.json
JetlinkEngineReady             PERSISTENT, STRING             sha256 the Jetson has built
JetlinkSpec                    PERSISTENT, JSON               the parsed model spec
```

`JetlinkEngineReady` and `JetlinkSpec` are deliberately not
`CLEAR_ON_MANAGER_START`: readiness has to survive a reboot or every ignition
cycle rebuilds a three minute engine.

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

### Open gap: a stock device cannot parse every model's spec

`onnx_meta.parse_file` tries tinygrad, then the `onnx` package. AGNOS ships
tinygrad and not `onnx`, and tinygrad rejects the `org.tinygrad` domain outright
(`ValueError: 'org.tinygrad' is not a valid Domain`). So on a stock device a model
carrying one of those nodes fails at spec time, before the Jetson is ever asked:

```
RuntimeError: could not read model metadata; need tinygrad or the onnx package.
```

It does not bite once `JetlinkSpec` is cached, because `provision()` takes its fast
path and never reopens the file. Anything that invalidates that cache puts the
device back in the retry loop.

The clean fix is to have the server return the spec over the protocol. It already
parses the ONNX at build time and has `onnx` in the container, which removes the
device-side dependency entirely, at the cost of a protocol addition. Not done.

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

Lebowski runs and is numerically correct, but 49.48 ms against a 50 ms deadline is
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
#    run from the comma, over the cable
python3 scripts/bench_link.py --ffs --spec spec.json --wait-host 60 --n 1200 --rate 20

# 2. are the numbers right. TensorRT vs onnxruntime on the unmodified ONNX,
#    per output slice. This is what validates any graph surgery.
python3 scripts/verify_parity.py capture   --spec spec.json --dir out --ffs --n 4   # on the comma
python3 scripts/verify_parity.py reference --spec spec.json --onnx model.onnx --dir out
python3 scripts/verify_parity.py compare   --spec spec.json --dir out

# 3. does modeld work. real segment, real warp, real modelV2 parsing.
#    lives in the fork, not here.
tools/jetlink_replay.py --segment /data/media/0/realdata/<seg> --frames 60
```

`jetlinkd` owns the link offroad, so stop it first or all three time out waiting
for a gadget that is already held.

Dump a spec for the first two from the device param:

```python
json.dumps(Params().get("JetlinkSpec"))
```

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

**manager never respawns a process that exited on its own.** After stopping
`jetlinkd` by hand it stays down until manager restarts. Restarting
`comma.service` can land on the factory reset screen if the touchscreen reads taps
at boot, so prefer starting it detached during a bench session.

## Working on the Jetson

The server runs in a Docker container named `jetlink`. `/opt/jetlink` is **baked
into the image**; only `/mnt/data/jetlink`, `/dev/bus/usb` and `/sys` are mounts.
So a code change means:

```bash
sudo docker cp jetlink/onnx_patch.py jetlink:/opt/jetlink/jetlink/onnx_patch.py
sudo docker restart jetlink
sudo docker logs --tail 40 jetlink
```

That survives a container restart, not an image rebuild. Fold anything you keep
back into `docker/Dockerfile`.

The system python outside the container has neither tensorrt nor onnx. Run
`verify_engine.py` and anything else touching an engine with `docker exec`.

Engine cache is `/mnt/data/jetlink/{engines,models}`. Plans are keyed by TensorRT
version and GPU arch (`<oid16>.trt10.3.0.Orin-sm87.plan`) and are not portable
across either. The sidecar json records `build_seconds`, which is where
`models.json` gets its number.

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
