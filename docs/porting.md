# Add Jetlink to an openpilot fork

This page is for whoever keeps an openpilot fork. It lists every line a fork
adds so the comma can run the large driving model over Jetlink. The snippets
are for stock openpilot's files (the 0.11 layout); move them to wherever your
fork keeps the same code.

Jetlink never imports openpilot. Your fork has one adapter module that does,
and every hook line calls Jetlink through it.

## Get the code

Add Jetlink as a submodule at a release tag (replace `vX.Y.Z`), or vendor a
copy. The comma needs `jetlink/`, `LICENSE`, `pyproject.toml` and `README.md`
and nothing else; a release build that filters files must keep them.

```bash
git submodule add https://github.com/zoompilot/jetlink.git jetlink_repo
git -C jetlink_repo checkout vX.Y.Z
```

Make `import jetlink` work with a link beside the tinygrad and opendbc ones in
`launch_chffrplus.sh`, and ignore it in `.gitignore` (`/jetlink`):

```bash
ln -sfn jetlink_repo/jetlink jetlink
```

Right after it, cut the comma's draw from the USB-C port while the link is on,
so the comma never runs off the host's port from boot (the owner only starts
about 20 s later). With the link on, the comma must be powered from its own
power port (the car harness, or the comma four's power port), never this one:

```bash
sudo -n bash jetlink/comma/jetlink-root.sh boot /data/params/d/JetlinkLink &   # on AGNOS only
```

## Params

Add these to `common/params_keys.h` (`IsOffroad` is already there). A fork
that ships a prebuilt params library must rebuild it.

```cpp
    {"JetlinkLink", {PERSISTENT | BACKUP, INT, "0"}},
    {"JetlinkProgress", {CLEAR_ON_MANAGER_START, JSON}},
    {"JetlinkSpec", {PERSISTENT, JSON}},
    {"JetlinkChargePhone", {PERSISTENT, BOOL, "0"}},
    {"Offroad_JetlinkUnavailable", {CLEAR_ON_MANAGER_START, JSON}},
```

`JetlinkLink` is the setting: 0 off, 1 USB (a Jetson, PC, Mac or Android
phone), 2 iOS (an iPhone or iPad). `JetlinkSpec` must survive a reboot, or
every drive rebuilds the model for minutes. These names are `Stock.KEYS`; a
fork that names them differently sets its own `KEYS` in the adapter.

The offroad alert also needs a line in `selfdrive/selfdrived/alerts_offroad.json`:

```json
  "Offroad_JetlinkUnavailable": {"text": "Jetlink unavailable: %1", "severity": 0},
```

## The adapter module

Put this in `openpilot/jetlink_adapter.py`. manager imports it to build its
process list and runs it as the gadget owner, so its top level imports the
standard library only. `Stock` covers stock openpilot's paths; override only
what your fork does differently.

```python
import functools
import os

API = 3              # the jetlink.openpilot API this adapter is written to
OWNER = 'jetlinkd'   # manager's name for the gadget owner

def _adapter_class():
  from jetlink.openpilot.stock import Stock

  class Adapter(Stock):
    """Stock openpilot. Override KEYS, IN_CONTROL_SERVICES and engaged() here if yours differ."""

  return Adapter

def adapter():
  return _adapter_class()()

def owner_config():
  return _adapter_class().owner_config()

def main():
  _adapter_class().main()

def link_on() -> bool:
  """The Jetlink setting, read as a file, so it works with no jetlink at all."""
  prefix = os.environ.get('OPENPILOT_PREFIX', '')
  root = os.environ.get('PARAMS_ROOT') or ('/data/params' if os.path.isfile('/AGNOS') else
                                           os.path.expanduser(f'~/.comma{prefix}/params'))
  try:
    with open(os.path.join(root, os.environ.get('OPENPILOT_PREFIX', 'd'), 'JetlinkLink')) as f:
      return f.read().strip() not in ('', '0')
  except OSError:
    return False

def _cloudlog():
  from openpilot.common.swaglog import cloudlog
  return cloudlog

@functools.cache
def hooks():
  """What every hook line calls, one per process. Never raises."""
  try:
    from jetlink.openpilot.hooks import Hooks
    return Hooks(API, adapter, link_on=link_on, log=_cloudlog)
  except Exception as e:
    try:
      from jetlink.openpilot.hooks import Off
    except Exception:
      return _NoJetlink(f"jetlink is missing: {e}")
    return Off(str(e), link_on=link_on)

def should_run(started, params, CP) -> bool:
  return hooks().should_run(started, params, CP)
```

When Jetlink fails, every hook answers as if the link were off, so a Jetlink
bug never takes manager, modeld, hardwared or the UI down. `Hooks` raises for
an API it does not accept, and `Off` gives the reason as the offroad alert to
someone who turned the link on. With `link_on()` saying the link is off,
only the UI's `status()`, hardwared's shutdown hooks and the model manager's
catalog hooks bind Jetlink to your adapter (`BINDS_WITH_LINK_OFF` in
`jetlink/openpilot/hooks.py`); every other hook answers as the link off
without it, modeld's `prepare()` among them.
The snippets below call the result `HOOKS`:
`HOOKS = jetlink_adapter.hooks()`. A checkout that is missing or too old has
no `Off`, so it needs your own fallback. It only answers the hooks a link
that never runs reaches; `attach()` waits for a yes from `prepare()`:

```python
class _NoJetlink:
  """The hooks with no usable jetlink checkout: the link off on every one."""
  def __init__(self, why): self._why = why
  def should_run(self, started, params, CP): return False
  def status(self): return None
  def reason(self): return self._why if link_on() else None
  def prepare(self, chestnut): return False
  def handed_over(self, model): return False
  def state(self, model): return 'none'
  def shutdown_step(self, should_shutdown, reason=''): return should_shutdown
  def runtime(self): return None
  def swap_events(self, tick_s): return None
```

A fork with sunnypilot's model manager also sets `KEYS.big_model` and
`KEYS.catalog` (the model library and the user's choice). Where it fetches the
big-model library it asks `lists_big_models()`, and when that is true (no
chestnut fitted) takes `big_catalog(selector)`, with `selector` its
`REQUIRED_JSON_VERSION`: the model index as a catalog in sunnypilot's shape, in place
of sunnypilot's own; `None` is a fetch that failed, and it keeps the list it
has. A pick a new list no longer names carries over to `listed_ref(ref)`, the
ref the list names the same ONNX under. Without them Jetlink runs the model
chosen on the Jetson or in an app, else its default big model.

A fork with a model list of its own that knows each model by its ONNX's
sha256, not by a comma commit, sets only `KEYS.big_model`, a JSON param it
writes `{"sha256": ..., "displayName": ...}` into when the user chooses (and
removes for no choice). Jetlink sizes the model from the model index. A newer choice made on the Jetson or in an app is written back into
that param, with the model's `sha256` and `ref` where it has them, so the
fork's list can show what drives. The param's write time is when the choice
was made: write it only when the user chooses.

## manager

Upstream's `PythonProcess` never starts a process again once it has died. The
owner needs one that does. In `system/manager/process.py`:

```python
class RestartingPythonProcess(PythonProcess):
  """Starts its process again after it dies; upstream's start() leaves a dead one dead."""
  def start(self) -> None:
    if self.proc is not None and self.proc.exitcode is not None:
      self.stop()   # reaps it, so the start below runs it again
    super().start()
```

In `system/manager/process_config.py`:

```python
  RestartingPythonProcess(jetlink_adapter.OWNER, "openpilot.jetlink_adapter", jetlink_adapter.should_run),
```

A dead owner sends the drive back to the small model; it must not block
engaging. In `selfdrive/selfdrived/selfdrived.py`, leave it out of
`processNotRunning`:

```python
    not_running = {p.name for p in self.sm['managerState'].processes
                   if not p.running and p.shouldBeRunning} - {jetlink_adapter.OWNER}
```

## modeld

In `selfdrive/modeld/modeld.py`'s `main()`, and in every other modeld your
fork has. First the chestnut test: a Jetson or an iPhone on the USB-C port
looks like a chestnut to `cable_connected()`, so count the cable only while
Jetlink is off:

```python
  HOOKS = jetlink_adapter.hooks()
  CHESTNUT = chestnut_compiled() and (chestnut_present() or (cable_connected() and not jetlink_adapter.link_on()))
```

Then, once `small_model` is built:

```python
  # before going realtime: the GPU thread it starts would inherit SCHED_FIFO 54 on core 7
  if HOOKS.prepare(CHESTNUT):
    # the joining model drives as small_model until the big model is ready
    model = HOOKS.attach(model, small_model, vipc_client_main.width, vipc_client_main.height)
  config_realtime_process(7, 54)
```

In the frame loop. Upstream never writes `frame_drop_ratio` onto the model,
and without it a lagging big model is never handed back. A swap or a
hand-back stalls a frame or two, so those drops restart the filter's warmup:

```python
    frame_drop_ratio = frames_dropped / (1 + frames_dropped)
    model.frame_drop_ratio = frame_drop_ratio   # before model.run()

    if HOOKS.handed_over(model):                # after model.run()
      run_count = 0
```

`HOOKS.state(model)` is the big model's state, one of `STATES`. Publish it if
your fork has a message for it (zoompilot: `modelDataV2SP.acceleratorState`);
without one, `HOOKS.runtime()` carries it.

## selfdrived

The big model joins mid-drive and can leave again. The swap events tell the
driver and hold the errors a switch causes. Build the tracker once, link on
or off; it binds nothing, and your fallback for no jetlink answers `None`:

```python
    self.jetlink_swaps = HOOKS.swap_events(DT_CTRL)   # in __init__
```

```python
    if self.jetlink_swaps is not None:   # in update_events()
      model_ok = self.sm.alive['modelV2'] and self.sm.valid['modelV2']
      # in control: the same rule as your adapter's engaged() (zoompilot: self.enabled or self.mads.enabled)
      flags = self.jetlink_swaps.update(model_ok, model_ok and self.sm['modelV2'].big, self.enabled)
      if flags.no_entry:
        self.events.add(EventName.bigModelLoading)   # no engaging for the second after a swap
```

The third argument must be the same rule as your adapter's `engaged()`, the
one the swap itself waits on: openpilot engaged, or, in a fork with a
lateral-only mode, that mode armed or paused (zoompilot passes
`self.enabled or self.mads.enabled`, MADS paused included). The tracker
reads the big model's state from modeld's runtime record a few times a
second. A fork that publishes the state on a message of its own passes it as
`state=`, with the first argument saying both messages are alive and valid.

`offer`, `link_lost` and `chime` need events of your own; `jetlink/openpilot/events.py`
says what each means. `settling` is a switch's dropped camera frame, which
would raise a comm or localizer error. Hold them as for a native big model's
load:

```python
    big_model_settling = self.big_model_loading or time.monotonic() < self.big_model_ready_t + warmup_sec \
      or (self.jetlink_swaps is not None and self.jetlink_swaps.settling)
```

If your fork chimes on a rising `modelV2.big`, gate that chime on the state
being `none`, or drop it on `no_entry`.

## hardwared

A Jetson that is always on should power off with the comma. Ask on
DoShutdown only, never on a reboot: a Jetson powered off at a comma reboot
does not come back. In `system/hardware/hardwared.py`, with
`shutdown_since = None` before the loop and `JETLINK_SHUTDOWN_CAP_S = 30.` at
the top (Jetlink's `SHUTDOWN_TIMEOUT` of 25 s plus 5 s, written out so it holds
without Jetlink):

```python
    should_shutdown = power_monitor.should_shutdown(onroad_conditions["ignition"], in_car, off_ts, started_seen)
    if should_shutdown and shutdown_since is None:
      shutdown_since = time.monotonic()
    capped = shutdown_since is not None and time.monotonic() - shutdown_since > JETLINK_SHUTDOWN_CAP_S
    if HOOKS.shutdown_step(should_shutdown, f"offroad since {off_ts}") or capped:
      cloudlog.warning(f"shutting device down, offroad since {off_ts}")
      params.put_bool("DoShutdown", True, block=True)
```

`shutdown_step()` says yes on any failure; the cap is your own guarantee on
top. Block a drive start while it waits:
`startup_conditions["not_powering_off"] = shutdown_since is None`.

The offroad alert, in the same loop:

```python
    reason = HOOKS.reason()   # why the link the user turned on cannot run; None with it off
    set_offroad_alert_if_changed("Offroad_JetlinkUnavailable", reason is not None, extra_text=reason)
```

## The UI

`HOOKS.status()` is one snapshot, or `None` when Jetlink cannot run at all.
It is files and params, so the UI can ask five times a second.

```python
    status = HOOKS.status()
    if status is not None and status.shown:
      chip = status.phase                          # 'off', 'preparing', 'active', ...
      line = (status.progress or {}).get('msg')    # the English line
      port = status.port_state                     # 'connected', 'empty', 'device' or None
      adb_allowed = not status.holds_port          # the link holds the USB port
```

`phase` is one of `PHASES` in `jetlink/openpilot/status.py`, which only grows:
map one you do not know to your default. The English is in `msg` and
`reason`; a fork with its own words reads `code`, `args` and
`status.reason_code` (`PROGRESS_CODES`, `REASON_CODES`).
`status.icon(started, model_seen, running_big, state)` is the chestnut icon's.

## The fork's tests

Jetlink publishes the checks it holds a fork to (`jetlink/openpilot/testing.py`).
Run `check_seam` on every modeld, and `check_fallback` on your own fallback
with every hook your lines call, under a params store of the test's own
(`PARAMS_ROOT` or `OPENPILOT_PREFIX`), since `link_on()` reads the setting.

```python
from jetlink.openpilot.testing import check_adapter, check_fallback, check_seam

def test_the_adapter():
  assert check_adapter(jetlink_adapter, heavy=('openpilot.common.params', 'openpilot.common.swaglog', 'openpilot.cereal')) == []

def test_the_modeld_seam():
  assert check_seam('openpilot/selfdrive/modeld/modeld.py') == []

def test_the_fallback():
  hooks = ('should_run', 'status', 'reason', 'prepare', 'handed_over', 'state', 'shutdown_step', 'runtime',
           'swap_events')
  assert check_fallback(jetlink_adapter._NoJetlink('jetlink is missing'), hooks, link_on=jetlink_adapter.link_on) == []
```

Install the guard in your `conftest.py`, so no test reaches the device's own
params, and run every test that touches params under its own `OPENPILOT_PREFIX`:

```python
from jetlink.openpilot.testing import install_guard

GUARD = install_guard()

@pytest.fixture(autouse=True)
def no_machine_files():
  GUARD.start()
  yield
  assert GUARD.stop() == []
```

`FakeJoinedModel` and `RecordingLog` in the same module stand in for the
joining model and cloudlog in your modeld tests, and `FakeSubMaster` for the
messages your `engaged()` reads.

## Versions

Pin a release tag. `API` in `jetlink/openpilot/__init__.py` changes only for a
breaking change, and keeps the previous one working for one minor release, so
you can move the pin, then the adapter. A hook added in a minor release is
not in an older checkout, and calling it there raises: if your lines call
one, pin that release or newer. Read the **Forks** group of the
[CHANGELOG](../CHANGELOG.md) before you move the pin.
