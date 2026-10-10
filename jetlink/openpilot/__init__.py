"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

jetlink on an openpilot device: everything between the fork and the comma's
device layer (jetlink.comma).

The fork implements Openpilot (interface.py) once, in an adapter module that
holds every openpilot import, binds it here (bind) and calls the methods of
the Jetlink that comes back. That is the whole contract between the two repos:
the names in __all__, the Jetlink methods and Status, which
tests/openpilot/test_api.py and test_interface.py pin. API names its version:
the adapter checks it exactly and treats any other value as jetlink being
absent, an offroad alert and no link rather than a crash. Everything else in
this package is jetlink's own and changes without notice.

API changes only for a breaking change to the contract. A new Status field, a
new Jetlink method, or a new interface member that jetlink reads with getattr
and a fallback, is additive and keeps it.

Imported by the resident gadget owner, so nothing heavy at module level: the
parts a heavy process uses are imported when it first uses them.
"""
from __future__ import annotations

import time

from jetlink.comma import gadget
from jetlink.openpilot.interface import MODES, STATES, Keys, ModelFace, Openpilot, OwnerConfig, conformance
from jetlink.openpilot.parts import Parts, for_this_process
from jetlink.openpilot.status import Status

# 2: modeld writes in_control onto the joining model before every frame, where
# jetlink polled the adapter's engagement() (2026-10-05). An adapter of API 1
# never writes it, and its large model would wait for a window forever.
# 3: the joining model asks the adapter's in_control() before every frame
# instead (2026-10-10), so modeld keeps one write; an adapter of API 2 has none
API = 3

__all__ = ['API', 'MODES', 'STATES', 'Jetlink', 'Keys', 'ModelFace', 'Openpilot', 'OwnerConfig', 'Status', 'bind',
           'conformance']

# how long a power-off request waits for the owner's run to shut the Jetson
# down before it is withdrawn. Wake from suspend is ~8 s to a server
SHUTDOWN_TIMEOUT = 25.0


def bind(op) -> Jetlink:
  """jetlink for this process, over the fork's adapter. One per process: it
  keeps the process's caches (parts.for_this_process)."""
  return Jetlink(for_this_process(op))


class Jetlink:
  """What the fork calls: one per process, from bind(). Its public methods are
  the API; the parts behind it are not."""

  def __init__(self, parts: Parts):
    self._parts = parts
    self._log = parts.log
    # prepare() said yes in this process, so attach() may join modeld
    self._prepared = False
    # what -> the last failure logged reading it; cleared by a read that works
    self._failures: dict[str, str] = {}
    # when this process asked for the Jetson to power off, until it was taken
    self._shutdown_asked: float | None = None

  def _read(self, what: str, read, fallback):
    """read(), or fallback(error) when it raises. The readers never raise:
    they are asked several times a second, from threads that have nothing to
    do with jetlink. A failure is logged once per distinct error, and again
    after a read that worked."""
    try:
      value = read()
    except Exception as e:
      error = f"{type(e).__name__}: {e}"
      if self._failures.get(what) != error:
        self._failures[what] = error
        self._log.exception("jetlink: could not read %s", what)
      return fallback(error)
    self._failures.pop(what, None)
    return value

  def _mode(self) -> str:
    """The link setting, or 'off' when it cannot be read."""
    return self._read('the link setting', lambda: self._parts.settings.mode(), lambda error: 'off')

  def enabled(self) -> bool:
    """Has the user turned the link on, with no chestnut fitted? The setting,
    never link state or readiness. A chestnut runs the big model natively and
    the link stays off beside it, so jetlinkd never takes the USB controller
    from it. manager's should_run for jetlinkd, on every device."""
    mode = self._mode()
    return self._read('whether the link is on', lambda: self._parts.enabled(mode), lambda error: False)

  def status(self) -> Status:
    """One snapshot for the UI and the panels."""
    from jetlink.openpilot import status
    mode = self._mode()
    return self._read('the status', lambda: status.read(self._parts, mode), lambda error: status.failed(error, mode))

  def model_state(self, ref: str) -> str | None:
    """One catalog model for the picker's list: 'ready' (built on the Jetson),
    'downloaded' (on the comma) or None. Asked when a list opens."""
    from jetlink.openpilot import status
    return self._read('a model\'s state', lambda: status.model_state(self._parts, ref), lambda error: None)

  def reason(self) -> str | None:
    """Why the link the user asked for cannot run: hardwared's offroad alert.
    None with the link off. The files the gadget and the build leave, nothing
    else: hardwared asks twice a second on every device."""
    from jetlink.openpilot import status
    mode = self._mode()
    return self._read('why the link cannot run', lambda: status.reason(self._parts, mode),
                      lambda error: status.failure(error, mode))

  def prepare(self) -> bool:
    """Will the link join this modeld? modeld only, before it goes realtime:
    enabled(), then the process-wide setup that has to happen before then,
    which is also a last veto."""
    from jetlink.openpilot.status import NO_WARP
    from jetlink.openpilot.warp import init_device
    self._prepared = False
    if not self.enabled():
      return False
    # the link is not worth waiting for: attach() joins in the background.
    # enabled() is the setting alone, so this is where a device that cannot
    # present a gadget at all says so; nothing here would ever reach a Jetson
    if not gadget.link_configured():
      self._log.warning("jetlink: no usable gadget (%s), staying on the small model",
                        gadget.gadget_error() or 'not set up')
      return False
    # the warp is a build product and nothing compiles one at runtime, so one
    # missing now stays missing, and saying no keeps modeld on the plain small
    # model; the offroad alert has said why
    if not self._parts.warps.built():
      self._log.warning("jetlink: %s, staying on the small model", NO_WARP)
      return False
    # the last hook before modeld goes SCHED_FIFO on core 7, and the GPU's init
    # spawns a thread that would inherit that. See warp.init_device
    init_device(self._log)
    self._prepared = True
    return True

  def attach(self, small, cam_w: int, cam_h: int):
    """Join the link to modeld, once the camera is up and `small` is built:
    the joining model, which drives as `small` until the Jetson is there.
    Before every run() it asks the adapter's in_control() (is openpilot or
    MADS in control? it swaps only while not), and modeld writes onto it
    `frame_drop_ratio` (its share of dropped camera frames, which hands a
    lagging large model back).

    None unless prepare() said yes in this process: without it the GPU's
    thread would start on modeld's realtime core. If the joining model cannot
    be built, `small`, and the failure is logged.
    """
    if not self._prepared:
      return None
    from jetlink.openpilot.joining import join
    try:
      return join(self._parts, cam_w, cam_h, small)
    except Exception:
      self._log.exception("jetlink load failed")
      return small

  def request_shutdown(self, reason: str = '') -> bool:
    """The device is powering off for good: ask for the Jetson to go down
    with it, and return at once. True when the request now waits for the
    owner, which shutdown_pending() follows; False when there was nothing to
    ask, with the link off, beside a chestnut, or with no Jetson known to be
    there. Never raises.

    hardwared calls this once and goes on publishing deviceState, putting
    DoShutdown once shutdown_pending() clears or SHUTDOWN_TIMEOUT has passed.
    The owner wakes a sleeping Jetson and starts a run that asks it; the
    wake and one round trip take ~10 s.
    """
    try:
      return self._ask_for_power_off(reason)
    except Exception:
      self._log.exception("jetlink: shutdown request failed")
      return False

  def shutdown_pending(self) -> bool:
    """Has the owner still to take the power-off request? A stat, for a
    caller's loop. Never raises.

    Once SHUTDOWN_TIMEOUT has passed since this process asked, the request is
    withdrawn and False: a comma that outlives its DoShutdown must not have
    the Jetson powered off later, when nobody wants it off, and the log says
    the request went unanswered."""
    try:
      pending = gadget.SHUTDOWN_REQUEST.exists()
    except Exception:
      return False
    asked, now = self._shutdown_asked, time.monotonic()
    if asked is None:
      return pending
    if not pending:
      # the run removes it whether or not the Jetson answered; its log says which
      self._log.warning("jetlink: the owner took the shutdown request after %.1f s", now - asked)
    elif now - asked >= SHUTDOWN_TIMEOUT:
      self._log.warning("jetlink: nobody took the shutdown request within %.0f s", SHUTDOWN_TIMEOUT)
      gadget.finish_shutdown()
      pending = False
    if not pending:
      self._shutdown_asked = None
    return pending

  def _ask_for_power_off(self, reason: str) -> bool:
    """Hand the request to the owner: nothing else can touch the link, and
    the request is a file (gadget.SHUTDOWN_REQUEST) the owner's step looks
    for before anything else. Skipped when no Jetson is known to be there
    (dormant counts as there). A run busy in a long provision is stopped for
    it; shutdown_pending()'s deadline covers a Jetson that never answers."""
    from jetlink.openpilot.status import Presence
    if not self.enabled():
      return False
    # a presence of its own, off the files: one this process's readers keep
    # refreshed, or the owner's with its hold, would count a Jetson seen
    # seconds before the power-off, and the run would wait 20 s for a host
    # that has gone
    if not Presence().present():
      return False
    self._log.warning("jetlink: asking the jetson to power off: %s", reason)
    if not gadget.request_shutdown(reason):
      return False
    self._shutdown_asked = time.monotonic()
    return True

  def should_extend_catalog(self) -> bool:
    """Should the model manager's big-model catalog carry the newer catalogs'
    models? With no chestnut fitted. Hardware, not the link setting: the model
    manager drops a pick its catalog does not list."""
    return not self._parts.chestnut_fitted()

  def extend_catalog(self, catalog: dict) -> dict:
    """The big-model catalog the model manager fetched, with the models newer
    catalogs list folded in. Never raises."""
    return self._parts.models.big_catalog(catalog)
