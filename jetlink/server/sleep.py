"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Suspend the Jetson when nobody is talking to it.

On an always-on supply the Jetson sits at ~7 W idle, which over a long park is
a flat battery. Deep suspend keeps the loaded engine resident, so the wake
path is ~6 s to a kernel and no plan reload, against ~65 s for a cold boot.

USB is the wake source and either edge wakes it: the comma presenting the
gadget, and the comma dropping it. The second is what makes this a loop
rather than a "go to sleep" command from the comma. Ignition-off pulls the
gadget and wakes us, so the only sane policy is: awake with no gadget for
SLEEP_AFTER seconds means nobody wants us, sleep again. The same rule handles
the reverse, a mid-drive disconnect longer than the timeout, because the next
enumeration is a wake.

Writing /sys/power/state blocks until resume, so the server process simply
stops in the middle of its poll loop and continues from there. The engine
host, the CUDA context and the libusb state all survive; measured on the
bench 2026-09-04 (31.2 ms mean over 90 s after a resume, no reload).

Not every attempt sleeps. The freezer can fail (a process that will not
freeze; a bench ssh session did it) and the kernel then returns EBUSY without
sleeping, or a wake edge can land between our check and the write. Neither is
detectable from the write's return value alone, so the suspend_stats success
counter is the proof, and a failed attempt backs off rather than hammering
the freezer, which stops every process on the box for its 20 s timeout each
time it tries.
"""
from __future__ import annotations

import errno
import logging
import time
from pathlib import Path

log = logging.getLogger('jetlink.sleep')

# Longer than the gadget's re-enumeration at the jetlinkd/modeld handover,
# observed at 45 to 70 s. Sleeping inside that gap costs the next connect a
# resume.
SLEEP_AFTER = 120.0

RETRY_MIN = 10.0
RETRY_MAX = 300.0

# The kernel's own freezer timeout; an attempt cannot take longer than this
# and still have failed to freeze.
FREEZER_TIMEOUT = 20.0


def _boottime() -> float:
  # Counts through a suspend, unlike CLOCK_MONOTONIC, so the difference across
  # the write is how long we were actually asleep.
  clock = getattr(time, 'CLOCK_BOOTTIME', None)
  return time.clock_gettime(clock) if clock is not None else time.monotonic()


class Sleeper:
  """Decides when the server has been orphaned long enough to suspend.

  `touch()` whenever a gadget is seen; `idle()` from the poll loop whenever
  it is not. `idle()` returns without doing anything until the orphan timeout
  has run out, then suspends and returns once the box is back.
  """

  def __init__(self, after: float = SLEEP_AFTER, power: str | Path = '/sys/power'):
    self.after = float(after)
    self.power = Path(power)
    self.enabled = True
    self._last_seen = time.monotonic()
    self._retry_at = 0.0
    self._backoff = RETRY_MIN
    self.slept = 0
    self.failed = 0

  def touch(self) -> None:
    self._last_seen = time.monotonic()
    self._backoff = RETRY_MIN

  def orphaned(self) -> bool:
    now = time.monotonic()
    return (self.enabled and now - self._last_seen >= self.after
            and now >= self._retry_at)

  def idle(self) -> bool:
    """Called with no gadget present. Returns True if we slept."""
    if not self.orphaned():
      return False
    ok = self.suspend()
    now = time.monotonic()
    if ok:
      # Woken by an edge; give whatever caused it the full timeout to show up.
      self._last_seen = now
      self._backoff = RETRY_MIN
    else:
      self._retry_at = now + self._backoff
      self._backoff = min(self._backoff * 2, RETRY_MAX)
    return ok

  # -- sysfs -----------------------------------------------------------------

  def _read(self, name: str) -> str:
    try:
      return (self.power / name).read_text().strip()
    except OSError:
      return ''

  def _read_int(self, path: str) -> int:
    try:
      return int((self.power / path).read_text())
    except (OSError, ValueError):
      return -1

  def _select_deep(self) -> bool:
    """s2idle keeps the CPUs in idle states and saves nothing worth having."""
    modes = self._read('mem_sleep')
    if not modes:
      # No mem_sleep at all: "mem" means whatever the platform does.
      return True
    if '[deep]' in modes:
      return True
    if 'deep' not in modes.split():
      log.error("deep suspend is not available (mem_sleep: %s), not sleeping", modes)
      return False
    try:
      (self.power / 'mem_sleep').write_text('deep')
    except OSError as e:
      log.error("could not select deep suspend: %s", e)
      return False
    return True

  def suspend(self) -> bool:
    if not self._select_deep():
      self.enabled = False
      return False
    before = self._read_int('suspend_stats/success')
    t0 = _boottime()
    log.info("no gadget for %.0f s, suspending", self.after)
    try:
      self._enter()
    except OSError as e:
      if e.errno in (errno.EACCES, errno.EPERM, errno.EROFS, errno.ENOENT):
        # Configuration, not weather: /sys/power is not writable in here.
        # run.sh and the unit mount it read-write; see docs/transport.md.
        log.error("cannot write %s (%s); sleep disabled", self.power / 'state', e)
        self.enabled = False
        self.failed += 1
        return False
      # EBUSY is the freezer giving up, EINVAL a mode the platform refused.
      # Both are worth another try later.
      log.warning("suspend failed: %s (%s)", e, self._failure())
      self.failed += 1
      return False
    asleep = _boottime() - t0
    after = self._read_int('suspend_stats/success')
    if before >= 0 and after <= before:
      # The write returned cleanly but the counter did not move: we never
      # left. Happens when a wake edge lands during the freeze.
      log.warning("suspend returned after %.1f s without sleeping (%s)",
                  asleep, self._failure())
      self.failed += 1
      return False
    self.slept += 1
    log.info("resumed after %.0f s asleep", asleep)
    return True

  def _enter(self) -> None:
    # Blocks until resume. Split out so a test can stand in for the kernel.
    (self.power / 'state').write_text('mem')

  def _failure(self) -> str:
    step = self._read('suspend_stats/last_failed_step') or '?'
    dev = self._read('suspend_stats/last_failed_dev')
    return f"last failed step {step}" + (f" in {dev}" if dev else '')
