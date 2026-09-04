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

# How long a sleep may last with nothing else waking us. USB is the wake
# source and it is not guaranteed: on 2026-09-04 the comma presented the
# gadget to a sleeping Jetson and got a bus reset and no enumeration, held
# there through four connect cycles, and neither a second bind nor a
# wake-on-LAN magic packet brought it back - the box needed its button. In
# the car that is a whole drive on the small model with no way to recover,
# because the only thing that could ask is the comma and it is already
# asking. An RTC alarm is the one wake source that does not depend on the
# path that just failed, so arm one before every sleep and bound the outage
# to this. The cost is a wake and SLEEP_AFTER awake per period, ~7 W for
# 120 s in 30 min, well under a tenth of a watt averaged.
WAKE_BACKSTOP = 1800.0
RTC = '/sys/class/rtc/rtc0'

# Where the hubs live, and the class code that says a USB device is one.
USB_DEVICES = '/sys/bus/usb/devices'
HUB_CLASS = '09'


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

  def __init__(self, after: float = SLEEP_AFTER, power: str | Path = '/sys/power',
               rtc: str | Path = RTC, backstop: float = WAKE_BACKSTOP,
               usb: str | Path = USB_DEVICES):
    self.after = float(after)
    self.power = Path(power)
    self.rtc = Path(rtc)
    self.backstop = float(backstop)
    self.usb = Path(usb)
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
    self._check_usb_wakeup()
    armed = self._arm_backstop()
    try:
      self._enter()
    except OSError as e:
      self._disarm_backstop(armed)
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
    self._disarm_backstop(armed)
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

  def _check_usb_wakeup(self) -> list[str]:
    """Refuse to sleep quietly if nothing can wake us.

    The comma is the gadget and hangs off the onboard Realtek hub, so a
    connect on a downstream port has to be signalled up by that hub before the
    root hub or tegra-xusb ever hear about it. Those three ship with wakeup
    enabled; the SuperSpeed hub does not, and at SuperSpeed it is the one in
    our path. Measured 2026-09-04 with it disarmed: the comma presented the
    gadget to a sleeping Jetson and got a bus reset with no SET_ADDRESS, its
    UDC sat at "default" through four connect cycles and fifteen minutes, a
    wake-on-LAN did not reach us either, and the box took its button. Armed,
    the same test resumed 4 s after the bind.

    Arming them is the host's job - `99-jetlink-usb-wakeup.rules` at boot and
    `jetlink-wake-setup.sh` from the unit's ExecStartPre - because `/sys` is
    mounted read-only in this container and the write from in here is a no-op
    under both shipped configurations. So try, because a deployment that
    mounts it read-write exists, and say so loudly when it fails: a line in
    the log is the difference between finding this in a minute and finding it
    after a drive.

    Which hub carries us depends on the negotiated speed - at 480 Mbps it is
    the USB 2.0 hub, which happens to ship armed - so check every hub rather
    than the one we can see. That speed dependence is why this looked like it
    worked before.
    """
    disarmed = []
    try:
      devices = sorted(self.usb.iterdir())
    except OSError:
      return disarmed          # no USB tree visible; nothing to say about it
    for dev in devices:
      try:
        if (dev / 'bDeviceClass').read_text().strip() != HUB_CLASS:
          continue
        wakeup = dev / 'power' / 'wakeup'
        if wakeup.read_text().strip() != 'disabled':
          continue
      except OSError:
        # Not a hub we can judge: this directory also holds interfaces
        # ("2-1:1.0"), which have bInterfaceClass and no bDeviceClass at all,
        # so reading it raises. Treating that as a hub we failed to arm named
        # six interfaces in an error that told the reader to go fix a udev
        # rule that was already installed and already working.
        continue
      try:
        wakeup.write_text('enabled\n')
      except OSError:
        disarmed.append(dev.name)
    if disarmed:
      log.error("hub(s) %s are not armed for remote wakeup and could not be armed from "
                "in here (/sys is read-only): the comma presenting its gadget may not "
                "wake this box. Install 99-jetlink-usb-wakeup.rules on the host.",
                ', '.join(disarmed))
    return disarmed

  def _arm_backstop(self) -> bool:
    """Set an RTC alarm so a wake we do not control still happens.

    Against the RTC's own count, not the wall clock: this box boots with an
    unset clock and only learns the time from NTP, which in the car never
    happens, so since_epoch may be years out and it does not matter as long
    as both sides of the comparison come from the same counter.
    """
    if self.backstop <= 0:
      return False
    try:
      alarm = self.rtc / 'wakealarm'
      now = int((self.rtc / 'since_epoch').read_text().strip())
      alarm.write_text('0\n')            # a stale alarm blocks setting a new one
      alarm.write_text(f"{now + int(self.backstop)}\n")
      return True
    except (OSError, ValueError) as e:
      # No RTC, no alarm support, or not writable from in here. The USB edge
      # is still the wake source; this was only the backstop.
      log.warning("could not arm the %.0f s wake backstop: %s", self.backstop, e)
      return False

  def _disarm_backstop(self, armed: bool) -> None:
    if not armed:
      return
    try:
      (self.rtc / 'wakealarm').write_text('0\n')
    except OSError:
      pass   # it has either fired or it fires once and is spent

  def _enter(self) -> None:
    # Blocks until resume. Split out so a test can stand in for the kernel.
    (self.power / 'state').write_text('mem')

  def _failure(self) -> str:
    step = self._read('suspend_stats/last_failed_step') or '?'
    dev = self._read('suspend_stats/last_failed_dev')
    return f"last failed step {step}" + (f" in {dev}" if dev else '')
