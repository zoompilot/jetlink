"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

One CPU core kept busy while frames arrive, for `ane`'s one-session layout.

At 20 frames a second the CPU drops its clocks in the gaps between frames,
and CoreML's share of the next prediction then runs slowly. On an M1 Pro at
20 Hz (2026-09-25), with Cinque Terre V2 in one session with every unit
allowed, one busy core took the round trip from 29.2 ms mean and 32.3 ms p99
to 28.7 and 31.9, interleaved over 1,200 frames each. The split `ane` runs
for a stateful graph measured no faster with it, so it gets none. Spinning
only in the 10 ms before each expected frame spun a tenth as much and
matched the mean, but one block in three reached a 38 ms p99, so it is not
used.

The spinning is a separate process, not a thread: a Python thread spinning
in the worker would hold the GIL the frame loop needs, and the worker is a
daemonic process, which multiprocessing does not let have children. So it is
a plain subprocess, fed a byte a frame on its stdin. It spins while those
keep coming, blocks in select() once they stop for IDLE_SECONDS, and exits
when the pipe closes, so it never outlives the worker.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys

log = logging.getLogger('jetlink.ort')

IDLE_SECONDS = 1.0

_SPINNER = r'''
import os, select, sys, time
fd, idle = sys.stdin.fileno(), float(sys.argv[1])
last = -1e9
while True:
  busy = time.monotonic() - last < idle
  ready, _, _ = select.select([fd], [], [], 0 if busy else None)
  if ready:
    if not os.read(fd, 4096):
      break
    last = time.monotonic()
  if time.monotonic() - last < idle:
    end = time.monotonic() + 0.002
    while time.monotonic() < end:
      pass
'''


class CpuKeepWarm:
  def __init__(self):
    self._proc = subprocess.Popen([sys.executable, '-c', _SPINNER, str(IDLE_SECONDS)],
                                  stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, close_fds=True)
    self._fd = self._proc.stdin.fileno()
    os.set_blocking(self._fd, False)

  def pulse(self) -> None:
    try:
      os.write(self._fd, b'.')
    except (BlockingIOError, BrokenPipeError, OSError):
      # A full pipe is a spinner that is already busy; a closed one is a
      # spinner that died, and inference carries on without it.
      pass

  def close(self) -> None:
    proc, self._proc = self._proc, None
    if proc is None:
      return
    try:
      proc.stdin.close()
    except OSError:
      pass
    try:
      proc.wait(2)
    except subprocess.TimeoutExpired:
      proc.kill()
      proc.wait(2)


def create_cpu_keepwarm(sessions: list[tuple[str, list]]) -> CpuKeepWarm | None:
  """A keep-warm for a CoreML session with every compute unit allowed (`ane`
  on a graph whose history the server queues), on a Mac; None otherwise, or
  with JETLINK_CPU_KEEPWARM=0."""
  if sys.platform != 'darwin' or os.environ.get('JETLINK_CPU_KEEPWARM', '1') == '0':
    return None
  units = [p[1].get('MLComputeUnits') if isinstance(p, tuple) else None
           for _, providers in sessions for p in providers
           if (p[0] if isinstance(p, tuple) else p) == 'CoreMLExecutionProvider']
  if not units or any(unit != 'ALL' for unit in units):
    return None
  try:
    return CpuKeepWarm()
  except OSError as e:
    log.warning('CPU keep-warm unavailable; continuing inference without it: %s', e)
    return None
