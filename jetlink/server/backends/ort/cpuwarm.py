"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

One CPU core kept busy while frames arrive, for the `ane` device.

At 20 frames a second the CPU drops its clocks in the gaps between frames,
and CoreML's share of the next prediction then runs slowly. On an M1 Pro,
with the vision trunk on the Neural Engine and the policy on the GPU, one
busy core took a paced run from 33.8 ms mean and 41.6 ms p99, with the GPU
already kept awake (metal.py), to 27.3 and 28.7 (1,200 frames, 2026-09-25).

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
  """A keep-warm for a CoreML session with every compute unit allowed (the
  `ane` device), on a Mac; None otherwise, or with JETLINK_CPU_KEEPWARM=0."""
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
