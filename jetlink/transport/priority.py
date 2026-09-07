"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.
"""
from __future__ import annotations

import os


def all_cpus() -> set[int]:
  return set(range(os.cpu_count() or 1))


def widen_affinity() -> None:
  """Let the calling thread run on every CPU.

  A thread created inside modeld inherits the frame loop's single-core pin,
  and a pinned helper competes with the loop for that core. If the inherited
  mask is a single core, prefer every *other* core: simply widening was
  measured not to help the reader, because the balancer keeps waking a thread
  on the core it last ran. Best effort; jetlink must not import openpilot, so
  this open-codes what common.realtime.set_core_affinity would do.
  """
  try:
    everything = all_cpus()
    inherited = os.sched_getaffinity(0)
    if len(inherited) == 1 and everything - inherited:
      os.sched_setaffinity(0, everything - inherited)   # off the frame-loop core
    elif everything - inherited:
      os.sched_setaffinity(0, everything)               # unpinned already; just widen
  except (OSError, AttributeError):
    # No affinity call (macOS), or a kernel that will not move us.
    pass


def background_thread() -> None:
  """Drop the calling thread to SCHED_OTHER 0 on every CPU.

  Threads created after config_realtime_process inherit SCHED_FIFO and the
  core pin, so a library thread that only waits on a condition would still
  take the frame loop's core, at equal priority, for every wake until it
  blocks again. Call this first thing in any thread the library starts; the
  process that created it cannot lend it realtime by accident. No-op where
  the scheduler calls do not exist, and best effort where they are refused.
  """
  try:
    os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
  except (OSError, AttributeError, ValueError):
    pass
  widen_affinity()
