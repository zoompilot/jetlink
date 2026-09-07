"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.
"""
import threading

from jetlink.transport.watchdog import WriteWatchdog


def test_successful_writes_reuse_one_worker_without_aborting():
  aborted = threading.Event()
  guard = WriteWatchdog(aborted.set)
  worker = guard.thread
  try:
    for _ in range(100):
      assert guard.arm(10)
      assert guard.disarm()
      assert guard.thread is worker
    assert not aborted.is_set()
  finally:
    guard.close()
    worker.join(1)
  assert not worker.is_alive()
  assert not guard.arm(1)


def test_expiry_wins_and_disarm_does_not_wait_for_abort_io():
  entered, release = threading.Event(), threading.Event()

  def abort():
    entered.set()
    assert release.wait(2)

  guard = WriteWatchdog(abort)
  try:
    assert guard.arm(0)
    assert entered.wait(1)
    assert not guard.disarm()
    assert not guard.arm(10)
    guard.close()
  finally:
    release.set()
    guard.thread.join(1)
  assert not guard.thread.is_alive()


def test_cancelled_deadline_cannot_abort_a_later_write():
  aborted = threading.Event()
  guard = WriteWatchdog(aborted.set)
  try:
    # Hold the lock so the first deadline expires before the worker can
    # observe it, then cancel and replace it. No stale Timer may survive.
    with guard._cv:
      assert guard.arm(0)
      assert guard.disarm()
      assert guard.arm(10)
    assert not aborted.wait(0.05)
    assert guard.disarm()
  finally:
    guard.close()
    guard.thread.join(1)


def test_worker_drops_realtime_before_it_waits(monkeypatch):
  """The worker is created from whatever thread opens the link, in modeld the
  SCHED_FIFO 54 frame thread pinned to core 7, and inherits both. It must
  drop them itself, on its own thread, before it first waits."""
  import os

  from jetlink.transport import watchdog

  calls = []
  monkeypatch.setattr(os, 'sched_setscheduler', lambda _pid, policy, _param: calls.append(('sched', threading.get_ident(), policy)), raising=False)
  monkeypatch.setattr(os, 'sched_getaffinity', lambda _pid: {7}, raising=False)
  monkeypatch.setattr(os, 'sched_setaffinity', lambda _pid, mask: calls.append(('affinity', threading.get_ident(), set(mask))), raising=False)
  monkeypatch.setattr(os, 'SCHED_OTHER', 0, raising=False)
  monkeypatch.setattr(os, 'sched_param', lambda prio: prio, raising=False)
  monkeypatch.setattr(os, 'cpu_count', lambda: 8)

  seen_at_first_wait = []

  class Guarded(threading.Condition):
    def wait(self, timeout=None):
      if not seen_at_first_wait and threading.current_thread().name == 'jetlink-write-guard':
        seen_at_first_wait.append(list(calls))
      return super().wait(timeout)

  monkeypatch.setattr(watchdog.threading, 'Condition', Guarded)
  guard = watchdog.WriteWatchdog(lambda: None)
  try:
    assert guard.arm(10) and guard.disarm()   # forces at least one wait cycle
  finally:
    guard.close()
    guard.thread.join(1)
  assert not guard.thread.is_alive()
  assert seen_at_first_wait and seen_at_first_wait[0] == calls, "scheduling changed after the worker started waiting"
  assert calls == [('sched', guard.thread.ident, 0), ('affinity', guard.thread.ident, set(range(7)))]
