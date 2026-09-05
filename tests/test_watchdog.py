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
