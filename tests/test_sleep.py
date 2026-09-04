"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The sleep-when-orphaned policy against a fake /sys/power.
"""
from __future__ import annotations

import errno

import pytest

from jetlink.server import sleep as S


def power(tmp_path, mem_sleep='s2idle [deep]', success=3):
  p = tmp_path / 'power'
  (p / 'suspend_stats').mkdir(parents=True)
  (p / 'mem_sleep').write_text(mem_sleep + '\n')
  (p / 'state').write_text('freeze mem\n')
  (p / 'suspend_stats' / 'success').write_text(f'{success}\n')
  (p / 'suspend_stats' / 'last_failed_step').write_text('\n')
  (p / 'suspend_stats' / 'last_failed_dev').write_text('\n')
  return p


class Kernel(S.Sleeper):
  """A Sleeper whose write to /sys/power/state behaves like the kernel we tell it to."""

  def __init__(self, *a, outcome='sleep', **kw):
    super().__init__(*a, **kw)
    self.outcome = outcome
    self.entered = 0

  def _enter(self):
    self.entered += 1
    stats = self.power / 'suspend_stats'
    if self.outcome == 'sleep':
      n = int((stats / 'success').read_text())
      (stats / 'success').write_text(f'{n + 1}\n')
    elif self.outcome == 'freezer':
      (stats / 'last_failed_step').write_text('freeze\n')
      raise OSError(errno.EBUSY, 'Device or resource busy')
    elif self.outcome == 'readonly':
      raise OSError(errno.EROFS, 'Read-only file system')
    elif self.outcome == 'woke-early':
      pass  # returned cleanly, counter untouched


@pytest.fixture
def clock(monkeypatch):
  now = [1000.0]
  monkeypatch.setattr(S.time, 'monotonic', lambda: now[0])
  return now


def test_not_orphaned_until_the_timeout(tmp_path, clock):
  s = Kernel(after=120, power=power(tmp_path))
  clock[0] += 119
  assert not s.idle()
  assert s.entered == 0
  clock[0] += 1
  assert s.idle()
  assert s.entered == 1
  assert s.slept == 1


def test_a_gadget_resets_the_clock(tmp_path, clock):
  s = Kernel(after=120, power=power(tmp_path))
  clock[0] += 100
  s.touch()
  clock[0] += 100
  assert not s.idle()
  clock[0] += 20
  assert s.idle()


def test_waking_gives_the_gadget_the_full_timeout_again(tmp_path, clock):
  s = Kernel(after=120, power=power(tmp_path))
  clock[0] += 120
  assert s.idle()
  assert not s.idle()  # just woke: whoever woke us has 120 s to show up
  clock[0] += 119
  assert not s.idle()
  clock[0] += 1
  assert s.idle()
  assert s.slept == 2


def test_freezer_failure_backs_off_and_retries(tmp_path, clock):
  s = Kernel(after=120, power=power(tmp_path), outcome='freezer')
  clock[0] += 120
  assert not s.idle()
  assert s.failed == 1 and s.enabled
  # Not again straight away: each attempt freezes every process on the box.
  assert not s.idle()
  clock[0] += S.RETRY_MIN
  assert not s.idle()
  assert s.entered == 2
  clock[0] += S.RETRY_MIN  # backoff doubled
  assert s.entered == 2 and not s.idle()
  clock[0] += S.RETRY_MIN
  assert not s.idle()
  assert s.entered == 3
  # A gadget showing up in between clears the backoff.
  s.outcome = 'sleep'
  s.touch()
  clock[0] += 120
  assert s.idle()


def test_returning_without_sleeping_counts_as_a_failure(tmp_path, clock):
  s = Kernel(after=1, power=power(tmp_path), outcome='woke-early')
  clock[0] += 1
  assert not s.idle()
  assert s.failed == 1 and s.slept == 0 and s.enabled


def test_unwritable_sysfs_disables_sleep_for_good(tmp_path, clock):
  s = Kernel(after=1, power=power(tmp_path), outcome='readonly')
  clock[0] += 1
  assert not s.idle()
  assert not s.enabled
  clock[0] += 1000
  assert not s.orphaned()
  assert s.entered == 1


def test_deep_is_selected_before_sleeping(tmp_path, clock):
  p = power(tmp_path, mem_sleep='[s2idle] deep')
  s = Kernel(after=1, power=p)
  clock[0] += 1
  assert s.idle()
  assert (p / 'mem_sleep').read_text() == 'deep'


def test_no_deep_mode_means_no_sleep(tmp_path, clock):
  s = Kernel(after=1, power=power(tmp_path, mem_sleep='[s2idle]'))
  clock[0] += 1
  assert not s.idle()
  assert not s.enabled and s.entered == 0


def test_serve_loop_sleeps_only_on_absence(tmp_path, clock, monkeypatch):
  """The server's loop: None from the opener is the only thing that ages the sleeper."""
  from tests.fake_trt import install_stubs
  install_stubs()
  from jetlink.server import main as M

  s = Kernel(after=120, power=power(tmp_path))
  polls = []

  class Stop(Exception):
    pass

  def opener():
    polls.append(S.time.monotonic())
    clock[0] += 60
    if len(polls) >= 4:
      raise Stop

  monkeypatch.setattr(M.time, 'sleep', lambda t: None)
  with pytest.raises(Stop):
    M._serve(None, opener, s)  # EngineHost(None, ...) is fine: it loads lazily
  # 60 s, 120 s: slept at the second poll, then woke and looked again at once.
  assert s.slept == 1
