"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.
"""
import os
import time

import pytest

from jetlink.server.backends.ort import cpuwarm


def cpu_seconds(pid: int) -> float:
  # user + system time of one process, from ps: portable across macOS and Linux
  out = os.popen(f'ps -o time= -p {pid}').read().strip()
  if not out:
    return 0.0
  parts = out.replace('-', ':').split(':')
  secs = 0.0
  for p in parts:
    secs = secs * 60 + float(p)
  return secs


@pytest.mark.parametrize(('platform', 'disabled', 'providers', 'enabled'), [
  ('darwin', False, [('CoreMLExecutionProvider', {'MLComputeUnits': 'ALL'})], True),
  ('darwin', True, [('CoreMLExecutionProvider', {'MLComputeUnits': 'ALL'})], False),
  ('linux', False, [('CoreMLExecutionProvider', {'MLComputeUnits': 'ALL'})], False),
  # the GPU path was not measured with it; the CPU provider has nothing to wait on
  ('darwin', False, [('CoreMLExecutionProvider', {'MLComputeUnits': 'CPUAndGPU'})], False),
  ('darwin', False, ['CPUExecutionProvider'], False),
])
def test_only_the_ane_device_keeps_a_core_warm(monkeypatch, platform, disabled, providers, enabled):
  monkeypatch.setattr(cpuwarm.sys, 'platform', platform)
  monkeypatch.delenv('JETLINK_CPU_KEEPWARM', raising=False)
  if disabled:
    monkeypatch.setenv('JETLINK_CPU_KEEPWARM', '0')
  sentinel = object()
  monkeypatch.setattr(cpuwarm, 'CpuKeepWarm', lambda: sentinel)
  assert cpuwarm.create_cpu_keepwarm([('model.onnx', providers)]) is (sentinel if enabled else None)


def test_it_spins_while_pulsed_and_sleeps_after(monkeypatch):
  monkeypatch.setattr(cpuwarm, 'IDLE_SECONDS', 0.3)
  keeper = cpuwarm.CpuKeepWarm()
  pid = keeper._proc.pid
  try:
    time.sleep(0.3)
    idle = cpu_seconds(pid)
    end = time.monotonic() + 1.0
    while time.monotonic() < end:
      keeper.pulse()
      time.sleep(0.05)
    busy = cpu_seconds(pid) - idle
    assert busy > 0.5, f'spun for {busy:.2f} s of a 1 s burst'
    time.sleep(0.8)
    before = cpu_seconds(pid)
    time.sleep(1.0)
    assert cpu_seconds(pid) - before < 0.1, 'still spinning a second after the last pulse'
  finally:
    keeper.close()
  assert keeper._proc is None


def test_it_goes_when_the_pipe_closes():
  keeper = cpuwarm.CpuKeepWarm()
  proc = keeper._proc
  keeper.close()
  assert proc.poll() is not None
  keeper.pulse()  # after close: nothing to write to, and no error
