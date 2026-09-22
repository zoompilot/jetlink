"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The optional Metal helper must expire without another inference request and
release its resources on its own thread, including after GPU failures.
"""
import threading

import pytest

from jetlink.server.backends.ort import metal


@pytest.fixture
def work(monkeypatch):
  class Work:
    def __init__(self):
      self.owner = threading.get_ident()
      self.started = threading.Event()
      self.gate = threading.Semaphore(0)
      self.closed = threading.Event()
      self.calls = 0
      instances.append(self)

    def run(self):
      assert threading.get_ident() == self.owner
      self.calls += 1
      self.started.set()
      assert self.gate.acquire(timeout=5)

    def close(self):
      assert threading.get_ident() == self.owner
      self.closed.set()

  instances = []
  monkeypatch.setattr(metal, '_MetalWork', Work)
  keeper = metal.MetalKeepAlive()
  instance = instances[0]
  assert keeper._idle.wait(1)
  assert instance.calls == 0
  try:
    yield keeper, instance
  finally:
    keeper.pause()
    instance.gate.release()
    keeper.close()
  assert instance.closed.is_set()
  assert not keeper._thread.is_alive()


def test_expiry_without_another_request_and_resume(work, monkeypatch):
  keeper, gpu = work
  clock = [100.0]
  monkeypatch.setattr(metal.time, 'monotonic', lambda: clock[0])
  keeper.pulse()
  assert gpu.started.wait(1)
  clock[0] += metal.IDLE_SECONDS + 1
  gpu.gate.release()
  assert keeper._idle.wait(1)
  assert gpu.calls == 1
  gpu.started.clear()
  keeper.pulse()
  assert gpu.started.wait(1)
  assert gpu.calls == 2
  keeper.pause()
  gpu.gate.release()
  assert keeper._idle.wait(1)


def test_pulse_extends_the_active_stream(work, monkeypatch):
  keeper, gpu = work
  clock = [100.0]
  monkeypatch.setattr(metal.time, 'monotonic', lambda: clock[0])
  keeper.pulse()
  assert gpu.started.wait(1)
  clock[0] += metal.IDLE_SECONDS * .75
  keeper.pulse()
  clock[0] += metal.IDLE_SECONDS * .75
  gpu.started.clear()
  gpu.gate.release()
  assert gpu.started.wait(1)
  assert gpu.calls == 2
  keeper.pause()
  gpu.gate.release()
  assert keeper._idle.wait(1)


def test_close_is_idempotent_and_cannot_restart(work):
  keeper, gpu = work
  keeper.close()
  keeper.close()
  keeper.pulse()
  assert gpu.closed.is_set()
  assert gpu.calls == 0


def test_gpu_error_stops_optional_work_and_releases_resources(work, monkeypatch, caplog):
  keeper, gpu = work

  def fail():
    raise RuntimeError('device lost')

  monkeypatch.setattr(gpu, 'run', fail)
  keeper.pulse()
  assert gpu.closed.wait(1)
  keeper.close()
  keeper.pulse()
  assert isinstance(keeper._error, RuntimeError)
  assert 'continuing inference without it: device lost' in caplog.text


@pytest.mark.parametrize(('platform', 'disabled', 'providers', 'enabled'), [
  ('darwin', False, [('CoreMLExecutionProvider', {'MLComputeUnits': 'CPUAndGPU'})], True),
  ('darwin', True, [('CoreMLExecutionProvider', {'MLComputeUnits': 'CPUAndGPU'})], False),
  ('linux', False, [('CoreMLExecutionProvider', {'MLComputeUnits': 'CPUAndGPU'})], False),
  ('darwin', False, [('CoreMLExecutionProvider', {'MLComputeUnits': 'ALL'})], False),
  ('darwin', False, [('CoreMLExecutionProvider', {'MLComputeUnits': 'CPUAndNeuralEngine'})], False),
  ('darwin', False, ['CPUExecutionProvider'], False),
  ('darwin', False, ['CoreMLExecutionProvider'], False),
])
def test_only_coreml_gpu_sessions_enable_keepalive(monkeypatch, platform, disabled, providers, enabled):
  monkeypatch.setattr(metal.sys, 'platform', platform)
  monkeypatch.delenv('JETLINK_METAL_KEEPALIVE', raising=False)
  if disabled:
    monkeypatch.setenv('JETLINK_METAL_KEEPALIVE', '0')
  sentinel = object()
  monkeypatch.setattr(metal, 'MetalKeepAlive', lambda: sentinel)
  assert metal.create_keepalive([('model.onnx', providers)]) is (sentinel if enabled else None)


def test_initialization_failure_is_optional(monkeypatch, caplog):
  def fail():
    raise RuntimeError('no Metal device')

  monkeypatch.setattr(metal.sys, 'platform', 'darwin')
  monkeypatch.delenv('JETLINK_METAL_KEEPALIVE', raising=False)
  monkeypatch.setattr(metal, '_MetalWork', fail)
  assert metal.create_keepalive([('model.onnx', [
    ('CoreMLExecutionProvider', {'MLComputeUnits': 'CPUAndGPU'}),
  ])]) is None
  assert 'no Metal device' in caplog.text
  assert not any(t.name == 'jetlink-metal-keepalive' for t in threading.enumerate())


def test_ane_trunk_with_gpu_policy_does_not_enable_keepalive(monkeypatch):
  monkeypatch.setattr(metal.sys, 'platform', 'darwin')
  monkeypatch.delenv('JETLINK_METAL_KEEPALIVE', raising=False)

  def unexpected():
    pytest.fail('the mixed ANE/GPU chain must not start a keep-alive')

  monkeypatch.setattr(metal, 'MetalKeepAlive', unexpected)
  assert metal.create_keepalive([
    ('trunk.onnx', [('CoreMLExecutionProvider', {'MLComputeUnits': 'ALL'})]),
    ('policy.onnx', [('CoreMLExecutionProvider', {'MLComputeUnits': 'CPUAndGPU'})]),
  ]) is None
