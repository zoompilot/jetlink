"""Health IO must not own the inference thread or publish stale samples."""
import threading
from types import SimpleNamespace

import pytest

from jetlink.server import telemetry


def test_cache_rate_expiry_and_slow_sample_age(monkeypatch):
  now = [10.0]
  monkeypatch.setattr(telemetry, 'time', SimpleNamespace(monotonic=lambda: now[0]))
  entered, release = threading.Event(), threading.Event()

  class Sensor:
    calls = 0

    def read(self):
      self.calls += 1
      if self.calls == 2:
        entered.set()
        assert release.wait(2.0)
      return {'temp_c': 60 + self.calls}

  source = Sensor()
  cache = telemetry.CachedTelemetry(source)
  try:
    assert cache.read() == {}
    with cache._cv:
      assert cache._cv.wait_for(lambda: cache._sample_time == 10.0, timeout=1.0)
    for _ in range(100):
      assert cache.read() == {'temp_c': 61}
    assert source.calls == 1
    now[0] = 10.2
    assert cache.read() == {'temp_c': 61}
    assert entered.wait(1.0)
    now[0] = 12.0
    for _ in range(100):
      assert cache.read() == {}
    assert source.calls == 2
    release.set()
    with cache._cv:
      assert cache._cv.wait_for(lambda: cache._sample_time == 10.2, timeout=1.0)
    assert cache.read() == {}  # sample was already old when the read finished
  finally:
    release.set()
    cache.close()
    cache.thread.join(1.0)
  assert not cache.thread.is_alive()
  assert cache.read() == {}


def test_failed_sensor_expires_health_and_can_retry(monkeypatch):
  now = [1.0]
  monkeypatch.setattr(telemetry, 'time', SimpleNamespace(monotonic=lambda: now[0]))

  class Sensor:
    calls = 0

    def read(self):
      self.calls += 1
      if self.calls == 1:
        raise OSError('sensor unavailable')
      return {'temp_c': 62}

  cache = telemetry.CachedTelemetry(Sensor())
  try:
    cache.read()
    with cache._cv:
      assert cache._cv.wait_for(lambda: cache._sample_time == 1.0, timeout=1.0)
    assert cache.read() == {}
    now[0] = 2.0
    cache.read()
    with cache._cv:
      assert cache._cv.wait_for(lambda: cache._sample_time == 2.0, timeout=1.0)
    assert cache.read() == {'temp_c': 62}
  finally:
    cache.close()
    cache.thread.join(1.0)


@pytest.mark.parametrize('period,max_age', [(0, 1), (-1, 1), (2, 1)])
def test_invalid_cache_limits(period, max_age):
  with pytest.raises(ValueError):
    telemetry.CachedTelemetry(None, period=period, max_age=max_age)


def test_no_telemetry_is_an_empty_sample_not_zeros():
  """An empty dict is what the client already treats as no health; zeros
  would log a board at 0 C drawing nothing."""
  assert telemetry.NoTelemetry().read() == {}


def test_nvml_maps_onto_the_tegra_keys(monkeypatch):
  import sys
  import types

  class NVMLError(Exception):
    pass

  calls = []
  fake = types.SimpleNamespace(
    NVMLError=NVMLError, NVML_TEMPERATURE_GPU=0, NVML_CLOCK_GRAPHICS=1,
    nvmlInit=lambda: calls.append('init'),
    nvmlDeviceGetHandleByIndex=lambda i: f'handle{i}',
    nvmlDeviceGetEnforcedPowerLimit=lambda h: 115_000,
    nvmlDeviceGetTemperature=lambda h, kind: 67,
    nvmlDeviceGetPowerUsage=lambda h: 42_500,
    nvmlDeviceGetUtilizationRates=lambda h: types.SimpleNamespace(gpu=83, memory=40),
    nvmlDeviceGetClockInfo=lambda h, kind: 2100,
    nvmlDeviceGetFanSpeed=lambda h: (_ for _ in ()).throw(NVMLError('no fan on a laptop')),
  )
  monkeypatch.setitem(sys.modules, 'pynvml', fake)
  src = telemetry.NvmlTelemetry()
  assert calls == ['init']
  assert src.read() == {'temp_c': 67.0, 'power_w': 42.5, 'power_limit_w': 115.0,
                        'gpu_load_pct': 83, 'gpu_clock_mhz': 2100, 'fan_pct': 0}


def test_pick_source_prefers_tegra_then_nvml_then_nothing(monkeypatch):
  import sys

  from jetlink.server import platform
  monkeypatch.setattr(platform, 'is_jetson', lambda: True)
  assert isinstance(telemetry.pick_source('trt'), telemetry.Telemetry)
  monkeypatch.setattr(platform, 'is_jetson', lambda: False)
  monkeypatch.setitem(sys.modules, 'pynvml', None)   # import raises
  assert isinstance(telemetry.pick_source('trt'), telemetry.NoTelemetry)
  assert isinstance(telemetry.pick_source('ort'), telemetry.NoTelemetry)
