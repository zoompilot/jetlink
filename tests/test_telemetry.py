"""Health IO must not own the inference thread or publish stale samples."""
from types import SimpleNamespace
import threading

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
