"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Jetson health, shaped to fit openpilot's chestnutState.

The comma publishes this as chestnutState, so the sidebar, alerts and logging
work unchanged and the Jetson reads as an attached accelerator; field names
mirror ChestnutState in log.capnp. Read from sysfs rather than tegrastats: no
subprocess per poll, and it works unprivileged inside the container.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

THERMAL = Path('/sys/devices/virtual/thermal')
GPU = Path('/sys/devices/platform/bus@0/17000000.gpu')
GPU_DEVFREQ = Path('/sys/class/devfreq/17000000.gpu')
HWMON = Path('/sys/class/hwmon')


class CachedTelemetry:
  """One sensor worker per server, with bounded refresh rate and sample age.

  Sensor IO never holds the snapshot lock, so inference cannot wait on a stuck
  read and reconnects spawn no more workers. Age starts before sampling, or a
  slow read would look fresh when it finally returns.
  """

  def __init__(self, source, period: float = 0.1, max_age: float = 1.0):
    if period <= 0 or max_age < period:
      raise ValueError('require 0 < period <= max_age')
    self.source = source
    self.period = period
    self.max_age = max_age
    self._cv = threading.Condition()
    self._sample: dict = {}
    self._sample_time = float('-inf')
    self._next_read = 0.0
    self._requested = False
    self._closed = False
    self.thread = threading.Thread(target=self._run, name='jetlink-health', daemon=True)
    self.thread.start()

  def read(self) -> dict:
    now = time.monotonic()
    with self._cv:
      if self._closed:
        return {}
      if now >= self._next_read:
        self._requested = True
        self._cv.notify()
      return dict(self._sample) if now - self._sample_time <= self.max_age else {}

  def close(self) -> None:
    # Do not join a worker that could be blocked in a sensor's kernel driver.
    with self._cv:
      self._closed = True
      self._cv.notify()

  def _run(self) -> None:
    while True:
      with self._cv:
        while not self._requested and not self._closed:
          self._cv.wait()
        if self._closed:
          return
        self._requested = False
        started = time.monotonic()
        self._next_read = float('inf')
      try:
        sample = self.source.read()
      except Exception:
        sample = {}
      with self._cv:
        self._sample = sample
        self._sample_time = started
        self._next_read = time.monotonic() + self.period
        self._cv.notify_all()


def _read(path: Path, default=None):
  # Broad on purpose: the Orin Nano's cv*-thermal zones have no sensor wired up
  # and raise TypeError out of the codec layer, not OSError.
  try:
    return path.read_text().strip()
  except Exception:
    return default


def _read_int(path: Path, default: int = 0) -> int:
  v = _read(path)
  try:
    return int(v)
  except (TypeError, ValueError):
    return default


def _thermal_zones() -> dict[str, float]:
  out = {}
  for z in sorted(THERMAL.glob('thermal_zone*')):
    name = _read(z / 'type')
    raw = _read(z / 'temp')
    if not name or raw in (None, ''):
      continue
    try:
      out[name] = int(raw) / 1000.0
    except ValueError:
      pass
  return out


def _hwmon(name: str) -> Path | None:
  for h in sorted(HWMON.glob('hwmon*')):
    if _read(h / 'name') == name:
      return h
  return None


class Telemetry:
  """Caches the sysfs paths once; polling is then a handful of small reads."""

  def __init__(self, power_limit_w: float = 25.0):
    self.ina = _hwmon('ina3221')
    self.fan = _hwmon('pwm_tach')
    self.power_limit_w = power_limit_w

  def read(self) -> dict:
    zones = _thermal_zones()
    # tj-thermal is the junction temperature the throttle point is defined on
    # (~92 C); it is the closest analogue to chestnut's GPU hotspot.
    temp = zones.get('tj-thermal') or zones.get('gpu-thermal') or 0.0
    mem_temp = zones.get('soc0-thermal') or zones.get('cpu-thermal') or 0.0

    mv = ma = 0
    if self.ina is not None:
      # Channel 1 is VDD_IN: the whole board's input rail.
      mv = _read_int(self.ina / 'in1_input')
      ma = _read_int(self.ina / 'curr1_input')

    load_permille = _read_int(GPU / 'load')  # 0..1000
    freq_hz = _read_int(GPU_DEVFREQ / 'cur_freq')

    return {
      'temp_c': round(temp, 1),
      'memory_temp_c': round(mem_temp, 1),
      'power_w': round(mv * ma / 1e6, 2),
      'power_limit_w': self.power_limit_w,
      'gpu_load_pct': min(100, load_permille // 10),
      'gpu_clock_mhz': freq_hz // 1_000_000,
      'fan_rpm': _read_int(self.fan / 'rpm') if self.fan else 0,
      'supply_mv': mv,
      'supply_ma': ma,
    }
