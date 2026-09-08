"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The TensorRT backend without a GPU: keys, the timing cache, and the two
TensorRT generations it has to build on.

The one thing that must never move is the cache key. The Jetson has plans
named '<sha16>.trt10.3.0.Orin-sm87.plan' and they have to load after any
refactor without a three minute rebuild.
"""
from __future__ import annotations

import types

import pytest

from tests.fake_trt import install_stubs

install_stubs()

import tensorrt as trt  # noqa: E402

from jetlink.server.backends.trt import TrtBackend  # noqa: E402
from jetlink.server.backends.trt import build as B  # noqa: E402
from jetlink.server.cache import EngineCache  # noqa: E402

SHA = 'c' * 64


def test_the_key_is_what_the_jetson_already_has_on_disk(tmp_path, monkeypatch):
  monkeypatch.setattr(B, 'device_tag', lambda device=0: 'Orin-sm87')
  monkeypatch.setattr(trt, '__version__', '10.3.0')
  cache = EngineCache(tmp_path, TrtBackend())
  entry = cache.entry(SHA)
  assert entry.path.name == f"{SHA[:16]}.trt10.3.0.Orin-sm87.plan"
  assert entry.meta_path.name == f"{SHA[:16]}.trt10.3.0.Orin-sm87.json"
  assert B.timing_cache_path(cache.engines).name == 'timing.trt10.3.0.Orin-sm87.cache'


def test_the_version_is_sanitized_into_the_key(monkeypatch):
  monkeypatch.setattr(B, 'device_tag', lambda device=0: 'RTX_4060-sm89')
  monkeypatch.setattr(trt, '__version__', '11.2.1+cu12')
  assert TrtBackend().tag() == 'trt11.2.1_cu12.RTX_4060-sm89'


def test_describe_names_the_backend_and_the_device(monkeypatch):
  monkeypatch.setattr(B, 'device_tag', lambda device=0: 'Orin-sm87')
  monkeypatch.setattr(trt, '__version__', '10.3.0')
  assert TrtBackend().describe() == {'backend': 'trt', 'runtime_version': '10.3.0', 'device': 'Orin-sm87'}


def test_a_device_index_is_accepted_and_auto_is_zero():
  assert TrtBackend('auto').device == 0
  assert TrtBackend('1').device == 1
  with pytest.raises(ValueError):
    TrtBackend('METAL')


class TestTwoTensorRTGenerations:
  """TensorRT 10 builds a weakly typed network with the FP16 flag; 11 removed
  both and types the network from the ONNX. The Jetson is on 10.3 and its
  build must not change; a laptop on 11 must build at all."""

  class Config:
    def __init__(self):
      self.flags = []

    def set_flag(self, flag):
      self.flags.append(flag)

  def test_tensorrt_10_sets_the_fp16_flag_and_no_network_flags(self, monkeypatch):
    monkeypatch.setattr(trt, 'BuilderFlag', types.SimpleNamespace(FP16='fp16'))
    assert not B.strongly_typed_only()
    assert B.network_flags() == 0
    cfg = self.Config()
    assert B.configure_precision(cfg, fp16=True) == 'fp16 enabled'
    assert cfg.flags == ['fp16']

  def test_tensorrt_11_asks_for_a_strongly_typed_network_and_sets_no_flag(self, monkeypatch):
    monkeypatch.setattr(trt, 'BuilderFlag', types.SimpleNamespace())
    monkeypatch.setattr(trt, 'NetworkDefinitionCreationFlag', types.SimpleNamespace(STRONGLY_TYPED=1),
                        raising=False)
    assert B.strongly_typed_only()
    assert B.network_flags() == 1 << 1
    cfg = self.Config()
    assert 'strongly typed' in B.configure_precision(cfg, fp16=True)
    assert cfg.flags == []

  def test_a_tensorrt_with_neither_flag_still_builds_with_zero(self, monkeypatch):
    monkeypatch.setattr(trt, 'BuilderFlag', types.SimpleNamespace())
    monkeypatch.delattr(trt, 'NetworkDefinitionCreationFlag', raising=False)
    assert B.network_flags() == 0


class TestTimingCache:
  """TensorRT re-times candidate kernels on every build; a warm cache cut a
  Lebowski build from 254 s to 173 s. Advisory, so every path fails open: a bad
  cache costs a slow build, never a wrong engine."""

  class FakeCache:
    def __init__(self, blob=b'timings'):
      self.blob = blob

    def serialize(self):
      return self.blob

  class FakeConfig:
    def __init__(self, raises=None):
      self.raises = raises
      self.seeded = None
      self.set_with = None

    def create_timing_cache(self, blob):
      if self.raises:
        raise self.raises
      self.seeded = blob
      return TestTimingCache.FakeCache()

    def set_timing_cache(self, cache, ignore_mismatch):
      self.set_with = (cache, ignore_mismatch)

  def test_a_previous_cache_seeds_the_build(self, tmp_path):
    p = tmp_path / 'timing.cache'
    p.write_bytes(b'previous')
    cfg = self.FakeConfig()
    cache = B._load_timing_cache(cfg, p)
    assert cfg.seeded == b'previous'
    assert cfg.set_with is not None and cache is not None

  def test_a_missing_cache_still_builds(self, tmp_path):
    cfg = self.FakeConfig()
    assert B._load_timing_cache(cfg, tmp_path / 'nope.cache') is not None
    assert cfg.seeded == b''   # empty seed, build runs cold

  def test_a_cache_tensorrt_rejects_does_not_fail_the_build(self, tmp_path):
    # A cache from another TensorRT version, or truncated by a killed build.
    p = tmp_path / 'timing.cache'
    p.write_bytes(b'garbage')
    cfg = self.FakeConfig(raises=RuntimeError('version mismatch'))
    assert B._load_timing_cache(cfg, p) is None

  def test_the_cache_is_written_atomically(self, tmp_path):
    p = tmp_path / 'sub' / 'timing.cache'
    B._save_timing_cache(self.FakeCache(b'fresh'), p)
    assert p.read_bytes() == b'fresh'
    # No .tmp left for the next build to mistake for a cache.
    assert list(p.parent.glob('*.tmp')) == []

  def test_an_unwritable_cache_is_not_an_error(self, tmp_path):
    B._save_timing_cache(self.FakeCache(), tmp_path / 'nodir' / 'x' / 'c.cache')
    B._save_timing_cache(None, tmp_path / 'c.cache')   # nothing to write


def test_workspace_is_sized_from_free_memory_with_a_cap(monkeypatch):
  monkeypatch.setattr(B, 'available_bytes', lambda: 0)
  assert B.workspace_bytes() == B.MAX_WORKSPACE_BYTES
  monkeypatch.setattr(B, 'available_bytes', lambda: 1 << 30)
  assert B.workspace_bytes() == int((1 << 30) * B.WORKSPACE_FRACTION)
  monkeypatch.setattr(B, 'available_bytes', lambda: 100 << 30)
  assert B.workspace_bytes() == B.MAX_WORKSPACE_BYTES


def test_the_old_import_paths_still_resolve():
  """Playbooks and container images import these names; keep them one release."""
  from jetlink.server import builder, cudart, engine
  assert builder.EngineCache is EngineCache
  assert builder.build_engine is B.build_engine
  assert builder.device_tag is B.device_tag
  assert engine.TrtEngine.__name__ == 'TrtEngine'
  assert callable(cudart.malloc)
  with pytest.raises(AttributeError):
    builder.no_such_thing  # noqa: B018
