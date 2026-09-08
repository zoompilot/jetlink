"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The engine cache: keys, pruning, and what survives a restart.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from jetlink.server.cache import CacheEntry, EngineCache
from jetlink.spec import ModelSpec
from tests.fake_backend import FakeBackend

SHA = 'b' * 64


def make_spec() -> ModelSpec:
  return ModelSpec(sha256=SHA, nbytes=1234, frame_skip=4,
                   input_shapes={'img': (1, 12, 128, 256)},
                   output_shapes={'outputs': (1, 18452)},
                   output_slices={'hidden_state': slice(2066, 18450)}, checkpoint=None)


def _artifact(cache: EngineCache, name: str, mtime: float, suffix: str = '.fake') -> Path:
  """An artifact and its sidecar, stamped at `mtime`."""
  art = cache.engines / f"{name}{suffix}"
  art.write_bytes(b'plan')
  art.with_suffix('.json').write_text('{}')
  os.utime(art, (mtime, mtime))
  return art


def test_the_key_is_the_model_and_the_backend_tag(tmp_path):
  cache = EngineCache(tmp_path, FakeBackend(version='2.0', device='gpu9'))
  entry = cache.entry(SHA)
  assert entry.path == cache.engines / f"{SHA[:16]}.fake2.0.gpu9.fake"
  assert entry.meta_path == cache.engines / f"{SHA[:16]}.fake2.0.gpu9.json"
  assert cache.model_path(SHA) == cache.models / f"{SHA[:16]}.onnx"


@pytest.mark.parametrize('identity', ['../escape', '/tmp/escape', '', 'a' * 63, 'z' * 64])
def test_model_identity_cannot_escape_the_cache(tmp_path, identity):
  cache = EngineCache(tmp_path, FakeBackend())
  for resolve in (cache.entry, cache.model_path):
    with pytest.raises(ValueError, match='SHA-256'):
      resolve(identity)


def test_two_backends_keep_separate_artifacts_for_one_model(tmp_path):
  a = EngineCache(tmp_path, FakeBackend(version='1'))
  b = EngineCache(tmp_path, FakeBackend(version='2'))
  spec = make_spec()
  a.entry(SHA).path.write_bytes(b'a')
  a.entry(SHA).write_meta({'spec': spec.to_dict()})
  assert a.inventory() == [SHA]
  assert b.inventory() == []
  assert not b.entry(SHA).exists


def test_prune_keeps_the_newest_artifacts_of_this_backend_only(tmp_path):
  cache = EngineCache(tmp_path, FakeBackend())
  for i in range(4):
    _artifact(cache, f"m{i}", 1_000_000 + i)
  other = _artifact(cache, 'theirs', 1, suffix='.plan')
  cache.prune(keep=2)
  left = sorted(p.stem for p in cache.engines.glob('*.fake'))
  assert left == ['m2', 'm3']
  assert not (cache.engines / 'm0.json').exists()
  assert other.exists(), 'another backend\'s artifact was pruned'


def test_prune_never_drops_the_artifact_just_built(tmp_path):
  """The Jetson boots at 1970 with no network, so a fresh plan can be the
  oldest file on disk. Pruning by mtime would delete the build that just
  finished and leave the caller reading a sidecar that no longer exists."""
  cache = EngineCache(tmp_path, FakeBackend())
  _artifact(cache, 'old_a', 2_000_000)
  _artifact(cache, 'old_b', 2_000_001)
  fresh = _artifact(cache, 'fresh', 1)          # 1970, but it is the new one

  cache.prune(keep=2, protect=fresh)

  assert fresh.is_file()
  assert fresh.with_suffix('.json').is_file()
  assert len(list(cache.engines.glob('*.fake'))) == 2


def test_prune_removes_a_directory_artifact_whole(tmp_path):
  """onnxruntime's artifact is a directory: the model plus a compiled cache."""
  backend = FakeBackend()
  backend.suffix = '.dir'
  cache = EngineCache(tmp_path, backend)
  for i in range(3):
    d = cache.engines / f"m{i}.dir"
    (d / 'inner').mkdir(parents=True)
    (d / 'inner' / 'model').write_bytes(b'x')
    d.with_suffix('.json').write_text('{}')
    os.utime(d, (1_000_000 + i, 1_000_000 + i))
  cache.prune(keep=1)
  assert sorted(p.name for p in cache.engines.iterdir()) == ['m2.dir', 'm2.json']


def test_entry_remove_takes_both_halves(tmp_path):
  cache = EngineCache(tmp_path, FakeBackend())
  art = _artifact(cache, 'gone', 1)
  entry = CacheEntry(art, art.with_suffix('.json'))
  assert entry.exists
  entry.remove()
  assert not entry.exists and not art.exists() and not entry.meta_path.exists()
  entry.remove()   # nothing left to remove is not an error


def test_sweep_temp_drops_only_stale_build_dirs(tmp_path):
  cache = EngineCache(tmp_path, FakeBackend())
  stale = cache.engines / 'tmpstale'
  fresh = cache.engines / 'tmpfresh'
  for d in (stale, fresh):
    d.mkdir()
    (d / 'engine.plan').write_bytes(b'x')
  os.utime(stale, (time.time() - 7 * 3600, time.time() - 7 * 3600))

  cache.sweep_temp()

  assert not stale.exists()
  assert fresh.exists()


def test_last_loaded_survives_and_names_the_backend(tmp_path):
  cache = EngineCache(tmp_path, FakeBackend())
  assert cache.last_loaded() is None
  cache.remember_loaded(SHA, 4)
  assert cache.last_loaded() == (SHA, 4)
  assert json.loads((tmp_path / 'last-loaded.json').read_text())['backend'] == 'fake'


def test_a_marker_written_before_backends_still_reads(tmp_path):
  """The Jetson's marker has no backend field; it must still preload."""
  cache = EngineCache(tmp_path, FakeBackend())
  (tmp_path / 'last-loaded.json').write_text(json.dumps({'sha256': SHA, 'frame_skip': 4}))
  assert cache.last_loaded() == (SHA, 4)


def test_inventory_requires_a_compatible_artifact_and_spec(tmp_path):
  spec = make_spec()
  cache = EngineCache(tmp_path, FakeBackend())
  entry = cache.entry(spec.sha256)
  entry.path.write_bytes(b'plan')
  entry.write_meta({'spec': spec.to_dict()})
  assert cache.inventory() == [spec.sha256]
  entry.path.unlink()
  assert cache.inventory() == []


def test_a_sidecar_from_before_backends_counts(tmp_path):
  """A Jetson's sidecars carry trt_version and no backend field."""
  spec = make_spec()
  cache = EngineCache(tmp_path, FakeBackend())
  entry = cache.entry(spec.sha256)
  entry.path.write_bytes(b'plan')
  entry.write_meta({'trt_version': '10.3.0', 'device': 'Orin-sm87', 'spec': spec.to_dict()})
  assert cache.inventory() == [spec.sha256]


def test_the_cache_root_defaults_to_the_platform_cache_dir(tmp_path, monkeypatch):
  monkeypatch.setenv('JETLINK_CACHE', str(tmp_path / 'env'))
  cache = EngineCache(backend=FakeBackend())
  assert cache.root == tmp_path / 'env'
  assert cache.engines.is_dir() and cache.models.is_dir()
