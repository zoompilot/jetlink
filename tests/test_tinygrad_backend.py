"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The tinygrad backend on the CPU, with the tiny driving-shaped model:
build, pickle, load in a fresh engine, run, and agree with numpy.

Needs tinygrad importable; skipped otherwise, so the suite still runs on a
comma or a Jetson image without it.
"""
from __future__ import annotations

import numpy as np
import pytest

tinygrad = pytest.importorskip('tinygrad')
onnx = pytest.importorskip('onnx')

from jetlink.server.backends.base import ArtifactInvalid, infer  # noqa: E402
from jetlink.server.backends.tinygrad import TinygradBackend  # noqa: E402
from jetlink.server.backends.tinygrad import build as B  # noqa: E402
from tests import tiny_model  # noqa: E402


@pytest.fixture(scope='module')
def backend():
  try:
    return TinygradBackend('CPU')
  except Exception as e:  # a tinygrad without a working CPU device on this box
    pytest.skip(f'tinygrad CPU device unavailable: {e}')


@pytest.fixture(scope='module')
def built(backend, tmp_path_factory):
  d = tmp_path_factory.mktemp('tg')
  model = tiny_model.write(d / 'tiny.onnx')
  stages = []
  out = backend.build(model, d / 'tiny.pkl', report=lambda s, f, m: stages.append((s, f)),
                      meta_extra={'spec': {'sha256': 'x'}})
  return out, stages


def test_the_tag_names_tinygrad_and_the_device(backend):
  tag = backend.tag()
  assert tag.startswith('tg') and '.CPU-' in tag
  assert backend.describe()['backend'] == 'tinygrad'
  assert backend.runtime_version == B.tinygrad_identity()


def test_build_writes_the_artifact_and_a_sidecar(built):
  out, stages = built
  assert out.is_file() and out.stat().st_size > 1000
  meta = out.with_suffix('.json')
  assert meta.is_file() and '"backend": "tinygrad"' in meta.read_text()
  names = [s for s, _ in stages]
  assert names[0] == 'parse' and 'build' in names and names[-1] == 'save'
  assert stages[-1][1] == 1.0


def test_uint8_images_are_staged_as_fp16_and_the_rest_as_declared(backend, built):
  engine = backend.load(built[0])
  try:
    assert engine.inputs['img'].dtype == np.float16
    assert engine.inputs['features_buffer'].dtype == np.float16
    assert engine.host_input('img').shape == tiny_model.SHAPES['img']
    assert engine.outputs['outputs'].shape == (1, tiny_model.N_OUT)
  finally:
    engine.close()


def test_the_loaded_jit_agrees_with_numpy_frame_after_frame(backend, built):
  engine = backend.load(built[0])
  try:
    print(engine.warm())
    for seed in range(4):
      inputs = tiny_model.random_inputs(seed)
      out = np.asarray(infer(engine, inputs)['outputs'], np.float32).reshape(-1)
      ref = tiny_model.reference(inputs)
      assert np.all(np.isfinite(out))
      assert np.corrcoef(out, ref)[0, 1] > 0.999, seed
      np.testing.assert_allclose(out, ref, atol=0.05, rtol=0.02)
      assert engine.last_gpu_us >= 0
  finally:
    engine.close()


def test_a_second_engine_from_the_same_pickle_is_independent(backend, built):
  a = backend.load(built[0])
  b = backend.load(built[0])
  try:
    ia, ib = tiny_model.random_inputs(1), tiny_model.random_inputs(2)
    oa = np.asarray(infer(a, ia)['outputs'], np.float32).reshape(-1)
    ob = np.asarray(infer(b, ib)['outputs'], np.float32).reshape(-1)
    assert not np.array_equal(oa, ob)
    np.testing.assert_allclose(oa, tiny_model.reference(ia), atol=0.05, rtol=0.02)
  finally:
    a.close()
    b.close()


def test_garbage_is_artifact_invalid_not_a_crash(backend, tmp_path):
  bad = tmp_path / 'bad.pkl'
  bad.write_bytes(b'\x00' * 40)
  with pytest.raises(ArtifactInvalid):
    backend.load(bad)
  bad.write_bytes(b'')
  with pytest.raises(ArtifactInvalid):
    backend.load(bad)


def test_a_pickle_for_another_device_or_format_is_artifact_invalid(backend, tmp_path):
  p = tmp_path / 'other.pkl'
  with open(p, 'wb') as f:
    B.dump_oob({'format': B.FORMAT, 'device': 'METAL', 'inputs': [], 'outputs': [], 'jit': None}, f)
  with pytest.raises(ArtifactInvalid, match='METAL'):
    backend.load(p)
  with open(p, 'wb') as f:
    B.dump_oob({'format': B.FORMAT + 1, 'device': 'CPU'}, f)
  with pytest.raises(ArtifactInvalid, match='format'):
    backend.load(p)


def test_oob_pickle_round_trips_buffers():
  payload = {'a': np.arange(10, dtype=np.float16), 'b': b'bytes', 'n': 3}
  import io
  f = io.BytesIO()
  f.name = '/tmp/x'   # dump_oob stages next to the file it writes
  B.dump_oob(payload, f)
  f.seek(0)
  back = B.load_oob(f)
  assert back['n'] == 3 and back['b'] == b'bytes'
  np.testing.assert_array_equal(back['a'], payload['a'])


def test_beam_becomes_part_of_the_tag(backend, monkeypatch):
  plain = backend.tag()
  monkeypatch.setenv('BEAM', '2')
  assert backend.tag() == plain + '.beam2'
  monkeypatch.setenv('BEAM', '0')
  assert backend.tag() == plain


def test_identity_carries_a_git_sha_when_the_source_is_a_checkout():
  ident = B.tinygrad_identity()
  assert ident and ident != 'unknown'
