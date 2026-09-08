"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The onnxruntime backend on the CPU provider, with the tiny driving-shaped
model. CoreML itself needs a Mac and twenty minutes; what is checked here is
everything around it: the model surgery, the directory artifact, the load
guard, and that the numbers agree with numpy.
"""
from __future__ import annotations

import numpy as np
import pytest

ort = pytest.importorskip('onnxruntime')
onnx = pytest.importorskip('onnx')

from jetlink.server.backends.base import ArtifactInvalid, infer  # noqa: E402
from jetlink.server.backends.ort import (  # noqa: E402
  MODEL,
  OrtBackend,
  _cache_key,
  _pick_device,
)
from tests import tiny_model  # noqa: E402


@pytest.fixture(scope='module')
def backend():
  return OrtBackend('cpu')


@pytest.fixture(scope='module')
def built(backend, tmp_path_factory):
  d = tmp_path_factory.mktemp('ort')
  model = tiny_model.write(d / 'tiny.onnx')
  stages = []
  out = backend.build(model, d / 'tiny.ortcache', report=lambda s, f, m: stages.append((s, f, m)),
                      meta_extra={'spec': {'sha256': 'x'}})
  return out, stages


def test_device_selection(monkeypatch):
  assert _pick_device(ort, 'cpu') == 'cpu'
  with pytest.raises(ValueError, match='one of'):
    _pick_device(ort, 'metal')
  monkeypatch.setattr(ort, 'get_available_providers', lambda: ['CPUExecutionProvider'])
  assert _pick_device(ort, 'auto') == 'cpu'
  with pytest.raises(RuntimeError, match='CoreMLExecutionProvider'):
    _pick_device(ort, 'coreml')


def test_the_tag_and_describe(backend):
  assert backend.tag().startswith(f"ort{ort.__version__}".replace('+', '_')) or backend.tag().startswith('ort')
  assert '.cpu-' in backend.tag()
  assert backend.describe()['backend'] == 'ort'
  assert backend.suffix == '.ortcache'


def test_the_cache_key_is_alphanumeric_and_short(tmp_path):
  key = _cache_key(tmp_path / 'abcdef0123456789.ort1.29.0.coreml-Apple_M1_Pro.ortcache')
  assert key.isalnum() and len(key) < 64
  assert key.startswith('abcdef0123456789ort1290coreml')


def test_build_makes_a_directory_with_the_prepared_model(built):
  out, stages = built
  assert out.is_dir()
  prepared = onnx.load(str(out / MODEL))
  assert not any(n.domain == 'org.tinygrad' for n in prepared.graph.node), 'Contiguous was not stripped'
  img = next(i for i in prepared.graph.input if i.name == 'img')
  assert img.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16, 'uint8 images were not retyped'
  assert any(p.key == 'CACHE_KEY' for p in prepared.metadata_props)
  meta = out.with_suffix('.json').read_text()
  assert '"backend": "ort"' in meta and '"spec"' in meta
  assert [s for s, _, _ in stages][0] == 'patch' and stages[-1][1] == 1.0


def test_the_loaded_session_agrees_with_numpy(backend, built):
  engine = backend.load(built[0])
  try:
    assert engine.inputs['img'].dtype == np.float16   # patched, so the queues' fp16 lands directly
    print(engine.warm())
    for seed in range(4):
      inputs = tiny_model.random_inputs(seed)
      out = np.asarray(infer(engine, inputs)['outputs'], np.float32).reshape(-1)
      assert out.dtype == np.float32
      ref = tiny_model.reference(inputs)
      assert np.corrcoef(out, ref)[0, 1] > 0.999
      np.testing.assert_allclose(out, ref, atol=0.05, rtol=0.02)
  finally:
    engine.close()


def test_a_directory_without_a_model_is_artifact_invalid(backend, tmp_path):
  d = tmp_path / 'empty.ortcache'
  d.mkdir()
  with pytest.raises(ArtifactInvalid, match=MODEL):
    backend.load(d)


def test_an_empty_coreml_cache_is_artifact_invalid(backend, built, monkeypatch):
  """Otherwise onnxruntime recompiles for twenty minutes under 'loading engine'."""
  monkeypatch.setattr(backend, 'device', 'coreml')
  with pytest.raises(ArtifactInvalid, match='CoreML cache'):
    backend.load(built[0])


def test_a_model_without_tinygrad_ops_or_uint8_inputs_builds_too(backend, tmp_path):
  model = tiny_model.write(tmp_path / 'plain.onnx', with_contiguous=False)
  out = backend.build(model, tmp_path / 'plain.ortcache')
  engine = backend.load(out)
  try:
    inputs = tiny_model.random_inputs(9)
    out = np.asarray(infer(engine, inputs)['outputs'], np.float32).reshape(-1)
    np.testing.assert_allclose(out, tiny_model.reference(inputs), atol=0.05, rtol=0.02)
  finally:
    engine.close()
