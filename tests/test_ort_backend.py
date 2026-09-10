"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The onnxruntime backend on the CPU provider, with the tiny driving-shaped
model. CoreML itself needs a Mac and ten minutes; what is checked here is
everything around it: the model surgery, the directory artifact, the load
guard, the worker process, and that the numbers agree with numpy.
"""
from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import pytest

# onnxruntime is never imported here: its telemetry thread aborted the suite
# at exit (see backends.ort.quiet). Presence is checked without importing.
if importlib.util.find_spec('onnxruntime') is None:
  pytest.skip('needs onnxruntime', allow_module_level=True)
onnx = pytest.importorskip('onnx')

from onnx import numpy_helper  # noqa: E402

from jetlink.server.backends.base import ArtifactInvalid, infer  # noqa: E402
from jetlink.server.backends.ort import (  # noqa: E402
  MANIFEST,
  OrtBackend,
  _cache_key,
  _pick_device,
  available_providers,
  runtime_version,
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


def test_device_selection():
  cpu_only = ['CPUExecutionProvider']
  assert _pick_device(cpu_only, 'cpu') == 'cpu'
  with pytest.raises(ValueError, match='one of'):
    _pick_device(cpu_only, 'metal')
  assert _pick_device(cpu_only, 'auto') == 'cpu'
  with pytest.raises(RuntimeError, match='CoreMLExecutionProvider'):
    _pick_device(cpu_only, 'coreml')
  assert _pick_device(cpu_only + ['CUDAExecutionProvider'], 'auto') == 'cuda'
  with_coreml = cpu_only + ['CoreMLExecutionProvider']
  assert _pick_device(with_coreml, 'ane') == 'ane'
  assert _pick_device(with_coreml, 'auto') != 'ane', 'the Neural Engine is the measured opt-in, never auto'


def test_providers_are_probed_in_a_child_and_the_version_read_from_metadata():
  assert 'onnxruntime' not in sys.modules, 'the test process must not load onnxruntime'
  found = available_providers()
  assert 'CPUExecutionProvider' in found
  assert 'onnxruntime' not in sys.modules
  assert runtime_version()[0].isdigit()


def test_the_tag_and_describe(backend):
  assert backend.tag().startswith(f"ort{runtime_version()}".replace('+', '_'))
  assert '.cpu-' in backend.tag()
  assert backend.describe()['backend'] == 'ort'
  assert backend.suffix == '.ortcache'


def test_the_cache_key_is_alphanumeric_and_short(tmp_path):
  key = _cache_key(tmp_path / 'abcdef0123456789.ort1.29.0.coreml-Apple_M1_Pro.ortcache', 'trunk')
  assert key.isalnum() and len(key) < 64
  assert key.startswith('abcdef0123456789ort1290coreml') and key.endswith('trunk')
  assert key != _cache_key(tmp_path / 'abcdef0123456789.ort1.29.0.coreml-Apple_M1_Pro.ortcache', 'policy')


def test_build_makes_a_directory_with_the_prepared_model(built):
  out, stages = built
  assert out.is_dir()
  manifest = json.loads((out / MANIFEST).read_text())
  assert [e['model'] for e in manifest] == ['model.onnx']
  prepared = onnx.load(str(out / 'model.onnx'))
  assert not any(n.domain == 'org.tinygrad' for n in prepared.graph.node), 'Contiguous was not stripped'
  img = next(i for i in prepared.graph.input if i.name == 'img')
  assert img.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16, 'uint8 images were not retyped'
  # The name onnxruntime reads; under any other it keys the cache on the path.
  assert any(p.key == 'COREML_CACHE_KEY' for p in prepared.metadata_props)
  assert not any(p.key == 'CACHE_KEY' for p in prepared.metadata_props)
  meta = out.with_suffix('.json').read_text()
  assert '"backend": "ort"' in meta and '"spec"' in meta and '"sessions"' in meta
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


def test_a_directory_without_a_manifest_is_artifact_invalid(backend, tmp_path):
  """Also what an artifact from before the manifest looks like: it rebuilds."""
  d = tmp_path / 'empty.ortcache'
  d.mkdir()
  with pytest.raises(ArtifactInvalid, match=MANIFEST):
    backend.load(d)
  (d / MANIFEST).write_text('[]')
  with pytest.raises(ArtifactInvalid, match='no sessions'):
    backend.load(d)
  (d / MANIFEST).write_text(json.dumps([{'model': 'gone.onnx', 'units': None, 'cache': None}]))
  with pytest.raises(ArtifactInvalid, match='gone.onnx'):
    backend.load(d)


def test_an_empty_coreml_cache_is_artifact_invalid(backend, built):
  """Otherwise onnxruntime recompiles for minutes under 'loading engine'."""
  out = built[0]
  manifest = json.loads((out / MANIFEST).read_text())
  manifest[0]['cache'] = 'coreml'
  (out / 'coreml').mkdir(exist_ok=True)
  (out / MANIFEST).write_text(json.dumps(manifest))
  try:
    with pytest.raises(ArtifactInvalid, match='CoreML cache'):
      backend.load(out)
  finally:
    manifest[0]['cache'] = None
    (out / MANIFEST).write_text(json.dumps(manifest))


def test_a_manifest_of_two_sessions_runs_as_a_chain(tmp_path):
  """The worker runs sessions back to back, feeding one's outputs to the next
  by name; a split graph was measured through this and lost on a Mac, but the
  chain stays generic and this keeps it honest."""
  import onnx.utils
  backend = OrtBackend('cpu')
  model = tiny_model.write(tmp_path / 'tiny.onnx', with_contiguous=False)
  d = tmp_path / 'chain.ortcache'
  d.mkdir()
  onnx.utils.extract_model(str(model), str(d / 'trunk.onnx'), ['img', 'big_img'], ['img_mean'])
  onnx.utils.extract_model(str(model), str(d / 'policy.onnx'),
                           ['img_mean', 'desire_pulse', 'traffic_convention', 'action_t', 'features_buffer'], ['outputs'])
  (d / MANIFEST).write_text(json.dumps([{'model': 'trunk.onnx', 'units': None, 'cache': None},
                                        {'model': 'policy.onnx', 'units': None, 'cache': None}]))
  engine = backend.load(d)
  try:
    assert set(engine.inputs) == set(tiny_model.SHAPES), 'the boundary tensor leaked into the staged inputs'
    assert len(engine.providers) == 2
    print(engine.warm())
    for seed in range(3):
      inputs = tiny_model.random_inputs(seed)
      got = np.asarray(infer(engine, inputs)['outputs'], np.float32).reshape(-1)
      np.testing.assert_allclose(got, tiny_model.reference(inputs), atol=0.05, rtol=0.02)
  finally:
    engine.close()


def test_the_neural_engine_device_gets_the_rewrites_and_the_gpu_does_not(tmp_path):
  """The two graph rewrites exist for the Neural Engine; the GPU path ships
  the graph as exported apart from what TensorRT also does to it."""
  providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
  path = tiny_model.write(tmp_path / 'tiny.onnx')
  # a LayerNormalization on the policy side, over the concatenated features
  m = onnx.load(str(path))
  g = m.graph
  g.initializer.append(numpy_helper.from_array(np.ones(tiny_model.FEATURES, np.float16), 'ln_scale'))
  matmul = next(n for n in g.node if n.op_type == 'MatMul')
  src = matmul.input[0]
  g.node.insert(list(g.node).index(matmul),
                onnx.helper.make_node('LayerNormalization', [src, 'ln_scale'], ['normed'], axis=-1, name='ln_policy'))
  matmul.input[0] = 'normed'
  onnx.save(m, str(path))
  for device, want_units, want_casts in (('coreml', 'CPUAndGPU', 0), ('ane', 'ALL', 1)):
    backend = OrtBackend(device, providers=providers)
    staged = tmp_path / f'staged_{device}'
    staged.mkdir()
    manifest = backend._stage(path, staged, tmp_path / f'{device}.ortcache')
    assert manifest[0]['units'] == want_units
    prepared = onnx.load(str(staged / 'model.onnx'))
    casts = [n for n in prepared.graph.node if n.op_type == 'Cast' and n.name.endswith('__cast_in')]
    assert len(casts) == want_casts, device
