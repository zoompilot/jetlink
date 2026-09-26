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
  COREML_CACHE_KEY,
  MANIFEST,
  PREPARE_VERSION,
  OrtBackend,
  _cache_key,
  _pick_device,
  available_providers,
  runtime_version,
)
from tests import tiny_model  # noqa: E402

COREML_PROVIDERS = ['CoreMLExecutionProvider', 'CPUExecutionProvider']


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
  monkeypatch.setattr('jetlink.server.backends.ort.sys.platform', 'linux')
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
  assert _pick_device(with_coreml, 'coreml') == 'coreml'
  # auto on a Mac: the Neural Engine split on Apple silicon, the GPU on
  # Intel, and never the CPU, so auto moves on to tinygrad instead
  monkeypatch.setattr('jetlink.server.backends.ort.sys.platform', 'darwin')
  monkeypatch.setattr('jetlink.server.backends.ort.is_apple_silicon', lambda: True)
  assert _pick_device(with_coreml, 'auto') == 'ane'
  monkeypatch.setattr('jetlink.server.backends.ort.is_apple_silicon', lambda: False)
  assert _pick_device(with_coreml, 'auto') == 'coreml'
  with pytest.raises(RuntimeError, match='none of'):
    _pick_device(cpu_only, 'auto')


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


def test_a_coreml_artifact_prepared_under_another_version_is_artifact_invalid(tmp_path):
  """One version for every CoreML build: a load of an artifact prepared under
  another is refused, and the host rebuilds it from the ONNX in seconds. That
  covers every older layout, and the builds from before the Gemm weight
  rewrite whose loads took minutes."""
  d = tmp_path / 'old.ortcache'
  d.mkdir()
  (d / MANIFEST).write_text(json.dumps([{'model': 'model.onnx', 'units': 'ALL', 'cache': 'coreml'}]))
  for device in ('coreml', 'ane'):
    backend = OrtBackend(device, providers=COREML_PROVIDERS)
    for version in (None, 2, 3, 4):
      d.with_suffix('.json').write_text(json.dumps({'backend': 'ort'} if version is None else {'prepare': version}))
      with pytest.raises(ArtifactInvalid, match=f"prepared as version {version or 1}"):
        backend.load(d)
    # at the current version it gets past that, to the next thing it lacks
    d.with_suffix('.json').write_text(json.dumps({'prepare': PREPARE_VERSION}))
    with pytest.raises(ArtifactInvalid, match='model.onnx'):
      backend.load(d)


def test_coreml_sessions_ask_for_fast_prediction():
  for device in ('coreml', 'ane'):
    opts = OrtBackend(device, providers=COREML_PROVIDERS)._providers('CPUAndGPU', None)[0][1]
    assert opts['SpecializationStrategy'] == 'FastPrediction'


def test_a_coreml_cache_without_its_compile_is_artifact_invalid(backend, built):
  """Otherwise onnxruntime recompiles for minutes under 'loading engine'."""
  out = built[0]
  manifest = json.loads((out / MANIFEST).read_text())
  manifest[0]['cache'] = 'coreml'
  (out / 'coreml').mkdir(exist_ok=True)
  (out / MANIFEST).write_text(json.dumps(manifest))
  try:
    with pytest.raises(ArtifactInvalid, match='no CoreML compile'):
      backend.load(out)
  finally:
    manifest[0]['cache'] = None
    (out / MANIFEST).write_text(json.dumps(manifest))


def _staged_on_the_cpu(device, path, tmp_path):
  """What `device` stages, loadable by the CPU backend: the same sessions with
  CoreML's units and caches taken out."""
  artifact = tmp_path / f'{device}.ortcache'
  artifact.mkdir()
  manifest = OrtBackend(device, providers=COREML_PROVIDERS)._stage(path, artifact, artifact)
  for entry in manifest:
    (artifact / entry['cache']).rmdir()
    entry['units'] = entry['cache'] = None
  (artifact / MANIFEST).write_text(json.dumps(manifest))
  return artifact


def test_the_neural_engine_split_computes_what_the_gpu_graph_does(backend, tmp_path):
  """On the same provider, the cut changes where the graph runs, never what it
  computes. To within fp16 rounding, not bit for bit: whether two layouts of
  one graph round alike depends on how onnxruntime fuses and orders them.
  1.29 gave the two identical results; 1.30 fuses the whole graph across the
  cut and lands one fp16 step (0.03125) from the split. A miswired split is
  off by whole units, hundreds of steps."""
  path = tiny_model.write(tmp_path / 'tiny.onnx')
  whole = backend.load(_staged_on_the_cpu('coreml', path, tmp_path))
  split = backend.load(_staged_on_the_cpu('ane', path, tmp_path))
  try:
    assert len(split.providers) == 2
    assert set(split.inputs) == set(whole.inputs), 'the hand-off leaked into the staged inputs'
    assert set(split.outputs) == set(whole.outputs) == {'outputs'}
    for seed in range(3):
      inputs = tiny_model.random_inputs(seed)
      want = np.asarray(infer(whole, inputs)['outputs'])
      got = np.asarray(infer(split, inputs)['outputs'])
      # four fp16 steps at the largest output
      step = 2.0 ** (np.floor(np.log2(np.max(np.abs(want)))) - 10)
      np.testing.assert_allclose(got, want, rtol=0, atol=4 * step)
  finally:
    split.close()
    whole.close()


def test_the_neural_engine_split_of_a_stateful_graph_loops_its_state(tmp_path):
  """The image queue comes out of the first session and the other queues out
  of the second; the StateLoop has to find all three, frame after frame."""
  from jetlink.spec import spec_from_onnx
  from tests.test_stateful import drive, reference
  path = tiny_model.write_stateful(tmp_path / 'stateful.onnx')
  engine = OrtBackend('cpu').load(_staged_on_the_cpu('ane', path, tmp_path))
  try:
    assert set(engine.outputs) == {'outputs', *tiny_model.STATE_PAIRS.values()}
    frames = tiny_model.stateful_frames(7, seed=5)
    got = drive(engine, spec_from_onnx(str(path)), frames, reset_at=(4,))
    for g, want in zip(got, reference(frames[:4]) + reference(frames[4:]), strict=True):
      np.testing.assert_allclose(g, want, atol=0.02, rtol=0.02)
  finally:
    engine.close()


def test_a_manifest_of_two_sessions_runs_as_a_chain(tmp_path):
  """The worker runs sessions back to back, feeding one's outputs to the next
  by name; any manifest may be a chain, not only the Neural Engine's."""
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


def _with_policy_layernorm(path):
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
  return path


def test_each_device_stages_its_sessions(tmp_path):
  """`coreml` is one session on the GPU. `ane` is the trunk on the Neural
  Engine and the rest on the GPU, whether the server queues the graph's
  history or the graph keeps its own, each keyed for its own compile cache."""
  queued = tiny_model.write(tmp_path / 'tiny.onnx')
  stateful = tiny_model.write_stateful(tmp_path / 'stateful.onnx')
  split = [{'model': 'vision.onnx', 'units': 'CPUAndNeuralEngine', 'cache': 'coreml-vision'},
           {'model': 'policy.onnx', 'units': 'CPUAndGPU', 'cache': 'coreml-policy'}]
  cases = [
    ('coreml', queued, [{'model': 'model.onnx', 'units': 'CPUAndGPU', 'cache': 'coreml-model'}]),
    ('ane', queued, split),
    ('ane', stateful, split),
  ]
  for k, (device, path, manifest_want) in enumerate(cases):
    backend = OrtBackend(device, providers=COREML_PROVIDERS)
    staged = tmp_path / f'staged_{k}'
    staged.mkdir()
    out = tmp_path / f'{k}.ortcache'
    manifest = backend._stage(path, staged, out)
    assert manifest == manifest_want == json.loads((staged / MANIFEST).read_text()), (device, path.name)
    for entry in manifest:
      assert (staged / entry['cache']).is_dir()
      prepared = onnx.load(str(staged / entry['model']))
      key = _cache_key(out, entry['model'].removesuffix('.onnx'))
      assert (COREML_CACHE_KEY, key) in {(p.key, p.value) for p in prepared.metadata_props}


def test_the_split_keeps_the_images_in_the_trunk_and_the_norms_in_the_policy(tmp_path):
  """The Neural Engine's fp16 LayerNormalization is not precise enough for
  the residual MLP after the trunk, so a norm there goes with the policy."""
  path = _with_policy_layernorm(tiny_model.write(tmp_path / 'tiny.onnx'))
  staged = tmp_path / 'staged'
  staged.mkdir()
  OrtBackend('ane', providers=COREML_PROVIDERS)._stage(path, staged, tmp_path / 'ane.ortcache')
  assert [i.name for i in onnx.load(str(staged / 'vision.onnx')).graph.input] == ['img', 'big_img']
  assert 'LayerNormalization' in {n.op_type for n in onnx.load(str(staged / 'policy.onnx')).graph.node}
