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
  _has_cache_key,
  _pick_device,
  available_providers,
  repair_coreml_cache,
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


def test_device_selection(monkeypatch):
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
  # auto on a Mac: the Neural Engine split on Apple silicon, the GPU on Intel
  monkeypatch.setattr('jetlink.server.backends.ort.sys.platform', 'darwin')
  monkeypatch.setattr('jetlink.server.backends.ort.is_apple_silicon', lambda: True)
  assert _pick_device(with_coreml, 'auto') == 'ane'
  monkeypatch.setattr('jetlink.server.backends.ort.is_apple_silicon', lambda: False)
  assert _pick_device(with_coreml, 'auto') == 'coreml'


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


class TestRepairCoreMLCache:
  """An artifact built under the old metadata key carries two compiles, one
  under a hash of the build's temp path and one under a hash of the final
  path, and neither is found by key. The repair renames the one that belongs
  to this artifact and drops the rest, without a rebuild. CoreML is not
  needed to check any of that, only the directory layout it leaves behind.
  """

  @staticmethod
  def _layout(tmp_path, compiles):
    """An artifact whose coreml/ holds `{dir name: what model.txt says}`."""
    artifact = tmp_path / 'a086.ort1.29.0.coreml-Apple_M1_Pro.ortcache'
    artifact.mkdir()
    onnx.save(onnx.load(str(tiny_model.write(tmp_path / 'src.onnx'))), str(artifact / 'model.onnx'))
    cache = artifact / 'coreml'
    cache.mkdir()
    for name, recorded in compiles.items():
      d = cache / name
      (d / '0_dynamic_mlprogram' / 'model' / 'compiled_model.mlmodelc' / 'weights').mkdir(parents=True)
      (d / '0_dynamic_mlprogram' / 'model' / 'compiled_model.mlmodelc' / 'model.mil').write_text('program(1.3)')
      (d / 'model.txt').write_text(str(recorded))
    return artifact, cache, _cache_key(artifact, 'model')

  def test_the_compile_for_this_artifact_is_kept_under_the_key(self, tmp_path):
    artifact, cache, key = self._layout(tmp_path, {
      '9229090538059370699': tmp_path / 'tmpzr7xwcwm' / 'artifact' / 'model.onnx',
      '15625364929042385060': tmp_path / 'a086.ort1.29.0.coreml-Apple_M1_Pro.ortcache' / 'model.onnx',
    })
    repair_coreml_cache(artifact, 'model.onnx', cache, key)
    assert [d.name for d in sorted(cache.iterdir())] == [key], 'one directory, named by the key'
    mil = cache / key / '0_dynamic_mlprogram' / 'model' / 'compiled_model.mlmodelc' / 'model.mil'
    assert mil.read_text() == 'program(1.3)', 'the compiled program was rebuilt, not renamed'

  def test_the_key_is_written_into_the_model(self, tmp_path):
    artifact, cache, key = self._layout(tmp_path, {})
    model = onnx.load(str(artifact / 'model.onnx'))
    assert not _has_cache_key(model, key)
    repair_coreml_cache(artifact, 'model.onnx', cache, key)
    assert _has_cache_key(onnx.load(str(artifact / 'model.onnx')), key)

  def test_a_compile_for_another_artifact_is_removed(self, tmp_path):
    artifact, cache, key = self._layout(tmp_path, {'999': tmp_path / 'somewhere' / 'else.onnx'})
    repair_coreml_cache(artifact, 'model.onnx', cache, key)
    assert list(cache.iterdir()) == [], 'a stale compile was left to be loaded'

  def test_a_second_repair_changes_nothing(self, tmp_path):
    artifact, cache, key = self._layout(tmp_path, {
      '15625364929042385060': tmp_path / 'a086.ort1.29.0.coreml-Apple_M1_Pro.ortcache' / 'model.onnx',
    })
    repair_coreml_cache(artifact, 'model.onnx', cache, key)
    before = sorted(p.name for p in cache.rglob('*'))
    repair_coreml_cache(artifact, 'model.onnx', cache, key)
    assert sorted(p.name for p in cache.rglob('*')) == before

  def test_a_compile_already_under_the_key_survives_a_stale_neighbour(self, tmp_path):
    artifact, cache, key = self._layout(tmp_path, {'999': tmp_path / 'somewhere' / 'else.onnx'})
    (cache / key).mkdir()
    (cache / key / 'model.txt').write_text(str(artifact / 'model.onnx'))
    repair_coreml_cache(artifact, 'model.onnx', cache, key)
    assert [d.name for d in cache.iterdir()] == [key]

  def test_a_compile_with_no_model_txt_is_treated_as_stale(self, tmp_path):
    artifact, cache, key = self._layout(tmp_path, {'999': tmp_path / 'x.onnx'})
    (cache / '999' / 'model.txt').unlink()
    repair_coreml_cache(artifact, 'model.onnx', cache, key)
    assert list(cache.iterdir()) == []


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


def test_a_coreml_artifact_from_before_the_weight_rewrite_is_artifact_invalid(tmp_path):
  """Its MIL carries the weights as text and a load parsed it for minutes; the
  host rebuilds an invalid artifact in seconds instead. The sidecar of a build
  since the rewrite records compile_bytes, so its absence is the marker."""
  coreml = OrtBackend('coreml', providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
  d = tmp_path / 'old.ortcache'
  d.mkdir()
  (d / MANIFEST).write_text(json.dumps([{'model': 'model.onnx', 'units': 'CPUAndGPU', 'cache': 'coreml'}]))
  d.with_suffix('.json').write_text(json.dumps({'backend': 'ort', 'build_seconds': 526.4}))
  with pytest.raises(ArtifactInvalid, match='weight rewrite'):
    coreml.load(d)


def test_every_neural_engine_artifact_from_before_the_split_is_artifact_invalid(tmp_path):
  """One version mechanism: a load of an `ane` artifact prepared under an older
  version is refused, and the host rebuilds it from the ONNX. 1 is the one
  session from before #8, 2 is #8's one session with Expand as Tile."""
  ane = OrtBackend('ane', providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
  d = tmp_path / 'old.ortcache'
  d.mkdir()
  (d / MANIFEST).write_text(json.dumps([{'model': 'model.onnx', 'units': 'ALL', 'cache': 'coreml'}]))
  sidecar = {'backend': 'ort', 'compile_bytes': 1}
  for version in (None, 2):
    d.with_suffix('.json').write_text(json.dumps(sidecar if version is None else {**sidecar, 'prepare': version}))
    with pytest.raises(ArtifactInvalid, match=f"prepared as version {version or 1}"):
      ane.load(d)
  # at the current version it gets past that, to the next thing it lacks
  d.with_suffix('.json').write_text(json.dumps({**sidecar, 'prepare': 3}))
  with pytest.raises(ArtifactInvalid, match='model.onnx'):
    ane.load(d)


def test_the_ane_session_asks_for_fast_prediction():
  ane = OrtBackend('ane', providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
  coreml = OrtBackend('coreml', providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
  assert ane._providers(None, None)[0][1]['SpecializationStrategy'] == 'FastPrediction'
  assert 'SpecializationStrategy' not in coreml._providers(None, None)[0][1]


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


COREML_PROVIDERS = ['CoreMLExecutionProvider', 'CPUExecutionProvider']


def _staged_on_the_cpu(device, path, tmp_path, split=False):
  """What `device` stages, loadable by the CPU backend: the same sessions with
  CoreML's units and caches taken out. `split` stages the Neural Engine split
  whatever the graph."""
  artifact = tmp_path / f'{device}.ortcache'
  artifact.mkdir()
  backend = OrtBackend(device, providers=COREML_PROVIDERS)
  manifest = (backend._stage_split if split else backend._stage)(path, artifact, artifact)
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
  split = backend.load(_staged_on_the_cpu('ane', path, tmp_path, split=True))
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


def _fp32_norms(model):
  return [n for n in model.graph.node if n.op_type == 'Cast' and n.name.endswith('__cast_in')]


def test_each_device_stages_its_sessions(tmp_path):
  """`coreml` is one session on the GPU. `ane` is one session with every unit
  allowed for a graph whose history the server queues, its policy's
  LayerNormalizations in fp32 so CoreML keeps the policy on the GPU (#8). A
  graph that keeps its own history is the trunk on the Neural Engine and the
  policy on the GPU instead, each keyed for its own compile cache, with no
  fp32 LayerNormalization: that is for a policy CoreML might otherwise put on
  the Neural Engine, and this one never runs there."""
  queued = _with_policy_layernorm(tiny_model.write(tmp_path / 'tiny.onnx'))
  stateful = tiny_model.write_stateful(tmp_path / 'stateful.onnx')
  cases = [
    ('coreml', queued, [{'model': 'model.onnx', 'units': 'CPUAndGPU', 'cache': 'coreml'}], 0),
    ('ane', queued, [{'model': 'model.onnx', 'units': 'ALL', 'cache': 'coreml'}], 1),
    ('ane', stateful, [{'model': 'vision.onnx', 'units': 'CPUAndNeuralEngine', 'cache': 'coreml-vision'},
                       {'model': 'policy.onnx', 'units': 'CPUAndGPU', 'cache': 'coreml-policy'}], 0),
  ]
  for k, (device, path, manifest_want, fp32_want) in enumerate(cases):
    backend = OrtBackend(device, providers=COREML_PROVIDERS)
    staged = tmp_path / f'staged_{k}'
    staged.mkdir()
    out = tmp_path / f'{k}.ortcache'
    manifest = backend._stage(path, staged, out)
    assert manifest == manifest_want == json.loads((staged / MANIFEST).read_text()), (device, path.name)
    fp32 = 0
    for entry in manifest:
      assert (staged / entry['cache']).is_dir()
      prepared = onnx.load(str(staged / entry['model']))
      assert _has_cache_key(prepared, _cache_key(out, entry['model'].removesuffix('.onnx')))
      fp32 += len(_fp32_norms(prepared))
    assert fp32 == fp32_want, (device, path.name)


def test_the_split_keeps_the_images_in_the_trunk_and_the_norms_in_the_policy(tmp_path):
  path = _with_policy_layernorm(tiny_model.write(tmp_path / 'tiny.onnx'))
  staged = tmp_path / 'staged'
  staged.mkdir()
  OrtBackend('ane', providers=COREML_PROVIDERS)._stage_split(path, staged, tmp_path / 'ane.ortcache')
  assert [i.name for i in onnx.load(str(staged / 'vision.onnx')).graph.input] == ['img', 'big_img']
  policy = onnx.load(str(staged / 'policy.onnx'))
  assert 'LayerNormalization' in {n.op_type for n in policy.graph.node}
  assert not _fp32_norms(policy)
