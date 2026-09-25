"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

A graph that keeps its own history (openpilot #38916, Cinque Terre V3 on).

What has to hold: the wire is the frame and the scalars, the server feeds
each next_state_ output back as its state_ input every frame, a reset empties
the queues, and every backend agrees with a numpy run of the same graph frame
after frame. The TensorRT loop is checked against a recorded CUDA stream,
since there is no GPU here.
"""
from __future__ import annotations

import importlib.util
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

onnx = pytest.importorskip('onnx')

from onnx import TensorProto, helper  # noqa: E402

from jetlink.client import JetlinkClient  # noqa: E402
from jetlink.onnx_patch import needs_patch, patch_uint8_inputs  # noqa: E402
from jetlink.queues import PolicyQueues, StateLoop, for_model  # noqa: E402
from jetlink.server.backends.base import IO  # noqa: E402
from jetlink.spec import ModelSpec, spec_from_onnx  # noqa: E402
from jetlink.transport.tcp import TcpTransport  # noqa: E402
from tests import tiny_model  # noqa: E402
from tests.test_session import ready_session  # noqa: E402

IMAGES = tiny_model.STATEFUL_IMAGES


@pytest.fixture(scope='module')
def model_path(tmp_path_factory):
  return tiny_model.write_stateful(tmp_path_factory.mktemp('stateful') / 'tiny_stateful.onnx')


@pytest.fixture(scope='module')
def spec(model_path):
  return spec_from_onnx(str(model_path))


def packed_for(frame: dict) -> np.ndarray:
  return np.concatenate([frame['desire'].ravel(), frame['traffic_convention'].ravel(),
                         frame['action_t'].ravel()]).astype(np.float32)


def reference(frames: list[dict]) -> list[np.ndarray]:
  state, outs = tiny_model.empty_state(), []
  for f in frames:
    out, state = tiny_model.stateful_step(state, **f)
    outs.append(out)
  return outs


class NumpyEngine:
  """The tiny stateful graph as an engine, images staged in fp16 the way the
  patched backends stage them. Knows nothing about looping: that is the
  StateLoop's job, and what these tests check."""

  def __init__(self, image_dtype=np.float16):
    def dtype(n):
      return image_dtype if n in IMAGES else np.float32
    self._host = {n: np.zeros(s, dtype(n)) for n, s in tiny_model.STATEFUL_SHAPES.items()}
    self.inputs = {n: IO(n, a.shape, a.dtype) for n, a in self._host.items()}
    self.outputs = {'outputs': IO('outputs', (1, tiny_model.N_OUT), np.dtype(np.float32))}
    self.outputs.update({nxt: IO(nxt, tiny_model.STATEFUL_SHAPES[n], np.dtype(dtype(n)))
                         for n, nxt in tiny_model.STATE_PAIRS.items()})
    self.last_gpu_us = 1
    self.calls = 0

  def host_input(self, name):
    return self._host[name]

  def run(self):
    self.calls += 1
    h = self._host
    state = {n: h[n].astype(np.uint8) if n in IMAGES else h[n] for n in tiny_model.STATE_PAIRS}
    out, nxt = tiny_model.stateful_step(state, h['new_img'].astype(np.uint8), h['desire'],
                                        h['traffic_convention'], h['action_t'])
    return {'outputs': out.reshape(1, -1).astype(np.float32),
            **{tiny_model.STATE_PAIRS[n]: v.astype(h[n].dtype) for n, v in nxt.items()}}

  def warm(self):
    self.run()
    return 'numpy'

  def close(self):
    pass


def drive(engine, spec, frames, reset_at=()):
  """Frames through a StateLoop and an engine, as the session does."""
  loop = for_model(spec, engine)
  dest = {n: engine.host_input(n) for n in engine.inputs}
  loop.reset()
  outs = []
  for i, f in enumerate(frames):
    if i in reset_at:
      loop.reset()
    loop.step_into(f['new_img'], packed_for(f), dest)
    got = engine.run()
    outs.append(np.asarray(got['outputs'], np.float32).reshape(-1).copy())
    loop.after_run(got, dest)
  return outs


class TestSpec:
  def test_the_layout_is_read_off_the_graph(self, spec):
    assert spec.stateful
    assert spec.state_pairs == tiny_model.STATE_PAIRS
    assert spec.warped_shape == (2, 6, 8, 16)
    assert spec.model_hw == (8, 16)

  def test_the_wire_carries_the_scalars_and_no_hidden_state(self, spec):
    assert list(spec.packed_shapes) == ['desire', 'traffic_convention', 'action_t']
    assert spec.packed_nelem == 8 + 2 + 2
    assert spec.output_nelem == tiny_model.N_OUT

  def test_the_wire_form_round_trips(self, spec):
    again = ModelSpec.from_dict(spec.to_dict())
    assert again.stateful and again.state_pairs == spec.state_pairs
    assert again.infer_req_nbytes == spec.infer_req_nbytes

  def test_a_queued_graph_is_unchanged(self, tmp_path):
    queued = spec_from_onnx(str(tiny_model.write(tmp_path / 'tiny.onnx')))
    assert not queued.stateful and queued.state_pairs == {}
    assert list(queued.packed_shapes) == ['desire', 'traffic_convention', 'action_t', 'prev_feat']
    assert isinstance(for_model(queued, None), PolicyQueues)


class TestHostLoop:
  def test_frame_after_frame_matches_the_graph(self, spec):
    frames = tiny_model.stateful_frames(9)
    for got, want in zip(drive(NumpyEngine(), spec, frames), reference(frames), strict=True):
      np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)

  def test_uint8_staging_is_exact_too(self, spec):
    frames = tiny_model.stateful_frames(6, seed=3)
    for got, want in zip(drive(NumpyEngine(np.uint8), spec, frames), reference(frames), strict=True):
      np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)

  def test_the_history_matters(self, spec):
    """Without the loop the output would not depend on earlier frames."""
    frames = tiny_model.stateful_frames(6)
    looped = drive(NumpyEngine(), spec, frames)
    alone = drive(NumpyEngine(), spec, frames[-1:])
    assert not np.allclose(looped[-1], alone[0])

  def test_a_reset_starts_from_empty_queues(self, spec):
    frames = tiny_model.stateful_frames(8)
    outs = drive(NumpyEngine(), spec, frames, reset_at=(5,))
    for got, want in zip(outs[5:], reference(frames[5:]), strict=True):
      np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)

  def test_an_engine_that_loops_itself_is_left_to_it(self, spec):
    engine = NumpyEngine()
    calls = []
    engine.loop_state = lambda pairs: calls.append(('loop', dict(pairs))) or True
    engine.reset_state = lambda: calls.append(('reset',))
    loop = StateLoop(spec, engine)
    assert loop.on_engine and calls == [('loop', tiny_model.STATE_PAIRS)]
    engine.host_input('state_feat_q')[...] = 7
    loop.after_run({'next_state_feat_q': np.zeros((4, 1, 16))}, {'state_feat_q': engine.host_input('state_feat_q')})
    assert (engine.host_input('state_feat_q') == 7).all()
    loop.reset()
    assert calls[-1] == ('reset',)

  def test_wrong_sizes_are_refused(self, spec):
    engine = NumpyEngine()
    loop = StateLoop(spec, engine)
    dest = {n: engine.host_input(n) for n in engine.inputs}
    with pytest.raises(ValueError, match='warped'):
      loop.step_into(np.zeros((2, 6, 8, 8), np.uint8), np.zeros(12, np.float32), dest)
    with pytest.raises(ValueError, match='packed'):
      loop.step_into(np.zeros((2, 6, 8, 16), np.uint8), np.zeros(13, np.float32), dest)


class TestPatch:
  def test_the_queue_is_retyped_end_to_end(self, model_path):
    model = onnx.load(str(model_path))
    assert needs_patch(model)
    g = patch_uint8_inputs(model).graph
    types = {vi.name: vi.type.tensor_type.elem_type for vi in list(g.input) + list(g.output)}
    assert types['new_img'] == types['state_img_q'] == types['next_state_img_q'] == TensorProto.FLOAT16
    assert types['state_feat_q'] == types['next_state_feat_q'] == TensorProto.FLOAT
    # the head Cast is gone; the fp16 -> fp32 one after it is the model's own
    assert [n.input[0] for n in g.node if n.op_type == 'Cast'] == ['imgs']
    onnx.checker.check_model(model)

  def test_arithmetic_on_the_uint8_bytes_is_refused(self):
    graph = helper.make_graph(
      [helper.make_node('Add', ['new_img', 'new_img'], ['twice']),
       helper.make_node('Cast', ['twice'], ['out'], to=TensorProto.FLOAT16)],
      'g', [helper.make_tensor_value_info('new_img', TensorProto.UINT8, (2, 6, 8, 16))],
      [helper.make_tensor_value_info('out', TensorProto.FLOAT16, (2, 6, 8, 16))])
    with pytest.raises(ValueError, match='reads the uint8 images'):
      patch_uint8_inputs(helper.make_model(graph))


def _agrees(backend, model_path, spec, tmp_path, suffix):
  artifact = backend.build(model_path, tmp_path / f'tiny{suffix}', meta_extra={'spec': spec.to_dict()})
  engine = backend.load(artifact)
  try:
    assert set(engine.outputs) >= {'outputs', *tiny_model.STATE_PAIRS.values()}
    frames = tiny_model.stateful_frames(7, seed=5)
    for got, want in zip(drive(engine, spec, frames, reset_at=(4,)),
                         reference(frames[:4]) + reference(frames[4:]), strict=True):
      assert np.all(np.isfinite(got))
      assert np.corrcoef(got, want)[0, 1] > 0.999
      np.testing.assert_allclose(got, want, atol=0.02, rtol=0.02)
  finally:
    engine.close()


@pytest.mark.skipif(importlib.util.find_spec('onnxruntime') is None, reason='needs onnxruntime')
def test_onnxruntime_loops_the_state_on_the_host(model_path, spec, tmp_path):
  from jetlink.server.backends.ort import OrtBackend
  _agrees(OrtBackend('cpu'), model_path, spec, tmp_path, '.ortcache')


@pytest.mark.skipif(importlib.util.find_spec('tinygrad') is None, reason='needs tinygrad')
def test_tinygrad_returns_the_queues_and_loops_them(model_path, spec, tmp_path):
  from jetlink.server.backends.tinygrad import TinygradBackend
  try:
    backend = TinygradBackend('CPU')
  except Exception as e:
    pytest.skip(f'tinygrad CPU device unavailable: {e}')
  _agrees(backend, model_path, spec, tmp_path, '.pkl')


class TestOverTheLink:
  """A real client and Session over TCP, only the engine faked."""

  @pytest.fixture
  def link(self, spec, tmp_path):
    srv = TcpTransport.listen('127.0.0.1', 0)
    client_t = TcpTransport.connect('127.0.0.1', srv.getsockname()[1])
    server_t, _ = TcpTransport.accept(srv)
    srv.close()
    session, _ = ready_session(spec, server_t, engine=NumpyEngine(), cache=tmp_path)
    thread = threading.Thread(target=session.serve_forever, daemon=True)
    thread.start()
    client = JetlinkClient(client_t, deadline=10.0)
    client.spec = spec
    yield client
    client.close()
    server_t.close()
    thread.join(1.0)
    session.host.close()

  def test_the_state_carries_from_frame_to_frame(self, link, spec):
    frames = tiny_model.stateful_frames(8, seed=2)
    want = reference(frames[:3]) + reference(frames[3:])
    for i, (f, w) in enumerate(zip(frames, want, strict=True)):
      out = link.infer(f['new_img'], packed_for(f), frame_id=i + 1, reset=i == 3)
      assert out.shape == (spec.output_nelem,)
      np.testing.assert_allclose(out, w, rtol=1e-5, atol=1e-5)

  def test_a_request_is_the_frame_and_twelve_floats(self, link, spec):
    from jetlink import protocol as P
    assert spec.infer_req_nbytes == P.INFER_REQ_SIZE + spec.warped_nbytes + 12 * 4
    seq = link.infer_begin(np.zeros(spec.warped_shape, np.uint8), np.zeros(spec.packed_nelem, np.float32), 1)
    assert link.infer_end(seq).shape == (spec.output_nelem,)


class TestTensorRTLoop:
  """The per-frame stream TrtEngine records, with the CUDA calls captured."""

  @pytest.fixture
  def engine(self, monkeypatch):
    from tests.fake_trt import install_stubs
    install_stubs()
    from jetlink.server.backends.trt import engine as E
    calls = []
    for name in ('memcpy_h2d_async', 'memcpy_d2h_async', 'memcpy_d2d_async'):
      monkeypatch.setattr(E.cudart, name, lambda dst, src, n, s, _k=name: calls.append((_k, dst, src, n)))
    monkeypatch.setattr(E.cudart, 'memset_async', lambda p, v, n, s: calls.append(('memset', p, v, n)))
    monkeypatch.setattr(E.cudart, 'stream_sync', lambda s: None)

    eng = E.TrtEngine.__new__(E.TrtEngine)
    shapes = dict(tiny_model.STATEFUL_SHAPES)
    outs = {'outputs': (1, 64), **{nxt: shapes[n] for n, nxt in tiny_model.STATE_PAIRS.items()}}
    bindings, ptr = {}, 0x1000
    for is_input, table in ((True, shapes), (False, outs)):
      for name, shape in table.items():
        dtype = np.dtype(np.float16 if name.removeprefix('next_') in IMAGES else np.float32)
        nbytes = int(np.prod(shape)) * dtype.itemsize
        bindings[name] = E.Binding(name, shape, dtype, nbytes, ptr, ptr + 1, np.zeros(shape, dtype), is_input)
        ptr += 0x100000
    eng.bindings = bindings
    eng.inputs = {n: b for n, b in bindings.items() if b.is_input}
    eng.outputs = {n: b for n, b in bindings.items() if not b.is_input}
    eng.context = SimpleNamespace(execute_async_v3=lambda s: calls.append(('execute',)) or True)
    eng.stream = 1
    eng.graph_exec = None
    eng.last_gpu_us = 0
    eng.looped, eng._zero_state = {}, False
    return eng, calls

  def test_the_queues_stay_on_the_gpu(self, engine):
    eng, calls = engine
    assert eng.loop_state(tiny_model.STATE_PAIRS)
    out = eng.run()
    assert set(out) == {'outputs'}
    kinds = [c[0] for c in calls]
    # zeroed once, before the first frame
    assert kinds[:3] == ['memset'] * 3
    h2d = {c[1] for c in calls if c[0] == 'memcpy_h2d_async'}
    assert h2d == {eng.inputs[n].device_ptr for n in ('new_img', 'desire', 'traffic_convention', 'action_t')}
    assert [c[2] for c in calls if c[0] == 'memcpy_d2h_async'] == [eng.outputs['outputs'].device_ptr]
    d2d = {(c[1], c[2], c[3]) for c in calls if c[0] == 'memcpy_d2d_async'}
    assert d2d == {(eng.inputs[n].device_ptr, eng.outputs[x].device_ptr, eng.inputs[n].nbytes)
                   for n, x in tiny_model.STATE_PAIRS.items()}
    assert kinds.index('execute') < kinds.index('memcpy_d2d_async')

    calls.clear()
    eng.run()
    assert 'memset' not in [c[0] for c in calls]
    eng.reset_state()
    calls.clear()
    eng.run()
    assert [c[0] for c in calls][:3] == ['memset'] * 3

  def test_a_pair_that_does_not_match_is_left_to_the_host(self, engine):
    eng, _ = engine
    assert not eng.loop_state({'state_feat_q': 'next_state_img_q'})
    assert eng.looped == {} and set(eng.run()) == set(eng.outputs)

  def test_it_must_come_before_the_graph(self, engine):
    eng, _ = engine
    eng.graph_exec = object()
    with pytest.raises(RuntimeError, match='captured'):
      eng.loop_state(tiny_model.STATE_PAIRS)

  def test_the_session_hands_the_loop_to_the_engine(self, engine, spec):
    eng, _ = engine
    loop = StateLoop(spec, eng)
    assert loop.on_engine and eng.looped == tiny_model.STATE_PAIRS


class TestBenchTools:
  """verify_parity and verify_engine on a stateful graph: the reference loops
  the graph's state itself, and a capture replays through the server's loop.
  Neither imports onnxruntime here; a stand-in session runs the tiny graph."""

  @staticmethod
  def script(name):
    import importlib.util
    import sys
    scripts = str(Path(__file__).resolve().parents[1] / 'scripts')
    if scripts not in sys.path:
      sys.path.insert(0, scripts)   # verify_engine imports its sibling
    spec_ = importlib.util.spec_from_file_location(name, f'{scripts}/{name}.py')
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return mod

  @staticmethod
  def capture(spec, d, frames):
    """What verify_parity's capture would write, the link played by the server's loop."""
    (d / 'spec.json').write_text(json.dumps(spec.to_dict()))
    for i, (f, out) in enumerate(zip(frames, drive(NumpyEngine(), spec, frames), strict=True)):
      np.save(d / f'in_warped_{i}.npy', f['new_img'])
      np.save(d / f'in_packed_{i}.npy', packed_for(f))
      np.save(d / f'out_link_{i}.npy', out)

  def test_the_reference_loops_the_state_itself(self, spec, tmp_path):
    vp = self.script('verify_parity')

    class Session:
      def get_inputs(self):
        return [SimpleNamespace(name=n, shape=list(s), type='tensor(uint8)' if n in IMAGES else 'tensor(float)')
                for n, s in tiny_model.STATEFUL_SHAPES.items()]

      def get_outputs(self):
        return [SimpleNamespace(name=n) for n in ('outputs', *tiny_model.STATE_PAIRS.values())]

      def run(self, _, feed):
        state = {n: feed[n] for n in tiny_model.STATE_PAIRS}
        out, nxt = tiny_model.stateful_step(state, feed['new_img'], feed['desire'],
                                            feed['traffic_convention'], feed['action_t'])
        return [out.reshape(1, -1), *(nxt[n] for n in tiny_model.STATE_PAIRS)]

    frames = tiny_model.stateful_frames(6, seed=9)
    self.capture(spec, tmp_path, frames)
    assert vp.reference_stateful(spec, Session(), tmp_path, len(frames)) == 0
    for i, want in enumerate(reference(frames)):
      np.testing.assert_allclose(np.load(tmp_path / f'out_ref_{i}.npy'), want, rtol=1e-6, atol=1e-6)
    assert not vp.feeds_hidden_back(spec)

  def test_a_capture_replays_bit_for_bit_and_a_corrupt_one_does_not(self, spec, tmp_path):
    ve = self.script('verify_engine')
    frames = tiny_model.stateful_frames(5, seed=4)
    self.capture(spec, tmp_path, frames)
    assert ve.replay_capture(NumpyEngine(), tmp_path) == 0
    bad = np.load(tmp_path / 'out_link_3.npy')
    bad[0] += 1
    np.save(tmp_path / 'out_link_3.npy', bad)
    assert ve.replay_capture(NumpyEngine(), tmp_path) == 2
