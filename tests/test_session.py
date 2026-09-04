"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

End-to-end: a real client talking to a real Session over a real socket.

Only the TensorRT engine is faked. Everything else - framing, the infer
request/response encoding, the history queues, the hidden-state feedback, the
piggybacked telemetry - is the code that will run in the car. This is the test
that covers the paths the car would otherwise be the first to execute.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from tests.fake_trt import install_stubs

install_stubs()

from jetlink import protocol as P                          # noqa: E402
from jetlink.client import JetlinkClient                   # noqa: E402
from jetlink.queues import PolicyQueues                    # noqa: E402
from jetlink.server.builder import EngineCache             # noqa: E402
from jetlink.server.session import EngineHost, Loaded, Request, Session  # noqa: E402
from jetlink.spec import ModelSpec                         # noqa: E402
from jetlink.transport.base import LinkError               # noqa: E402
from jetlink.transport.tcp import TcpTransport             # noqa: E402

BIG = {
  'img': (1, 12, 128, 256), 'big_img': (1, 12, 128, 256),
  'desire_pulse': (1, 33, 8), 'traffic_convention': (1, 2),
  'action_t': (1, 2), 'features_buffer': (1, 32, 32, 512),
}


def make_spec() -> ModelSpec:
  return ModelSpec(sha256='b' * 64, nbytes=1234, frame_skip=4, input_shapes=BIG,
                   output_shapes={'outputs': (1, 18452)},
                   output_slices={'hidden_state': slice(2066, 18450),
                                  'plan': slice(917, 1907)},
                   checkpoint=None)


class FakeEngine:
  """Returns something deterministic that depends on the inputs, so a wiring
  mistake between the queues and the engine cannot pass unnoticed."""

  def __init__(self, spec: ModelSpec):
    self.spec = spec
    self.inputs = {k: None for k in spec.input_shapes}
    self._host = {k: np.zeros(v, np.float16) for k, v in spec.input_shapes.items()}
    self._out = np.zeros(spec.output_nelem, np.float16)
    self.last_gpu_us = 1234
    self.calls = 0
    self.nonfinite = False

  def host_input(self, name):
    return self._host[name]

  def run(self):
    self.calls += 1
    # Fold a couple of inputs into the output so the test can verify the queues
    # actually fed the engine. Sample rather than reduce: a float16 sum over
    # 393216 elements overflows to inf, which the server correctly rejects.
    self._out[:] = self._host['img'][0, 0, 0, 0]
    self._out[0] = self._host['features_buffer'][0, 0, 0, 0]
    self._out[1] = np.float16(self.calls)
    self._out[2] = self._host['img'][0, 6, 0, 0]
    if self.nonfinite:
      self._out[5] = np.float16('nan')
    return {'outputs': self._out}


def ready_session(spec, transport, engine=None, cache='/tmp/jetlink-test-cache'):
  host = EngineHost(EngineCache(cache))
  session = Session(transport, host)
  engine = engine or FakeEngine(spec)
  host.loaded = Loaded(spec.sha256, spec, engine, PolicyQueues(spec),
                       {n: engine.host_input(n) for n in spec.input_shapes})
  session.request = Request(spec.sha256, spec.nbytes, spec.frame_skip)
  return session, engine


@pytest.fixture
def link():
  spec = make_spec()
  srv = TcpTransport.listen('127.0.0.1', 0)
  port = srv.getsockname()[1]
  client_t = TcpTransport.connect('127.0.0.1', port)
  server_t, _ = TcpTransport.accept(srv)
  srv.close()

  session, engine = ready_session(spec, server_t)
  thread = threading.Thread(target=session.serve_forever, daemon=True)
  thread.start()

  client = JetlinkClient(client_t, deadline=10.0)
  client.spec = spec
  yield client, session, engine, spec
  client.close()
  server_t.close()


def test_infer_round_trip(link):
  client, session, engine, spec = link
  rng = np.random.default_rng(0)
  warped = rng.integers(0, 256, spec.warped_shape, dtype=np.uint8)
  packed = np.zeros(spec.packed_nelem, np.float32)

  out = client.infer(warped, packed, frame_id=7)
  assert out.shape == (spec.output_nelem,)
  assert out.dtype == np.float32
  assert engine.calls == 1
  assert out[1] == 1  # the engine saw exactly one call
  gpu_us, queue_us, total_us = client.last_timings
  assert gpu_us == 1234 and total_us >= 0


def test_hidden_state_feeds_back_into_the_queues(link):
  """prev_feat goes out, comes back as features_buffer on the next frame."""
  client, session, engine, spec = link
  warped = np.zeros(spec.warped_shape, np.uint8)
  packed = np.zeros(spec.packed_nelem, np.float32)
  hidden = spec.output_slices['hidden_state']

  client.infer(warped, packed, frame_id=1)
  packed[-(hidden.stop - hidden.start):] = 3.0   # a distinctive prev_feat
  client.infer(warped, packed, frame_id=2)

  # feat_q is sampled at logical [0, 4, ... 124] of 128, so a value pushed now
  # takes several frames to appear. Drive it until it does.
  seen = False
  for i in range(3, 200):
    out = client.infer(warped, packed, frame_id=i)
    if float(out[0]) == 3.0:
      seen = True
      break
  assert seen, "prev_feat never reached the engine's features_buffer"


def test_queues_reset_flag_clears_history(link):
  client, session, engine, spec = link
  warped = np.full(spec.warped_shape, 9, np.uint8)
  packed = np.zeros(spec.packed_nelem, np.float32)
  for i in range(6):
    client.infer(warped, packed, frame_id=i)
  assert session.host.loaded.queues.img_q.buf.any()
  client.infer(np.zeros(spec.warped_shape, np.uint8), packed, frame_id=99, reset=True)
  assert not session.host.loaded.queues.img_q.buf.any()


def test_nonfinite_output_is_reported_not_returned(link):
  """openpilot treats a non-finite big-model output as fatal; so must we."""
  client, session, engine, spec = link
  engine.nonfinite = True
  with pytest.raises(LinkError, match='NOT_FINITE'):
    client.infer(np.zeros(spec.warped_shape, np.uint8),
                 np.zeros(spec.packed_nelem, np.float32))


def test_telemetry_piggybacks_without_an_extra_round_trip(link):
  client, session, engine, spec = link
  warped = np.zeros(spec.warped_shape, np.uint8)
  packed = np.zeros(spec.packed_nelem, np.float32)

  client.infer(warped, packed, want_state=False)
  assert client.last_state is None
  client.infer(warped, packed, want_state=True)
  assert client.last_state is not None
  assert 'temp_c' in client.last_state and 'power_w' in client.last_state


def test_not_ready_is_reported_rather_than_crashing():
  spec = make_spec()
  srv = TcpTransport.listen('127.0.0.1', 0)
  port = srv.getsockname()[1]
  client_t = TcpTransport.connect('127.0.0.1', port)
  server_t, _ = TcpTransport.accept(srv)
  srv.close()
  session = Session(server_t, EngineHost(EngineCache('/tmp/jetlink-test-cache')))  # nothing loaded
  threading.Thread(target=session.serve_forever, daemon=True).start()

  client = JetlinkClient(client_t, deadline=10.0)
  client.spec = spec
  try:
    with pytest.raises(LinkError, match='NOT_READY'):
      client.infer(np.zeros(spec.warped_shape, np.uint8),
                   np.zeros(spec.packed_nelem, np.float32))
  finally:
    client.close()
    server_t.close()


def test_ping_and_state_requests(link):
  client, session, engine, spec = link
  assert client.ping(timeout=5) < 5.0
  state = client.state(timeout=5)
  assert state['engine_state'] == 'ready'
  assert 'frames_served' in state
  hello = client.hello(timeout=5)
  assert hello['protocol'] == P.VERSION


def test_shutdown_replies_before_leaving_the_flag(link, tmp_path):
  from jetlink.server import power
  client, session, engine, spec = link
  session.host.cache.root = tmp_path
  resp = client.shutdown('car battery', timeout=5)
  assert resp['ok'] is True
  flag = power.flag_path(tmp_path)
  deadline = time.monotonic() + 2.0
  while not flag.exists() and time.monotonic() < deadline:
    time.sleep(0.01)
  assert 'car battery' in flag.read_text()
  # The link is still usable: the host is what goes down, not the session.
  assert client.ping(timeout=5) < 5.0


def test_a_stale_poweroff_flag_is_removed_at_startup(tmp_path):
  from jetlink.server import power
  assert power.request_poweroff(tmp_path, 'test')
  power.clear_stale_flag(tmp_path)
  assert not power.flag_path(tmp_path).exists()
  power.clear_stale_flag(tmp_path)  # and nothing to do is not an error


def test_wrong_sized_request_is_rejected(link):
  """A client on a different model must be told, not silently fed garbage.

  The server reads `packed` at an offset derived from its own spec, so a
  mismatched request would have the scalars and the 16384-float hidden state
  read out of the middle of the image - and the result would look finite.
  """
  client, session, engine, spec = link
  seq = client._next_seq()
  client.t.send(P.Msg.INFER_REQ, seq, (P.pack_infer_req(1, 0), b'\x00' * 1000))
  msg = client._expect(P.Msg.INFER_RESP, seq, 5.0)
  _, status, _, _, _ = P.unpack_infer_resp(msg.payload)
  assert status == P.Status.BAD_SHAPE
  assert engine.calls == 0, "the engine must not have run on a malformed request"


# -- the engine cache -------------------------------------------------------

def _plan(cache: EngineCache, name: str, mtime: float) -> Path:
  """A plan and its sidecar, stamped at `mtime`."""
  plan = cache.engines / f"{name}.plan"
  plan.write_bytes(b'plan')
  plan.with_suffix('.json').write_text('{}')
  os.utime(plan, (mtime, mtime))
  return plan


def test_prune_keeps_the_newest_plans(tmp_path):
  cache = EngineCache(tmp_path)
  for i in range(4):
    _plan(cache, f"m{i}", 1_000_000 + i)
  cache.prune(keep=2)
  left = sorted(p.stem for p in cache.engines.glob('*.plan'))
  assert left == ['m2', 'm3']
  assert not (cache.engines / 'm0.json').exists()


def test_prune_never_drops_the_plan_just_built(tmp_path):
  """The Jetson boots at 1970 with no network, so a fresh plan can be the
  oldest file on disk. Pruning by mtime would delete the build that just
  finished and leave the caller reading a sidecar that no longer exists."""
  cache = EngineCache(tmp_path)
  _plan(cache, 'old_a', 2_000_000)
  _plan(cache, 'old_b', 2_000_001)
  fresh = _plan(cache, 'fresh', 1)          # 1970, but it is the new one

  cache.prune(keep=2, protect=fresh)

  assert fresh.is_file()
  assert fresh.with_suffix('.json').is_file()
  assert len(list(cache.engines.glob('*.plan'))) == 2


def test_sweep_temp_drops_only_stale_build_dirs(tmp_path):
  cache = EngineCache(tmp_path)
  stale = cache.engines / 'tmpstale'
  fresh = cache.engines / 'tmpfresh'
  for d in (stale, fresh):
    d.mkdir()
    (d / 'engine.plan').write_bytes(b'x')
  os.utime(stale, (time.time() - 7 * 3600, time.time() - 7 * 3600))

  cache.sweep_temp()

  assert not stale.exists()
  assert fresh.exists()


class TestPreload:
  """A fresh server process has no engine loaded, and the first client pays the
  deserialize. Offroad jetlinkd absorbs that, but at an ignition-on cold start
  manager runs no jetlinkd, so it lands on modeld's join - at the end, after
  the comma has already waited out the Jetson's boot."""

  def _cache(self, tmp_path, spec, with_spec=True):
    cache = EngineCache(tmp_path)
    entry = cache.entry(spec.sha256)
    entry.plan_path.write_bytes(b'plan')
    meta = {'spec': spec.to_dict()} if with_spec else {'trt_version': 'x'}
    entry.write_meta(meta)
    cache.remember_loaded(spec.sha256, spec.frame_skip)
    return cache

  def test_the_last_loaded_engine_is_remembered_and_preloaded(self, tmp_path):
    spec = make_spec()
    cache = self._cache(tmp_path, spec)
    assert cache.last_loaded() == (spec.sha256, spec.frame_skip)

    host = EngineHost(cache)
    started = []
    host._start = lambda job, req, entry, mp, sp: started.append((job.sha256, job.load_only, sp.frame_skip))
    host.preload()
    assert started == [(spec.sha256, True, spec.frame_skip)]

  def test_nothing_is_preloaded_without_a_marker(self, tmp_path):
    host = EngineHost(EngineCache(tmp_path))
    started = []
    host._start = lambda *a: started.append(a)
    host.preload()
    assert started == []

  def test_a_plan_whose_sidecar_has_no_spec_is_left_alone(self, tmp_path):
    # Deriving one means parsing the ONNX, which is real work to do on a guess.
    spec = make_spec()
    cache = self._cache(tmp_path, spec, with_spec=False)
    host = EngineHost(cache)
    started = []
    host._start = lambda *a: started.append(a)
    host.preload()
    assert started == []

  def test_a_loaded_engine_is_not_preloaded_over(self, tmp_path):
    spec = make_spec()
    cache = self._cache(tmp_path, spec)
    host = EngineHost(cache)
    host.loaded = Loaded(spec.sha256, spec, object(), None, {})
    started = []
    host._start = lambda *a: started.append(a)
    host.preload()
    assert started == []

  def test_a_different_frame_skip_is_not_served_from_the_loaded_engine(self, tmp_path):
    """request() matched on sha alone, so a client asking for a different
    frame_skip got the spec the previous one asked for. A preload guesses the
    frame_skip from the marker, which makes that mismatch reachable."""
    spec = make_spec()
    host = EngineHost(EngineCache(tmp_path))
    engine = FakeEngine(spec)
    host.loaded = Loaded(spec.sha256, spec, engine, PolicyQueues(spec),
                         {n: engine.host_input(n) for n in spec.input_shapes})
    same = host.request(Request(spec.sha256, spec.nbytes, spec.frame_skip), None)
    assert same['state'] == 'ready'

    started = []
    host._start = lambda *a: started.append(a)
    other = host.request(Request(spec.sha256, spec.nbytes, spec.frame_skip + 1), None)
    assert other['state'] != 'ready' or started, "served a spec the client did not ask for"


class TestTimingCache:
  """TensorRT re-times candidate kernels on every build. Most of those timings
  do not depend on the model: a warm cache cut a Lebowski build from 254 s to
  173 s. It is advisory, so every path here has to fail open - a bad cache
  costs a slow build, never a wrong engine."""

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
    from jetlink.server import builder as B
    p = tmp_path / 'timing.cache'
    p.write_bytes(b'previous')
    cfg = self.FakeConfig()
    cache = B._load_timing_cache(cfg, p)
    assert cfg.seeded == b'previous'
    assert cfg.set_with is not None and cache is not None

  def test_a_missing_cache_still_builds(self, tmp_path):
    from jetlink.server import builder as B
    cfg = self.FakeConfig()
    assert B._load_timing_cache(cfg, tmp_path / 'nope.cache') is not None
    assert cfg.seeded == b''   # empty seed, build runs cold

  def test_a_cache_tensorrt_rejects_does_not_fail_the_build(self, tmp_path):
    # A cache from another TensorRT version, or truncated by a killed build.
    from jetlink.server import builder as B
    p = tmp_path / 'timing.cache'
    p.write_bytes(b'garbage')
    cfg = self.FakeConfig(raises=RuntimeError('version mismatch'))
    assert B._load_timing_cache(cfg, p) is None

  def test_the_cache_is_written_atomically(self, tmp_path):
    from jetlink.server import builder as B
    p = tmp_path / 'sub' / 'timing.cache'
    B._save_timing_cache(self.FakeCache(b'fresh'), p)
    assert p.read_bytes() == b'fresh'
    # No .tmp left for the next build to mistake for a cache.
    assert list(p.parent.glob('*.tmp')) == []

  def test_an_unwritable_cache_is_not_an_error(self, tmp_path):
    from jetlink.server import builder as B
    B._save_timing_cache(self.FakeCache(), tmp_path / 'nodir' / 'x' / 'c.cache')
    B._save_timing_cache(None, tmp_path / 'c.cache')   # nothing to write

  def test_the_cache_is_keyed_like_the_plans(self, tmp_path):
    from jetlink.server.builder import EngineCache
    name = EngineCache(tmp_path).timing_cache().name
    # A timing from another TensorRT version or another chip is not a timing.
    assert name.startswith('timing.trt') and name.endswith('.cache')
