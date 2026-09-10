"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

End-to-end: a real client talking to a real Session over a real socket.

Only the engine is faked, behind the backend seam. Everything else - framing,
the infer request/response encoding, the history queues, the hidden-state
feedback, the piggybacked telemetry - is the code that will run in the car.
This is the test that covers the paths the car would otherwise be the first to
execute, and it imports no inference runtime at all.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from jetlink import protocol as P
from jetlink.client import JetlinkClient
from jetlink.queues import PolicyQueues
from jetlink.server.cache import EngineCache
from jetlink.server.session import EngineHost, Job, Loaded, Request, Session
from jetlink.spec import ModelSpec
from jetlink.transport.base import LinkError
from jetlink.transport.tcp import TcpTransport
from tests.fake_backend import FakeBackend, FakeEngine

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


@pytest.mark.parametrize('identity', ['../escape', '/tmp/escape', '', 'a'*63, 'z'*64])
def test_a_request_for_a_bad_model_identity_is_refused(identity):
  with pytest.raises(ValueError, match='SHA-256'):
    Request(identity, 100, 4)


@pytest.mark.parametrize('payload', [b'', (1234).to_bytes(8, 'little') + b'x'])
def test_invalid_upload_does_not_create_a_file(tmp_path, payload):
  from types import SimpleNamespace
  sent = []
  spec = make_spec()
  session, _ = ready_session(spec, SimpleNamespace(send=lambda *args: sent.append(args)), cache=tmp_path)
  session.on_upload_chunk(SimpleNamespace(seq=1, payload=memoryview(payload)))
  assert sent[0][0] == P.Msg.ERROR
  assert not session.host.cache.model_path(spec.sha256).exists()


@pytest.mark.parametrize('kind', ['wrong_frame', 'short_header', 'short_output'])
def test_invalid_inference_response_abandons_the_stream(kind):
  from types import SimpleNamespace
  spec = make_spec()
  payload = P.pack_infer_resp(42 if kind == 'wrong_frame' else 7, P.Status.OK, 0, 0, 0)
  payload += bytes(spec.output_nbytes)
  if kind == 'short_header':
    payload = payload[:P.INFER_RESP_SIZE-1]
  elif kind == 'short_output':
    payload = payload[:-1]
  transport = SimpleNamespace(send=lambda *a, **kw: None,
                              recv=lambda **kw: SimpleNamespace(msg_type=P.Msg.INFER_RESP, seq=1,
                                                               payload=memoryview(payload)))
  client = JetlinkClient(transport)
  client.spec = spec
  seq = client.infer_begin(bytes(spec.warped_nbytes), bytes(spec.packed_nbytes), frame_id=7)
  with pytest.raises(LinkError):
    client.infer_end(seq)
  assert client.dead
  with pytest.raises(LinkError, match='previously failed'):
    client.infer_begin(bytes(spec.warped_nbytes), bytes(spec.packed_nbytes), frame_id=8)


class FakeSensor:
  """Health as a Jetson's sysfs would report it; the server no longer reads
  zeros off a host with no sensors."""

  def read(self):
    return {'temp_c': 48.5, 'power_w': 7.2, 'gpu_load_pct': 12}


def ready_session(spec, transport, engine=None, cache='/tmp/jetlink-test-cache'):
  host = EngineHost(EngineCache(cache, FakeBackend(spec)), telemetry=FakeSensor())
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
  thread.join(1.0)
  session.host.close()


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

  # Sampling is asynchronous; startup may legitimately carry no health yet.
  health = session.host.telemetry
  health.read()
  with health._cv:
    assert health._cv.wait_for(lambda: bool(health._sample), timeout=1.0)

  client.infer(warped, packed, want_state=False)
  assert client.last_state is None
  client.infer(warped, packed, want_state=True)
  assert client.last_state is not None
  assert 'temp_c' in client.last_state and 'power_w' in client.last_state


def test_blocked_sensor_does_not_delay_following_inferences(link):
  from jetlink.server.telemetry import CachedTelemetry

  client, session, engine, spec = link
  entered, release = threading.Event(), threading.Event()

  class Sensor:
    calls = 0

    def read(self):
      self.calls += 1
      entered.set()
      assert release.wait(3.0)
      return {'temp_c': 65.0}

  sensor = Sensor()
  session.host.telemetry.close()
  health = CachedTelemetry(sensor)
  session.telemetry = session.host.telemetry = health
  try:
    health.read()
    assert entered.wait(1.0)
    warped = np.zeros(spec.warped_shape, np.uint8)
    packed = np.zeros(spec.packed_nelem, np.float32)
    for frame in range(3):
      client.infer(warped, packed, frame_id=frame, want_state=True, deadline=0.5)
      assert client.last_state == {}  # unavailable health must not look valid
    assert engine.calls == 3
    assert sensor.calls == 1
  finally:
    release.set()
    health.close()


def test_not_ready_is_reported_rather_than_crashing():
  spec = make_spec()
  srv = TcpTransport.listen('127.0.0.1', 0)
  port = srv.getsockname()[1]
  client_t = TcpTransport.connect('127.0.0.1', port)
  server_t, _ = TcpTransport.accept(srv)
  srv.close()
  session = Session(server_t, EngineHost(EngineCache('/tmp/jetlink-test-cache', FakeBackend())))  # nothing loaded
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


def test_frame_deadline_includes_time_spent_sending(link, monkeypatch):
  client, _, _, spec = link
  send = client.t.send

  def delayed_send(*args, **kwargs):
    send(*args, **kwargs)
    time.sleep(0.08)

  monkeypatch.setattr(client.t, 'send', delayed_send)
  client.deadline = 0.05
  with pytest.raises(LinkError, match='link abandoned'):
    client.infer(np.zeros(spec.warped_nbytes // 4, np.float32),
                 np.zeros(spec.packed_nelem, np.float32))
  assert client.dead


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

  The server reads `packed` at an offset from its own spec, so a mismatched
  request reads the scalars out of the middle of the image, and looks finite.
  """
  client, session, engine, spec = link
  seq = client._next_seq()
  client.t.send(P.Msg.INFER_REQ, seq, (P.pack_infer_req(1, 0), b'\x00' * 1000))
  msg = client._expect(P.Msg.INFER_RESP, seq, 5.0)
  _, status, _, _, _ = P.unpack_infer_resp(msg.payload)
  assert status == P.Status.BAD_SHAPE
  assert engine.calls == 0, "the engine must not have run on a malformed request"


class TestPreload:
  """A fresh server process has no engine loaded and the first client pays the
  deserialize. Offroad jetlinkd absorbs that; at an ignition-on cold start there
  is no jetlinkd and it lands on modeld's join instead."""

  def _cache(self, tmp_path, spec, with_spec=True):
    cache = EngineCache(tmp_path, FakeBackend(spec))
    entry = cache.entry(spec.sha256)
    entry.path.write_bytes(b'plan')
    meta = {'spec': spec.to_dict()} if with_spec else {'trt_version': 'x'}
    entry.write_meta(meta)
    cache.remember_loaded(spec.sha256, spec.frame_skip)
    return cache

  @pytest.mark.parametrize('requested_sha', ['b' * 64, 'c' * 64])
  def test_request_during_load_returns_without_locking_out_completion(self, tmp_path, requested_sha):
    host = EngineHost(EngineCache(tmp_path, FakeBackend()))
    host.job = Job('b' * 64, load_only=True)
    result = []
    worker = threading.Thread(target=lambda: result.append(
      host.request(Request(requested_sha, 1234, 4), None)), daemon=True)
    worker.start()
    worker.join(1.0)
    assert not worker.is_alive(), 'request deadlocked while an engine was loading'
    assert result[0]['state'] == 'building'
    assert host.lock.acquire(timeout=1.0), 'the load worker cannot publish its result'
    host.lock.release()

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
    host = EngineHost(EngineCache(tmp_path, FakeBackend()))
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
    host = EngineHost(EngineCache(tmp_path, FakeBackend()))
    engine = FakeEngine(spec)
    host.loaded = Loaded(spec.sha256, spec, engine, PolicyQueues(spec),
                         {n: engine.host_input(n) for n in spec.input_shapes})
    same = host.request(Request(spec.sha256, spec.nbytes, spec.frame_skip), None)
    assert same['state'] == 'ready'

    started = []
    host._start = lambda *a: started.append(a)
    other = host.request(Request(spec.sha256, spec.nbytes, spec.frame_skip + 1), None)
    assert other['state'] != 'ready' or started, "served a spec the client did not ask for"


class TestRequestWithAPartialModel:
  """While the server has no engine the comma asks again every few seconds, and
  an upload cut short (a jetlinkd to modeld handover, a dropped link) leaves
  its first chunks in models/. Every one of those requests found the file,
  parsed it, and logged a traceback for the upload it then asked for anyway."""

  def _host(self, tmp_path, spec, nbytes_on_disk, with_entry=False):
    host = EngineHost(EngineCache(tmp_path, FakeBackend(spec)))
    host.cache.model_path(spec.sha256).write_bytes(b'x' * nbytes_on_disk)
    if with_entry:
      # An artifact whose sidecar predates specs: the one case that needs the ONNX
      entry = host.cache.entry(spec.sha256)
      entry.path.write_bytes(b'plan')
      entry.write_meta({'trt_version': 'x'})
    return host

  @pytest.mark.parametrize('with_entry', [False, True])
  def test_a_partial_upload_is_not_parsed(self, tmp_path, caplog, with_entry):
    spec = make_spec()
    host = self._host(tmp_path, spec, spec.nbytes // 2, with_entry)
    parsed, started = [], []
    host._derive_spec = lambda *a: parsed.append(a)
    host._start = lambda *a: started.append(a)
    resp = host.request(Request(spec.sha256, spec.nbytes, spec.frame_skip), None)
    assert resp['state'] == 'need_upload'
    assert f'have {spec.nbytes // 2}' in resp['detail']
    assert parsed == [] and started == []
    assert 'could not derive' not in caplog.text

  def test_a_whole_upload_is_parsed_and_built(self, tmp_path):
    spec = make_spec()
    host = self._host(tmp_path, spec, spec.nbytes)
    host._derive_spec = lambda *a: spec
    started = []
    host._start = lambda job, req, entry, mp, sp: started.append((job.load_only, sp))
    host.request(Request(spec.sha256, spec.nbytes, spec.frame_skip), None)
    assert started == [(False, spec)]


def test_slow_reply_send_is_logged_even_when_inference_is_fast(tmp_path, monkeypatch, caplog):
  from types import SimpleNamespace

  from jetlink.server import session as session_module
  from jetlink.transport.base import Message

  clock = [1.0]

  def send(*args):
    clock[0] += .02

  spec = make_spec()
  session, engine = ready_session(spec, SimpleNamespace(send=send), cache=tmp_path)
  monkeypatch.setattr(session_module, 'time', SimpleNamespace(perf_counter=lambda: clock[0]))
  payload = P.pack_infer_req(42, 0) + bytes(spec.warped_nbytes + spec.packed_nbytes)
  try:
    session.on_infer(Message(P.Msg.INFER_REQ, 1, 0, memoryview(payload)))
    assert engine.calls == 1
    assert 'slow frame 42:' in caplog.text
    assert 'total 0.0 send 20.0 ms' in caplog.text
  finally:
    session.host.close()


class TestHello:
  """A hello means a new client process, whatever the seq says.

  A comma process that takes the gadget over from another one starts its seqs
  at 1, and the server may still be inside the session the last one left: the
  Jetson's hub driver can keep the usb_device across a rebind, so nothing here
  ever saw a disconnect. Dropping that hello as a replay left modeld with a
  drained write and no answer, and the drive on the small model.
  """

  def _session(self, tmp_path, sleep_after=0.0):
    sent = []
    host = EngineHost(EngineCache(tmp_path, FakeBackend(make_spec())),
                      telemetry=FakeSensor(), sleep_after=sleep_after)
    return Session(SimpleNamespace(send=lambda *args: sent.append(args)), host), sent

  @staticmethod
  def _hello(seq, name='modeld', nonce='deadbeef'):
    payload = json.dumps({'client': {'nonce': nonce, 'name': name}}).encode()
    return SimpleNamespace(msg_type=P.Msg.HELLO_REQ, seq=seq, payload=memoryview(payload))

  @staticmethod
  def _resp(sent):
    return json.loads(bytes(sent[0][2][0]))

  def test_a_hello_is_answered_on_a_session_that_outlived_its_client(self, tmp_path):
    session, sent = self._session(tmp_path)
    spec = make_spec()
    session.last_seq = 5000
    session.frames = 42
    session.request = Request(spec.sha256, spec.nbytes, spec.frame_skip)

    session.handle(self._hello(1))

    assert sent, "the hello was dropped as a replay"
    assert sent[0][0] == P.Msg.HELLO_RESP and sent[0][1] == 1
    assert session.last_seq == 1, "the new client's seqs are counted from its own hello"
    assert session.request is None and session.frames == 0

  def test_replays_are_still_dropped_after_the_reset(self, tmp_path):
    session, sent = self._session(tmp_path)
    session.handle(self._hello(1))
    sent.clear()
    ping = SimpleNamespace(msg_type=P.Msg.PING, seq=1, payload=memoryview(b''))
    session.handle(ping)
    assert not sent, "a message at the hello's own seq is a replay"
    session.handle(SimpleNamespace(msg_type=P.Msg.PING, seq=2, payload=memoryview(b'')))
    assert sent[0][0] == P.Msg.PONG

  def test_the_journal_names_the_client_that_is_talking(self, tmp_path, caplog):
    session, _ = self._session(tmp_path)
    with caplog.at_level(logging.INFO, logger='jetlink.server'):
      session.handle(self._hello(7, name='jetlinkd', nonce='0badcafe'))
      session.handle(SimpleNamespace(msg_type=P.Msg.PING, seq=7, payload=memoryview(b'')))
    assert 'jetlinkd/0badcafe' in caplog.text
    assert 'dropping replayed message' in caplog.text

  def test_a_client_that_names_nothing_is_still_served(self, tmp_path):
    session, sent = self._session(tmp_path)
    session.handle(SimpleNamespace(msg_type=P.Msg.HELLO_REQ, seq=3, payload=memoryview(b'')))
    assert sent[0][0] == P.Msg.HELLO_RESP

  @pytest.mark.parametrize('sleep_after', [0.0, 120.0])
  def test_the_hello_says_whether_this_server_sleeps(self, tmp_path, sleep_after):
    # The comma only lets go of the gadget when parked if letting go buys the
    # far end a suspend; see jetlinkd.go_dormant.
    session, sent = self._session(tmp_path, sleep_after=sleep_after)
    session.handle(self._hello(1))
    assert self._resp(sent)['sleep_after'] == sleep_after


class TestEngineStateWithoutAnUpload:
  """The server asked for a gigabyte it already had.

  Only request() consulted the cache, so every other answer for a plan on disk
  was need_upload: modeld carries no ONNX and waited out its whole
  build_timeout before falling back.
  """

  def _cached(self, tmp_path, spec):
    cache = EngineCache(tmp_path, FakeBackend(spec))
    entry = cache.entry(spec.sha256)
    entry.path.write_bytes(b'plan')
    entry.write_meta({'spec': spec.to_dict()})
    return cache

  def test_a_plan_on_disk_is_not_reported_as_needing_an_upload(self, tmp_path):
    spec = make_spec()
    host = EngineHost(self._cached(tmp_path, spec))
    # a preload that guessed another sha, finished and still holds the job slot
    host.job = Job('c' * 64, load_only=True, state='ready')
    assert host.status(spec.sha256, spec.frame_skip)['state'] == 'building'

  def test_a_model_that_really_is_absent_still_asks_for_the_upload(self, tmp_path):
    host = EngineHost(EngineCache(tmp_path, FakeBackend(make_spec())))
    assert host.status('d' * 64, 4)['state'] == 'need_upload'

  def test_a_finished_job_for_another_model_starts_the_one_asked_for(self, tmp_path):
    spec = make_spec()
    cache = self._cached(tmp_path, spec)
    host = EngineHost(cache, telemetry=FakeSensor())
    sent = []
    session = Session(SimpleNamespace(send=lambda *a: sent.append(a)), host)
    session.request = Request(spec.sha256, spec.nbytes, spec.frame_skip)
    other = FakeEngine(spec)
    host.loaded = Loaded('c' * 64, spec, other, PolicyQueues(spec), {})
    host.job = Job('c' * 64, load_only=True, state='ready')
    started = []
    host._start = lambda job, req, entry, mp, sp: started.append(job.sha256)

    session.engine_update()

    assert started == [spec.sha256], "nobody loaded the model this client asked for"
    assert sent[0][0] == P.Msg.ENGINE_RESP
    assert json.loads(bytes(sent[0][2][0]))['state'] == 'building'
