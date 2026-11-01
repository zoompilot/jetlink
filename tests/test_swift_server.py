"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The comma's client against the Swift server, live.

The comma is the peer that matters, so the contract is checked from its side:
jetlink.client over TCP against a real `jetlink-server` on onnxruntime's CPU
provider, serving tests/tiny_model.py's graphs (the committed copies the Swift
golden frames use). Outputs are held to Python's own staging (jetlink.queues)
and a numpy run of the graph, the hidden state fed back as the server feeds it.

The binary comes from JETLINK_SERVER_BIN, else the newest SwiftPM build in
JetlinkKit/.build; JETLINK_SERVER_BUILD=1 builds it first. Without one this
skips.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from unittest import mock
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from jetlink import protocol as P
from jetlink.client import EngineMissing, JetlinkClient
from jetlink.queues import PolicyQueues
from jetlink.spec import DRIVING_OUTPUT, ModelSpec
from jetlink.transport.base import LinkError, LinkTimeout
from jetlink.transport.tcp import TcpTransport
from tests import tiny_model

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'JetlinkKit'
FIXTURES = PACKAGE / 'Tests' / 'JetlinkServerTests' / 'Fixtures'
QUEUED, STATEFUL = FIXTURES / 'tiny_queued.onnx', FIXTURES / 'tiny_stateful.onnx'
HELLO_KEYS = {'protocol', 'backend', 'runtime_version', 'device', 'engine_state', 'loaded', 'frames_served', 'cached_models',
              'telemetry', 'sleep_after', 'pick', 'fetches'}


def _binary() -> Path | None:
  named = os.environ.get('JETLINK_SERVER_BIN')
  if named:
    if not os.access(named, os.X_OK):
      raise RuntimeError(f'JETLINK_SERVER_BIN={named} is not an executable')
    return Path(named)
  if os.environ.get('JETLINK_SERVER_BUILD') == '1':
    subprocess.run(['swift', 'build', '--package-path', str(PACKAGE), '--product', 'jetlink-server'], check=True)
  built = [p for p in (PACKAGE / '.build' / c / 'jetlink-server' for c in ('debug', 'release')) if os.access(p, os.X_OK)]
  return max(built, key=lambda p: p.stat().st_mtime, default=None)


BIN = _binary()
pytestmark = pytest.mark.skipif(BIN is None, reason='no jetlink-server: set JETLINK_SERVER_BIN, or build the jetlink-server product')


def spec_of(path: Path) -> ModelSpec:
  """The spec Python derives, as committed beside the graph: no onnx package
  needed here, so this runs anywhere numpy does."""
  return ModelSpec.load(path.with_suffix('.spec.json'))


@dataclass
class Server:
  proc: subprocess.Popen
  port: int
  cache: Path
  log: Path
  provisioned: dict = field(default_factory=dict)

  def connect(self) -> JetlinkClient:
    return JetlinkClient.open_tcp('127.0.0.1', self.port, deadline=10.0, name='test_swift_server')

  def tail(self) -> str:
    return '\n'.join(self.log.read_text(errors='replace').splitlines()[-40:])

  def provision(self, path: Path) -> tuple[ModelSpec, list[str]]:
    """Upload and build `path` once, as a provisioning run does; (served spec, stages)."""
    if path not in self.provisioned:
      spec, stages = spec_of(path), []
      client = self.connect()
      try:
        served = client.ensure_engine(spec.sha256, spec.nbytes, onnx_path=path, progress=lambda s, f, m: stages.append(s),
                                      build_timeout=120.0)
      finally:
        client.close()
      self.provisioned[path] = (served, stages)
    return self.provisioned[path]


def _free_port() -> int:
  with socket.socket() as s:
    s.bind(('127.0.0.1', 0))
    return s.getsockname()[1]


def _wait_listening(server: Server, timeout: float = 30.0) -> None:
  end = time.monotonic() + timeout
  while time.monotonic() < end:
    if server.proc.poll() is not None:
      raise AssertionError(f'jetlink-server exited with {server.proc.returncode}:\n{server.tail()}')
    try:
      socket.create_connection(('127.0.0.1', server.port), timeout=0.2).close()
      return
    except OSError:
      time.sleep(0.1)
  raise AssertionError(f'jetlink-server never listened on {server.port}:\n{server.tail()}')


@contextmanager
def running(tmp: Path, *extra: str, env: dict | None = None):
  cache = tmp / 'cache'
  cache.mkdir(parents=True)
  # Run without --poweroff, a shutdown must never power a machine off; a
  # stand-in on PATH catches a server that tried
  fake = tmp / 'bin'
  fake.mkdir()
  (fake / 'systemctl').write_text(f'#!/bin/sh\necho "$@" >> "{tmp / "systemctl.called"}"\n')
  (fake / 'systemctl').chmod(0o755)
  env = dict(os.environ, PATH=f'{fake}{os.pathsep}{os.environ.get("PATH", "")}', JETLINK_CACHE=str(cache), **(env or {}))
  log = tmp / 'server.log'
  port = _free_port()
  with open(log, 'wb') as out:
    proc = subprocess.Popen([str(BIN), 'serve', '--backend', 'ort', '--device', 'cpu', '--listen', '--host', '127.0.0.1',
                             '--port', str(port), '--cache', str(cache), '--log-level', 'debug', *extra],
                            stdout=out, stderr=subprocess.STDOUT, env=env)
  server = Server(proc, port, cache, log)
  try:
    _wait_listening(server)
    yield server
  finally:
    proc.send_signal(signal.SIGTERM)
    try:
      proc.wait(10.0)
    except subprocess.TimeoutExpired:
      proc.kill()
      proc.wait()


@pytest.fixture(scope='module')
def server(tmp_path_factory):
  with running(tmp_path_factory.mktemp('swift-server')) as s:
    yield s


@pytest.fixture(scope='module')
def bare(tmp_path_factory):
  """A server that has never seen a model."""
  with running(tmp_path_factory.mktemp('swift-bare'), '--no-preload') as s:
    yield s


@contextmanager
def ready(server: Server, path: Path):
  server.provision(path)
  spec = spec_of(path)
  client = server.connect()
  try:
    client.ensure_engine(spec.sha256, spec.nbytes, onnx_path=None, build_timeout=60.0)
    yield client
  finally:
    client.close()


@pytest.fixture
def queued(server):
  with ready(server, QUEUED) as client:
    yield client


@pytest.fixture
def stateful(server):
  with ready(server, STATEFUL) as client:
    yield client


def close_enough(got: np.ndarray, want: np.ndarray, atol: float) -> None:
  assert np.all(np.isfinite(got))
  assert np.corrcoef(got, want)[0, 1] > 0.999
  np.testing.assert_allclose(got, want, atol=atol, rtol=atol)


# -- the queued graph: the server keeps the history and the hidden state -------

def queued_frames(n: int, seed: int = 0):
  """(warped, packed) per frame.

  Pixels up to 15, not 255: at full range the image terms are 80 and fp16's
  rounding of them (0.1) is as big as a scalar the queues dropped (0.2).
  """
  spec, rng = spec_of(QUEUED), np.random.default_rng(seed)
  frames = []
  for _ in range(n):
    warped = rng.integers(0, 16, spec.warped_shape, dtype=np.uint8)
    packed = (rng.standard_normal(spec.packed_nelem) * 0.5).astype(np.float32)
    frames.append((warped, packed))
  return frames


def queued_reference(frames, resets=(0,), hellos=(), fed=None, dropped=()) -> list[np.ndarray]:
  """Python's staging (the reference the Swift is locked to) into a numpy run,
  each frame's hidden state fed into the next as the server feeds it: from
  `fed`, the server's own outputs as WANT_HIDDEN returned them, else from the
  reference's. `hellos` start the feedback over; `dropped` frames feed nothing."""
  queues, outs = PolicyQueues(spec_of(QUEUED)), []
  for i, (warped, packed) in enumerate(frames):
    if i in resets:
      queues.reset()
    if i in hellos:
      queues.new_client()
    outs.append(tiny_model.reference(queues.step(warped, packed)))
    if i not in dropped:
      queues.after_run({DRIVING_OUTPUT: outs[-1] if fed is None else fed[i]})
  return outs


def run(client, frames, resets=(0,)) -> list[np.ndarray]:
  return [client.infer(w, p, frame_id=i + 1, reset=i in resets) for i, (w, p) in enumerate(frames)]


def outside(out: np.ndarray, spec: ModelSpec) -> np.ndarray:
  """The output less hidden_state: what a reply carries."""
  start, stop = spec.hidden_range
  return np.concatenate([out[:start], out[stop:]])


def raw_infer(client, warped, packed, frame_id: int, flags: int = 0) -> tuple[int, np.ndarray]:
  """One INFER straight through the transport, past the client's rules:
  (status, the floats the reply carried)."""
  seq = client._next_seq()
  client.t.send(P.Msg.INFER_REQ, seq, (P.pack_infer_req(frame_id, flags), warped, packed))
  payload = client._expect(P.Msg.INFER_RESP, seq, 10.0).payload
  _, status, _, _, _ = P.unpack_infer_resp(payload)
  return status, np.frombuffer(payload, np.float32, offset=P.INFER_RESP_SIZE).copy()


def test_upload_build_and_load_through_the_link(server):
  spec, stages = server.provision(QUEUED)
  assert spec.to_dict() == spec_of(QUEUED).to_dict()   # the spec the server derives is Python's
  assert {'upload', 'load'} <= set(stages)
  # the sidecar carries the spec, so a later load needs neither the ONNX nor a parser
  sidecars = [json.loads(p.read_text()) for p in (server.cache / 'engines').glob(f'{spec.sha256[:16]}.*.json')]
  assert [m['backend'] for m in sidecars] == ['ort']
  assert ModelSpec.from_dict(sidecars[0]['spec']).to_dict() == spec.to_dict()
  assert (server.cache / 'models' / f'{spec.sha256[:16]}.onnx').stat().st_size == spec.nbytes


def test_infer_round_trip(queued):
  spec = queued.spec
  frames = queued_frames(10)
  outs = run(queued, frames)
  assert all(o.shape == (spec.output_nelem,) and o.dtype == np.float32 for o in outs)
  start, stop = spec.hidden_range
  # the server kept the hidden state: it reads as zeros, everything else as the graph's
  assert all(not o[start:stop].any() for o in outs)
  for got, want in zip(outs, queued_reference(frames), strict=True):
    close_enough(outside(got, spec), outside(want, spec), atol=0.02)
  gpu_us, queue_us, total_us = queued.last_timings
  assert gpu_us >= 0 and queue_us >= 0 and total_us >= gpu_us


def test_frames_sent_without_waiting_are_answered_in_order_and_drained(queued):
  """What the comma does while the small model drives: every frame goes out,
  none is waited for, and the answers are read a frame later. The server
  cannot tell them from driven frames: the history advances the same way."""
  spec = queued.spec
  frames = queued_frames(6, seed=3)
  for i, (w, p) in enumerate(frames[:4]):
    queued.infer_begin(w, p, frame_id=i + 1, reset=i == 0)
  assert queued.unanswered == 4
  deadline = time.monotonic() + 10.0
  while queued.unanswered and time.monotonic() < deadline:
    queued.drain()
    time.sleep(0.01)
  assert queued.unanswered == 0 and not queued.dead
  want = queued_reference(frames)
  close_enough(outside(queued.last_output, spec), outside(want[3], spec), atol=0.02)
  # the frames after them are the server's fifth and sixth of the same history
  outs = [queued.infer(w, p, frame_id=i + 5) for i, (w, p) in enumerate(frames[4:])]
  for got, w in zip(outs, want[4:], strict=True):
    close_enough(outside(got, spec), outside(w, spec), atol=0.02)
  # and a frame given up on is read quietly by the next one's wait. A hold
  # of 0 still takes a reply already in: with the tiny model answering in
  # under a millisecond, either outcome is right here
  w, p = queued_frames(1, seed=4)[0]
  seq = queued.infer_begin(w, p, frame_id=7)
  held = queued.infer_end(seq, hold=0.0)
  assert queued.unanswered == (1 if held is None else 0) and not queued.dead
  out = queued.infer(w, p, frame_id=8)
  assert queued.unanswered == 0
  close_enough(outside(out, spec), outside(queued_reference(frames + [(w, p), (w, p)])[7], spec), atol=0.02)


def test_hidden_state_feeds_back_into_the_queues(queued):
  """Each frame's hidden state comes back as features_buffer on the next,
  exactly the server's own: the reference fed the values WANT_HIDDEN returned
  matches frame for frame, and one fed nothing does not."""
  queued.want_hidden = True
  frames = queued_frames(12, seed=1)
  outs = run(queued, frames)
  assert all(np.abs(o[slice(*queued.spec.hidden_range)]).max() > 0.1 for o in outs)
  for got, want in zip(outs, queued_reference(frames, fed=outs), strict=True):
    close_enough(got, want, atol=0.02)
  without = queued_reference(frames, dropped=range(len(frames)))
  assert max(np.abs(o - w).max() for o, w in zip(outs, without, strict=True)) > 0.5, "the hidden state never reached the engine"


def test_the_reply_leaves_the_hidden_state_out_unless_asked(queued):
  spec = queued.spec
  warped, packed = queued_frames(1, seed=5)[0]
  status, compact = raw_infer(queued, warped, packed, 1, P.Flag.RESET_QUEUES)
  assert status == P.Status.OK and compact.size == spec.reply_nelem
  status, whole = raw_infer(queued, warped, packed, 2, P.Flag.RESET_QUEUES | P.Flag.WANT_HIDDEN)
  assert status == P.Status.OK and whole.size == spec.output_nelem
  assert np.array_equal(compact, outside(whole, spec))   # the same frame from the same history, bit for bit


def test_queues_reset_flag_clears_history(queued):
  frames = queued_frames(9, seed=2)
  queued.want_hidden = True
  outs = run(queued, frames, resets=(0, 6))
  for got, want in zip(outs, queued_reference(frames, resets=(0, 6), fed=outs), strict=True):
    close_enough(got, want, atol=0.02)


def test_a_hello_starts_the_feedback_over_and_keeps_the_queues(queued):
  """A new client's first frame feeds back zeros, as a new modeld's prev_feat
  did; the history stays until it resets."""
  spec = queued.spec
  frames = queued_frames(8, seed=3)
  queued.want_hidden = True
  outs = run(queued, frames[:4])
  queued.hello(timeout=5)   # as modeld joins: a hello, then the model again
  queued.ensure_engine(spec.sha256, spec.nbytes, onnx_path=None, build_timeout=60.0)
  outs += [queued.infer(w, p, frame_id=i + 5) for i, (w, p) in enumerate(frames[4:])]
  for got, want in zip(outs, queued_reference(frames, hellos=(4,), fed=outs), strict=True):
    close_enough(got, want, atol=0.02)
  # frame_skip 4: a row reaches features_buffer three frames after it is pushed
  fed_on = queued_reference(frames, fed=outs)
  assert np.abs(outs[7] - fed_on[7]).max() > 0.1, "the hello left the hidden state in place"


def test_a_frame_that_is_not_finite_feeds_nothing_back(queued):
  """The frame after a NOT_FINITE one takes the hidden state from the frame
  before it, as modeld fed back only what reached it. Kept, the NaNs would
  reach features_buffer three frames on (frame_skip 4); zeroed, frame 5 would
  read no hidden state where frame 0's belongs."""
  spec = queued.spec
  frames = queued_frames(7, seed=4)
  frames[1][1][spec.packed_layout['traffic_convention'][0]] = np.inf
  flags = [P.Flag.RESET_QUEUES | P.Flag.WANT_HIDDEN] + [P.Flag.WANT_HIDDEN] * 6
  replies = [raw_infer(queued, w, p, i + 1, f) for i, ((w, p), f) in enumerate(zip(frames, flags, strict=True))]
  assert [r[0] for r in replies] == [P.Status.OK, P.Status.NOT_FINITE] + [P.Status.OK] * 5
  outs = [r[1] for r in replies]
  with np.errstate(invalid='ignore'):   # frame 1's reference is not finite either
    want = queued_reference(frames, fed=outs, dropped=(1,))
  for i in (0, 2, 3, 4, 5, 6):
    close_enough(outs[i], want[i], atol=0.02)


def test_nonfinite_output_is_reported_not_returned(queued):
  # and the link stays: the session goes on, and the next frame is answered
  warped, packed = queued_frames(1)[0]
  good = packed.copy()
  packed[queued.spec.packed_layout['traffic_convention'][0]] = np.inf
  with pytest.raises(P.NotFinite, match='NOT_FINITE'):
    queued.infer(warped, packed, reset=True)
  assert not queued.dead
  assert np.all(np.isfinite(queued.infer(warped, good)))


def test_telemetry_piggybacks_only_when_asked(queued):
  warped, packed = queued_frames(1)[0]
  queued.infer(warped, packed, reset=True, want_state=False)
  assert queued.last_state is None
  queued.infer(warped, packed, want_state=True)
  assert isinstance(queued.last_state, dict)   # {} where the host has no sensors: never zeros


def test_frame_deadline_includes_time_spent_sending(queued, monkeypatch):
  send = queued.t.send

  def delayed_send(*args, **kwargs):
    send(*args, **kwargs)
    time.sleep(0.08)

  monkeypatch.setattr(queued.t, 'send', delayed_send)
  queued.deadline = 0.05
  warped, packed = queued_frames(1)[0]
  with pytest.raises(LinkError, match='link abandoned'):
    queued.infer(warped, packed)
  assert queued.dead


def test_wrong_sized_request_is_rejected(queued):
  seq = queued._next_seq()
  queued.t.send(P.Msg.INFER_REQ, seq, (P.pack_infer_req(1, 0), b'\x00' * 1000))
  _, status, _, _, _ = P.unpack_infer_resp(queued._expect(P.Msg.INFER_RESP, seq, 5.0).payload)
  assert status == P.Status.BAD_SHAPE


def _may_power_off() -> bool:
  """A booted systemd host outside CI: a server that got --poweroff wrong could take it down."""
  return Path('/run/systemd/system').exists() and not (os.environ.get('CI') or os.environ.get('JETLINK_TEST_SHUTDOWN'))


@pytest.mark.skipif(_may_power_off(), reason='a real systemd host; JETLINK_TEST_SHUTDOWN=1 runs it anyway')
def test_shutdown_replies_and_without_poweroff_stays_up(server, queued):
  resp = queued.shutdown('car battery', timeout=5)
  assert isinstance(resp.get('ok'), bool) and isinstance(resp.get('detail'), str)
  if sys.platform.startswith('linux'):
    assert resp['ok'] is True
  time.sleep(0.5)
  assert server.proc.poll() is None, server.tail()
  assert not (server.cache.parent / 'systemctl.called').exists(), 'a server without --poweroff called systemctl'
  assert queued.ping(timeout=5) < 5.0   # the host is what goes down, not the session


# -- the hello ------------------------------------------------------------------

def test_the_hello_carries_the_link_and_answers_with_what_the_comma_reads(server, monkeypatch):
  client = server.connect()
  try:
    monkeypatch.setattr(client.t, 'link_info', lambda: {'kind': 'cable', 'usb_speed': 'high-speed'})
    hello = client.hello(timeout=5)
  finally:
    client.close()
  assert HELLO_KEYS <= set(hello), f'missing {HELLO_KEYS - set(hello)}'
  assert hello['protocol'] == P.VERSION
  assert hello['backend'] == 'ort' and 'trt_version' not in hello
  # a missing sleep_after reads as 1.0 on the comma, which then lets the gadget go
  assert isinstance(hello['sleep_after'], (int, float)) and hello['sleep_after'] >= 0


def test_the_hello_settles_whose_pick_is_newer(tmp_path):
  # a model picked on either end drives on both: the comma's hello says what
  # it picked, how long ago and when, and the answer is the newer of that and
  # the server's, kept beside the cache
  with running(tmp_path) as s:
    client = s.connect()
    try:
      hello = client.hello(timeout=5)
      assert hello['pick'] is None and isinstance(hello['fetches'], bool)
      now = time.time()
      mine = {'ref': 'a' * 40, 'sha256': '1' * 64, 'nbytes': 766, 'name': 'Alpha', 'age_s': 30.0, 'at': now - 30}
      pick = client.hello(timeout=5, pick=mine)['pick']
      assert (pick['ref'], pick['sha256'], pick['by']) == ('a' * 40, '1' * 64, 'comma') and 29 <= pick['age_s'] < 40
      # the comma picks another: it wins, by what the comma named before, not by the server's clock
      other = dict(mine, ref='b' * 40, sha256='2' * 64, age_s=1.0, at=now - 1)
      assert client.hello(timeout=5, pick=other)['pick']['ref'] == 'b' * 40
      # one with no pick of its own takes the server's
      assert client.hello(timeout=5, pick=dict(mine, age_s=None))['pick']['ref'] == 'b' * 40
    finally:
      client.close()
  kept = json.loads((s.cache / 'pick.json').read_text())
  assert kept['ref'] == 'b' * 40 and 'heard' not in kept


def test_a_hello_restarts_the_seqs_and_replays_are_dropped(server):
  client = server.connect()
  try:
    client.seq = 5000
    client.state(timeout=5)
    client.seq = 0                  # a new process on the same link starts at 1
    client.hello(timeout=5)         # seq 1: answered, whatever came before
    client.t.send(P.Msg.PING, 1)    # the hello's own seq is a replay
    with pytest.raises(LinkTimeout):
      client._expect(P.Msg.PONG, 1, 0.5)
    assert client.ping(timeout=5) < 5.0
  finally:
    client.close()


# -- the stateful graph: the engine keeps the history ------------------------------

def test_the_state_carries_from_frame_to_frame(stateful):
  spec = stateful.spec
  frames = tiny_model.stateful_frames(8, seed=2)
  want = tiny_model.stateful_reference(frames[:3]) + tiny_model.stateful_reference(frames[3:])
  for i, (f, w) in enumerate(zip(frames, want, strict=True)):
    stateful.want_hidden = i % 2 == 1   # the whole vector every other frame
    out = stateful.infer(f['new_img'], tiny_model.packed_for(f), frame_id=i + 1, reset=i in (0, 3))
    assert out.shape == (spec.output_nelem,)
    if not stateful.want_hidden:
      assert not out[slice(*spec.hidden_range)].any()
      out, w = outside(out, spec), outside(w, spec)
    close_enough(out, w, atol=1e-4)   # float32 throughout: a frame off in the queue is far outside this


def test_a_request_is_the_frame_and_twelve_floats(stateful):
  spec = stateful.spec
  assert spec.infer_req_nbytes == P.INFER_REQ_SIZE + spec.warped_nbytes + 12 * 4
  seq = stateful.infer_begin(np.zeros(spec.warped_shape, np.uint8), np.zeros(spec.packed_nelem, np.float32), 1, reset=True)
  assert stateful.infer_end(seq).shape == (spec.output_nelem,)


# -- a server with nothing loaded -------------------------------------------------

def test_not_ready_is_reported_rather_than_crashing(bare):
  client = bare.connect()
  client.spec = spec_of(QUEUED)
  try:
    with pytest.raises(LinkError, match='NOT_READY'):
      client.infer(np.zeros(client.spec.warped_shape, np.uint8), np.zeros(client.spec.packed_nelem, np.float32))
  finally:
    client.close()


def test_state_hello_and_ping_need_no_engine(bare):
  client = bare.connect()
  try:
    state = client.state(timeout=5)
    assert state['engine_state'] == 'none' and 'frames_served' in state
    assert client.hello(timeout=5)['protocol'] == P.VERSION
    assert client.ping(timeout=5) < 5.0
  finally:
    client.close()


def test_a_ping_where_the_server_knows_no_client_raises(bare):
  # Between its join and the swap the comma only pings, over a link that can
  # outlive a server restart or a USB session the server reopened. The ping
  # must fail there, so the keep-alive rejoins, rather than the swap's first
  # frame come back NOT_READY.
  client = bare.connect()
  try:
    with pytest.raises(LinkError, match='no_hello'):
      client.ping(timeout=5)
    client.hello(timeout=5)
    assert client.ping(timeout=5) < 5.0
  finally:
    client.close()


def test_a_missing_engine_with_nothing_to_upload_is_engine_missing(bare):
  # modeld never carries the ONNX: this is what sends the comma back to provisioning
  spec = spec_of(STATEFUL)
  client = bare.connect()
  try:
    with pytest.raises(EngineMissing):
      client.ensure_engine(spec.sha256, spec.nbytes, onnx_path=None, build_timeout=10.0)
  finally:
    client.close()


# -- frames as datagrams: a phone's server over the cable ----------------------

@pytest.fixture(scope='module')
def cable(tmp_path_factory):
  """A server that takes loopback for the cable, so it offers frame datagrams."""
  with running(tmp_path_factory.mktemp('swift-cable'), env={'JETLINK_DATAGRAMS_ANY_PEER': '1'}) as s:
    yield s


@pytest.fixture
def cable_queued(cable, monkeypatch):
  monkeypatch.setattr(TcpTransport, 'on_the_cable', lambda self: True)
  # small pieces, so the tiny model's frames are cut as a big model's are
  monkeypatch.setattr(P, 'DATAGRAM_PAYLOAD', 700)
  with ready(cable, QUEUED) as client:
    client.hello()
    client.ensure_engine(client.spec.sha256 if client.spec else spec_of(QUEUED).sha256, spec_of(QUEUED).nbytes,
                         onnx_path=None, build_timeout=60.0)
    yield client


def lossy_run(client, frames, lose=()):
  """Frames as modeld sends them once it holds an output: the first on the
  stream, the rest as datagrams; those in `lose` with their middle piece
  dropped on the way. The outputs of the frames that were answered."""
  udp, token = client.t._udp
  real = udp.sendmsg
  outs = []
  for i, (w, p) in enumerate(frames):
    piece = [0]

    def sendmsg(bufs, i=i, piece=piece):
      piece[0] += 1
      if i in lose and piece[0] == 2:
        return sum(memoryview(b).nbytes for b in bufs)
      return real(bufs)
    client.t._udp = (mock.Mock(sendmsg=sendmsg), token)
    seq = client.infer_begin(w, p, frame_id=i + 1, reset=i == 0, skip_if_busy=i > 0)
    out = client.infer_end(seq, hold=0.5 if i > 0 else None)
    if out is not None:
      outs.append(out)
  client.t._udp = (udp, token)
  return outs


def test_the_hello_offers_frame_datagrams_only_over_the_cable(server, cable, monkeypatch):
  monkeypatch.setattr(TcpTransport, 'on_the_cable', lambda self: True)
  for s, offered in ((server, False), (cable, True)):
    client = s.connect()
    try:
      hello = client.hello()
      assert ('frame_port' in hello, 'frame_token' in hello) == (offered, offered)
      assert client.t.datagrams == offered
    finally:
      client.close()


def test_frames_as_datagrams_give_what_the_stream_gives(cable_queued):
  spec = cable_queued.spec
  frames = queued_frames(8, seed=11)
  assert len(P.datagram_pieces(spec.infer_req_nbytes + P.HEADER_SIZE)) > 1
  outs = lossy_run(cable_queued, frames)
  assert cable_queued.t.datagrams and cable_queued.frames_lost == 0
  for got, want in zip(outs, queued_reference(frames), strict=True):
    close_enough(outside(got, spec), outside(want, spec), atol=0.02)


def test_a_frame_that_lost_a_piece_never_ran_and_the_history_goes_on_without_it(cable_queued):
  spec = cable_queued.spec
  frames = queued_frames(8, seed=12)
  outs = lossy_run(cable_queued, frames, lose={3})
  assert len(outs) == 7
  assert cable_queued.frames_lost == 1 and not cable_queued.dead
  kept = [f for i, f in enumerate(frames) if i != 3]
  for got, want in zip(outs, queued_reference(kept), strict=True):
    close_enough(outside(got, spec), outside(want, spec), atol=0.02)
