"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Framing and transport tests. No Jetson, no CUDA - these run anywhere.
"""
from __future__ import annotations

import json
import struct
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from jetlink import protocol as P
from jetlink.client import JetlinkClient
from jetlink.spec import ModelSpec
from jetlink.transport.base import LinkError, LinkTimeout, Message, StreamTransport
from jetlink.transport.tcp import TcpTransport


@pytest.mark.parametrize('msg_type,payload', [(P.Msg.PONG, b''), (P.Msg.PROGRESS, b'{}')])
def test_unsolicited_messages_cannot_extend_a_reply_deadline(monkeypatch, msg_type, payload):
  from jetlink import client as client_module
  clock = [0.0]

  def recv(timeout):
    clock[0] += 0.02
    assert clock[0] < 1.0, 'client kept consuming messages past its deadline'
    return SimpleNamespace(msg_type=msg_type, seq=1, payload=memoryview(payload))

  monkeypatch.setattr(client_module, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
  client = client_module.JetlinkClient(SimpleNamespace(recv=recv))
  with pytest.raises(LinkTimeout):
    client._expect(P.Msg.INFER_RESP, 2, 0.05)
  assert clock[0] <= 0.08


def make_pair() -> tuple[TcpTransport, TcpTransport]:
  """A real loopback TCP pair, so the tests exercise the production path
  (including TCP_NODELAY and true stream fragmentation) rather than a
  socketpair, which is AF_UNIX and behaves differently."""
  srv = TcpTransport.listen('127.0.0.1', 0)
  port = srv.getsockname()[1]
  client = TcpTransport.connect('127.0.0.1', port)
  server = TcpTransport(srv.accept()[0])
  srv.close()
  return client, server


def test_header_roundtrip():
  raw = P.pack_header(P.Msg.INFER_REQ, 42, 1234, P.Flag.RESET_QUEUES, 99)
  assert len(raw) == P.HEADER_SIZE == 32
  _, _, mt, seq, flags, length, t = P.unpack_header(raw)
  assert (mt, seq, flags, length, t) == (P.Msg.INFER_REQ, 42, P.Flag.RESET_QUEUES, 1234, 99)


def test_bad_magic_is_diagnosed():
  with pytest.raises(P.ProtocolError, match='magic'):
    P.unpack_header(b'\x00' * 32)


@pytest.mark.parametrize('version', [1, P.VERSION - 1, P.VERSION + 1])
def test_incompatible_release_is_rejected_before_payload(version):
  raw = bytearray(P.pack_header(P.Msg.HELLO_RESP, 1, 0))
  raw[4:6] = version.to_bytes(2, 'little')
  with pytest.raises(P.ProtocolError, match='peer speaks protocol'):
    P.unpack_header(raw)


def test_payload_alignment_allows_zero_copy_views():
  """A float32 array must be readable straight out of the receive buffer."""
  a, b = make_pair()
  try:
    packed = np.arange(1000, dtype=np.float32)
    warped = np.zeros((2, 6, 8, 8), np.uint8)
    a.send(P.Msg.INFER_REQ, 1, (P.pack_infer_req(7, 0), warped, packed))
    msg = b.recv(timeout=2)
    off = P.INFER_REQ_SIZE + warped.nbytes
    got = np.frombuffer(msg.payload, np.float32, packed.size, off)
    assert np.array_equal(got, packed)
  finally:
    a.close()
    b.close()


def test_multipart_message_is_one_message():
  a, b = make_pair()
  try:
    parts = [b'x' * 10, b'y' * 100, b'z' * 1000]
    a.send(P.Msg.STATE_RESP, 5, parts)
    msg = b.recv(timeout=2)
    assert msg.seq == 5
    assert bytes(msg.payload) == b''.join(parts)
  finally:
    a.close()
    b.close()


def test_large_message_survives_stream_fragmentation():
  """A 460 KB request crosses many TCP segments; framing must reassemble it."""
  a, b = make_pair()
  got = []

  def reader():
    got.append(bytes(b.recv(timeout=10).payload))

  t = threading.Thread(target=reader)
  t.start()
  try:
    payload = np.random.default_rng(0).integers(0, 256, 460_000, dtype=np.uint8)
    a.send(P.Msg.INFER_REQ, 9, (payload,))
    t.join(15)
    assert got and got[0] == payload.tobytes()
  finally:
    a.close()
    b.close()


def test_timeout_midmessage_does_not_desync():
  """A missed deadline must cost a frame, not the stream.

  The buffer keeps whatever arrived, so the next recv resumes the same message
  instead of reading a header out of the middle of a payload.
  """
  a, b = make_pair()
  try:
    body = bytes(range(256)) * 800  # 204 KB, will not arrive in one segment
    a.send(P.Msg.INFER_RESP, 3, (body,))

    deadline_misses = 0
    for _ in range(200):
      try:
        msg = b.recv(timeout=0.001)
        break
      except LinkTimeout:
        deadline_misses += 1
    else:
      pytest.fail("message never completed")

    assert msg.seq == 3
    assert bytes(msg.payload) == body

    # And the stream is still usable afterwards.
    a.send(P.Msg.PONG, 4)
    assert b.recv(timeout=5).seq == 4
  finally:
    a.close()
    b.close()


def test_back_to_back_messages_keep_their_boundaries():
  a, b = make_pair()
  try:
    for i in range(20):
      a.send(P.Msg.PING, i, (bytes([i]) * (i * 1000 + 1),))
    for i in range(20):
      msg = b.recv(timeout=5)
      assert msg.seq == i
      assert bytes(msg.payload) == bytes([i]) * (i * 1000 + 1)
  finally:
    a.close()
    b.close()


def test_closed_peer_raises_link_error():
  a, b = make_pair()
  a.close()
  with pytest.raises(LinkError):
    b.recv(timeout=2)
  b.close()


def _spec(**kw) -> ModelSpec:
  base = {
    'sha256': 'a' * 64, 'nbytes': 765953504, 'frame_skip': 4,
    'input_shapes': {'img': (1, 12, 128, 256), 'big_img': (1, 12, 128, 256),
                     'desire_pulse': (1, 33, 8), 'traffic_convention': (1, 2),
                     'action_t': (1, 2), 'features_buffer': (1, 32, 32, 512)},
    'output_shapes': {'outputs': (1, 18452)},
    'output_slices': {'hidden_state': slice(2066, 18450)}, 'checkpoint': None}
  base.update(kw)
  return ModelSpec(**base)


# Cinque Terre V3's, as the fork's tests read them off its ONNX
STATEFUL = {'new_img': (2, 6, 128, 256), 'desire': (8,), 'traffic_convention': (1, 2), 'action_t': (1, 2),
            'state_img_q': (2, 5, 6, 128, 256), 'state_desire_q': (132, 1, 8), 'state_feat_q': (128, 1, 16384)}


def _stateful_spec() -> ModelSpec:
  states = {f'next_{n}': s for n, s in STATEFUL.items() if n.startswith('state_')}
  return _spec(input_shapes=STATEFUL, output_shapes={'outputs': (1, 18452), **states})


def test_spec_matches_the_shipped_big_model():
  """Guards the numbers the whole design is sized against."""
  s = _spec()
  assert s.warped_shape == (2, 6, 128, 256)
  assert s.warped_nbytes == 393_216
  assert s.feat_dim == 16_384                    # 32 * 512, == hidden_state length
  assert s.prev_feat_shape == (1, 16_384)        # kept on the server
  assert s.packed_nelem == 8 + 2 + 2
  assert s.packed_nbytes == 48
  assert s.output_nelem == 18_452
  assert s.hidden_range == (2066, 18450)
  assert s.reply_nelem == 2066 + 2
  assert s.img_buf_shape == (5, 6, 128, 256)     # frame_skip*(n_frames-1)+1
  assert s.feat_q_shape == (128, 1, 16_384)
  assert s.desire_q_shape == (132, 1, 8)
  # the messages with their 32-byte headers: 393,304 B up (409,600 once the
  # gadget pads it to 16 KB) and 8,324 B down, one 16 KB read on the comma;
  # 73,860 B, five reads, with the hidden state (WANT_HIDDEN)
  assert P.HEADER_SIZE + s.infer_req_nbytes == 393_304
  assert P.HEADER_SIZE + s.infer_resp_nbytes == 8_324
  assert P.HEADER_SIZE + P.INFER_RESP_SIZE + s.output_nbytes == 73_860


@pytest.mark.parametrize('hidden,expected', [
  (slice(2066, 18450), (2066, 18450)),
  (slice(-16386, -2), (2066, 18450)),        # counted from the back
  (slice(2068, None), (2068, 18452)),        # an open end
  (slice(None, 16384), (0, 16384)),
  (slice(2066, 99999), (2066, 18452)),       # past the end: clamped, as slicing does
  (slice(18450, 2066), None),                # takes nothing
  (slice(2066, 18450, 2), None),             # a step: not one run of floats
])
def test_hidden_state_ends_resolve_as_python_slices_them(hidden, expected):
  """The Swift server resolves them alike (NamedSlice.range(in:)); the layout
  conformance fixture holds the two to it."""
  s = _spec(output_slices={'hidden_state': hidden})
  assert s.hidden_range == expected
  assert s.reply_nelem == 18_452 - (expected[1] - expected[0] if expected else 0)


def test_a_stateful_model_sends_and_gets_what_a_queued_one_does():
  stateful, queued = _stateful_spec(), _spec()
  assert stateful.stateful and not queued.stateful
  assert (stateful.infer_req_nbytes, stateful.infer_resp_nbytes) == (queued.infer_req_nbytes, queued.infer_resp_nbytes)


def test_spec_handles_the_older_3d_features_buffer():
  s = _spec(input_shapes={'img': (1, 12, 128, 256), 'big_img': (1, 12, 128, 256),
                          'desire_pulse': (1, 25, 8), 'traffic_convention': (1, 2),
                          'action_t': (1, 2), 'features_buffer': (1, 24, 512)})
  assert s.feat_dim == 512
  assert s.prev_feat_shape == (1, 512)
  assert s.packed_nelem == 8 + 2 + 2


def test_rx_buffer_compaction_preserves_a_partial_message():
  """Force the buffer to slide a partial message over itself.

  Source and destination genuinely overlap here (dest [0:800], src [100:900]),
  which is the case a memcpy gets wrong and a memmove gets right. Corruption
  here would deliver a subtly wrong camera frame rather than raising.
  """
  from jetlink.transport.base import RxBuffer

  rx = RxBuffer(1024)
  payload = bytes((i * 7 + 3) % 251 for i in range(900))
  rx.writable()[:900] = payload
  rx.committed(900)
  rx.take(100)                     # start=100, end=900 -> 800 unconsumed
  assert rx.start == 100 and rx.available == 800

  rx.reserve(950)                  # will not fit after `start`; must slide
  assert rx.start == 0 and rx.available == 800
  assert bytes(rx.view[:800]) == payload[100:], "overlapping compaction corrupted the buffer"


def test_rx_buffer_grows_for_an_oversized_message():
  from jetlink.transport.base import RxBuffer

  rx = RxBuffer(64)
  rx.writable()[:32] = bytes(range(32))
  rx.committed(32)
  rx.reserve(4096)
  assert len(rx.buf) >= 4096
  assert bytes(rx.view[:32]) == bytes(range(32))


@pytest.mark.parametrize('garbage', [
  b'\xde\xad\xbe\xef' + bytes(60),
  struct.pack(P.HEADER_FMT, P.MAGIC, P.VERSION - 1, P.Msg.HELLO_RESP, 1, 0, 2, 0) + b'{}',
], ids=['bad_magic', 'another_version'])
def test_desync_is_a_link_error_not_a_process_killer(garbage):
  """Garbage on the wire must surface as LinkError so callers reconnect.

  ProtocolError is not a LinkError, and the server's accept loop only catches
  LinkError - so letting it escape would unwind out of main() and exit the
  process instead of dropping one connection. A header of another version is
  no special case: the caller reopens the link.
  """
  a, b = make_pair()
  try:
    a.sock.sendall(garbage)
    with pytest.raises(LinkError, match='protocol error'):
      b.recv(timeout=5)
    # And it stays failed rather than re-reading the same bad bytes forever.
    with pytest.raises(LinkError, match='desynced'):
      b.recv(timeout=5)
  finally:
    a.close()
    b.close()


def test_absurd_length_is_rejected_before_allocating():
  """A corrupt length field must not make us allocate gigabytes."""
  from jetlink.transport.base import MAX_MESSAGE

  a, b = make_pair()
  try:
    a.sock.sendall(P.pack_header(P.Msg.INFER_REQ, 1, 0xFFFFFFF0))
    with pytest.raises(LinkError):
      b.recv(timeout=5)
    assert len(b.rx.buf) <= MAX_MESSAGE, "buffer grew to fit a bogus length"
  finally:
    a.close()
    b.close()


def test_frame_timeout_is_a_link_failure():
  """A frame that does not come back inside FRAME_TIMEOUT means the far end is
  gone, so modeld falls back as it does for a chestnut. LinkError and not
  LinkTimeout, latched, so nothing upstream treats it as recoverable.
  """
  a, b = make_pair()
  spec = _spec()
  client = JetlinkClient(a, deadline=0.02)
  client.spec = spec
  try:
    with pytest.raises(LinkError, match='abandoned'):   # nothing is serving b
      client.infer(np.zeros(spec.warped_shape, np.uint8),
                   np.zeros(spec.packed_nelem, np.float32))
    assert client.dead
  finally:
    client.close()
    b.close()


def test_send_rejects_a_wrongly_sized_buffer():
  """Catch a model/spec skew here, not as a misparse on the far end."""
  a, b = make_pair()
  spec = _spec()
  client = JetlinkClient(a, deadline=0.05)
  client.spec = spec
  try:
    with pytest.raises(LinkError, match='bytes'):
      client.infer_begin(np.zeros((2, 6, 128, 128), np.uint8),   # half-sized
                         np.zeros(spec.packed_nelem, np.float32))
  finally:
    client.close()
    b.close()


# -- the reply -----------------------------------------------------------------

def reply(floats: np.ndarray, tail: bytes = b'', frame_id: int = 1) -> bytes:
  return P.pack_infer_resp(frame_id, P.Status.OK, 0, 0, 0) + np.asarray(floats, np.float32).tobytes() + tail


def replying(spec: ModelSpec, payload: bytes) -> JetlinkClient:
  """A client whose transport answers every frame with `payload`."""
  transport = SimpleNamespace(send=lambda *a, **kw: None, recv=lambda **kw: SimpleNamespace(
    msg_type=P.Msg.INFER_RESP, seq=1, payload=memoryview(payload)))
  client = JetlinkClient(transport, want_hidden=False)
  client.spec = spec
  return client


def frame(client: JetlinkClient, **kw) -> np.ndarray:
  spec = client.spec
  return client.infer(bytes(spec.warped_nbytes), bytes(spec.packed_nbytes), frame_id=1, **kw)


@pytest.mark.parametrize('kind', ['wrong_frame', 'short_header', 'short_output'])
def test_invalid_inference_response_abandons_the_stream(kind):
  spec = _spec()
  payload = reply(np.zeros(spec.reply_nelem), frame_id=42 if kind == 'wrong_frame' else 7)
  if kind == 'short_header':
    payload = payload[:P.INFER_RESP_SIZE - 1]
  elif kind == 'short_output':
    payload = payload[:-1]
  client = replying(spec, payload)
  seq = client.infer_begin(bytes(spec.warped_nbytes), bytes(spec.packed_nbytes), frame_id=7)
  with pytest.raises(LinkError):
    client.infer_end(seq)
  assert client.dead
  with pytest.raises(LinkError, match='previously failed'):
    client.infer_begin(bytes(spec.warped_nbytes), bytes(spec.packed_nbytes), frame_id=8)


def test_the_output_keeps_its_layout_with_the_hidden_state_left_out():
  """The glue slices by the spec's output_slices, as when the whole vector
  crossed: the reply is expanded back, hidden_state reading as zeros."""
  spec = _spec()
  sent = np.arange(spec.reply_nelem, dtype=np.float32) + 1
  out = frame(replying(spec, reply(sent)))
  assert out.shape == (18_452,)
  np.testing.assert_array_equal(out[:2066], sent[:2066])
  assert not out[2066:18450].any()
  np.testing.assert_array_equal(out[18450:], sent[2066:])


def test_want_hidden_returns_the_whole_vector_as_sent():
  spec = _stateful_spec()
  whole = np.arange(spec.output_nelem, dtype=np.float32)
  sent = []
  client = replying(spec, reply(whole))
  client.t.send = lambda *a, **kw: sent.append(a)
  client.want_hidden = True
  np.testing.assert_array_equal(frame(client), whole)
  _, flags = P.unpack_infer_req(sent[0][2][0])
  assert flags & P.Flag.WANT_HIDDEN


def test_openpilots_raw_predictions_switch_asks_for_the_hidden_state(monkeypatch):
  monkeypatch.delenv('SEND_RAW_PRED', raising=False)
  assert not JetlinkClient(SimpleNamespace()).want_hidden
  monkeypatch.setenv('SEND_RAW_PRED', '1')
  assert JetlinkClient(SimpleNamespace()).want_hidden


def test_a_reply_with_the_hidden_state_left_in_is_refused():
  """A server that says 3 and answers like 2 would have every float after
  hidden_state misread: refused, not guessed at."""
  spec = _spec()
  client = replying(spec, reply(np.zeros(spec.output_nelem)))
  with pytest.raises(LinkError, match='73828 bytes, expected 8292'):
    frame(client)
  assert client.dead


def test_telemetry_follows_the_outputs():
  spec = _spec()
  client = replying(spec, reply(np.zeros(spec.reply_nelem), json.dumps({'temp_c': 50}).encode()))
  frame(client, want_state=True)
  assert client.last_state == {'temp_c': 50}


@pytest.mark.parametrize('tail', [b'', b'{"temp_c": 50}', b'[1, 2]', b'{not json'])
def test_a_reply_laid_out_another_way_is_refused_when_telemetry_was_asked_for(tail):
  """With WANT_STATE the bytes after the outputs are telemetry, a JSON object.
  The hidden state left in (or resolved differently on the server) is floats
  there instead, and read as outputs they would reach modelV2: refused."""
  spec = _spec()
  client = replying(spec, reply(np.full(spec.output_nelem, 0.5, np.float32), tail))
  with pytest.raises(LinkError, match='bytes, expected 8292'):
    frame(client, want_state=True)
  assert client.dead
  assert client.last_state is None


@pytest.mark.parametrize('tail', [b'[1, 2]', b'"hot"', b'{not json'])
def test_telemetry_that_is_not_a_json_object_is_refused(tail):
  spec = _spec()
  client = replying(spec, reply(np.zeros(spec.reply_nelem), tail))
  with pytest.raises(LinkError, match='bytes, expected 8292'):
    frame(client, want_state=True)
  assert client.dead


class _TimedTransport(StreamTransport):
  """Records the deadline each write ran under, as TcpTransport's sendmsg
  loop has it."""

  def __init__(self):
    super().__init__()
    self.deadlines: list[float | None] = []

  def _write(self, bufs):
    self.deadlines.append(self._write_timeout())
    return sum(b.nbytes for b in bufs)

  def _read_into(self, dest, timeout):
    return 0

  def close(self) -> None:
    pass


def test_a_frame_that_may_be_held_still_goes_out_under_the_clients_deadline():
  # a transport that cannot tell whether its host is behind sends the frame,
  # and a phone that stopped reading must not hold the frame loop past it
  client = JetlinkClient(_TimedTransport(), want_hidden=False)
  client.spec, client.deadline = _spec(), 0.2
  spec = client.spec
  assert client.infer_begin(bytes(spec.warped_nbytes), bytes(spec.packed_nbytes), skip_if_busy=True) is not None
  assert client.t.deadlines and all(d is not None and 0 < d <= 0.2 for d in client.t.deadlines)


def test_a_closed_client_is_dead():
  """Whoever still holds a closed client must open a new one: a big model
  retired after a link loss closes its client, and modeld's link reused it."""
  a, b = make_pair()
  client = JetlinkClient(a)
  assert not client.dead
  client.close()
  b.close()
  assert client.dead


# -- provisioning --------------------------------------------------------------

class _ScriptedServer:
  """The server's side of ensure_engine, one reply per request: ENGINE_REQ is
  answered `building` while another model's preload holds the device, and
  then, unasked, `need_upload` once it is free; UPLOAD_DONE is `ready`."""

  def __init__(self, spec: ModelSpec):
    self.spec = spec
    self.inbox: list[SimpleNamespace] = []
    self.sent: list[int] = []

  def _push(self, seq: int, **state) -> None:
    state.update(sha256=self.spec.sha256, chunk=1 << 20)
    self.inbox.append(SimpleNamespace(msg_type=P.Msg.ENGINE_RESP, seq=seq, payload=memoryview(json.dumps(state).encode())))

  def send(self, msg_type, seq, parts=(), flags=0, timeout=None):
    self.sent.append(msg_type)
    if msg_type == P.Msg.ENGINE_REQ:
      self._push(seq, state='building', detail='another build is in progress (preloaded)')
      self._push(0, state='need_upload', detail='have 0 of the model')
    elif msg_type == P.Msg.UPLOAD_DONE:
      self._push(seq, state='ready', detail='', spec=self.spec.to_dict())

  def send_json(self, msg_type, seq, obj, flags=0):
    self.send(msg_type, seq, (json.dumps(obj).encode(),), flags)

  def recv(self, timeout=None):
    if not self.inbox:
      raise LinkTimeout('nothing more from the scripted server')
    return self.inbox.pop(0)


def test_need_upload_pushed_after_a_preload_is_engine_missing_without_a_file():
  """modeld and provisioning's first ask carry no file: a need_upload that
  comes after `building` must end the wait at once (provision.ensure then asks
  again with the file), not after build_timeout, 1800 s in provisioning."""
  from jetlink.client import EngineMissing
  spec = _spec()
  client = JetlinkClient(_ScriptedServer(spec))
  started = time.monotonic()
  with pytest.raises(EngineMissing, match='have 0 of the model'):
    client.ensure_engine(spec.sha256, spec.nbytes, onnx_path=None, build_timeout=30.0)
  assert time.monotonic() - started < 5.0


def test_need_upload_pushed_after_a_preload_uploads_when_the_caller_has_the_file(tmp_path):
  spec = _spec(nbytes=3000)
  model = tmp_path / 'model.onnx'
  model.write_bytes(bytes(3000))
  server = _ScriptedServer(spec)
  client = JetlinkClient(server)
  assert client.ensure_engine(spec.sha256, spec.nbytes, onnx_path=model, build_timeout=30.0) == spec
  assert server.sent == [P.Msg.ENGINE_REQ, P.Msg.UPLOAD_CHUNK, P.Msg.UPLOAD_DONE]


def test_a_datagram_header_round_trips_and_rejects_what_is_not_ours():
  raw = P.pack_datagram_header(77, 2 ** 32 - 1, 65000, 398376)
  assert len(raw) == P.DATAGRAM_HEADER_SIZE == 20
  assert P.unpack_datagram_header(raw) == (77, 2 ** 32 - 1, 65000, 398376)
  with pytest.raises(P.ProtocolError):
    P.unpack_datagram_header(P.pack_header(P.Msg.PING, 1, 0)[:20])


@pytest.mark.parametrize('total', [1, 32, 65000, 65001, 130001, 398376, 16 << 20])
def test_messages_are_cut_into_as_few_equal_datagrams_as_fit(total):
  pieces = P.datagram_pieces(total)
  assert pieces[0][0] == 0 and sum(size for _, size in pieces) == total
  assert all(a + n == b for (a, n), (b, _) in zip(pieces, pieces[1:], strict=False))
  sizes = {size for _, size in pieces}
  assert max(sizes) <= P.DATAGRAM_PAYLOAD and max(sizes) - min(sizes) <= 1
  assert len(pieces) == -(-total // P.DATAGRAM_PAYLOAD)
  assert P.DATAGRAM_HEADER_SIZE + max(sizes) <= 65507   # what UDP carries


def test_an_addons_progress_reaches_only_the_client_waiting_on_it():
  """The server sends an add-on build's progress to whoever holds the session:
  modeld, when a drive starts mid-build. Its driving model's line is not it."""
  heard = []
  client = JetlinkClient(SimpleNamespace(recv=None))
  client.progress_cb = lambda stage, frac, msg: heard.append((stage, frac))

  def progress(**fields):
    client._dispatch(Message(P.Msg.PROGRESS, 0, 0, memoryview(json.dumps({'stage': 'build', 'frac': 0.5, 'msg': '', **fields}).encode())))

  client._providing = 'd' * 64           # the driving model
  progress(addon=True, sha256='a' * 64)
  progress()                             # the driving model's own, or an older server's
  client._providing = 'a' * 64
  progress(addon=True, sha256='a' * 64)
  assert heard == [('build', 0.5), ('build', 0.5)]
