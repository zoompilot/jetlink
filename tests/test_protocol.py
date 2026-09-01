"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Framing and transport tests. No Jetson, no CUDA - these run anywhere.
"""
from __future__ import annotations

import socket
import threading

import numpy as np
import pytest

from jetlink import protocol as P
from jetlink.spec import ModelSpec
from jetlink.transport.base import LinkError, LinkTimeout
from jetlink.transport.tcp import TcpTransport


def make_pair() -> tuple[TcpTransport, TcpTransport]:
  """A real loopback TCP pair, so the tests exercise the production path
  (including TCP_NODELAY and true stream fragmentation) rather than a
  socketpair, which is AF_UNIX and behaves differently."""
  srv = TcpTransport.listen('127.0.0.1', 0)
  port = srv.getsockname()[1]
  client = TcpTransport.connect('127.0.0.1', port)
  server, _ = TcpTransport.accept(srv)
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
    a.close(); b.close()


def test_multipart_message_is_one_message():
  a, b = make_pair()
  try:
    parts = [b'x' * 10, b'y' * 100, b'z' * 1000]
    a.send(P.Msg.STATE_RESP, 5, parts)
    msg = b.recv(timeout=2)
    assert msg.seq == 5
    assert bytes(msg.payload) == b''.join(parts)
  finally:
    a.close(); b.close()


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
    a.close(); b.close()


def test_timeout_midmessage_does_not_desync():
  """A missed deadline must cost a frame, not the stream.

  The buffer keeps whatever arrived, so the next recv resumes the same message
  instead of trying to read a header out of the middle of a payload. Before the
  framing was shared this was three separate implementations and this bug lived
  in one of them.
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
    a.close(); b.close()


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
    a.close(); b.close()


def test_closed_peer_raises_link_error():
  a, b = make_pair()
  a.close()
  with pytest.raises(LinkError):
    b.recv(timeout=2)
  b.close()


def _spec(**kw) -> ModelSpec:
  base = dict(
    sha256='a' * 64, nbytes=765953504, frame_skip=4,
    input_shapes={'img': (1, 12, 128, 256), 'big_img': (1, 12, 128, 256),
                  'desire_pulse': (1, 33, 8), 'traffic_convention': (1, 2),
                  'action_t': (1, 2), 'features_buffer': (1, 32, 32, 512)},
    output_shapes={'outputs': (1, 18452)},
    output_slices={'hidden_state': slice(2066, 18450)}, checkpoint=None)
  base.update(kw)
  return ModelSpec(**base)


def test_spec_matches_the_shipped_big_model():
  """Guards the numbers the whole design is sized against."""
  s = _spec()
  assert s.warped_shape == (2, 6, 128, 256)
  assert s.warped_nbytes == 393_216
  assert s.feat_dim == 16_384                    # 32 * 512, == hidden_state length
  assert s.packed_nelem == 8 + 2 + 2 + 16_384
  assert s.packed_nbytes == 65_584
  assert s.output_nelem == 18_452
  assert s.img_buf_shape == (5, 6, 128, 256)     # frame_skip*(n_frames-1)+1
  assert s.feat_q_shape == (128, 1, 16_384)
  assert s.desire_q_shape == (132, 1, 8)
  # ~533 KB/frame, 85 Mbit/s at 20 Hz
  assert s.infer_req_nbytes + s.infer_resp_nbytes < 540_000


def test_spec_handles_the_older_3d_features_buffer():
  s = _spec(input_shapes={'img': (1, 12, 128, 256), 'big_img': (1, 12, 128, 256),
                          'desire_pulse': (1, 25, 8), 'traffic_convention': (1, 2),
                          'action_t': (1, 2), 'features_buffer': (1, 24, 512)})
  assert s.feat_dim == 512
  assert s.packed_nelem == 8 + 2 + 2 + 512


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


def test_desync_is_a_link_error_not_a_process_killer():
  """Garbage on the wire must surface as LinkError so callers reconnect.

  ProtocolError is not a LinkError, and the server's accept loop only catches
  LinkError - so letting it escape would unwind out of main() and exit the
  process instead of dropping one connection.
  """
  a, b = make_pair()
  try:
    a.sock.sendall(b'\xde\xad\xbe\xef' + bytes(60))
    with pytest.raises(LinkError):
      b.recv(timeout=5)
    # And it stays failed rather than re-reading the same bad bytes forever.
    with pytest.raises(LinkError):
      b.recv(timeout=5)
  finally:
    a.close(); b.close()


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
    a.close(); b.close()


def test_timeout_does_not_disable_the_link():
  """A missed deadline is a late frame, not a dead link.

  Latching here would drop the car to the small model for the rest of the drive
  on one GC pause, which is the most likely failure of the whole system.
  """
  from jetlink.client import JetlinkClient

  a, b = make_pair()
  spec = _spec()
  client = JetlinkClient(a, deadline=0.02)
  client.spec = spec
  try:
    with pytest.raises(LinkTimeout):   # nothing is serving b, so no reply comes
      client.infer(np.zeros(spec.warped_shape, np.uint8),
                   np.zeros(spec.packed_nelem, np.float32))
    assert not client.dead, "a timeout must not latch the link as dead"
  finally:
    client.close(); b.close()


def test_send_rejects_a_wrongly_sized_buffer():
  """Catch a model/spec skew here, not as a misparse on the far end."""
  from jetlink.client import JetlinkClient

  a, b = make_pair()
  spec = _spec()
  client = JetlinkClient(a, deadline=0.05)
  client.spec = spec
  try:
    with pytest.raises(LinkError, match='bytes'):
      client.infer_begin(np.zeros((2, 6, 128, 128), np.uint8),   # half-sized
                         np.zeros(spec.packed_nelem, np.float32))
  finally:
    client.close(); b.close()
