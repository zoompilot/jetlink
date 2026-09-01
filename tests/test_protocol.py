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
from jetlink.transport.base import LinkError
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
