"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The FunctionFS transport against FIFOs standing in for the endpoint files.

What matters here is the one thing the kernel will not give us: FunctionFS
reads block regardless of O_NONBLOCK once the host has enabled the endpoint,
so a deadline can only be honoured if the read is somewhere else. A FIFO with
nothing written to it blocks a reader exactly the same way.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

from jetlink import protocol as P
from jetlink.transport import ffs
from jetlink.transport.base import LinkTimeout
from jetlink.transport.ffs import FfsTransport


@pytest.fixture
def mount(tmp_path):
  for ep in ('ep0', 'ep1', 'ep2'):
    os.mkfifo(tmp_path / ep)
  return tmp_path


def test_recv_times_out_while_the_kernel_read_is_blocked(mount):
  t = FfsTransport(str(mount))
  try:
    assert t._reader is not None and t._reader.is_alive()
    t0 = time.monotonic()
    with pytest.raises(LinkTimeout):
      t.recv(timeout=0.2)
    assert time.monotonic() - t0 < 1.0, "the deadline must not wait on the blocked read"
    assert t._reader.is_alive(), "the reader is still parked in its read, as it should be"
  finally:
    t.close()


def test_data_written_by_the_host_arrives_through_the_reader(mount):
  t = FfsTransport(str(mount))
  try:
    payload = np.arange(3000, dtype=np.float32)
    host = os.open(mount / 'ep1', os.O_WRONLY)
    try:
      header = P.pack_header(P.Msg.INFER_RESP, 7, payload.nbytes)
      threading.Thread(target=lambda: os.write(host, header + payload.tobytes()), daemon=True).start()
      msg = t.recv(timeout=5.0)
    finally:
      os.close(host)
    assert msg.seq == 7
    assert np.array_equal(np.frombuffer(msg.payload, np.float32), payload)
  finally:
    t.close()


def test_padded_messages_survive_the_chunked_reader(mount):
  """A message whose header plus payload is a packet multiple carries a pad
  byte; the reader hands over whatever the read returned, so the pad must be
  consumed by framing and never leak into the next message."""
  t = FfsTransport(str(mount))
  try:
    host = os.open(mount / 'ep1', os.O_WRONLY)
    try:
      first = bytes(P.PACKET_MULTIPLE - P.HEADER_SIZE)
      second = b'second'
      wire = (P.pack_header(1, 1, len(first), P.Flag.PADDED) + first + b'\0'
              + P.pack_header(1, 2, len(second)) + second)
      threading.Thread(target=lambda: os.write(host, wire), daemon=True).start()
      a = t.recv(timeout=5.0)
      b = t.recv(timeout=5.0)
    finally:
      os.close(host)
    assert (a.seq, a.payload.nbytes) == (1, len(first))
    assert (b.seq, bytes(b.payload)) == (2, second)
  finally:
    t.close()


def _bare_transport(**attrs):
  """An FfsTransport with no gadget behind it, for the pure-Python paths."""
  t = FfsTransport.__new__(FfsTransport)
  t._ready_deadline = None
  t._had_host = False
  t.bound_udc = None
  t.gadget = None
  for k, v in attrs.items():
    setattr(t, k, v)
  return t


def test_a_host_that_has_not_configured_us_yet_gets_the_grace_period(tmp_path, monkeypatch):
  monkeypatch.setattr(ffs, 'EP_READY_TIMEOUT', 0.05)
  t = _bare_transport(bound_udc='udc0')
  monkeypatch.setattr(ffs, 'UDC_SYSFS', str(tmp_path))
  state = tmp_path / 'udc0' / 'state'
  state.parent.mkdir()
  state.write_text('powered\n')
  # Never talked to a host: ENODEV means "not yet", and we wait for it.
  assert t._wait_for_host_ready() is True


def test_a_host_that_disconnects_mid_session_ends_the_link_at_once(tmp_path, monkeypatch):
  monkeypatch.setattr(ffs, 'EP_READY_TIMEOUT', 10.0)
  t = _bare_transport(bound_udc='udc0', _had_host=True)
  monkeypatch.setattr(ffs, 'UDC_SYSFS', str(tmp_path))
  state = tmp_path / 'udc0' / 'state'
  state.parent.mkdir()
  state.write_text('not attached\n')
  # No 10 s of retries against a link the UDC already reports as gone.
  assert t._wait_for_host_ready() is False
  # But a host that is still configured (endpoints being re-enabled after a
  # reset, or the server not reading yet) keeps the grace period.
  state.write_text('configured\n')
  t._ready_deadline = None
  assert t._wait_for_host_ready() is True
