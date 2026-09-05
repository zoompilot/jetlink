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
from jetlink.transport.base import LinkError, LinkTimeout
from jetlink.transport.ffs import FfsTransport


@pytest.fixture
def mount(tmp_path):
  for ep in ('ep0', 'ep1', 'ep2'):
    os.mkfifo(tmp_path / ep)
  return tmp_path


def test_no_endpoint_is_opened_before_a_host_has_enabled_it(mount):
  """The one that costs the gadget until the comma is rebooted.

  ffs_epfile_io does not fail on an endpoint no host has enabled, it sleeps in
  wait_event_interruptible until a signal that never comes, and unbinding does
  not wake it. The thread stuck there holds the struct file, so ffs->opened
  never drops and every later ffs_ep0_open answers EBUSY - for jetlinkd and the
  next drive's modeld too, not just this process. Opening ep0 and nothing else
  until a host is actually there is what keeps that from happening.
  """
  t = FfsTransport(str(mount))
  try:
    assert t.ep0 >= 0
    assert t.ep_in == -1 and t.ep_out == -1, "endpoint files opened before a host"
    assert t._reader is None, "the reader thread started before a host"
  finally:
    t.close()
  # And that close really let go, which is the half that used to fail.
  again = FfsTransport(str(mount))
  again.close()


def test_recv_times_out_while_the_kernel_read_is_blocked(mount):
  t = FfsTransport(str(mount))
  try:
    t0 = time.monotonic()
    with pytest.raises(LinkTimeout):
      t.recv(timeout=0.2)
    assert time.monotonic() - t0 < 1.0, "the deadline must not wait on the blocked read"
    # Started by that first recv, not by the constructor, and still parked in
    # its read, which is exactly where it should be.
    assert t._reader is not None and t._reader.is_alive()
  finally:
    t.close()


def test_data_written_by_the_host_arrives_through_the_reader(mount):
  t = FfsTransport(str(mount))
  try:
    # The transport opens its endpoint files on the first read or write, and a
    # FIFO standing in for one blocks the writer until somebody is reading it.
    # The real ep1 is not a FIFO and needs none of this.
    t._ensure_epfiles()
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
    t._ensure_epfiles()   # see above: the FIFO needs a reader before a writer
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


def test_the_watchdog_drops_the_link_so_a_stuck_write_can_return(mount, monkeypatch):
  """FunctionFS writes cannot time out: the request sits on the endpoint until
  the host drains it. A hello to a Jetson that has enumerated but whose server
  is not reading blocked three minutes on a drive 2026-09-05, and the join loop
  cannot retry what it is blocked inside. Unbinding the UDC is the only lever,
  and it works here because the endpoint is enabled - unlike a read on an
  endpoint no host ever enabled, which unbind does not touch (_ensure_epfiles).
  """
  t = FfsTransport(str(mount))
  try:
    unbound = []
    monkeypatch.setattr(t, 'unbind', lambda: unbound.append(True))
    t._abort_write()
    assert t._write_aborted and unbound, "the watchdog must drop the link, not just flag it"

    # Once aborted, the write reports our own doing rather than retrying
    # through the host-ready grace period.
    t._ensure_epfiles()
    os.close(t.ep_in)
    t.ep_in = -1                      # any failure will do; the flag decides the message
    with pytest.raises(LinkError) as e:
      t._write([memoryview(b'x')])
    assert 'no reader' in str(e.value), str(e.value)
  finally:
    t.ep_in = -1                      # already closed; keep close() off it
    t.close()


def test_a_write_that_completes_leaves_the_link_alone(mount, monkeypatch):
  # The guard must not fire on a healthy write; unbinding a working link would
  # turn a slow frame into a dropped one.
  monkeypatch.setattr(ffs, 'WRITE_TIMEOUT', 5.0)
  t = FfsTransport(str(mount))
  try:
    t._ensure_epfiles()
    unbound = []
    monkeypatch.setattr(t, 'unbind', lambda: unbound.append(True))
    host = os.open(mount / 'ep2', os.O_RDONLY | os.O_NONBLOCK)
    try:
      t.send(P.Msg.HELLO_REQ, 1, (b'hello',))
    finally:
      os.close(host)
    assert not unbound
    assert not t._write_aborted
  finally:
    t.close()
