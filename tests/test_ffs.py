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


def test_inference_reply_spans_bounded_kernel_reads(mount, monkeypatch):
  from types import SimpleNamespace

  t = FfsTransport(str(mount))
  reads = []
  real_os = ffs.os

  def readv(fd, buffers):
    reads.append(sum(len(b) for b in buffers))
    return real_os.readv(fd, buffers)

  # Keep read requests below costly high-order kernel allocations, including
  # when a response is larger than the buffer. Framing must reassemble it.
  monkeypatch.setattr(ffs, 'os', SimpleNamespace(**{k: getattr(real_os, k) for k in dir(real_os) if k != 'readv'}, readv=readv))
  try:
    t._ensure_epfiles()
    payload = np.arange(18452, dtype=np.float32).tobytes()
    wire = P.pack_header(P.Msg.INFER_RESP, 7, len(payload)) + payload
    host = real_os.open(mount / 'ep1', real_os.O_WRONLY)

    def write_all():
      remaining = memoryview(wire)
      while remaining:
        remaining = remaining[real_os.write(host, remaining):]

    writer = threading.Thread(target=write_all, daemon=True)
    writer.start()
    try:
      msg = t.recv(timeout=5.0)
      writer.join(1.0)
      assert not writer.is_alive()
      assert bytes(msg.payload) == payload
      assert msg.seq == 7
      assert len(reads) >= 5
      assert max(reads) <= 16 * 1024
      assert all(n % ffs.SS_MAX_PACKET == 0 for n in reads)
    finally:
      real_os.close(host)
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
  t._state_fd = -1
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
    host = os.open(mount / 'ep2', os.O_RDONLY)
    # A padded message can exceed a FIFO's capacity (8 KB on macOS). Having
    # an fd open is not enough: the fake host must actually drain the write.
    def drain():
      remaining = P.GADGET_TX_ALIGN
      while remaining:
        remaining -= len(os.read(host, remaining))
    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    try:
      t.send(P.Msg.HELLO_REQ, 1, (b'hello',))
      reader.join(1.0)
      assert not reader.is_alive()
    finally:
      os.close(host)
    assert not unbound
    assert not t._write_aborted
  finally:
    t.close()


def test_send_deadline_aborts_a_blocked_kernel_write(mount, monkeypatch):
  from types import SimpleNamespace
  t = FfsTransport(str(mount))
  released = threading.Event()
  try:
    t._ensure_epfiles()
    monkeypatch.setattr(t, 'unbind', released.set)

    def blocked(*args):
      assert released.wait(1.0), 'write watchdog did not run'
      raise OSError(108, 'endpoint shutdown')

    # Replace this module's reference, not the process-wide os.writev.
    monkeypatch.setattr(ffs, 'os', SimpleNamespace(writev=blocked))
    started = time.monotonic()
    with pytest.raises(LinkError, match='no reader'):
      t.send(P.Msg.INFER_REQ, 1, (b'frame',), timeout=0.05)
    assert time.monotonic() - started < 0.5
    assert released.is_set()
  finally:
    monkeypatch.undo()
    t.close()


def test_receive_diagnostics_follow_chunks_and_reset_per_message(monkeypatch):
  from collections import deque
  from types import SimpleNamespace
  from jetlink.transport.base import StreamTransport

  t = _bare_transport()
  StreamTransport.__init__(t)
  t._ensure_epfiles = lambda: None
  t._cv = threading.Condition()
  t._reader_error = None
  monkeypatch.setattr(ffs, 'time', SimpleNamespace(monotonic=lambda: 2.0))
  payload = b'first response'
  wire = P.pack_header(P.Msg.INFER_RESP, 1, len(payload)) + payload
  # Header and payload can split anywhere. Partial consumption must retain
  # the originating chunk's timestamps, without changing the received bytes.
  t._chunks = deque([(memoryview(wire[:8]), 1.8, .01, .06),
                     (memoryview(wire[8:]), 1.9, .02, .03)])
  t._queued = len(wire)
  assert bytes(t.recv(timeout=1).payload) == payload
  assert t.last_receive == pytest.approx({'prepare': .02, 'read_wait': .06, 'handoff': .2})
  assert t._queued == 0

  payload = b'next response'
  wire = P.pack_header(P.Msg.INFER_RESP, 2, len(payload)) + payload
  t._chunks.append((memoryview(wire), 1.99, .001, .002))
  t._queued = len(wire)
  assert bytes(t.recv(timeout=1).payload) == payload
  assert t.last_receive == pytest.approx({'prepare': .001, 'read_wait': .002, 'handoff': .01})


def test_reader_widens_its_cpu_affinity_off_the_pinned_core(monkeypatch):
  """The reader is created from modeld's core-7-pinned frame thread and would
  inherit that pin, serialising it with the frame loop. It must widen to every
  core so a completed read is serviced on another core, not behind the loop."""
  from types import SimpleNamespace
  calls = {}

  def getaffinity(_pid):
    return {7}

  def setaffinity(_pid, mask):
    calls['mask'] = set(mask)

  monkeypatch.setattr(ffs, 'os', SimpleNamespace(cpu_count=lambda: 8,
                                                 sched_getaffinity=getaffinity,
                                                 sched_setaffinity=setaffinity))
  _bare_transport()._widen_affinity()
  assert calls['mask'] == set(range(7)), "reader did not move off the pinned frame-loop core"


def test_reader_affinity_is_a_noop_when_already_unpinned(monkeypatch):
  from types import SimpleNamespace
  calls = {}
  monkeypatch.setattr(ffs, 'os', SimpleNamespace(
    cpu_count=lambda: 8,
    sched_getaffinity=lambda _pid: set(range(8)),
    sched_setaffinity=lambda _pid, mask: calls.setdefault('set', True)))
  _bare_transport()._widen_affinity()
  assert 'set' not in calls, "widened affinity when it was already full"


def test_reader_affinity_survives_a_platform_without_the_call(monkeypatch):
  from types import SimpleNamespace
  monkeypatch.setattr(ffs, 'os', SimpleNamespace(cpu_count=lambda: 8))  # no sched_* (macOS)
  _bare_transport()._widen_affinity()  # must not raise


def test_send_resets_the_write_quantum_each_message(monkeypatch):
  """A prior ENOMEM shrink must not persist. Once write_chunk is halved, a
  request larger than the shrunk size splits across two writes and re-arms the
  dwc3 double-TRB replay; each new message must start at the full quantum again
  so the split window is only the frames actually under memory pressure."""
  t = _bare_transport()
  t.write_chunk = 256 * ffs.SS_MAX_PACKET   # as if _shrink_write had halved it once
  monkeypatch.setattr(ffs.StreamTransport, 'send', lambda self, *a, **k: None)
  t.send(P.Msg.INFER_REQ, 1, ())
  assert t.write_chunk == FfsTransport.write_chunk, "write quantum not reset for the next message"


def test_gadget_receive_buffer_is_not_oversized(mount):
  """The gadget only receives ~74 KB replies; a 2 MB start was resident memory
  the memory-tight comma did not need. RxBuffer still grows on demand."""
  t = FfsTransport(str(mount))
  try:
    assert len(t.rx.buf) <= 256 << 10, "gadget receive buffer larger than a reply needs"
    t.rx.reserve(400 << 10)   # a hypothetical bigger message still fits after a grow
    assert len(t.rx.buf) >= 400 << 10
  finally:
    t.close()


def test_reader_widens_when_inherited_mask_is_several_cores(monkeypatch):
  """Offroad jetlinkd is not pinned; the reader then just fills out the mask to
  every core rather than excluding one (there is no single frame-loop core)."""
  from types import SimpleNamespace
  calls = {}
  monkeypatch.setattr(ffs, 'os', SimpleNamespace(
    cpu_count=lambda: 8,
    sched_getaffinity=lambda _pid: {0, 1, 2, 3},   # four cores online, unpinned
    sched_setaffinity=lambda _pid, mask: calls.__setitem__('mask', set(mask))))
  _bare_transport()._widen_affinity()
  assert calls['mask'] == set(range(8)), "reader did not widen an unpinned mask to all cores"


def test_read_buffers_are_recycled_not_reallocated():
  """The hot receive path must not allocate a fresh buffer per read: under
  memory pressure that allocation reclaims (prepare 24 ms on the 2026-09-06
  bench). A fully consumed buffer returns to the reader's pool for reuse."""
  from collections import deque
  t = _bare_transport(ep_out=0, _read_size=ffs.READ_CHUNK, _queued=4,
                      _reader_error=None, _closing=False, _chunks=deque(), _free=deque(),
                      last_receive={'prepare': 0.0, 'read_wait': 0.0, 'handoff': 0.0})
  t._cv = threading.Condition()
  buf = bytearray(ffs.READ_CHUNK)
  buf[:4] = b'\x01\x02\x03\x04'
  t._chunks.append((memoryview(buf)[:4], time.monotonic(), 0.0, 0.0))
  dest = memoryview(bytearray(4))
  assert t._read_into(dest, 0) == 4
  assert bytes(dest) == b'\x01\x02\x03\x04'
  assert list(t._free) == [buf], "consumed buffer was not returned to the pool"


def test_recycle_pool_is_bounded_and_ignores_foreign_buffers():
  from collections import deque
  t = _bare_transport(ep_out=0, _read_size=ffs.READ_CHUNK, _queued=0,
                      _reader_error=None, _closing=False, _chunks=deque(),
                      _free=deque(bytearray(ffs.READ_CHUNK) for _ in range(ffs.FREE_BUFS)),
                      last_receive={'prepare': 0.0, 'read_wait': 0.0, 'handoff': 0.0})
  t._cv = threading.Condition()
  ours = bytearray(ffs.READ_CHUNK)
  foreign = b'\x00\x00\x00\x00'          # bytes, not a bytearray we allocated
  for payload, expect_pooled in ((memoryview(ours)[:4], False), (memoryview(foreign), False)):
    t._chunks.append((payload, time.monotonic(), 0.0, 0.0))
    t._queued += payload.nbytes
    t._read_into(memoryview(bytearray(4)), 0)
  assert len(t._free) == ffs.FREE_BUFS, "pool grew past its cap or pooled a foreign buffer"


def test_configured_holds_the_state_fd_open_and_sees_live_changes(tmp_path, monkeypatch):
  """The per-read config check must not open the sysfs file each time (that
  reclaim-stalled 24.9 ms mid-frame on the bench). It holds the fd open and
  re-reads it, which still reflects a live state change."""
  monkeypatch.setattr(ffs, 'UDC_SYSFS', str(tmp_path))
  (tmp_path / 'udc0').mkdir()
  state = tmp_path / 'udc0' / 'state'
  state.write_text('configured\n')
  t = _bare_transport(bound_udc='udc0', gadget='/x', _state_fd=-1)
  try:
    assert t._configured() is True
    fd = t._state_fd
    assert fd >= 0, "state fd not held open"
    assert t._configured() is True and t._state_fd == fd, "state fd reopened per check"
    state.write_text('not_attached\n')          # a live disconnect
    assert t._configured() is False and t._state_fd == fd, "held fd missed the live change"
  finally:
    if t._state_fd >= 0:
      os.close(t._state_fd)


def test_unbind_drops_the_held_state_fd(tmp_path, monkeypatch):
  monkeypatch.setattr(ffs, 'UDC_SYSFS', str(tmp_path))
  (tmp_path / 'udc0').mkdir()
  (tmp_path / 'udc0' / 'state').write_text('configured\n')
  gdir = tmp_path / 'gadget'
  gdir.mkdir()
  (gdir / 'UDC').write_text('udc0\n')
  t = _bare_transport(bound_udc='udc0', gadget=str(gdir), _state_fd=-1)
  t._configured()
  assert t._state_fd >= 0
  t.unbind()
  assert t._state_fd == -1, "unbind left the state fd open"
  assert t.bound_udc is None


def test_reader_raises_itself_to_realtime_below_the_frame_loop(monkeypatch):
  """The reader inherits SCHED_OTHER (created during warmup, before modeld goes
  realtime) and then waits on the run queue after each read completes, straight
  onto read_wait. It must lift itself to SCHED_FIFO, below modeld's loop (54)."""
  from types import SimpleNamespace
  calls = {}
  monkeypatch.setattr(ffs, 'os', SimpleNamespace(
    SCHED_FIFO=1, sched_param=lambda p: SimpleNamespace(sched_priority=p),
    sched_setscheduler=lambda pid, pol, par: calls.update(policy=pol, prio=par.sched_priority)))
  _bare_transport()._raise_reader_priority()
  assert calls['policy'] == 1, "reader did not switch to SCHED_FIFO"
  assert calls['prio'] == ffs.READER_RT_PRIORITY < 54, "reader priority not set below the frame loop"


def test_reader_priority_is_best_effort_without_permission(monkeypatch):
  from types import SimpleNamespace
  def denied(*a):
    raise PermissionError()
  monkeypatch.setattr(ffs, 'os', SimpleNamespace(
    SCHED_FIFO=1, sched_param=lambda p: SimpleNamespace(sched_priority=p),
    sched_setscheduler=denied))
  _bare_transport()._raise_reader_priority()   # must not raise on a box without RTPRIO
