"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The FunctionFS transport against FIFOs standing in for the endpoint files.

FunctionFS reads block regardless of O_NONBLOCK once the host has enabled the
endpoint, so a deadline is only honoured if the read is somewhere else. A FIFO
with nothing written to it blocks a reader the same way.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque

import numpy as np
import pytest

from jetlink import protocol as P
from jetlink.transport import ffs, priority
from jetlink.transport.base import LinkError, LinkTimeout
from jetlink.transport.ffs import FfsTransport


@pytest.fixture
def mount(tmp_path):
  for ep in ('ep0', 'ep1', 'ep2'):
    os.mkfifo(tmp_path / ep)
  return tmp_path


def test_no_endpoint_is_opened_before_a_host_has_enabled_it(mount):
  """The one that costs the gadget until the comma is rebooted.

  ffs_epfile_io sleeps on an endpoint no host has enabled, and unbind does not
  wake it. The stuck thread holds the struct file, so every later ffs_ep0_open
  answers EBUSY, jetlinkd and the next drive's modeld included.
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
    # started by that first recv, not the constructor, and still parked in its read
    assert t._reader is not None and t._reader.is_alive()
  finally:
    t.close()


def test_data_written_by_the_host_arrives_through_the_reader(mount):
  t = FfsTransport(str(mount))
  try:
    # a FIFO standing in for ep1 blocks its writer until somebody reads; the
    # real ep1 does not
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


def test_a_failure_names_the_controller_state_it_found(tmp_path, monkeypatch):
  """ENODEV reads the same for a cable falling out, a host resetting us and a
  bus the host suspended. A drive's worth of failures (2026-09-07: fourteen,
  every one a USB drop) said nothing about which; the UDC state does.
  """
  t = _bare_transport(bound_udc='udc0', _had_host=True, _closing=False, _queued=0,
                      _reader_error=None, _read_size=ffs.SS_MAX_PACKET, _free=deque(),
                      _cv=threading.Condition())
  monkeypatch.setattr(ffs, 'UDC_SYSFS', str(tmp_path))
  monkeypatch.setattr(t, '_widen_affinity', lambda: None)
  monkeypatch.setattr(t, '_raise_reader_priority', lambda: None)
  state = tmp_path / 'udc0' / 'state'
  state.parent.mkdir()
  state.write_text('not attached\n')
  # The reader checks the state before every read and stops on the first
  # miss; with no host there is no read to sleep in. On its own thread: the
  # loop blocks signals for the thread's life, and pytest's is not its own.
  reader = threading.Thread(target=t._read_loop)
  reader.start()
  reader.join(5.0)
  assert not reader.is_alive()
  assert t._reader_error == 'host dropped the gadget configuration (udc: not attached)'
  state.write_text('default\n')
  assert t._udc_note() == ' (udc: default)'
  # an empty read (the file mid-rewrite) is no state, not a host that left
  state.write_text('')
  assert t._udc_state() is None
  assert t._udc_note() == ''
  t.bound_udc = None
  assert t._udc_note() == ''


def test_the_watchdog_drops_the_link_so_a_stuck_write_can_return(mount, monkeypatch):
  """FunctionFS writes cannot time out: the request sits on the endpoint until the
  host drains it, and the join loop cannot retry what it is blocked inside.
  Unbinding works here only because the endpoint is enabled (see _ensure_epfiles).
  """
  t = FfsTransport(str(mount))
  try:
    unbound = []
    monkeypatch.setattr(t, 'unbind', lambda: unbound.append(True))
    t._abort_write()
    assert t._write_aborted and unbound, "the watchdog must drop the link, not just flag it"

    # once aborted, the write reports our own doing, not the host-ready grace period
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

  monkeypatch.setattr(priority, 'os', SimpleNamespace(cpu_count=lambda: 8,
                                                 sched_getaffinity=getaffinity,
                                                 sched_setaffinity=setaffinity))
  _bare_transport()._widen_affinity()
  assert calls['mask'] == set(range(7)), "reader did not move off the pinned frame-loop core"


def test_reader_affinity_is_a_noop_when_already_unpinned(monkeypatch):
  from types import SimpleNamespace
  calls = {}
  monkeypatch.setattr(priority, 'os', SimpleNamespace(
    cpu_count=lambda: 8,
    sched_getaffinity=lambda _pid: set(range(8)),
    sched_setaffinity=lambda _pid, mask: calls.setdefault('set', True)))
  _bare_transport()._widen_affinity()
  assert 'set' not in calls, "widened affinity when it was already full"


def test_reader_affinity_survives_a_platform_without_the_call(monkeypatch):
  from types import SimpleNamespace
  monkeypatch.setattr(priority, 'os', SimpleNamespace(cpu_count=lambda: 8))  # no sched_* (macOS)
  _bare_transport()._widen_affinity()  # must not raise


def test_send_resets_the_write_quantum_each_message(monkeypatch):
  """A prior ENOMEM shrink must not persist: a split write re-arms the dwc3
  double-TRB replay, so each message starts at the full quantum again and only
  the frames under memory pressure are ever split."""
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
  monkeypatch.setattr(priority, 'os', SimpleNamespace(
    cpu_count=lambda: 8,
    sched_getaffinity=lambda _pid: {0, 1, 2, 3},   # four cores online, unpinned
    sched_setaffinity=lambda _pid, mask: calls.__setitem__('mask', set(mask))))
  _bare_transport()._widen_affinity()
  assert calls['mask'] == set(range(8)), "reader did not widen an unpinned mask to all cores"


def test_read_buffers_are_recycled_not_reallocated():
  """The hot receive path must not allocate per read: under memory pressure that
  allocation reclaims, 24 ms on the bench. A consumed buffer returns to the pool."""
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
  for payload in (memoryview(ours)[:4], memoryview(foreign)):
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


def test_close_releases_the_endpoints_even_when_the_watchdog_hangs(mount, monkeypatch):
  """The leak that outlived the link.

  close() used to raise before touching an fd if the write guard had not come
  back within a second, which it has not when it is stuck inside its own
  abort's unbind. ep0 then stayed open for the life of the process and every
  descriptor write after it answered ESRCH: only a reboot brought the gadget
  back.
  """
  monkeypatch.setattr(ffs, 'READER_JOIN_TIMEOUT', 0.05)
  t = FfsTransport(str(mount))
  ep0 = t.ep0
  stuck = threading.Event()
  guard = threading.Thread(target=stuck.wait, daemon=True)
  guard.start()
  t._write_guard.thread = guard
  try:
    t.close()
  finally:
    stuck.set()

  assert t.ep0 == -1
  with pytest.raises(OSError):
    os.fstat(ep0)   # really closed, not merely forgotten

  # And the gadget opens again, which is the half the raise used to cost.
  again = FfsTransport(str(mount))
  again.close()


def test_rebind_bounces_a_gadget_no_host_ever_configured(mount, monkeypatch):
  """The one edge a Jetson that took the bind as a wake and then stopped needs.

  It answers with a bus reset and parks the UDC in default or addressed;
  nothing on the comma moves it but another connect.
  """
  monkeypatch.setattr(ffs, 'REBIND_SETTLE', 0.0)
  t = FfsTransport(str(mount))
  try:
    t.gadget = '/sys/kernel/config/usb_gadget/jetlink'
    t.bound_udc = 'udc0'
    calls = []
    monkeypatch.setattr(t, 'unbind', lambda: calls.append('unbind'))
    monkeypatch.setattr(t, 'bind', lambda udc=None: calls.append(f'bind {udc}'))
    assert t.rebind() is True
    assert calls == ['unbind', 'bind udc0'], calls
  finally:
    t.gadget = None
    t.close()


@pytest.mark.parametrize('state', ['no gadget', 'endpoints open', 'not bound'])
def test_rebind_is_refused_once_it_would_cost_a_working_link(mount, monkeypatch, state):
  monkeypatch.setattr(ffs, 'REBIND_SETTLE', 0.0)
  t = FfsTransport(str(mount))
  try:
    t.gadget = None if state == 'no gadget' else '/sys/kernel/config/usb_gadget/jetlink'
    t.bound_udc = None if state == 'not bound' else 'udc0'
    if state == 'endpoints open':
      t.ep_out = 999   # unbinding here completes the reader's read with ESHUTDOWN
    unbound = []
    monkeypatch.setattr(t, 'unbind', lambda: unbound.append(state))
    assert t.rebind() is False
    assert not unbound, 'unbound a link that was not stalled'
  finally:
    monkeypatch.undo()
    t.gadget, t.ep_out = None, -1
    t.close()


def test_only_the_gadget_can_bounce_itself():
  # tcp and libusb peers reconnect on their own; there is nothing to bounce.
  from jetlink.transport.tcp import TcpTransport
  srv = TcpTransport.listen('127.0.0.1', 0)
  try:
    t = TcpTransport.connect('127.0.0.1', srv.getsockname()[1])
    assert t.rebind() is False
    t.close()
  finally:
    srv.close()


def _udc(tmp_path, monkeypatch, state='configured', name='udc0'):
  monkeypatch.setattr(ffs, 'UDC_SYSFS', str(tmp_path))
  path = tmp_path / name / 'state'
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(state + '\n')
  return path


def test_a_borrowed_gadget_opens_no_ep0_and_binds_nothing(mount, tmp_path, monkeypatch):
  """The owner holds ep0 and the bind for its whole life, so the link changing
  hands is no longer an unplug the host has to recover from."""
  _udc(tmp_path / 'sys', monkeypatch)
  owner = FfsTransport(str(mount))          # stands in for jetlinkd
  t = FfsTransport.borrowed(str(mount), 'udc0')
  try:
    assert t.ep0 == -1 and t.gadget is None
    assert t.bound_udc == 'udc0'
    t.unbind()                              # a borrower must never take the gadget down
    assert owner.ep0 >= 0
  finally:
    t.close()
    owner.close()


def test_a_borrowed_gadget_still_waits_for_a_host(mount, tmp_path, monkeypatch):
  """The ep0 trap does not care who opened the endpoint: a read on one no host
  has enabled sleeps in the kernel and costs the gadget until a reboot."""
  monkeypatch.setattr(ffs, 'EP_OPEN_TIMEOUT', 0.05)
  state = _udc(tmp_path / 'sys', monkeypatch, state='powered')
  t = FfsTransport.borrowed(str(mount), 'udc0')
  try:
    with pytest.raises(LinkTimeout):
      t._ensure_epfiles()
    assert t.ep_out == -1 and t.ep_in == -1
    state.write_text('configured\n')
    t._ensure_epfiles()
    assert t.ep_out >= 0 and t.ep_in >= 0
  finally:
    t.close()


def test_a_borrowed_gadget_with_no_controller_named_is_refused(mount):
  # without one _configured() answers "assume ready" and the endpoints open
  # before any host has enabled them
  with pytest.raises(LinkError, match='controller'):
    FfsTransport.borrowed(str(mount), '')


def test_a_stuck_borrowed_write_asks_the_owner_to_free_it(mount, tmp_path, monkeypatch):
  """Unbinding is what dequeues a FunctionFS write nobody is reading, and on a
  borrowed gadget only the owner can do it."""
  _udc(tmp_path / 'sys', monkeypatch)
  asked = []
  t = FfsTransport.borrowed(str(mount), 'udc0', bounce=lambda: asked.append(True))
  try:
    monkeypatch.setattr(t, 'unbind', lambda: pytest.fail('a borrower took the gadget down'))
    t._abort_write()
    assert t._write_aborted and asked == [True]
  finally:
    monkeypatch.undo()
    t.close()


def test_a_gadget_is_lendable_only_with_nothing_open_on_it(mount, tmp_path, monkeypatch):
  """FunctionFS keeps a queued read queued until something completes it, so a
  reader here would sit in front of the borrower and take its reply."""
  _udc(tmp_path / 'sys', monkeypatch)
  t = FfsTransport(str(mount))
  try:
    assert not t.lendable, 'a gadget bound to nothing has nothing to hand over'
    t.gadget, t.bound_udc = '/sys/kernel/config/usb_gadget/jetlink', 'udc0'
    assert t.lendable
    t._ensure_epfiles()
    assert not t.lendable
  finally:
    t.gadget = None
    t.close()
  assert not t.lendable


def test_a_borrower_asks_the_owner_to_bounce_a_stalled_bus(mount, tmp_path, monkeypatch):
  # the same signature as an owned gadget's, and the same one edge; only the
  # process that can make it is a different one
  _udc(tmp_path / 'sys', monkeypatch)
  asked = []
  t = FfsTransport.borrowed(str(mount), 'udc0', bounce=lambda: asked.append(True) or True)
  try:
    assert t.rebind() is True
    assert asked == [True]
    t.ep_out = 999
    assert t.rebind() is False, 'bounced a link this end is reading'
  finally:
    t.ep_out = -1
    t.close()
