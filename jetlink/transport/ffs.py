"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

USB gadget side of the link: a FunctionFS vendor-specific bulk function.

On a comma this is the *comma* end. The roles look backwards and are not:
AGNOS has CONFIG_USB_F_FS and libcomposite built in, while a USB host needs no
kernel driver at all, which matters because L4T rootfs images are often
stripped of the gadget modules. See docs/transport.md.

Bring-up (scripts/setup_gadget.sh does all of this):
    configfs gadget -> functions/ffs.jetlink -> mount -t functionfs
    write descriptors + strings to ep0 -> ep1 (OUT) and ep2 (IN) appear
    echo <udc> > UDC
"""
from __future__ import annotations

import errno
import logging
import os
import signal
import struct
import threading
import time
from collections import deque

from jetlink import protocol as P
from jetlink.transport.base import LinkError, LinkTimeout, StreamTransport, take
from jetlink.transport.watchdog import WriteWatchdog

# --- FunctionFS ABI -------------------------------------------------------

FUNCTIONFS_DESCRIPTORS_MAGIC_V2 = 3
FUNCTIONFS_STRINGS_MAGIC = 2

FLAG_HAS_FS = 1 << 0
FLAG_HAS_HS = 1 << 1
FLAG_HAS_SS = 1 << 2

USB_DT_INTERFACE = 0x04
USB_DT_ENDPOINT = 0x05
USB_DT_SS_ENDPOINT_COMP = 0x30

USB_ENDPOINT_XFER_BULK = 0x02
EP_OUT = 0x01  # host -> device
EP_IN = 0x82   # device -> host

SS_MAX_PACKET = 1024
# FunctionFS kmallocs a contiguous kernel buffer for every read, even though
# our userspace buffer is already allocated. The 2026-09-05 drive hit order-6
# and order-5 allocation failures in ffs_epfile_read_iter, with 100-300 ms
# reply stalls. 16 KB (order-2) stopped the failures. Going smaller to dodge
# the rare *slow success* (an order-2 read reclaiming ~16 ms mid-frame under
# memory pressure) was tried and reverted: at one page per read a 74 KB reply
# takes ~19 syscalls and their wakeups added ~6 ms to every frame's baseline,
# far more than the occasional reclaim it removed. The userspace side of the
# per-read allocation is handled instead by recycling (see _read_loop); the
# kernel side stays at 16 KB and leans on the free-memory floor the setup
# script sets to keep order-2 off the direct-reclaim path.
READ_CHUNK = 16 * SS_MAX_PACKET
# How much the reader thread may queue before it stops reading. The inference
# path never needs more than one response; the cap only bounds memory if the
# consumer stalls.
MAX_QUEUED = 8 << 20
# How many read buffers to keep for reuse. The consumer copies each chunk out
# and hands the buffer back; steady state has one or two in flight, so this is
# only a ceiling for a transient backlog. 16 * 16 KB = 256 KB.
FREE_BUFS = 16
# SCHED_FIFO priority for the reader thread. modeld runs its frame loop at 54;
# the reader sits just below so the loop always wins a contended core, but it
# still preempts every SCHED_OTHER thread the way the loop does. See
# _raise_reader_priority for why the reader needs this at all.
READER_RT_PRIORITY = 51

# A host enumerating us is not the same as a host being ready to talk: it still
# has to open the device and claim the interface, and until it does the gadget's
# endpoints return EIO/ESHUTDOWN. The server polls sysfs every couple of
# seconds to notice us, so that gap is easily a second or two. Wait it out
# rather than reporting a dead link on the first frame after connect.
log = logging.getLogger('jetlink')

EP_READY_TIMEOUT = 10.0
# How long opening the endpoint files may wait for the host to configure us.
# Separate from EP_READY_TIMEOUT, which covers a host that has configured us
# but has not claimed the interface yet.
EP_OPEN_TIMEOUT = 10.0

# How long a write may sit with the host not reading before we take the link
# down ourselves. FunctionFS writes cannot time out: the request is queued on
# an enabled endpoint and os.writev returns when the host drains it, whenever
# that is. A hello sent to a Jetson that has enumerated but whose server is not
# reading blocks for as long as that stays true - measured 90 s while the
# Jetson booted, and three minutes on a drive 2026-09-05, which the driver saw
# as the big model simply never arriving. The join loop cannot retry what it is
# blocked inside.
#
# Unbinding the UDC is the only lever. It dequeues the pending request, the
# writev returns ESHUTDOWN, and the caller gets a LinkError it can retry -
# which is exactly what happened by hand when the driver cycled offroad. Note
# this works *because the endpoint is enabled*: unbind does not wake a read
# waiting on an endpoint no host ever enabled, which is a different bug with a
# different fix (see _ensure_epfiles).
#
# 15 s is far longer than any real write (a frame is ~3.6 ms) and short enough
# that a stuck link costs one retry rather than a drive. Churning the gadget a
# few times while a Jetson boots is harmless: the server polls for it and
# re-enumeration is milliseconds.
WRITE_TIMEOUT = 15.0
UDC_SYSFS = '/sys/class/udc'
# How long close() waits for the reader thread after unbinding, which is what
# wakes it. A read the kernel will not complete is left to die with the process.
READER_JOIN_TIMEOUT = 1.0
_NOT_READY = (errno.EIO, errno.ESHUTDOWN, errno.ENODEV)

# Every signal the thread doing endpoint IO must not take mid-transfer. A
# signal that interrupts a FunctionFS write is not a retry: ffs_epfile_io
# dequeues the request, which stops the transfer wherever it is, and returns
# EINTR, and Python's os.writev then re-issues the whole call. The packets
# already on the wire stay sent, so the host receives the start of the message
# twice and the stream is lost from there. modeld is exactly the process this
# happens to: msgq wakes its subscriber threads with SIGUSR2 for every camera
# frame, ~160 a second, and strace on a live modeld showed four of them landing
# inside writev in 120 s, one per link failure that run. Masking signals for
# the few milliseconds of a write just defers them; nothing in here waits on
# one. SIGKILL and SIGSTOP cannot be masked and are ignored by the call.
_IO_SIGNALS = signal.valid_signals()


def _interface_desc(n_endpoints: int = 2, i_interface: int = 1) -> bytes:
  return struct.pack('<BBBBBBBBB', 9, USB_DT_INTERFACE, 0, 0, n_endpoints,
                     0xFF, 0xFF, 0xFF, i_interface)


def _endpoint_desc(addr: int, max_packet: int) -> bytes:
  return struct.pack('<BBBBHB', 7, USB_DT_ENDPOINT, addr, USB_ENDPOINT_XFER_BULK, max_packet, 0)


def _ss_companion(max_burst: int = 15) -> bytes:
  """SuperSpeed endpoint companion.

  bMaxBurst is the number of *additional* packets per burst, so 15 means bursts
  of 16 x 1024 bytes. It matters enormously: without bursting the gadget sends
  one packet per handshake and the 459 KB request takes 167 ms (~2.75 MB/s),
  which blows the whole 50 ms frame budget on its own.
  """
  return struct.pack('<BBBBH', 6, USB_DT_SS_ENDPOINT_COMP, max_burst, 0, 0)


def build_descriptors() -> bytes:
  fs = _interface_desc() + _endpoint_desc(EP_OUT, 64) + _endpoint_desc(EP_IN, 64)
  hs = _interface_desc() + _endpoint_desc(EP_OUT, 512) + _endpoint_desc(EP_IN, 512)
  ss = (_interface_desc()
        + _endpoint_desc(EP_OUT, SS_MAX_PACKET) + _ss_companion()
        + _endpoint_desc(EP_IN, SS_MAX_PACKET) + _ss_companion())

  # struct usb_functionfs_descs_head_v2: magic, length, flags, then ALL the
  # per-speed counts together (one __le32 per flag set, in flag order), and only
  # then the descriptor blocks. Interleaving count/block per speed gets EINVAL.
  body = struct.pack('<III', 3, 3, 5) + fs + hs + ss
  flags = FLAG_HAS_FS | FLAG_HAS_HS | FLAG_HAS_SS
  return struct.pack('<III', FUNCTIONFS_DESCRIPTORS_MAGIC_V2, 12 + len(body), flags) + body


def build_strings(name: str = 'jetlink') -> bytes:
  s = name.encode() + b'\0'
  return (struct.pack('<IIII', FUNCTIONFS_STRINGS_MAGIC, 16 + 2 + len(s), 1, 1)
          + struct.pack('<H', 0x0409) + s)


class FfsTransport(StreamTransport):
  """
  Reads run on their own thread. FunctionFS ignores O_NONBLOCK once the host
  has enabled the endpoint: a synchronous read always waits for the USB
  request to complete, so a read on the caller's thread cannot honour any
  deadline and a Jetson that stops answering blocks modeld for as long as it
  stays silent (measured on a comma: a 0.5 s timeout returned after 14 s, when
  the server was resumed). The thread absorbs the blocking read and hands
  whole chunks over under a condition variable, which the caller waits on with
  a real timeout. Writes stay on the caller's thread: a write only blocks until
  the host has read it, and a host that is not reading is a dead link either
  way.
  """
  read_chunk = READ_CHUNK
  # One writev is one USB request is one TRB, and this controller (dwc3 on
  # AGNOS 4.9) will occasionally send a TRB twice. Measured on the bench with a
  # live modeld: about once in 400 frames the host read, right after a
  # complete message, another copy of one of that message's chunks. When the
  # request was split into two 256 KB writes the replay of the first chunk
  # carried a valid header, so the server ran the frame again from a payload
  # that was half the old request and half the next one, and a replay of the
  # second chunk put pixels where the next header should be. Either way the
  # stream was lost and modeld spent the drive rejoining. A message that fits
  # one request is replayed whole, with its sequence number, and the receiver
  # drops it (Session.handle). The largest inference request is 475 KB with
  # its padding; only the model upload is bigger, and a corrupted upload is
  # caught by its sha256 and sent again.
  write_chunk = 512 * SS_MAX_PACKET
  tx_align = P.GADGET_TX_ALIGN

  def __init__(self, mount: str = '/dev/ffs-jetlink', gadget: str | None = None,
               udc: str | None = None):
    # The gadget end only ever receives replies - an INFER_RESP is ~74 KB of
    # output plus telemetry and burst padding, well under 128 KB, and the
    # control messages are smaller still. The whole model upload goes the other
    # way (writes), so nothing large lands in this buffer. 256 KB is a 3x margin
    # that RxBuffer still grows past on demand if the wire format ever changes;
    # a 2 MB start was 1.75 MB of resident memory the memory-tight comma did not
    # need. (In the unsupported gadget-on-Jetson inversion this end would take
    # 458 KB requests - it grows to fit on the first one, once.)
    super().__init__(rx_size=256 << 10)
    self.mount = mount
    self.gadget = gadget
    self.bound_udc: str | None = None
    self.ep0 = self.ep_out = self.ep_in = -1
    self._state_fd = -1   # held-open UDC 'state' fd; see _udc_state
    self._ready_deadline: float | None = None
    self._read_size = READ_CHUNK
    self._cv = threading.Condition()
    self._chunks: deque[tuple[memoryview, float, float, float]] = deque()
    self._free: deque[bytearray] = deque()   # read buffers the consumer handed back; see _read_loop
    self.last_receive = {'prepare': 0.0, 'read_wait': 0.0, 'handoff': 0.0}
    self._queued = 0
    self._reader_error: str | None = None
    self._closing = False
    self._had_host = False
    self._open_lock = threading.Lock()
    self._write_aborted = False
    self._reader: threading.Thread | None = None
    self._write_guard = WriteWatchdog(self._abort_write)
    try:
      self.ep0 = os.open(os.path.join(mount, 'ep0'), os.O_RDWR)
      os.write(self.ep0, build_descriptors())
      os.write(self.ep0, build_strings())
      if gadget is not None:
        # Bind last: a FunctionFS gadget cannot attach to a controller until its
        # descriptors have been written, which is why setup_gadget.sh leaves UDC
        # empty and we finish the job here.
        self.bind(udc)
      # The endpoint files exist now, but opening them is deferred to
      # _ensure_epfiles: see there for why touching one before a host has
      # enabled it costs the gadget until the next reboot.
    except BaseException:
      self.close()   # otherwise a failed bring-up leaks the descriptors it did open
      raise

  def _udc_state(self) -> str | None:
    """The controller's current gadget state, off a held-open fd.

    The reader checks this before every readv - a readv on a deconfigured
    endpoint sleeps in the kernel until a signal that never comes (see
    _read_loop), so it cannot be skipped. Opening the sysfs file each time was
    the cost: `open` walks dentries and allocates a struct file, and under
    recording memory pressure that allocation reclaims - the 2026-09-06 bench
    saw it stall 25 ms mid-frame (prepare 24.9 ms), straight over budget.
    Holding the fd and re-reading it (lseek to 0, read) still re-runs the
    attribute's show() so the value is current - verified on the device against
    a fresh open across the state's values - but allocates nothing, so it cannot
    stall. The controller outlives our bind/unbind, so the fd stays valid; it is
    dropped on unbind and reopened lazily in case the controller ever differs.
    """
    if self.bound_udc is None:
      return None
    if self._state_fd < 0:
      try:
        self._state_fd = os.open(os.path.join(UDC_SYSFS, self.bound_udc, 'state'), os.O_RDONLY)
      except OSError:
        return None
    try:
      os.lseek(self._state_fd, 0, os.SEEK_SET)
      return os.read(self._state_fd, 64).decode().strip()
    except OSError:
      _close_quietly(self._state_fd)
      self._state_fd = -1
      return None

  def _configured(self) -> bool:
    """Has a host set our configuration? Only then are the endpoints enabled."""
    if self.bound_udc is None:
      return self.gadget is None   # no controller of ours to ask; assume ready
    return self._udc_state() == 'configured'

  def _ensure_epfiles(self) -> None:
    """Open ep1/ep2 and start the reader, once a host has enabled them.

    Not at open, which is where this used to be, and the difference is the
    whole gadget. ffs_epfile_io waits for the endpoint with
    wait_event_interruptible(ffs->wait, (ep = epfile->ep)), so a read on an
    endpoint no host has ever enabled does not return EIO - it sleeps, and
    only a signal wakes it. Unbinding does not: unbind completes a request the
    hardware already has, and there is no request here. The reader thread was
    started at open, so on any drive without a Jetson it went straight into
    that sleep and stayed there.

    That thread's fd is then unclosable while it sleeps, close() or no close():
    the syscall holds the struct file, ffs_epfile_release never runs, ffs->opened
    stays above zero and ffs_ep0_open answers EBUSY from then on - to this
    process, to jetlinkd, and to the next drive's modeld, until the comma is
    rebooted. Measured 2026-09-04: one open/close of a gadget no Jetson ever
    attached to, and every open after it failed with EBUSY.

    So wait for the UDC to say configured, which is the same edge that sets
    epfile->ep, and only then open them. A host that never arrives leaves ep0
    as the only fd, and closing that is clean.
    """
    if self.ep_out >= 0:
      return
    with self._open_lock:
      if self.ep_out >= 0:
        return
      deadline = time.monotonic() + EP_OPEN_TIMEOUT
      while not self._configured():
        if self._closing:
          raise LinkError("gadget closing")
        if time.monotonic() >= deadline:
          raise LinkTimeout(f"no host configured the gadget within {EP_OPEN_TIMEOUT:.0f}s")
        time.sleep(0.02)
      self.ep_out = os.open(os.path.join(self.mount, 'ep1'), os.O_RDWR)
      self.ep_in = os.open(os.path.join(self.mount, 'ep2'), os.O_RDWR)
      self._had_host = True
      self._reader = threading.Thread(target=self._read_loop, name='jetlink-ffs-read', daemon=True)
      self._reader.start()

  def bind(self, udc: str | None = None) -> None:
    if self.gadget is None:
      raise LinkError("no gadget path given")
    if udc is None:
      udcs = sorted(os.listdir(UDC_SYSFS))
      if not udcs:
        raise LinkError("no USB device controller found")
      udc = udcs[0]
    with open(os.path.join(self.gadget, 'UDC'), 'w') as f:
      f.write(udc + '\n')
    self.bound_udc = udc

  def unbind(self) -> None:
    if self.gadget is None or self.bound_udc is None:
      return
    try:
      with open(os.path.join(self.gadget, 'UDC'), 'w') as f:
        f.write('\n')
    except OSError:
      pass
    self.bound_udc = None
    if self._state_fd >= 0:   # reopens lazily against whatever udc we bind next
      _close_quietly(self._state_fd)
      self._state_fd = -1

  def _shrink(self, attr: str) -> bool:
    """Halve a transfer size after the kernel refused to allocate for one.

    ENOMEM here is about contiguous DMA memory, not about how much RAM is
    free, so it depends on how fragmented the machine is right now and a size
    that worked at boot can fail an hour in - which is exactly what handling a
    1.7 GB model does to a comma. Backing off keeps the link slow rather than
    broken. Both directions need it: reads hit this after a big upload just as
    writes hit it during one.
    """
    floor = (self.packet_size or SS_MAX_PACKET) * 16
    current = getattr(self, attr)
    if current <= floor:
      return False
    setattr(self, attr, max(floor, current // 2))
    if attr == '_read_size':
      self._free.clear()   # pooled buffers are the old, larger size now
    log.warning("jetlink: gadget could not allocate for %s, dropping to %d KB",
                attr, getattr(self, attr) >> 10)
    return True

  def _shrink_write(self) -> bool:
    return self._shrink('write_chunk')

  def _abort_write(self) -> None:
    """Take the link down so a write nobody is reading returns. See WRITE_TIMEOUT.

    Runs on the watchdog thread, never on the one stuck in writev - that thread
    cannot act, which is the whole problem.
    """
    self._write_aborted = True
    log.warning("jetlink: no reader for %.3f s, dropping the gadget to free the write",
                getattr(self, '_write_budget', WRITE_TIMEOUT))
    self.unbind()

  def send(self, *args, **kwargs) -> None:
    # Start every message at the full write quantum. _shrink_write halves it on
    # an ENOMEM, and once it has, a request larger than the shrunk size is split
    # across two writes - which re-arms the dwc3 double-TRB replay that splitting
    # was measured to cause (see write_chunk). The shrink must therefore last
    # only as long as the memory pressure that forced it: the contiguous memory
    # a 512 KB kmalloc failed for on one frame is almost always back by the next,
    # and a request that fits one write must go out as one write. Resetting here,
    # at each message boundary, keeps the split window to the frames actually
    # under ENOMEM instead of latching every request split for the rest of the
    # drive. A no-op for TCP, whose write_chunk is 0.
    self.write_chunk = type(self).write_chunk
    super().send(*args, **kwargs)

  def _write(self, bufs: list[memoryview]) -> int:
    self._ensure_epfiles()
    self._write_budget = self._write_timeout(WRITE_TIMEOUT)
    # See _IO_SIGNALS: a signal here duplicates data on the wire.
    was = signal.pthread_sigmask(signal.SIG_BLOCK, _IO_SIGNALS)
    try:
      if not self._write_guard.arm(self._write_budget):
        raise LinkError('gadget write watchdog already expired or closed')
      while True:
        try:
          n = os.writev(self.ep_in, bufs)
          self._had_host = True
          return n
        except OSError as e:
          if self._write_aborted:
            # Our own doing, not the host's: say so, because "no such device"
            # on its own reads like a cable falling out.
            raise LinkError(f"gadget write had no reader for {self._write_budget:.3f}s") from e
          # FunctionFS submits a write as one request, so a failed writev put
          # nothing on the wire and is safe to retry.
          if e.errno in _NOT_READY and self._wait_for_host_ready():
            continue
          if e.errno == errno.ENOMEM and self._shrink_write():
            # A short write is fine: send() loops until the message is out.
            bufs = take(bufs, self.write_chunk)
            continue
          raise LinkError(f"gadget write failed: {e}") from e
    finally:
      completed = self._write_guard.disarm()
      signal.pthread_sigmask(signal.SIG_SETMASK, was)
      if not completed:
        raise LinkError('gadget write had no reader before deadline; link abandoned')

  # -- the reader thread ---------------------------------------------------

  def _widen_affinity(self) -> None:
    """Move the reader off the one core modeld pinned its frame loop to.

    This thread is created wherever the first read or write happens, because
    that is where _ensure_epfiles runs. On the comma that is normally the
    caller's background thread (the joining state's join loop, or jetlinkd),
    which has already dropped to SCHED_OTHER on every core, so this is usually
    a no-op. It is the case where it is not that costs frames: modeld runs
    config_realtime_process(7, 54), and any thread created from a thread that
    inherited that gets SCHED_FIFO 54 *and* the single-core pin. Sharing that
    core with the frame loop serialises them - a read that has completed on the
    endpoint cannot be taken off it until the loop next blocks. Simply widening
    the mask to every core was measured not to help: the balancer keeps waking
    this thread on the core it last ran, next to the loop. So when the
    inherited mask is a single core, run on every *other* core instead; the
    frame loop keeps its core to itself and a completed read is serviced at
    once elsewhere. The priority is left to _raise_reader_priority below.
    jetlink must not import openpilot, so this open-codes what
    common.realtime.set_core_affinity would do.
    """
    try:
      everything = set(range(os.cpu_count() or 1))
      inherited = os.sched_getaffinity(0)
      if len(inherited) == 1 and everything - inherited:
        os.sched_setaffinity(0, everything - inherited)   # off the frame-loop core
      elif everything - inherited:
        os.sched_setaffinity(0, everything)               # unpinned already; just widen
    except (OSError, AttributeError):
      # No affinity call (macOS), or a kernel that will not move us. The
      # priority still lets a completed read preempt normal work on our core.
      pass

  def _raise_reader_priority(self) -> None:
    """Run the reader at realtime priority so a completed read is taken off the
    endpoint at once, not behind whatever else the comma is running.

    The thread that creates this one has dropped realtime deliberately (see the
    caller's background priority helper), so what this inherits is SCHED_OTHER
    at priority 0 - measured on the device. read_wait brackets the whole readv,
    including the time this thread waits on the run queue to return once the
    transfer is done, and under recording plus onroad
    load that wait was 17 ms mean and 23 ms max, landing straight on read_wait
    and over budget - the residual that looked like a kernel allocation stall
    but is scheduling. The reader only copies a chunk and notifies before it
    blocks in the next read, so realtime here preempts the contention without
    ever holding a core. Below modeld's frame loop (54) so the loop still wins.
    Best effort: a dev box without RTPRIO just keeps SCHED_OTHER and the slower
    tail. jetlink must not import openpilot, so this open-codes the setscheduler.
    """
    try:
      os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(READER_RT_PRIORITY))
    except (OSError, AttributeError, ValueError):
      pass

  def _read_loop(self) -> None:
    # The same hazard as _write, on the read side: an interrupted read drops
    # the packets it had already taken. This thread is ours and handles no
    # signals, so mask them for its whole life.
    signal.pthread_sigmask(signal.SIG_BLOCK, _IO_SIGNALS)
    self._widen_affinity()
    self._raise_reader_priority()
    # Fill the recycle pool up front. Recycling alone only removes the hot-path
    # allocation once the consumer has handed a buffer back; while a reply's
    # chunks are still arriving, a frame-thread that is a scheduling beat behind
    # (which CPU contention makes routine) has not recycled the last one yet, so
    # the reader allocates the next - and that allocation reclaims under memory
    # pressure (prepare 24 ms, over budget). Pre-allocating FREE_BUFS here, once,
    # off the frame path, means the reader draws from the pool through that gap
    # and only ever allocates if a stalled consumer lets it fall this far behind.
    self._free.extend(bytearray(self._read_size) for _ in range(FREE_BUFS))
    while not self._closing:
      with self._cv:
        while self._queued >= MAX_QUEUED and not self._closing:
          self._cv.wait(0.1)
      if self._closing:
        return
      prepare_started = time.monotonic()
      # A fresh buffer per read: the chunk is handed to the consumer as is, so
      # reusing one would overwrite bytes it has not copied out yet.
      if not self._configured():
        # Same trap as _ensure_epfiles, one loop later: the host can drop our
        # configuration between two reads, and a readv issued after that sleeps
        # in the kernel until a signal that is never coming. Treat it as the
        # endpoint error it would have been, and let the grace period decide.
        if self._wait_for_host_ready():
          continue
        self._fail("host dropped the gadget configuration")
        return
      # Reuse a buffer the consumer handed back rather than allocating one on
      # the hot path. A fresh bytearray here, like the kernel's own per-read
      # kmalloc, reclaims under recording memory pressure - the 2026-09-06 bench
      # measured this allocation stalling 20+ ms (prepare 24 ms) mid-frame, and
      # the read cannot start until it returns. The reader is the only thread
      # that pops the free list, so a bare check needs no lock.
      buf = self._free.popleft() if self._free else bytearray(self._read_size)
      read_started = time.monotonic()
      try:
        # Multiples of the packet size only: the OUT endpoint rejects anything
        # else, and _read_size is only ever halved from one.
        got = os.readv(self.ep_out, [buf])
      except OSError as e:
        if self._closing:
          return
        if e.errno in _NOT_READY and self._wait_for_host_ready():
          continue
        if e.errno == errno.ENOMEM and self._shrink('_read_size'):
          continue
        self._fail(f"gadget read failed: {e}")
        return
      if got == 0:
        self._fail("gadget read returned EOF (host disconnected)")
        return
      read_finished = time.monotonic()
      self._ready_deadline = None
      self._had_host = True
      with self._cv:
        self._chunks.append((memoryview(buf)[:got], read_finished,
                             read_started - prepare_started, read_finished - read_started))
        self._queued += got
        self._cv.notify_all()

  def _fail(self, why: str) -> None:
    with self._cv:
      self._reader_error = why
      self._cv.notify_all()

  def recv(self, timeout: float | None = None):
    # Per-message maxima, in seconds. read_wait includes waiting for the peer
    # and kernel IO; handoff includes waiting for the consumer to run. Neither
    # is a pure scheduler or USB transfer measurement. No logging on the reader.
    self.last_receive = {'prepare': 0.0, 'read_wait': 0.0, 'handoff': 0.0}
    return super().recv(timeout)

  def _read_into(self, dest: memoryview, timeout: float | None) -> int:
    self._ensure_epfiles()
    with self._cv:
      if not self._chunks and self._reader_error is None:
        self._cv.wait(timeout)
      if self._chunks:
        chunk, arrived, prepare, read_wait = self._chunks[0]
        self.last_receive['prepare'] = max(self.last_receive['prepare'], prepare)
        self.last_receive['read_wait'] = max(self.last_receive['read_wait'], read_wait)
        self.last_receive['handoff'] = max(self.last_receive['handoff'], time.monotonic() - arrived)
        n = min(chunk.nbytes, dest.nbytes)
        dest[:n] = chunk[:n]
        if n < chunk.nbytes:
          self._chunks[0] = (chunk[n:], arrived, prepare, read_wait)
        else:
          self._chunks.popleft()
          # The copy above is done and nothing else reads this chunk, so return
          # its buffer to the reader's pool. chunk.obj is the bytearray readv
          # filled (still so after a chunk[n:] reslice); skip anything that is
          # not one of ours at the current size - a shrunk buffer, or a bytes
          # object a test injected.
          buf = chunk.obj
          if type(buf) is bytearray and len(buf) == self._read_size and len(self._free) < FREE_BUFS:
            self._free.append(buf)
        self._queued -= n
        self._cv.notify_all()
        return n
      if self._reader_error is not None:
        raise LinkError(self._reader_error)
      return 0   # timed out with nothing new; _fill owns the deadline

  def _wait_for_host_ready(self) -> bool:
    """True while we are still inside the grace period for the host to claim us.

    Enumeration and readiness are different things: the host has to open the
    device and claim the interface before our endpoints work, and until then
    they return EIO. Starting the clock on the first such error rather than at
    open means the wait covers a re-enumeration too.
    """
    if self._host_gone():
      return False
    now = time.monotonic()
    if self._ready_deadline is None:
      self._ready_deadline = now + EP_READY_TIMEOUT
    if now < self._ready_deadline:
      time.sleep(0.005)
      return True
    return False

  def _host_gone(self) -> bool:
    """A host had us configured and no longer does.

    The endpoints answer ENODEV both before a host has configured us and after
    it disconnects, and only the first deserves the grace period above. The
    second used to get it too: on the 2026-09-04 drive a USB disconnect landed
    mid-write, the kernel disabled the endpoints at once, and modeld then sat
    here for the full EP_READY_TIMEOUT retrying a link the UDC already said
    was gone, 10 s and 201 frames before modeld fell back. Both errors look
    the same at the endpoint; the UDC state is what tells them apart, and it
    is set in the same interrupt that disabled the endpoint, so it is already
    current by the time the failed call returns.
    """
    if not self._had_host or self.bound_udc is None:
      return False
    state = self._udc_state()
    return state is not None and state != 'configured'

  def close(self) -> None:
    self._closing = True
    self._write_guard.close()
    self._write_guard.thread.join(READER_JOIN_TIMEOUT)
    if self._write_guard.thread.is_alive():
      # Keep ownership rather than allow a late abort to unbind a new owner.
      raise LinkError('gadget watchdog teardown did not finish')
    # Unbinding disables the endpoints, which completes the reader's pending
    # request with ESHUTDOWN and lets the thread exit before its fd goes away.
    self.unbind()
    reader, self._reader = self._reader, None
    if reader is not None and reader is not threading.current_thread():
      reader.join(READER_JOIN_TIMEOUT)
    with self._cv:
      self._cv.notify_all()
    # Clear each fd as it is closed: a second close() would otherwise shut
    # whatever those descriptor numbers had been recycled into.
    for name in ('_state_fd', 'ep_in', 'ep_out', 'ep0'):
      fd = getattr(self, name, -1)
      setattr(self, name, -1)
      if fd is None or fd < 0:
        continue
      if name == 'ep_out' and reader is not None and reader.is_alive():
        # The read did not come back (no gadget to unbind, or a kernel that
        # will not complete it). Linux lets close() return regardless, but
        # not every kernel does, so do not risk the caller on it.
        threading.Thread(target=_close_quietly, args=(fd,), daemon=True).start()
        continue
      _close_quietly(fd)


def _close_quietly(fd: int) -> None:
  try:
    os.close(fd)
  except OSError:
    pass
