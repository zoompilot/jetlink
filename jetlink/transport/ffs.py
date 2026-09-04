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
READ_CHUNK = 256 * SS_MAX_PACKET  # 256 KB
# How much the reader thread may queue before it stops reading. The inference
# path never needs more than one response; the cap only bounds memory if the
# consumer stalls.
MAX_QUEUED = 8 << 20

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
    super().__init__(rx_size=2 << 20)
    self.mount = mount
    self.gadget = gadget
    self.bound_udc: str | None = None
    self.ep0 = self.ep_out = self.ep_in = -1
    self._ready_deadline: float | None = None
    self._read_size = READ_CHUNK
    self._cv = threading.Condition()
    self._chunks: deque[memoryview] = deque()
    self._queued = 0
    self._reader_error: str | None = None
    self._closing = False
    self._had_host = False
    self._open_lock = threading.Lock()
    self._reader: threading.Thread | None = None
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

  def _configured(self) -> bool:
    """Has a host set our configuration? Only then are the endpoints enabled."""
    if self.bound_udc is None:
      return self.gadget is None   # no controller of ours to ask; assume ready
    try:
      with open(os.path.join(UDC_SYSFS, self.bound_udc, 'state')) as f:
        return f.read().strip() == 'configured'
    except OSError:
      return False

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
    log.warning("jetlink: gadget could not allocate for %s, dropping to %d KB",
                attr, getattr(self, attr) >> 10)
    return True

  def _shrink_write(self) -> bool:
    return self._shrink('write_chunk')

  def _write(self, bufs: list[memoryview]) -> int:
    self._ensure_epfiles()
    # See _IO_SIGNALS: a signal here duplicates data on the wire.
    was = signal.pthread_sigmask(signal.SIG_BLOCK, _IO_SIGNALS)
    try:
      while True:
        try:
          n = os.writev(self.ep_in, bufs)
          self._had_host = True
          return n
        except OSError as e:
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
      signal.pthread_sigmask(signal.SIG_SETMASK, was)

  # -- the reader thread ---------------------------------------------------

  def _read_loop(self) -> None:
    # The same hazard as _write, on the read side: an interrupted read drops
    # the packets it had already taken. This thread is ours and handles no
    # signals, so mask them for its whole life.
    signal.pthread_sigmask(signal.SIG_BLOCK, _IO_SIGNALS)
    while not self._closing:
      with self._cv:
        while self._queued >= MAX_QUEUED and not self._closing:
          self._cv.wait(0.1)
      if self._closing:
        return
      # A fresh buffer per read: the chunk is handed to the consumer as is, so
      # reusing one would overwrite bytes it has not copied out yet. 256 KB at
      # 20 Hz is nothing next to the frame itself.
      if not self._configured():
        # Same trap as _ensure_epfiles, one loop later: the host can drop our
        # configuration between two reads, and a readv issued after that sleeps
        # in the kernel until a signal that is never coming. Treat it as the
        # endpoint error it would have been, and let the grace period decide.
        if self._wait_for_host_ready():
          continue
        self._fail("host dropped the gadget configuration")
        return
      buf = bytearray(self._read_size)
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
      self._ready_deadline = None
      self._had_host = True
      with self._cv:
        self._chunks.append(memoryview(buf)[:got])
        self._queued += got
        self._cv.notify_all()

  def _fail(self, why: str) -> None:
    with self._cv:
      self._reader_error = why
      self._cv.notify_all()

  def _read_into(self, dest: memoryview, timeout: float | None) -> int:
    self._ensure_epfiles()
    with self._cv:
      if not self._chunks and self._reader_error is None:
        self._cv.wait(timeout)
      if self._chunks:
        chunk = self._chunks[0]
        n = min(chunk.nbytes, dest.nbytes)
        dest[:n] = chunk[:n]
        if n < chunk.nbytes:
          self._chunks[0] = chunk[n:]
        else:
          self._chunks.popleft()
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
    try:
      with open(os.path.join(UDC_SYSFS, self.bound_udc, 'state')) as f:
        return f.read().strip() != 'configured'
    except OSError:
      return False

  def close(self) -> None:
    self._closing = True
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
    for name in ('ep_in', 'ep_out', 'ep0'):
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
