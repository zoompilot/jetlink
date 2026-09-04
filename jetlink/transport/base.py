"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from jetlink import protocol as P

# One frame is ~460 KB. The cap is what stops a corrupt length field making
# RxBuffer allocate gigabytes before a single byte of it has been read.
MAX_MESSAGE = 16 << 20
_PAD = bytes(P.GADGET_TX_ALIGN)


class LinkError(IOError):
  """The link is unusable. Callers treat this as 'fall back to the small model'."""


class LinkTimeout(LinkError):
  """No complete message arrived in time. The stream is still in sync."""


@dataclass
class Message:
  msg_type: int
  seq: int
  flags: int
  payload: memoryview  # valid only until the next recv() on this transport


class Transport(ABC):
  """One framed, ordered, reliable message channel.

  Implementations must preserve message boundaries and ordering. recv() hands
  back a view into a reusable buffer: copy anything you need to keep.
  """

  @abstractmethod
  def send(self, msg_type: int, seq: int, parts=(), flags: int = 0) -> None:
    """Send one message. `parts` is an iterable of buffers, sent as one message."""

  @abstractmethod
  def recv(self, timeout: float | None = None) -> Message:
    """Block for one message. Raises LinkTimeout if `timeout` elapses."""

  @abstractmethod
  def close(self) -> None:
    ...

  def send_json(self, msg_type: int, seq: int, obj, flags: int = 0) -> None:
    import json
    self.send(msg_type, seq, (json.dumps(obj).encode(),), flags)


class RxBuffer:
  """Receive buffer for a byte stream carrying framed messages.

  Reads land directly in here and messages are handed out as views, so the
  steady state does no allocation and no copying between the wire and the
  caller. Crucially it is also *resumable*: a read that times out part way
  through a message leaves the bytes in place, so a missed deadline costs a
  frame rather than desyncing the stream.
  """

  def __init__(self, size: int = 1 << 20):
    self.buf = bytearray(size)
    self.view = memoryview(self.buf)
    self.start = 0  # first byte not yet consumed
    self.end = 0    # one past the last byte read

  @property
  def available(self) -> int:
    return self.end - self.start

  def reserve(self, need: int) -> None:
    """Guarantee room for `need` unconsumed bytes, compacting or growing."""
    if self.start and self.start + need > len(self.buf):
      # Slide the partial message to the front before considering a resize.
      # Go through the bytearray, not the memoryview: source and destination
      # overlap, and a memoryview slice assignment is a memcpy, which is
      # undefined on overlap. Slicing the bytearray materialises the source
      # first. Compaction is rare, so the copy is not worth avoiding.
      self.buf[:self.available] = self.buf[self.start:self.end]
      self.end -= self.start
      self.start = 0
    if need > len(self.buf):
      grown = bytearray(max(need, len(self.buf) * 2))
      grown[:self.available] = self.view[self.start:self.end]
      self.buf, self.view = grown, memoryview(grown)
      self.end -= self.start
      self.start = 0

  def writable(self) -> memoryview:
    return self.view[self.end:]

  def committed(self, n: int) -> None:
    self.end += n

  def take(self, n: int) -> memoryview:
    out = self.view[self.start:self.start + n]
    self.start += n
    return out

  def consumed(self) -> None:
    """Call once a message has been fully handed out."""
    if self.start == self.end:
      self.start = self.end = 0


class StreamTransport(Transport):
  """Framing over any ordered byte stream.

  TCP, USB bulk and FunctionFS are all streams, so they all need exactly this.
  Subclasses supply only the two primitives that differ.
  """

  # Extra capacity kept beyond the current message, for transports whose reads
  # must land in a buffer of at least a given size (FunctionFS OUT endpoints
  # want a multiple of the max packet size).
  read_slack = 0
  # Bulk endpoints reject a read whose buffer is not a whole number of packets.
  # 0 means "no constraint" (TCP).
  packet_size = 0
  read_chunk = 1 << 20
  # Pad every message sent to a multiple of this (0: the one-byte PADDED rule
  # instead), and expect the peer's messages padded likewise. Set on the two
  # USB transports; see protocol.GADGET_TX_ALIGN for the measurement behind it.
  tx_align = 0
  rx_align = 0
  # Largest single write to hand the kernel. FunctionFS turns one writev into
  # one USB request and has to allocate a contiguous buffer for it, so a big
  # write fails with ENOMEM on a device whose memory is fragmented - which the
  # inference path never sees, because its largest message is a few hundred KB,
  # and the model upload hits immediately at 4 MB a chunk. 0 means no cap (TCP).
  write_chunk = 0

  def __init__(self, rx_size: int = 1 << 20):
    self.rx = RxBuffer(rx_size)
    self._desynced = False

  # -- primitives a subclass must provide ----------------------------------

  @abstractmethod
  def _write(self, bufs: list[memoryview]) -> int:
    """Write from one or more buffers. Returns bytes written (may be partial)."""

  @abstractmethod
  def _read_into(self, dest: memoryview, timeout: float | None) -> int:
    """Read up to len(dest) bytes, returning how many arrived.

    May return short, including 0. Must NOT raise on a timeout: return whatever
    arrived and let _fill decide. Dropping partially transferred bytes is how a
    stream silently desyncs.
    """

  # -- framing -------------------------------------------------------------

  def send(self, msg_type: int, seq: int, parts=(), flags: int = 0) -> None:
    # cast('B') matters: slicing a memoryview of a float32 array in _advance
    # would step by elements, not bytes.
    bufs = [memoryview(p).cast('B') for p in parts]
    length = sum(b.nbytes for b in bufs)
    if self.tx_align:
      # Never a short packet in this direction; see protocol.GADGET_TX_ALIGN.
      pad = -(P.HEADER_SIZE + length) % self.tx_align
      if pad:
        bufs.append(memoryview(_PAD)[:pad])
    elif (P.HEADER_SIZE + length) % P.PACKET_MULTIPLE == 0:
      # A bulk transfer only ends on a short packet; see protocol.PACKET_MULTIPLE.
      flags |= P.Flag.PADDED
      bufs.append(memoryview(_PAD)[:1])
    header = P.pack_header(msg_type, seq, length, flags)
    bufs.insert(0, memoryview(header))
    while bufs:
      n = self._write(take(bufs, self.write_chunk) if self.write_chunk else bufs)
      if n <= 0:
        raise LinkError("peer went away during send")
      bufs = advance(bufs, n)

  def _clamp_read(self, dest: memoryview) -> int:
    """How many bytes this transport may ask for in one read."""
    n = min(dest.nbytes, self.read_chunk)
    return (n // self.packet_size) * self.packet_size if self.packet_size else n

  def _fill(self, need: int, timeout: float | None) -> None:
    """Read until `need` bytes are buffered, or the deadline passes.

    The deadline is per *message*, not per read: handing the full timeout to
    each read would let one 74 KB response take several times the caller's
    budget. Partial reads are kept, so a missed deadline costs a frame and
    leaves the stream in sync.
    """
    self.rx.reserve(need + self.read_slack)
    end = None if timeout is None else time.monotonic() + timeout
    while self.rx.available < need:
      dest = self.rx.writable()[:self._read_limit(need - self.rx.available)]
      if self._clamp_read(dest) == 0:
        # No room to post a whole packet, so every read from here returns 0 and
        # this loop would spin on a core forever while the peer blocks writing
        # the rest. A transport whose read_slack is too small gets here; say so
        # rather than hanging.
        raise LinkError(f"no room to read the rest of a {need} byte message "
                        + f"({self.rx.available} in hand); read_slack too small")
      remaining = None
      if end is not None:
        remaining = end - time.monotonic()
        if remaining <= 0:
          raise LinkTimeout(f"only {self.rx.available} of {need} bytes arrived in time")
      n = self._read_into(dest, remaining)
      if n:
        self.rx.committed(n)

  def _read_limit(self, missing: int) -> int:
    """The most one read may ask for, given `missing` bytes of the current
    message are still to come: exactly that, rounded up to a whole packet.

    On the USB host this is what keeps the stream in sync. The gadget's
    messages are burst-aligned (protocol.GADGET_TX_ALIGN), so a read that asks
    for exactly what is left completes by count, and there is never a read
    outstanding across the end of a message for the controller to fill with
    whatever it flushes at a short packet. Asking for more, the way a plain
    stream reader would, was measured to desync the link about once in 400
    frames. TCP and FunctionFS do not need the limit and are not hurt by it.
    """
    if self.packet_size:
      return -(-missing // self.packet_size) * self.packet_size
    return missing

  def drain(self, timeout: float) -> None:
    """After a desync, swallow whatever the peer is still sending, until the
    link drops or it goes quiet for `timeout`.

    The peer is mid-message and blocked writing the rest of it. Closing and
    reopening the link instead takes only what one read asks for each time,
    so on USB the peer's request drains a packet per reopen at ~100 Hz, its
    frame timeout never fires, and modeld sits inside one send for the whole
    drive. Measured. Reading it out lets the peer finish, time out, and
    reconnect cleanly.
    """
    scratch = memoryview(bytearray(self.read_chunk))
    last = time.monotonic()
    while time.monotonic() - last < timeout:
      try:
        n = self._read_into(scratch, timeout)
      except LinkError:
        return
      if n:
        last = time.monotonic()

  def recv(self, timeout: float | None = None) -> Message:
    if self._desynced:
      raise LinkError("stream desynced; the link must be reopened")
    end = None if timeout is None else time.monotonic() + timeout
    self._fill(P.HEADER_SIZE, timeout)
    try:
      _, _, msg_type, seq, flags, length, _reserved = P.unpack_header(
        self.rx.view[self.rx.start:self.rx.start + P.HEADER_SIZE])
      if length > MAX_MESSAGE:
        raise P.ProtocolError(f"message claims {length} bytes, over the {MAX_MESSAGE} cap")
    except P.ProtocolError as e:
      # Nothing can resynchronise a byte stream mid-message, and the bad bytes
      # are still buffered. Latch it and report a link failure: callers
      # reconnect on LinkError, whereas a ProtocolError escaping from here
      # unwinds out of the server's accept loop and kills the process.
      self._desynced = True
      raise LinkError(f"protocol error, link unusable: {e}") from e
    # The remainder of the caller's budget, not a second full one, so one recv
    # can never block for twice what it was given.
    if self.rx_align:
      pad = -(P.HEADER_SIZE + length) % self.rx_align
    else:
      pad = 1 if flags & P.Flag.PADDED else 0
    self._fill(P.HEADER_SIZE + length + pad,
               None if end is None else max(0.0, end - time.monotonic()))
    self.rx.take(P.HEADER_SIZE)
    payload = self.rx.take(length)
    self.rx.take(pad)
    self.rx.consumed()
    return Message(msg_type, seq, flags, payload)


def take(bufs: list[memoryview], n: int) -> list[memoryview]:
  """The first `n` bytes across a list of buffers, without copying."""
  out: list[memoryview] = []
  for mv in bufs:
    if n <= 0:
      break
    out.append(mv if mv.nbytes <= n else mv[:n])
    n -= out[-1].nbytes
  return out


def advance(bufs: list[memoryview], n: int) -> list[memoryview]:
  """Drop the first `n` bytes across a list of buffers, returning what is left."""
  out: list[memoryview] = []
  for mv in bufs:
    if n:
      if n >= mv.nbytes:
        n -= mv.nbytes
        continue
      mv = mv[n:]
      n = 0
    out.append(mv)
  return out
