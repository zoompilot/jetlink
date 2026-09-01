"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

TCP transport.

This is the development and benchmarking transport, and the one to use if you
attach the Jetson over ethernet instead of USB. It is NOT usable over a USB-C
cable to a comma 3X: AGNOS's kernel has no host-side CDC-NCM/ECM/RNDIS driver
(only CDC_SUBSET/ZAURUS/QMI_WWAN are built in), so a USB ethernet gadget will
not enumerate there. Use `transport.usbbulk` for that. See docs/transport.md.
"""
from __future__ import annotations

import socket

from jetlink import protocol as P
from jetlink.transport.base import FramedBuffer, LinkError, LinkTimeout, Message, Transport, now_ns

DEFAULT_PORT = 5599


def _tune(sock: socket.socket) -> None:
  # NODELAY is the one that matters: without it a 32-byte header and a 460 KB
  # body can be split across an RTT. It cannot fail on a real TCP socket, but
  # a transport must not die in a setsockopt, so it is guarded like the rest.
  try:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
  except OSError:
    pass
  # A half-megabyte request must not be split across RTTs by a small window.
  for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF):
    try:
      sock.setsockopt(socket.SOL_SOCKET, opt, 4 << 20)
    except OSError:
      pass
  try:  # keepalive so a yanked cable surfaces as an error, not a hang
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
  except OSError:
    pass


class TcpTransport(Transport):
  def __init__(self, sock: socket.socket):
    self.sock = sock
    _tune(self.sock)
    self.fb = FramedBuffer()
    self._timeout: float | None = None

  @classmethod
  def connect(cls, host: str, port: int = DEFAULT_PORT, timeout: float = 5.0) -> TcpTransport:
    sock = socket.create_connection((host, port), timeout=timeout)
    return cls(sock)

  @classmethod
  def listen(cls, host: str = '0.0.0.0', port: int = DEFAULT_PORT, backlog: int = 1) -> socket.socket:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(backlog)
    return srv

  @classmethod
  def accept(cls, srv: socket.socket) -> tuple[TcpTransport, tuple]:
    conn, addr = srv.accept()
    return cls(conn), addr

  def _set_timeout(self, timeout: float | None) -> None:
    if timeout != self._timeout:
      self.sock.settimeout(timeout)
      self._timeout = timeout

  def send(self, msg_type: int, seq: int, parts=(), flags: int = 0) -> None:
    # cast('B') matters: slicing a memoryview of a float32 array in _advance
    # would step by elements, not bytes.
    parts = [memoryview(p).cast('B') for p in parts]
    length = sum(p.nbytes for p in parts)
    header = P.pack_header(msg_type, seq, length, flags, now_ns())
    # sendmsg keeps the header and a 512 KB payload in one syscall, and with
    # TCP_NODELAY that goes out as one segment train rather than a small header
    # packet followed by the body.
    bufs = [memoryview(header), *parts]
    self._set_timeout(None)
    try:
      while bufs:
        n = self.sock.sendmsg(bufs)
        if n == 0:
          raise LinkError("peer closed during send")
        bufs = _advance(bufs, n)
    except socket.timeout as e:
      raise LinkTimeout("send timed out") from e
    except OSError as e:
      raise LinkError(f"send failed: {e}") from e

  def _recv_exact(self, view: memoryview) -> None:
    got = 0
    n = view.nbytes
    while got < n:
      try:
        r = self.sock.recv_into(view[got:], n - got)
      except socket.timeout as e:
        raise LinkTimeout("recv timed out") from e
      except OSError as e:
        raise LinkError(f"recv failed: {e}") from e
      if r == 0:
        raise LinkError("peer closed the connection")
      got += r

  def recv(self, timeout: float | None = None) -> Message:
    self._set_timeout(timeout)
    self.fb.ensure(P.HEADER_SIZE)
    self._recv_exact(self.fb.view[:P.HEADER_SIZE])
    msg_type, seq, flags, length, t_mono_ns = self.fb.parse_header()
    view = self.fb.ensure(P.HEADER_SIZE + length)
    if length:
      self._recv_exact(view[P.HEADER_SIZE:P.HEADER_SIZE + length])
    return Message(msg_type, seq, flags, t_mono_ns, view[P.HEADER_SIZE:P.HEADER_SIZE + length])

  def close(self) -> None:
    try:
      self.sock.close()
    except OSError:
      pass


def _advance(bufs: list[memoryview], n: int) -> list[memoryview]:
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
