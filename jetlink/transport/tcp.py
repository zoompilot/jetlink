"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

TCP transport.

Used over ethernet, and for benchmarking. Note it cannot run over a USB cable
to a comma: AGNOS has no host-side USB-ethernet driver, so a USB ethernet
gadget will not enumerate there. See docs/transport.md.
"""
from __future__ import annotations

import socket

from jetlink.transport.base import LinkError, StreamTransport

DEFAULT_PORT = 5599


class TcpTransport(StreamTransport):
  def __init__(self, sock: socket.socket):
    super().__init__()
    self.sock = sock
    self._timeout: float | None = -1.0  # force the first settimeout
    _tune(sock)

  @classmethod
  def connect(cls, host: str, port: int = DEFAULT_PORT, timeout: float = 5.0) -> TcpTransport:
    return cls(socket.create_connection((host, port), timeout=timeout))

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

  def _write(self, bufs: list[memoryview]) -> int:
    # sendmsg keeps the header and a 460 KB body in one syscall, and with
    # TCP_NODELAY that goes out as one segment train rather than a small header
    # packet followed by the body.
    self._set_timeout(None)
    try:
      return self.sock.sendmsg(bufs)
    except socket.timeout as e:
      raise LinkTimeout("send timed out") from e
    except OSError as e:
      raise LinkError(f"send failed: {e}") from e

  def _read_into(self, dest: memoryview, timeout: float | None) -> int:
    self._set_timeout(timeout)
    try:
      n = self.sock.recv_into(dest, dest.nbytes)
    except socket.timeout:
      return 0  # recv_into delivers nothing on timeout; _fill owns the deadline
    except OSError as e:
      raise LinkError(f"recv failed: {e}") from e
    if n == 0:
      raise LinkError("peer closed the connection")
    return n

  def close(self) -> None:
    try:
      self.sock.close()
    except OSError:
      pass


def _tune(sock: socket.socket) -> None:
  # NODELAY is the one that matters: without it a 32-byte header and a 460 KB
  # body can be split across an RTT. It cannot fail on a real TCP socket, but a
  # transport must not die in a setsockopt, so it is guarded like the rest.
  for level, opt, value in ((socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),
                            (socket.SOL_SOCKET, socket.SO_SNDBUF, 4 << 20),
                            (socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20),
                            (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)):
    try:
      sock.setsockopt(level, opt, value)
    except OSError:
      pass
