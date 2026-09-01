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


class LinkError(IOError):
  """The link is unusable. Callers treat this as 'fall back to the small model'."""


class LinkTimeout(LinkError):
  pass


@dataclass
class Message:
  msg_type: int
  seq: int
  flags: int
  t_mono_ns: int
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

  # -- shared helpers ------------------------------------------------------

  def __enter__(self):
    return self

  def __exit__(self, *exc):
    self.close()

  def send_json(self, msg_type: int, seq: int, obj, flags: int = 0) -> None:
    import json
    self.send(msg_type, seq, (json.dumps(obj).encode(),), flags)

  @staticmethod
  def json_of(msg: Message):
    import json
    return json.loads(bytes(msg.payload))


class FramedBuffer:
  """Reassembles the 32-byte-header framing out of a byte stream.

  Bulk USB and TCP are both streams, so both need this. The buffer grows to the
  largest message seen and is then reused, which is what keeps the steady state
  allocation-free.
  """

  def __init__(self, initial: int = 1 << 20):
    self.buf = bytearray(initial)
    self.view = memoryview(self.buf)

  def ensure(self, n: int) -> memoryview:
    if len(self.buf) < n:
      self.buf = bytearray(max(n, len(self.buf) * 2))
      self.view = memoryview(self.buf)
    return self.view

  def parse_header(self) -> tuple[int, int, int, int, int]:
    _, _, msg_type, seq, flags, length, t_mono_ns = P.unpack_header(self.view[:P.HEADER_SIZE])
    return msg_type, seq, flags, length, t_mono_ns


def now_ns() -> int:
  return time.monotonic_ns()
