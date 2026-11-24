"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The comma's end of an add-on: ask the host, and let the caller run its own
model on any doubt.

AddonLink owns one add-on's interface (a jetlink.comma.functions
GadgetFunction). It opens the endpoints only once a host has configured the
gadget, and on any read or write error closes both at once and waits for that
again. infer() returns None, for the caller's own model, on any doubt:
  - no hello in HELLO_FRESH_S, or one that does not list the model,
  - the host yielding its GPU to the driving model: a YIELD answer for that
    request alone, a YIELD hello until the next plain one,
  - no answer within REPLY_TIMEOUT_S, a send still in flight, or an answer of
    the wrong size; it rests MISS_HOLD_S after a bad answer, or MISS_REST_AFTER
    timeouts in MISS_WINDOW_S, so a slow host cannot hold a caller at the faster
    cadence it keeps for the host, while one late answer costs that request alone.

HostModel wraps any runtime's model as a call, and JetsonNet stands in for a
cv2.dnn net (setInput/forward): only the forward pass moves, and the caller's
own model runs whenever the host does not answer.

Where the comma's cable is a network (an iPhone), the link listens on the
cable address at the function's port instead and the phone dials it; a TCP
stream has no transfers, so answers are framed by their headers there.

FunctionFS transfers are synchronous and a signal inside one loses bytes
(jetlink.transport.ffs), so both I/O threads block every signal. FunctionFS
also allocates one contiguous kernel buffer per call, so reads go in 16 KB
pieces, a message ending at the first short one, and writes in 8 KB pieces.
An endpoint file is closed only once no thread is inside a call on it, so its
number is never reused under one. numpy and the standard library only. From Peter Nguyen's V-ASM and sign-vision clients
(trung791997/openpilot).
"""
from __future__ import annotations

import collections
import logging
import os
import queue
import signal
import socket
import threading
import time

import numpy as np

from jetlink import protocol as P
from jetlink.addons import wire as W
from jetlink.comma import functions

HELLO_FRESH_S = 2.5
REPLY_TIMEOUT_S = 0.05
MISS_HOLD_S = 3.0
# as the server's yieldAfter in yieldFor: a lone timeout is a transfer's tail, not a slow host
MISS_REST_AFTER = 3
MISS_WINDOW_S = 2.0
REOPEN_S = 2.0
WRITE_CHUNK = 8192
# as jetlink.transport.ffs reads: a multiple of the packet size, an order-2 allocation
READ_CHUNK = 16 * P.USB_MAX_PACKET
REPORT_S = 30.0


def _host_attached() -> bool:
  from jetlink.comma import gadget
  return gadget.host_attached()


def _mask_signals():
  signal.pthread_sigmask(signal.SIG_BLOCK, signal.valid_signals())


class _Endpoint:
  """One open endpoint file, closed only once no thread is inside a call on
  it: a number closed under a read or write could be reused by any open."""

  def __init__(self, fd: int):
    self.fd, self.users, self.closed = fd, 0, False
    self._lock = threading.Lock()

  def acquire(self) -> int:
    """The fd for one call, -1 once closed; release() after."""
    with self._lock:
      if self.closed:
        return -1
      self.users += 1
      return self.fd

  def release(self):
    with self._lock:
      self.users -= 1
      last = self.closed and self.users == 0
    if last:
      self._close_fd()

  def close(self):
    with self._lock:
      if self.closed:
        return
      self.closed = True
      idle = self.users == 0
    if idle:
      self._close_fd()

  def _close_fd(self):
    try:
      os.close(self.fd)
    except OSError:
      pass


class RoundTrips:
  """Answered requests' round trips, and the Jetson's own share of each (server_us)."""

  def __init__(self, keep: int = 4096):
    self.total_us: collections.deque = collections.deque(maxlen=keep)
    self.server_us: collections.deque = collections.deque(maxlen=keep)
    self._lock = threading.Lock()   # the caller adds, the reader takes

  def add(self, total_us: int, server_us: int):
    with self._lock:
      self.total_us.append(total_us)
      self.server_us.append(server_us)

  def take(self) -> dict:
    with self._lock:
      total = np.asarray(self.total_us, np.float64) / 1000.0
      server = np.asarray(self.server_us, np.float64) / 1000.0
      self.total_us.clear()
      self.server_us.clear()
    if total.size == 0:
      return {"n": 0}
    return {"n": int(total.size), "total_ms": round(float(np.percentile(total, 50)), 2),
            "total_p99_ms": round(float(np.percentile(total, 99)), 2), "jetson_ms": round(float(np.percentile(server, 50)), 2)}


class AddonLink:
  def __init__(self, function, *, host_ready=None, report=None, clock=time.monotonic, cable_network=None):
    """function: the add-on's GadgetFunction. report(dict) hears the link's
    numbers every REPORT_S, for a fork's status page; None drops them."""
    self.function, self.addon = function, getattr(function, 'name', 'addon')
    self.host_ready = host_ready if host_ready is not None else _host_attached
    self.cable_network = cable_network if cable_network is not None else functions.cable_network
    self.report_to, self.clock = report, clock
    self.tags: frozenset[int] = frozenset()
    self.wanted: frozenset[int] = frozenset()   # the models this comma asks for, by tag; replaced, never changed
    self.last_hello = -1e9
    self.yield_seen = False
    self.hold_until = -1e9
    self.misses: collections.deque = collections.deque(maxlen=MISS_REST_AFTER)
    self.stats = {"eligible_attempts": 0, "sent": 0, "answered": 0, "timeouts": 0, "skipped_busy": 0, "unknown_model": 0, "bad_answers": 0,
                  "yields": 0, "reopens": 0}
    self.round_trips = RoundTrips()
    self._lock = threading.Lock()
    self._seq = 0
    self._waiting: dict[int, list] = {}
    self._requests: queue.Queue = queue.Queue(maxsize=1)
    self._writing = threading.Event()
    self._rx: _Endpoint | None = None   # ep1, host to comma
    self._tx: _Endpoint | None = None   # ep2, comma to host
    self._stream = False                 # a TCP connection, framed by its headers, not FunctionFS transfers
    self._listener: socket.socket | None = None
    self._conn: socket.socket | None = None
    self._next_open = 0.0
    self._threads_started = False
    self._open_lock = threading.Lock()   # one opener, for nets on several threads
    self._report_lock = threading.Lock()
    self._next_report = clock() + REPORT_S

  # -- public ------------------------------------------------------------
  def want(self, model_tag: int) -> None:
    """A model this comma will ask for, so the report can say the host lacks it."""
    if model_tag not in self.wanted:
      self.wanted = self.wanted | {model_tag}

  def live(self, model_tag: int) -> bool:
    return model_tag in self.tags and self.clock() - self.last_hello < HELLO_FRESH_S and not self.yield_seen

  def note_attempt(self, model_tag: int):
    with self._lock:
      if self.live(model_tag):
        self.stats["eligible_attempts"] += 1

  def ready(self, model_tag: int) -> bool:
    return self.live(model_tag) and self.clock() >= self.hold_until and not self._writing.is_set()

  def poll(self, model_tag: int) -> bool:
    """May a request for the model go now? Cheap, and it opens the link and
    reports as infer() does, so a caller can skip building its image when not."""
    self.want(model_tag)
    self._ensure_open()
    self._maybe_report()
    if not self.ready(model_tag):
      if self._writing.is_set():
        self.stats["skipped_busy"] += 1
      return False
    return True

  def infer(self, model_tag: int, image: np.ndarray, out_size: int, encoding: int | None = None) -> np.ndarray | None:
    """The model's outputs, flat float32 in output-name order, from the host;
    None for the caller's own model. image is [c, h, w], uint8 or float32,
    whose values the host gives the model as they are; encoding W.ENC_U8
    instead has it divide uint8 by 255."""
    if not self.poll(model_tag):
      return None
    encoding = encoding if encoding is not None else {np.dtype(np.uint8): W.ENC_BYTES, np.dtype(np.float32): W.ENC_F32}.get(image.dtype)
    if encoding is None or W.ITEM[encoding] != image.itemsize or image.ndim != 3 or image.nbytes + W.REQ.size >= W.MAX_REQUEST:
      return None
    with self._lock:
      self._seq = (self._seq + 1) % W.HELLO_SEQ
      seq = self._seq
      slot = [threading.Event(), None]
      self._waiting[seq] = slot
    t0 = time.monotonic_ns()
    msg = W.pack_request(seq, model_tag, np.ascontiguousarray(image).tobytes(), image.shape, encoding)
    try:
      self._requests.put_nowait(msg)
    except queue.Full:
      self._drop(seq)
      return None
    self.stats["sent"] += 1
    if not slot[0].wait(REPLY_TIMEOUT_S):
      self._drop(seq)
      self.stats["timeouts"] += 1
      self._timed_out()
      return None
    status, server_us, out = slot[1]
    if status == W.STATUS_UNKNOWN_MODEL:
      self.stats["unknown_model"] += 1
      return None
    if status != W.STATUS_OK or out.size != out_size or not np.isfinite(out).all():
      if status != W.STATUS_YIELD:
        self.stats["bad_answers"] += 1
        self.hold_until = self.clock() + MISS_HOLD_S
      return None
    self.round_trips.add((time.monotonic_ns() - t0) // 1000, server_us)
    self.stats["answered"] += 1
    return out

  def _timed_out(self):
    now = self.clock()
    self.misses.append(now)
    if len(self.misses) == MISS_REST_AFTER and now - self.misses[0] < MISS_WINDOW_S:
      self.misses.clear()
      self.hold_until = now + MISS_HOLD_S

  def report(self) -> dict:
    """models: the host's, by tag; missing: those this comma asks for that a
    connected host does not have, a different file on each end most often."""
    connected = self.clock() - self.last_hello < HELLO_FRESH_S
    tags = self.tags
    return {"present": True, "connected": connected, "yielding": self.yield_seen, "models": sorted(tags),
            "missing": sorted(self.wanted - tags) if connected else [], "link": dict(self.stats),
            "round_trips": self.round_trips.take()}

  # -- internals ---------------------------------------------------------
  def _maybe_report(self):
    """From the caller and from each hello, so a link the caller is not using
    still reports."""
    with self._report_lock:
      now = self.clock()
      if now < self._next_report:
        return
      self._next_report = now + REPORT_S
    if self.report_to is not None:
      try:
        self.report_to(self.report())
      except Exception:
        logging.getLogger(__name__).exception("jetlink: %s's report failed", self.addon)

  def _drop(self, seq: int):
    with self._lock:
      self._waiting.pop(seq, None)

  def _ensure_open(self):
    if self._rx is not None or self.clock() < self._next_open:
      return
    if not self._open_lock.acquire(blocking=False):
      return
    try:
      if self._rx is not None:
        return
      self._next_open = self.clock() + REOPEN_S
      stream = self.cable_network()
      fds = self._accept() if stream else self._open_endpoints()
      if fds is None:
        return
      with self._lock:
        self._rx, self._tx = _Endpoint(fds[0]), _Endpoint(fds[1])
        self._stream = stream
      self.stats["reopens"] += 1
      if not self._threads_started:
        self._threads_started = True
        threading.Thread(target=self._forever, args=(self._read_forever, 'reader'), name=f"{self.addon}-jetson-rx", daemon=True).start()
        threading.Thread(target=self._forever, args=(self._write_forever, 'writer'), name=f"{self.addon}-jetson-tx", daemon=True).start()
    finally:
      self._open_lock.release()

  def _open_endpoints(self) -> tuple[int, int] | None:
    self._stop_listening()
    # never before a host has configured the gadget: that costs the gadget until a reboot
    if not functions.present(self.function) or not self.host_ready():
      return None
    try:
      fd_in = os.open(self.function.ep_out, os.O_RDONLY)
    except OSError:
      return None
    try:
      fd_out = os.open(self.function.ep_in, os.O_WRONLY)
    except OSError:
      os.close(fd_in)
      return None
    return fd_in, fd_out

  def _accept(self) -> tuple[int, int] | None:
    """The phone's connection to this add-on's port on the cable address, if one came."""
    from jetlink.transport.tcp import CABLE_ADDRESS
    if self._listener is None:
      listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((CABLE_ADDRESS, self.function.port))
        listener.listen(1)
        listener.setblocking(False)
      except OSError:
        listener.close()
        return None
      self._listener = listener
    try:
      conn, _ = self._listener.accept()
    except OSError:
      return None
    conn.setblocking(True)
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    self._conn = conn
    return os.dup(conn.fileno()), os.dup(conn.fileno())

  def _stop_listening(self):
    listener, self._listener = self._listener, None
    if listener is not None:
      listener.close()

  def _close(self, failed: _Endpoint | None = None):
    """Both endpoints at once, so the owner's ep0 is never left busy by one of
    ours. With `failed`, only while that endpoint is still the open one: a late
    error from a thread never closes the pair opened after it."""
    with self._lock:
      if failed is not None and failed is not self._rx and failed is not self._tx:
        return
      endpoints, self._rx, self._tx = (self._rx, self._tx), None, None
      conn, self._conn = self._conn, None
      self.tags, self.yield_seen = frozenset(), False
      self._next_open = self.clock() + REOPEN_S
    if conn is not None:
      # ends a read blocked on it, so its endpoint can close
      try:
        conn.shutdown(socket.SHUT_RDWR)
      except OSError:
        pass
      conn.close()
    for endpoint in endpoints:
      if endpoint is not None:
        endpoint.close()

  def handle(self, buf) -> None:
    """One message from the Jetson."""
    try:
      seq, model_tag, status, server_us, count, payload = W.unpack_answer(buf)
    except ValueError:
      return
    self.last_hello = self.clock()   # an answer is as good as a hello
    if seq == W.HELLO_SEQ:
      self.tags = W.hello_tags(payload, count)
      if status == W.STATUS_YIELD and not self.yield_seen:
        self.stats["yields"] += 1
      self.yield_seen = status == W.STATUS_YIELD
      self._maybe_report()
      return
    if status == W.STATUS_YIELD:
      self.stats["yields"] += 1
    with self._lock:
      slot = self._waiting.pop(seq, None)
    if slot is not None:
      slot[1] = (status, server_us, np.frombuffer(payload, np.float32).copy())
      slot[0].set()

  def _forever(self, body, what: str):
    # the only reader and writer: no error in a message may end either
    while True:
      try:
        body()
      except Exception:
        logging.getLogger(__name__).exception("jetlink: %s's %s failed, reopening the link", self.addon, what)
        self._close()

  def _read_forever(self):
    _mask_signals()
    buf = bytearray(READ_CHUNK)
    message = bytearray()
    while True:
      endpoint, stream = self._rx, self._stream
      fd = endpoint.acquire() if endpoint is not None else -1
      if fd < 0:
        message.clear()
        time.sleep(0.2)
        continue
      try:
        got = os.readv(fd, [buf])
      except OSError:
        got = 0
      finally:
        endpoint.release()
      if got == 0 or len(message) + got > W.MAX_ANSWER + 1:
        # unbound, no longer configured, or no message ever ends: reopened once configured again
        message.clear()
        self._close(endpoint)
        continue
      message += memoryview(buf)[:got]
      if stream:
        # each whole answer at the front of a TCP stream; a bad header ends the link
        try:
          while len(message) >= W.RESP.size and len(message) >= (size := W.answer_length(message)):
            self.handle(bytes(message[:size]))
            del message[:size]
        except ValueError:
          message.clear()
          self._close(endpoint)
        continue
      # a transfer goes on through whole pieces and ends at a short one: the pad byte sees to that
      if got < READ_CHUNK:
        self.handle(bytes(message))
        message.clear()

  def _write_forever(self):
    _mask_signals()
    while True:
      msg = self._requests.get()
      endpoint = self._tx
      fd = endpoint.acquire() if endpoint is not None else -1
      if fd < 0:
        continue
      failed = False
      self._writing.set()
      try:
        view = memoryview(msg)
        for off in range(0, len(view), WRITE_CHUNK):
          piece = view[off:off + WRITE_CHUNK]
          while piece:
            piece = piece[os.write(fd, piece):]
      except OSError:
        failed = True
      finally:
        endpoint.release()
        self._writing.clear()
      if failed:
        self._close(endpoint)




def blob_to_u8(blob: np.ndarray) -> np.ndarray:
  """blobFromImage(u8, 1/255) back to the u8 [3, H, W] it came from: exact for those values."""
  return np.rint(blob[0] * 255.0).clip(0, 255).astype(np.uint8)


class ScaledU8:
  """A uint8 blob [1, C, H, W] that stands for blob / 255, as blobFromImage(img,
  1/255) makes it: the host gets its bytes as they are and scales them, and the
  float blob is made only when the comma runs the model itself, so neither end
  converts the image back. For a HostModel or JetsonNet only; scaled_input()
  gives a plain net its float blob."""

  __slots__ = ("u8",)

  def __init__(self, u8: np.ndarray):
    self.u8 = u8

  def scaled(self) -> np.ndarray:
    return self.u8.astype(np.float32) / 255.0


def scaled_input(net, u8: np.ndarray):
  """The blob for net from a uint8 [1, C, H, W] image scaled by 1/255:
  ScaledU8 for a net that asks a host, the float blob for any other."""
  return ScaledU8(u8) if isinstance(net, HostModel) else ScaledU8(u8).scaled()


class HostModel:
  """Any runtime's model, asking the host first: model(blob) gives every
  output in `outputs` order, from the host when it answers and from
  local(blob), the caller's own runner (onnxruntime, tinygrad, cv2), otherwise.

  outputs: the model's (name, shape) for every graph output; the host sends
  them in name order. scaled_u8: the caller promises the blob is a u8 image
  scaled by 1/255 (blobFromImage(img, 1/255)), so it goes as its u8 bytes, a
  quarter of the size and exact. Any other float blob must leave it False,
  the default, and goes as f32; a uint8 blob goes as its bytes, which the
  host gives the model as the same values. A blob cv2's setInput would scale or shift runs on the
  comma. enabled: a cheap callable read on every call, so a
  setting turns the host off and on without a reload."""

  def __init__(self, link: AddonLink, sha256: str, outputs, scaled_u8: bool = False, enabled=None, local=None):
    self.link, self.tag, self.local = link, W.tag(sha256), local
    self.outputs = tuple((name, tuple(shape)) for name, shape in outputs)
    self.wire = sorted(self.outputs)
    self.sizes = [int(np.prod(shape)) for _, shape in self.wire]
    self.scaled_u8 = scaled_u8
    self.enabled = enabled
    link.want(self.tag)

  def on(self) -> bool:
    try:
      return self.enabled is None or bool(self.enabled())
    except Exception:
      return False

  def ready(self) -> bool:
    """Would the host answer now: the setting on, and the host serving this
    model? For a fork that runs more often while it does."""
    return self.on() and self.link.ready(self.tag)

  def ask(self, blob) -> dict | None:
    """Every output by name from the host; None for the caller's own model."""
    # the image is built only for a host that will take it: no cost on the comma without one
    if blob is None or not self.on():
      return None
    self.link.note_attempt(self.tag)
    if not self.link.poll(self.tag):
      return None
    if isinstance(blob, ScaledU8):
      if blob.u8.dtype != np.uint8 or blob.u8.ndim != 4 or blob.u8.shape[0] != 1:
        return None
      return self._values(np.ascontiguousarray(blob.u8[0]), W.ENC_U8)
    blob = np.asarray(blob)
    # the host takes one image, [1, C, H, W]: a batch runs here, whole
    if blob.ndim != 4 or blob.shape[0] != 1:
      return None
    if blob.dtype == np.uint8:
      # its values as they are, whatever the model's input type
      image, encoding = np.ascontiguousarray(blob[0]), W.ENC_BYTES
    elif self.scaled_u8:
      image, encoding = blob_to_u8(blob), W.ENC_U8
    else:
      image, encoding = np.ascontiguousarray(blob[0], np.float32), W.ENC_F32
    return self._values(image, encoding)

  def _values(self, image: np.ndarray, encoding: int) -> dict | None:
    flat = self.link.infer(self.tag, image, sum(self.sizes), encoding)
    if flat is None:
      return None
    values, start = {}, 0
    for (name, shape), size in zip(self.wire, self.sizes, strict=True):
      values[name] = flat[start:start + size].reshape(shape)
      start += size
    return values

  def __call__(self, blob):
    values = self.ask(blob)
    if values is None and isinstance(blob, ScaledU8):
      blob = blob.scaled()
    return self.local(blob) if values is None else tuple(values[name] for name, _ in self.outputs)


class JetsonNet(HostModel):
  """A cv2.dnn net whose forward() asks the host first (HostModel). The first
  of `outputs` is what forward() with no names returns, as cv2 returns the
  graph's first."""

  _OWN = frozenset({"net", "link", "tag", "local", "outputs", "wire", "sizes", "scaled_u8", "enabled", "blob", "pending"})

  def __init__(self, net, link: AddonLink, sha256: str, outputs, scaled_u8: bool = False, enabled=None):
    self.net, self.blob, self.pending = net, None, None
    super().__init__(link, sha256, outputs, scaled_u8=scaled_u8, enabled=enabled)

  def setInput(self, blob, *args, **kwargs):
    # setInput(blob, name, scalefactor, mean): the host is sent the blob as it is, so one
    # that cv2 would scale or shift runs on the comma, where it is
    scale = kwargs.get("scalefactor", args[1] if len(args) > 1 else 1.0)
    mean = kwargs.get("mean", args[2] if len(args) > 2 else None)
    plain = scale == 1.0 and (mean is None or not np.any(mean))
    self.pending = None
    if isinstance(blob, ScaledU8):
      if plain:
        # the float blob reaches the net only if forward() runs it here
        self.blob, self.pending = blob, (args, kwargs)
        return
      blob = blob.scaled()
    self.blob = blob if plain else None
    self.net.setInput(blob, *args, **kwargs)

  def forward(self, names=None):
    wanted = [self.outputs[0][0]] if names is None else ([names] if isinstance(names, str) else list(names))
    values = self.ask(self.blob) if set(wanted) <= {name for name, _ in self.outputs} else None
    if values is not None:
      return values[wanted[0]] if names is None or isinstance(names, str) else tuple(values[name] for name in wanted)
    if self.pending is not None:
      args, kwargs = self.pending
      self.pending = None
      self.net.setInput(self.blob.scaled(), *args, **kwargs)
    return self.net.forward() if names is None else self.net.forward(names)

  def __getattr__(self, name):
    return getattr(self.net, name)

  def __setattr__(self, name, value):
    # a stand-in both ways: anything not the wrapper's own lands on the net it wraps
    if name in self._OWN:
      object.__setattr__(self, name, value)
    else:
      setattr(self.net, name, value)


_shared: dict[str, AddonLink] = {}
_shared_lock = threading.Lock()


def shared_link(function, **kwargs) -> AddonLink:
  """This process's one link on the add-on's interface: two nets on one
  interface share its endpoints. kwargs (AddonLink's) apply when it is made."""
  with _shared_lock:
    link = _shared.get(function.name)
    if link is None:
      link = _shared[function.name] = AddonLink(function, **kwargs)
    return link


def with_host(net, function, sha256: str, outputs, *, scaled_u8: bool = False, report=None, enabled=None):
  """net, asking the host over the add-on's interface first; net itself if
  the wrapper cannot be made. Whether the gadget carries the interface yet is
  the link's to find out on each call, so a net loaded before the owner built
  the gadget still moves once it has."""
  try:
    return JetsonNet(net, shared_link(function, report=report), sha256, outputs, scaled_u8=scaled_u8, enabled=enabled)
  except Exception:
    logging.getLogger(__name__).exception("jetlink: %s runs on the comma alone", getattr(function, 'name', 'add-on'))
    return net


def host_model(local, function, sha256: str, outputs, *, scaled_u8: bool = False, report=None, enabled=None):
  """local, any runtime's model as a call, asking the host over the add-on's
  interface first (HostModel); local itself if the wrapper cannot be made."""
  try:
    return HostModel(shared_link(function, report=report), sha256, outputs, scaled_u8=scaled_u8, enabled=enabled, local=local)
  except Exception:
    logging.getLogger(__name__).exception("jetlink: %s runs on the comma alone", getattr(function, 'name', 'add-on'))
    return local
