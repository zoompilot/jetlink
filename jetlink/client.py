"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The comma side of the link.

Deliberately knows nothing about openpilot: it takes the warped frame and the
packed scalars, and returns the model output. The openpilot glue lives in the
fork (sunnypilot/jetlink/), so this package stays importable by any fork.

Failure policy: every error is raised as LinkError. openpilot's modeld already
wraps the model call in try/except and falls back to the small model, so a
link that raises inherits that path for free. A frame is treated the way a
chestnut frame is: the call blocks until the answer is there, and only a stall
long enough to mean the far end is gone (FRAME_TIMEOUT, chestnut's HCQ wait)
becomes a failure.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from jetlink import protocol as P
from jetlink.spec import CHUNK, DEFAULT_FRAME_SKIP, ModelSpec
from jetlink.transport.base import LinkError, LinkTimeout, Message, Transport


def _name(enum_cls, value) -> str:
  """Enum name for a value off the wire, which may be anything."""
  try:
    return enum_cls(value).name
  except ValueError:
    return f'unknown({value})'

log = logging.getLogger('jetlink.client')

ProgressFn = Callable[[str, float, str], None]

# How long one frame may take before the link is declared dead. Not a frame
# budget: a frame past 50 ms is a dropped camera frame and modeld already
# accounts for those, exactly as it does when a chestnut runs long. This is the
# analogue of chestnut's HCQDEV_WAIT_TIMEOUT_MS (3000): past it the far end is
# not slow, it is gone, and modeld falls back to the small model.
FRAME_TIMEOUT = 3.0

StopFn = Callable[[], bool]


class EngineMissing(LinkError):
  """The server has no engine for this model and we have nothing to upload.

  Raised in modeld, which never carries the ONNX: the Jetson's cache was
  pruned, re-flashed or swapped since jetlinkd recorded it as ready.
  """


class JetlinkClient:
  def __init__(self, transport: Transport, deadline: float = FRAME_TIMEOUT):
    self.t = transport
    self.deadline = deadline
    self.seq = 0
    self.spec: ModelSpec | None = None
    self.progress_cb: ProgressFn | None = None
    self._engine_state: dict | None = None
    self._should_stop: StopFn = lambda: False
    self.dead = False
    self.last_timings = (0, 0, 0)  # gpu_us, queue_us, total_us as measured server-side
    self.last_state: dict | None = None  # most recent piggybacked telemetry
    self._infer_started = 0.0
    self._infer_frame_id: int | None = None

  # -- construction ---------------------------------------------------------

  @classmethod
  def open_usb(cls, **kw) -> JetlinkClient:
    """This end is the USB host (libusb)."""
    from jetlink.transport.usbbulk import UsbBulkTransport
    return cls(UsbBulkTransport.open(), **kw)

  @classmethod
  def open_ffs(cls, mount: str = '/dev/ffs-jetlink', gadget: str | None = None,
               udc: str | None = None, **kw) -> JetlinkClient:
    """This end is the USB gadget (FunctionFS).

    Which end is which is decided by what the two kernels support, not by which
    is the "client". On a comma 3X the comma is the gadget: AGNOS has F_FS and
    libcomposite built in, while a Jetson host needs no driver at all because
    libusb goes through usbfs. See docs/transport.md.
    """
    from jetlink.transport.ffs import FfsTransport
    return cls(FfsTransport(mount, gadget=gadget, udc=udc), **kw)

  @classmethod
  def open_tcp(cls, host: str, port: int = 5599, **kw) -> JetlinkClient:
    from jetlink.transport.tcp import TcpTransport
    return cls(TcpTransport.connect(host, port), **kw)

  # -- plumbing -------------------------------------------------------------

  def _next_seq(self) -> int:
    self.seq = (self.seq + 1) & 0xFFFFFFFF
    return self.seq

  def _dispatch(self, msg: Message) -> None:
    """Handle the messages the server may send at any time."""
    if msg.msg_type == P.Msg.PROGRESS:
      p = json.loads(bytes(msg.payload))
      if self.progress_cb:
        self.progress_cb(p.get('stage', ''), float(p.get('frac', 0.0)), p.get('msg', ''))
    elif msg.msg_type == P.Msg.ENGINE_RESP:
      state = json.loads(bytes(msg.payload))
      # The server may finish a build for a model we have since moved off.
      # Accepting that as our readiness would leave us inferring against an
      # engine the server will answer NOT_READY for, every frame.
      wanted = (self._engine_state or {}).get('sha256')
      if wanted is not None and state.get('sha256') not in (None, wanted):
        log.warning("ignoring engine state for %s (we want %s)",
                    str(state.get('sha256'))[:16], wanted[:16])
      else:
        self._engine_state = state
    elif msg.msg_type == P.Msg.ERROR:
      e = json.loads(bytes(msg.payload))
      raise LinkError(f"server error: {e.get('error')}: {e.get('detail')}")

  def _expect(self, msg_type: int, seq: int, timeout: float | None) -> Message:
    """Wait for one specific reply, servicing anything unsolicited on the way."""
    end = None if timeout is None else time.monotonic() + timeout
    while True:
      remaining = None if end is None else end - time.monotonic()
      if remaining is not None and remaining <= 0:
        raise LinkTimeout(f'timed out waiting for message type={msg_type} seq={seq}')
      msg = self.t.recv(timeout=remaining)
      if msg.msg_type == msg_type and msg.seq == seq:
        return msg
      if msg.msg_type in (P.Msg.PROGRESS, P.Msg.ENGINE_RESP, P.Msg.ERROR):
        self._dispatch(msg)
        continue
      # A reply to an earlier request, which only happens after a caller gave up
      # on one. Drop it and keep looking, otherwise every subsequent frame would
      # read one response behind.
      log.warning("discarding stale %s seq=%d (waiting for %s seq=%d)",
                  _name(P.Msg, msg.msg_type), msg.seq, _name(P.Msg, msg_type), seq)

  # -- handshake ------------------------------------------------------------

  def hello(self, timeout: float = 5.0) -> dict:
    seq = self._next_seq()
    self.t.send_json(P.Msg.HELLO_REQ, seq, {})
    return json.loads(bytes(self._expect(P.Msg.HELLO_RESP, seq, timeout).payload))

  def state(self, timeout: float = 2.0) -> dict:
    seq = self._next_seq()
    self.t.send_json(P.Msg.STATE_REQ, seq, {})
    return json.loads(bytes(self._expect(P.Msg.STATE_RESP, seq, timeout).payload))

  def ping(self, timeout: float = 1.0) -> float:
    seq = self._next_seq()
    t0 = time.perf_counter()
    self.t.send(P.Msg.PING, seq)
    self._expect(P.Msg.PONG, seq, timeout)
    return time.perf_counter() - t0

  def shutdown(self, reason: str = '', timeout: float = 5.0) -> dict:
    """Ask the Jetson to power off. Off, not asleep: only a DC cycle or the
    power button brings it back, so this is for the comma's own low-battery
    shutdown and nothing else. The reply comes before the box goes down."""
    seq = self._next_seq()
    self.t.send_json(P.Msg.SHUTDOWN_REQ, seq, {'reason': reason})
    return json.loads(bytes(self._expect(P.Msg.SHUTDOWN_RESP, seq, timeout).payload))

  # -- model provisioning ---------------------------------------------------

  def ensure_engine(self, sha256: str, nbytes: int, onnx_path: str | Path | None = None,
                    frame_skip: int = DEFAULT_FRAME_SKIP,
                    progress: ProgressFn | None = None,
                    build_timeout: float = 900.0,
                    should_stop: StopFn | None = None) -> ModelSpec:
    """Make the server ready to run this model, uploading and building if needed.

    Blocks until the engine is ready or the build fails, and returns the spec
    the server derived from the ONNX. The comma never parses the model: the
    server has the onnx package and the file, so it is the one source of truth
    for shapes, and a device without a parser for a given export (tinygrad
    rejects the org.tinygrad domain) is no longer stuck.

    `onnx_path` is what gets uploaded if the server asks; None means the
    caller cannot upload (modeld) and a missing engine is EngineMissing.
    Progress is reported through `progress(stage, frac, msg)` with stage in
    upload/patch/parse/build/load. `should_stop` is polled during the long
    waits so a daemon told to exit can let go of the link promptly.
    """
    self.progress_cb = progress
    self._should_stop = should_stop or (lambda: False)
    self.spec = None
    self._engine_state = None

    seq = self._next_seq()
    self.t.send_json(P.Msg.ENGINE_REQ, seq, {'sha256': sha256, 'nbytes': nbytes, 'frame_skip': frame_skip})
    resp = json.loads(bytes(self._expect(P.Msg.ENGINE_RESP, seq, 60.0).payload))
    self._engine_state = resp
    log.info("server engine state: %s (%s)", resp['state'], resp.get('detail', ''))

    if resp['state'] == 'need_upload':
      if onnx_path is None:
        raise EngineMissing(f"server has no engine for {sha256[:16]} ({resp.get('detail', '')})")
      self._upload(Path(onnx_path), nbytes, int(resp.get('chunk') or CHUNK))

    self._await_ready(build_timeout)
    spec = ModelSpec.from_dict(self._engine_state['spec'])
    if spec.sha256 != sha256:
      raise LinkError(f"server answered for {spec.sha256[:16]}, we asked for {sha256[:16]}")
    self.spec = spec
    return spec

  def _stopped(self) -> None:
    if self._should_stop():
      raise LinkError("stopped while waiting for the engine")

  def _upload(self, onnx_path: Path, nbytes: int, chunk: int) -> None:
    log.info("uploading %s (%d MB)", onnx_path.name, nbytes >> 20)
    t0 = time.time()
    sent = 0
    with open(onnx_path, 'rb') as f:
      while data := f.read(chunk):
        self._stopped()
        self.t.send(P.Msg.UPLOAD_CHUNK, self._next_seq(),
                    (sent.to_bytes(8, 'little'), data))
        sent += len(data)
        if self.progress_cb:
          self.progress_cb('upload', sent / nbytes, f'{sent >> 20}/{nbytes >> 20} MB')
    seq = self._next_seq()
    self.t.send_json(P.Msg.UPLOAD_DONE, seq, {})
    resp = json.loads(bytes(self._expect(P.Msg.ENGINE_RESP, seq, 300.0).payload))
    self._engine_state = resp
    rate = sent / max(1e-6, time.time() - t0) / 1e6
    log.info("uploaded %d MB at %.1f MB/s -> %s", sent >> 20, rate, resp['state'])
    if resp['state'] == 'failed':
      raise LinkError(f"upload rejected: {resp.get('detail')}")

  def _await_ready(self, timeout: float) -> None:
    end = time.monotonic() + timeout
    while True:
      st = (self._engine_state or {}).get('state')
      if st == 'ready':
        if 'spec' not in (self._engine_state or {}):
          raise LinkError("server reported ready without a model spec")
        return
      if st == 'failed':
        raise LinkError(f"engine build failed: {self._engine_state.get('detail')}")
      if time.monotonic() > end:
        raise LinkTimeout(f"engine not ready after {timeout:.0f}s (state={st})")
      self._stopped()
      try:
        self._dispatch(self.t.recv(timeout=min(1.0, max(0.1, end - time.monotonic()))))
      except LinkTimeout:
        continue

  # -- inference ------------------------------------------------------------

  def infer_begin(self, warped: np.ndarray, packed: np.ndarray, frame_id: int = 0,
                  reset: bool = False, want_state: bool = False, deadline: float | None = None) -> int:
    """Send a frame and return immediately with its sequence number.

    Split from infer_end so the caller can do useful work while the Jetson is
    busy - openpilot publishes chestnutState in exactly this window.
    """
    if self.spec is None:
      raise LinkError("ensure_engine() first")
    if self.dead:
      raise LinkError("link previously failed")
    warped = _as_bytes(warped, self.spec.warped_nbytes, 'warped')
    packed = _as_bytes(packed, self.spec.packed_nbytes, 'packed')
    seq = self._next_seq()
    flags = (P.Flag.RESET_QUEUES if reset else 0) | (P.Flag.WANT_STATE if want_state else 0)
    try:
      self._infer_started = time.monotonic()
      self._infer_frame_id = frame_id
      self.t.send(P.Msg.INFER_REQ, seq, (P.pack_infer_req(frame_id, flags), warped, packed),
                  timeout=self.deadline if deadline is None else deadline)
    except LinkError:
      self.dead = True
      raise
    return seq

  def infer_end(self, seq: int, deadline: float | None = None) -> np.ndarray:
    """Block for the frame's output, as modeld blocks on a chestnut.

    A frame that runs long is a dropped camera frame, which modeld counts and
    tolerates. Only a stall past `deadline` (FRAME_TIMEOUT by default) is a
    failure, and then the link is done: the far end has stopped answering, and
    the small model is the right place to be.
    """
    try:
      budget = self.deadline if deadline is None else deadline
      remaining = budget - (time.monotonic() - self._infer_started)
      if remaining <= 0:
        raise LinkTimeout('frame deadline elapsed during send')
      msg = self._expect(P.Msg.INFER_RESP, seq, remaining)
    except LinkTimeout as e:
      self.dead = True
      raise LinkError(f"no answer for frame in {self.deadline:.1f}s; link abandoned") from e
    except LinkError:
      self.dead = True
      raise
    if msg.payload.nbytes < P.INFER_RESP_SIZE:
      self.dead = True
      raise LinkError('inference response is missing its header')
    fid, status, gpu_us, queue_us, total_us = P.unpack_infer_resp(msg.payload)
    self.last_timings = (gpu_us, queue_us, total_us)
    if status != P.Status.OK:
      self.dead = True
      raise LinkError(f"inference failed: {_name(P.Status, status)} (frame {fid})")
    if fid != self._infer_frame_id:
      self.dead = True
      raise LinkError(f'inference response frame {fid}, expected {self._infer_frame_id}')
    end = P.INFER_RESP_SIZE + self.spec.output_nbytes
    if msg.payload.nbytes < end:
      self.dead = True
      raise LinkError('inference response is missing model outputs')
    if msg.payload.nbytes > end:  # piggybacked telemetry
      try:
        self.last_state = json.loads(bytes(msg.payload[end:]))
      except ValueError:
        pass
    receive = getattr(self.t, 'last_receive', None)
    if receive is not None and time.monotonic() - self._infer_started > 0.05:
      log.warning('frame %d receive maxima: prepare %.1f read_wait %.1f handoff %.1f ms; '
                  'server gpu %.1f queue %.1f total %.1f ms', fid,
                  receive['prepare'] * 1e3, receive['read_wait'] * 1e3, receive['handoff'] * 1e3,
                  gpu_us / 1e3, queue_us / 1e3, total_us / 1e3)
    # copy: the payload is a view into the transport's reusable receive buffer.
    return np.frombuffer(msg.payload, np.float32, self.spec.output_nelem, P.INFER_RESP_SIZE).copy()

  def infer(self, warped: np.ndarray, packed: np.ndarray, frame_id: int = 0,
            reset: bool = False, deadline: float | None = None,
            want_state: bool = False) -> np.ndarray:
    """One frame. Returns the model output as float32, shaped (n,).

    `warped` is (2, 6, H, W) uint8 straight off openpilot's warp; `packed` is
    the float32 packed_npy_inputs buffer. Either may be a numpy array or a raw
    buffer - a tinygrad `Tensor.data()` memoryview goes straight to the wire
    with no numpy round trip. Sent as-is: no copy on the TCP path, one on USB.
    """
    return self.infer_end(self.infer_begin(warped, packed, frame_id, reset, want_state, deadline), deadline)

  def close(self) -> None:
    self.t.close()


def _as_bytes(buf, expect: int, name: str) -> memoryview:
  """Byte view over a numpy array or any buffer, size-checked.

  Checking bytes rather than shape lets the caller hand over whatever it
  already has, and turns a model/protocol mismatch into a clear error here instead
  of a misparse on the far end.
  """
  mv = memoryview(buf)
  if not mv.contiguous:
    raise LinkError(f"{name} must be contiguous")
  mv = mv.cast('B')
  if mv.nbytes != expect:
    raise LinkError(f"{name} is {mv.nbytes} bytes, expected {expect}")
  return mv
