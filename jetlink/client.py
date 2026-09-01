"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The comma side of the link.

Deliberately knows nothing about openpilot: it takes the warped frame and the
packed scalars, and returns the model output. The openpilot glue lives in the
fork (sunnypilot/jetlink/), so this package stays importable by any fork.

Failure policy: every error is raised as LinkError/LinkTimeout. openpilot's
modeld already wraps the model call in try/except and falls back to the small
model, so a link that raises inherits that path for free.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from jetlink import protocol as P
from jetlink.spec import CHUNK, ModelSpec, sha256_file, spec_from_onnx
from jetlink.transport.base import LinkError, LinkTimeout, Message, Transport


def _name(enum_cls, value) -> str:
  """Enum name for a value off the wire, which may be anything."""
  try:
    return enum_cls(value).name
  except ValueError:
    return f'unknown({value})'

log = logging.getLogger('jetlink.client')

ProgressFn = Callable[[str, float, str], None]

# The model must be back well inside modeld's 50 ms frame. 19.5 ms of compute
# plus transport leaves plenty of room; anything past this is a stall, and a
# stall is worse than the small model.
DEFAULT_DEADLINE = 0.035


class JetlinkClient:
  def __init__(self, transport: Transport, deadline: float = DEFAULT_DEADLINE):
    self.t = transport
    self.deadline = deadline
    self.seq = 0
    self.spec: ModelSpec | None = None
    self.progress_cb: ProgressFn | None = None
    self._engine_state: dict | None = None
    self.dead = False
    self.last_timings = (0, 0, 0)  # gpu_us, queue_us, total_us as measured server-side
    self.last_state: dict | None = None  # most recent piggybacked telemetry

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

  @staticmethod
  def usb_present() -> bool:
    from jetlink.transport.usbbulk import UsbBulkTransport
    return UsbBulkTransport.present()

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
      # Accepting that as our readiness would leave us inferring against a slot
      # the server will answer NOT_READY for, every frame.
      if self.spec is not None and state.get('sha256') not in (None, self.spec.sha256):
        log.warning("ignoring engine state for %s (we want %s)",
                    str(state.get('sha256'))[:16], self.spec.sha256[:16])
      else:
        self._engine_state = state
    elif msg.msg_type == P.Msg.ERROR:
      e = json.loads(bytes(msg.payload))
      raise LinkError(f"server error: {e.get('error')}: {e.get('detail')}")

  def _expect(self, msg_type: int, seq: int, timeout: float | None) -> Message:
    """Wait for one specific reply, servicing anything unsolicited on the way."""
    end = None if timeout is None else time.monotonic() + timeout
    while True:
      remaining = None if end is None else max(0.001, end - time.monotonic())
      msg = self.t.recv(timeout=remaining)
      if msg.msg_type == msg_type and msg.seq == seq:
        return msg
      if msg.msg_type in (P.Msg.PROGRESS, P.Msg.ENGINE_RESP, P.Msg.ERROR):
        self._dispatch(msg)
        continue
      # A late reply to an earlier request. Drop it and keep looking, otherwise
      # every subsequent frame would read one response behind.
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

  # -- model provisioning ---------------------------------------------------

  def ensure_engine(self, onnx_path: str | Path, spec: ModelSpec | None = None,
                    progress: ProgressFn | None = None,
                    build_timeout: float = 900.0) -> ModelSpec:
    """Make the server ready to run this model, uploading and building if needed.

    Blocks until the engine is ready or the build fails. Progress is reported
    through `progress(stage, frac, msg)` with stage in
    upload/patch/parse/build/load, so the caller can surface it the same way a
    model download is surfaced.
    """
    onnx_path = Path(onnx_path)
    self.progress_cb = progress
    if spec is None:
      spec = spec_from_onnx(str(onnx_path))
    self.spec = spec

    seq = self._next_seq()
    self.t.send_json(P.Msg.ENGINE_REQ, seq, {
      'sha256': spec.sha256,
      'nbytes': spec.nbytes,
      'frame_skip': spec.frame_skip,
      'checkpoint': spec.checkpoint,
      'input_shapes': {k: list(v) for k, v in spec.input_shapes.items()},
      'output_shapes': {k: list(v) for k, v in spec.output_shapes.items()},
      'output_slices': {k: [v.start, v.stop] for k, v in spec.output_slices.items()},
    })
    resp = json.loads(bytes(self._expect(P.Msg.ENGINE_RESP, seq, 60.0).payload))
    self._engine_state = resp
    log.info("server engine state: %s (%s)", resp['state'], resp.get('detail', ''))

    if resp['state'] == 'need_upload':
      self._upload(onnx_path, spec, int(resp.get('chunk') or CHUNK))

    self._await_ready(build_timeout)
    return spec

  def _upload(self, onnx_path: Path, spec: ModelSpec, chunk: int) -> None:
    log.info("uploading %s (%d MB)", onnx_path.name, spec.nbytes >> 20)
    t0 = time.time()
    sent = 0
    with open(onnx_path, 'rb') as f:
      while data := f.read(chunk):
        self.t.send(P.Msg.UPLOAD_CHUNK, self._next_seq(),
                    (sent.to_bytes(8, 'little'), data))
        sent += len(data)
        if self.progress_cb:
          self.progress_cb('upload', sent / spec.nbytes, f'{sent >> 20}/{spec.nbytes >> 20} MB')
    seq = self._next_seq()
    self.t.send_json(P.Msg.UPLOAD_DONE, seq, {'sha256': spec.sha256})
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
        return
      if st == 'failed':
        raise LinkError(f"engine build failed: {self._engine_state.get('detail')}")
      if time.monotonic() > end:
        raise LinkTimeout(f"engine not ready after {timeout:.0f}s (state={st})")
      try:
        self._dispatch(self.t.recv(timeout=min(5.0, max(0.1, end - time.monotonic()))))
      except LinkTimeout:
        continue

  # -- inference ------------------------------------------------------------

  def infer_begin(self, warped: np.ndarray, packed: np.ndarray, frame_id: int = 0,
                  reset: bool = False, want_state: bool = False) -> int:
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
      self.t.send(P.Msg.INFER_REQ, seq, (P.pack_infer_req(frame_id, flags), warped, packed))
    except LinkTimeout:
      raise           # the stream is still in sync; the next frame recovers
    except LinkError:
      self.dead = True
      raise
    return seq

  def infer_end(self, seq: int, deadline: float | None = None) -> np.ndarray:
    try:
      msg = self._expect(P.Msg.INFER_RESP, seq, self.deadline if deadline is None else deadline)
    except LinkTimeout:
      # Do NOT latch `dead` here. One frame overrunning a 35 ms deadline is the
      # single most likely thing to happen on a drive, and the buffer keeps the
      # stream in sync: the late reply is discarded as stale by the next
      # _expect. Latching would drop the car to the small model permanently on
      # the first GC pause.
      raise
    except LinkError:
      self.dead = True
      raise
    fid, status, gpu_us, queue_us, total_us = P.unpack_infer_resp(msg.payload)
    self.last_timings = (gpu_us, queue_us, total_us)
    if status != P.Status.OK:
      raise LinkError(f"inference failed: {_name(P.Status, status)} (frame {fid})")
    end = P.INFER_RESP_SIZE + self.spec.output_nbytes
    if msg.payload.nbytes > end:  # piggybacked telemetry
      try:
        self.last_state = json.loads(bytes(msg.payload[end:]))
      except ValueError:
        pass
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
    return self.infer_end(self.infer_begin(warped, packed, frame_id, reset, want_state), deadline)

  def reset(self, timeout: float = 1.0) -> None:
    seq = self._next_seq()
    self.t.send(P.Msg.RESET_REQ, seq)
    self._expect(P.Msg.RESET_RESP, seq, timeout)

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


def spec_for(onnx_path: str | Path, frame_skip: int | None = None) -> ModelSpec:
  """Convenience: hash and parse a model in one call."""
  from jetlink.spec import DEFAULT_FRAME_SKIP
  sha, nbytes = sha256_file(str(onnx_path))
  return spec_from_onnx(str(onnx_path), frame_skip or DEFAULT_FRAME_SKIP, sha, nbytes)
