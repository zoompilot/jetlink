"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The request loop.

One client at a time, one message at a time, with the engine build on a worker
thread so progress keeps flowing while a 160 s build runs. The inference path
does no allocation and no logging: numpy views are laid over the received
buffer in place.
"""
from __future__ import annotations

import json
import hashlib
import logging
import threading
import time

import numpy as np

from jetlink import protocol as P
from jetlink.server.builder import EngineCache, build_engine
from jetlink.server.engine import TrtEngine
from jetlink.server.telemetry import Telemetry
from jetlink.spec import ModelSpec
from jetlink.transport.base import LinkError, LinkTimeout, Message, Transport

log = logging.getLogger('jetlink.server')

PROGRESS_MIN_INTERVAL = 0.25  # s; the comma only needs a progress bar, not every step


class ModelSlot:
  """A model the server has been asked to run, and how far along it is."""

  def __init__(self, spec: ModelSpec):
    self.spec = spec
    self.state = 'unknown'     # unknown|need_upload|building|ready|failed
    self.detail = ''
    self.engine: TrtEngine | None = None
    self.queues = None
    self.host_inputs: dict = {}
    self.received = 0


class Session:
  def __init__(self, transport: Transport, cache: EngineCache, telemetry: Telemetry | None = None):
    self.t = transport
    self.cache = cache
    self.telemetry = telemetry or Telemetry()
    self.slot: ModelSlot | None = None
    self.send_lock = threading.Lock()
    self.build_thread: threading.Thread | None = None
    self._last_progress = 0.0
    self.frames = 0

  # -- plumbing -------------------------------------------------------------

  def _send(self, msg_type: int, seq: int, parts=(), flags: int = 0) -> None:
    with self.send_lock:
      self.t.send(msg_type, seq, parts, flags)

  def _send_json(self, msg_type: int, seq: int, obj, flags: int = 0) -> None:
    self._send(msg_type, seq, (json.dumps(obj).encode(),), flags)

  def _error(self, seq: int, error: str, detail: str = '') -> None:
    log.error("%s: %s", error, detail)
    self._send_json(P.Msg.ERROR, seq, {'error': error, 'detail': detail})

  def _progress(self, stage: str, frac: float, msg: str = '', force: bool = False) -> None:
    now = time.monotonic()
    if not force and frac < 1.0 and now - self._last_progress < PROGRESS_MIN_INTERVAL:
      return
    self._last_progress = now
    try:
      self._send_json(P.Msg.PROGRESS, 0, {'stage': stage, 'frac': round(frac, 4), 'msg': msg})
    except LinkError:
      pass  # the comma may have given up and fallen back; the build continues

  # -- handlers -------------------------------------------------------------

  def serve_forever(self) -> None:
    while True:
      try:
        msg = self.t.recv()
      except LinkTimeout:
        continue
      except LinkError as e:
        log.info("link closed: %s", e)
        return
      try:
        self.handle(msg)
      except LinkError:
        raise
      except Exception as e:  # a bad request must not take the server down
        log.exception("handler failed")
        self._error(msg.seq, type(e).__name__, str(e))

  def handle(self, msg: Message) -> None:
    mt = msg.msg_type
    if mt == P.Msg.INFER_REQ:
      self.on_infer(msg)
    elif mt == P.Msg.PING:
      self._send(P.Msg.PONG, msg.seq)
    elif mt == P.Msg.HELLO_REQ:
      self.on_hello(msg)
    elif mt == P.Msg.ENGINE_REQ:
      self.on_engine_req(msg)
    elif mt == P.Msg.UPLOAD_CHUNK:
      self.on_upload_chunk(msg)
    elif mt == P.Msg.UPLOAD_DONE:
      self.on_upload_done(msg)
    elif mt == P.Msg.RESET_REQ:
      self.on_reset(msg)
    elif mt == P.Msg.STATE_REQ:
      self.on_state(msg)
    else:
      self._error(msg.seq, 'unknown_message', f'type {mt}')

  def on_hello(self, msg: Message) -> None:
    import tensorrt as trt
    from jetlink.server.builder import device_tag
    self._send_json(P.Msg.HELLO_RESP, msg.seq, {
      'protocol': P.VERSION,
      'trt_version': trt.__version__,
      'device': device_tag(),
      'engine_state': self.slot.state if self.slot else 'none',
      'frames_served': self.frames,
      'telemetry': self.telemetry.read(),
    })

  def on_engine_req(self, msg: Message) -> None:
    req = json.loads(bytes(msg.payload))
    spec = ModelSpec(
      sha256=req['sha256'], nbytes=req['nbytes'], frame_skip=req['frame_skip'],
      input_shapes={k: tuple(v) for k, v in req['input_shapes'].items()},
      output_shapes={k: tuple(v) for k, v in req['output_shapes'].items()},
      output_slices={k: slice(*v) for k, v in req['output_slices'].items()},
      checkpoint=req.get('checkpoint'),
    )
    slot = ModelSlot(spec)
    self.slot = slot

    entry = self.cache.entry(spec.sha256)
    if entry.exists:
      self._start_worker(slot, load_only=True)
      self._respond_engine(msg.seq)
      return

    model_path = self.cache.model_path(spec.sha256)
    have = model_path.stat().st_size if model_path.exists() else 0
    if have == spec.nbytes:
      self._start_worker(slot, load_only=False)
    else:
      slot.state = 'need_upload'
      slot.received = 0  # a partial file is not resumable without a chunk digest
      slot.detail = f'have {have} of {spec.nbytes} bytes'
    self._respond_engine(msg.seq)

  def _respond_engine(self, seq: int) -> None:
    slot = self.slot
    assert slot is not None
    self._send_json(P.Msg.ENGINE_RESP, seq, {
      'state': slot.state,
      'detail': slot.detail,
      'offset': slot.received,
      'sha256': slot.spec.sha256,
      'chunk': 4 << 20,
    })

  def on_upload_chunk(self, msg: Message) -> None:
    slot = self.slot
    if slot is None:
      return self._error(msg.seq, 'no_model', 'send ENGINE_REQ first')
    offset = int.from_bytes(bytes(msg.payload[:8]), 'little')
    data = msg.payload[8:]
    path = self.cache.model_path(slot.spec.sha256)
    mode = 'r+b' if path.exists() and offset else 'wb'
    with open(path, mode) as f:
      f.seek(offset)
      f.write(data)
    slot.received = offset + data.nbytes
    frac = slot.received / max(1, slot.spec.nbytes)
    self._progress('upload', frac, f'{slot.received >> 20} / {slot.spec.nbytes >> 20} MB')

  def on_upload_done(self, msg: Message) -> None:
    slot = self.slot
    if slot is None:
      return self._error(msg.seq, 'no_model', 'send ENGINE_REQ first')
    path = self.cache.model_path(slot.spec.sha256)
    h = hashlib.sha256()
    with open(path, 'rb') as f:
      while chunk := f.read(1 << 20):
        h.update(chunk)
    if h.hexdigest() != slot.spec.sha256:
      path.unlink(missing_ok=True)
      slot.state = 'failed'
      slot.detail = 'sha256 mismatch after upload'
      self._respond_engine(msg.seq)
      return
    self._progress('upload', 1.0, 'verified', force=True)
    self._start_worker(slot, load_only=False)
    self._respond_engine(msg.seq)

  def _start_worker(self, slot: ModelSlot, load_only: bool) -> None:
    if self.build_thread is not None and self.build_thread.is_alive():
      return
    slot.state = 'building'
    slot.detail = 'loading engine' if load_only else 'building engine'
    self.build_thread = threading.Thread(
      target=self._build_and_load, args=(slot, load_only), daemon=True,
      name='jetlink-build')
    self.build_thread.start()

  def _build_and_load(self, slot: ModelSlot, load_only: bool) -> None:
    from jetlink.queues import PolicyQueues
    entry = self.cache.entry(slot.spec.sha256)
    try:
      if not load_only:
        build_engine(self.cache.model_path(slot.spec.sha256), entry.plan_path,
                     report=self._progress)
        self.cache.prune()
      self._progress('load', 0.0, 'deserializing engine', force=True)
      engine = TrtEngine(str(entry.plan_path))
      queues = PolicyQueues(slot.spec)

      self._check_shapes(engine, slot.spec)

      # One warm run so the first real frame is not the one that pays for
      # lazily-initialised CUDA state.
      host_inputs = {n: engine.host_input(n) for n in engine.inputs}
      warped = np.zeros(slot.spec.warped_shape, np.uint8)
      packed = np.zeros(slot.spec.packed_nelem, np.float32)
      queues.step_into(warped, packed, host_inputs)
      engine.run()
      queues.reset()

      slot.engine, slot.queues, slot.host_inputs = engine, queues, host_inputs
      slot.state, slot.detail = 'ready', ''
      self._progress('load', 1.0, 'ready', force=True)
      log.info("engine ready: %s", entry.plan_path)
    except Exception as e:
      log.exception("engine preparation failed")
      slot.state, slot.detail = 'failed', f'{type(e).__name__}: {e}'
      self._progress('failed', 1.0, slot.detail, force=True)
    finally:
      try:
        self._send_json(P.Msg.ENGINE_RESP, 0, {
          'state': slot.state, 'detail': slot.detail,
          'offset': slot.received, 'sha256': slot.spec.sha256, 'chunk': 4 << 20})
      except LinkError:
        pass

  @staticmethod
  def _check_shapes(engine: TrtEngine, spec: ModelSpec) -> None:
    """The engine is ground truth for what will execute; the spec came over the
    wire. Disagreement means the comma and Jetson are on different models."""
    for name, shape in engine.input_shapes.items():
      want = spec.input_shapes.get(name)
      if want is None:
        raise ValueError(f"engine input {name!r} is not in the model spec")
      if int(np.prod(shape)) != int(np.prod(want)):
        raise ValueError(f"input {name}: engine {shape} vs spec {want}")
    out = next(iter(engine.output_shapes.values()))
    if int(np.prod(out)) != spec.output_nelem:
      raise ValueError(f"output: engine {out} vs spec {spec.output_nelem}")

  # -- the hot path ---------------------------------------------------------

  def on_infer(self, msg: Message) -> None:
    slot = self.slot
    if slot is None or slot.state != 'ready' or slot.engine is None:
      self._send(P.Msg.INFER_RESP, msg.seq,
                 (P.pack_infer_resp(0, P.Status.NOT_READY, 0, 0, 0),))
      return

    t0 = time.perf_counter()
    spec = slot.spec
    frame_id, flags = P.unpack_infer_req(msg.payload)
    if flags & P.Flag.RESET_QUEUES:
      slot.queues.reset()

    off = P.INFER_REQ_SIZE
    warped = np.frombuffer(msg.payload, np.uint8, spec.warped_nbytes, off).reshape(spec.warped_shape)
    off += spec.warped_nbytes
    packed = np.frombuffer(msg.payload, np.float32, spec.packed_nelem, off)

    # Gathers land straight in TensorRT's pinned input buffers: between the
    # wire and the GPU there is exactly one copy.
    slot.queues.step_into(warped, packed, slot.host_inputs)
    queue_us = int((time.perf_counter() - t0) * 1e6)

    outputs = slot.engine.run()
    out = next(iter(outputs.values())).reshape(-1)

    # openpilot treats a non-finite big-model output as a hard failure and drops
    # to the small model. Check here so the comma does not have to rescan 18452
    # floats, and so the reason survives in the response.
    status = P.Status.OK if np.all(np.isfinite(out)) else P.Status.NOT_FINITE
    out32 = out.astype(np.float32)

    total_us = int((time.perf_counter() - t0) * 1e6)
    parts = [P.pack_infer_resp(frame_id, status, slot.engine.last_gpu_us, queue_us, total_us),
             out32]
    if flags & P.Flag.WANT_STATE:
      parts.append(json.dumps(self.telemetry.read()).encode())
    self._send(P.Msg.INFER_RESP, msg.seq, parts)
    self.frames += 1

  def on_reset(self, msg: Message) -> None:
    if self.slot is not None and self.slot.queues is not None:
      self.slot.queues.reset()
    self._send(P.Msg.RESET_RESP, msg.seq)

  def on_state(self, msg: Message) -> None:
    slot = self.slot
    self._send_json(P.Msg.STATE_RESP, msg.seq, {
      **self.telemetry.read(),
      'engine_state': slot.state if slot else 'none',
      'detail': slot.detail if slot else '',
      'frames_served': self.frames,
    })
