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
import logging
import threading
import time

import numpy as np

from jetlink import protocol as P
from jetlink.server.builder import EngineCache, build_engine
from jetlink.server.engine import TrtEngine
from jetlink.server.telemetry import Telemetry
from jetlink.spec import CHUNK, ModelSpec, sha256_file
from jetlink.transport.base import LinkError, LinkTimeout, Message, Transport

log = logging.getLogger('jetlink.server')

PROGRESS_MIN_INTERVAL = 0.25  # s; the comma only needs a progress bar, not every step

# Process-wide, not per Session: main.py builds a fresh Session for every
# connection, so a per-Session guard would let a reconnect start a second
# concurrent TensorRT build. Two builders each asking for a 4 GB workspace will
# OOM an Orin, and both would move a plan onto the same path.
_BUILD_LOCK = threading.Lock()


class ModelSlot:
  """A model the server has been asked to run, and how far along it is."""

  def __init__(self, spec: ModelSpec):
    self.spec = spec
    self.state = 'unknown'     # unknown|need_upload|building|ready|failed
    self.detail = ''
    self.engine: TrtEngine | None = None
    self.queues = None
    self.host_inputs: dict = {}


class Session:
  def __init__(self, transport: Transport, cache: EngineCache, telemetry: Telemetry | None = None):
    self.t = transport
    self.cache = cache
    self.telemetry = telemetry or Telemetry()
    self.slot: ModelSlot | None = None
    self.send_lock = threading.Lock()
    self._last_progress = 0.0
    self.frames = 0
    # Primed here so the first health publish carries real values. Session
    # construction is not latency sensitive; on_infer is.
    self._telemetry_cache = json.dumps(self.telemetry.read()).encode()

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

  def close(self) -> None:
    """Release the GPU allocations this session owns.

    TrtEngine deliberately has no __del__ (its pinned buffers are aliased by
    numpy views), so without this a reconnect would leak an engine's device
    memory every time.
    """
    slot, self.slot = self.slot, None
    if slot is not None and slot.engine is not None:
      slot.engine.close()

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
    spec = ModelSpec.from_dict(json.loads(bytes(msg.payload)))
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
      slot.detail = f'have {have} of {spec.nbytes} bytes'
    self._respond_engine(msg.seq)

  def _respond_engine(self, seq: int) -> None:
    slot = self.slot
    assert slot is not None
    self._send_json(P.Msg.ENGINE_RESP, seq, {
      'state': slot.state,
      'detail': slot.detail,
      'sha256': slot.spec.sha256,
      'chunk': CHUNK,
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
    # Deliberately silent. The client streams chunks without reading between
    # them, and over USB a gadget only accepts data while its peer has a read
    # posted - so a progress write here stalls for the whole transfer timeout
    # on every attempt. The client reports its own upload progress anyway.

  def on_upload_done(self, msg: Message) -> None:
    slot = self.slot
    if slot is None:
      return self._error(msg.seq, 'no_model', 'send ENGINE_REQ first')
    path = self.cache.model_path(slot.spec.sha256)
    if sha256_file(str(path))[0] != slot.spec.sha256:
      path.unlink(missing_ok=True)
      slot.state = 'failed'
      slot.detail = 'sha256 mismatch after upload'
      self._respond_engine(msg.seq)
      return
    self._progress('upload', 1.0, 'verified', force=True)
    self._start_worker(slot, load_only=False)
    self._respond_engine(msg.seq)

  def _start_worker(self, slot: ModelSlot, load_only: bool) -> None:
    if not _BUILD_LOCK.acquire(blocking=False):
      # Say so rather than silently reporting this slot as building: the client
      # would otherwise wait for an engine nobody is making.
      slot.state = 'building'
      slot.detail = 'another build is already in progress'
      return
    slot.state = 'building'
    slot.detail = 'loading engine' if load_only else 'building engine'
    threading.Thread(target=self._build_and_load, args=(slot, load_only),
                     daemon=True, name='jetlink-build').start()

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
      if engine.capture_graph():
        engine.run()   # first replay, so the steady state is never the first
        log.info("cuda graph captured")
      else:
        log.info("cuda graph unavailable, enqueueing per frame")
      queues.reset()

      previous = slot.engine
      slot.engine, slot.queues, slot.host_inputs = engine, queues, host_inputs
      if previous is not None and previous is not engine:
        previous.close()   # only once nothing points at its pinned buffers
      slot.state, slot.detail = 'ready', ''
      self._progress('load', 1.0, 'ready', force=True)
      log.info("engine ready: %s", entry.plan_path)
    except Exception as e:
      log.exception("engine preparation failed")
      slot.state, slot.detail = 'failed', f'{type(e).__name__}: {e}'
      self._progress('failed', 1.0, slot.detail, force=True)
    finally:
      _BUILD_LOCK.release()
      if slot is not self.slot:
        # The client moved to a different model while this was building. The
        # result is still cached on disk; just do not announce it as current.
        log.info("build finished for a superseded model %s", slot.spec.sha256[:16])
        return
      try:
        self._respond_engine(0)
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
    if msg.payload.nbytes != spec.infer_req_nbytes:
      # The offsets below come from our spec, not from the wire. A client with
      # a different model would otherwise have its scalars read out of the
      # middle of the image, and the result would look perfectly finite.
      self._send(P.Msg.INFER_RESP, msg.seq,
                 (P.pack_infer_resp(0, P.Status.BAD_SHAPE, 0, 0, 0),))
      return
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

    # asarray, not astype: a no-op when the engine already outputs float32,
    # instead of a 74 KB copy and an allocation every frame. Check finiteness on
    # the result - isfinite is ~7x faster on float32 than on float16, and the
    # non-finites map across the cast exactly.
    #
    # openpilot treats a non-finite big-model output as a hard failure and drops
    # to the small model. Checking here saves the comma rescanning 18452 floats
    # and keeps the reason in the response.
    out32 = np.asarray(out, dtype=np.float32)
    status = P.Status.OK if np.all(np.isfinite(out32)) else P.Status.NOT_FINITE

    total_us = int((time.perf_counter() - t0) * 1e6)
    parts = [P.pack_infer_resp(frame_id, status, slot.engine.last_gpu_us, queue_us, total_us),
             out32]
    if flags & P.Flag.WANT_STATE:
      parts.append(self._telemetry_cache)
    self._send(P.Msg.INFER_RESP, msg.seq, parts)
    self.frames += 1
    if flags & P.Flag.WANT_STATE:
      # ~30 sysfs reads. Refresh after replying, never between the GPU result
      # and the wire: health data must not cost a frame.
      self._telemetry_cache = json.dumps(self.telemetry.read()).encode()

  def on_state(self, msg: Message) -> None:
    slot = self.slot
    self._send_json(P.Msg.STATE_RESP, msg.seq, {
      **self.telemetry.read(),
      'engine_state': slot.state if slot else 'none',
      'detail': slot.detail if slot else '',
      'frames_served': self.frames,
    })
