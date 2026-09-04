"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The request loop, and the engine it serves.

One client at a time, one message at a time, with the engine build on a worker
thread so progress keeps flowing while a 160 s build runs. The inference path
does no allocation and no logging: numpy views are laid over the received
buffer in place.

The engine outlives the connection. The comma reconnects at every handover
between jetlinkd and modeld, and at the top of every drive that reconnect has
to fit inside modeld's 60 s budget together with a USB re-enumeration that has
been seen take 70 s. Reloading a 770 MB plan on each connect (13 to 25 s
measured) is what that budget cannot afford, so EngineHost owns the one loaded
engine and the one build in flight for the life of the process, and a Session
is only a view onto it.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jetlink import protocol as P
from jetlink.server.builder import CacheEntry, EngineCache, build_engine
from jetlink.server.telemetry import Telemetry
from jetlink.spec import CHUNK, DEFAULT_FRAME_SKIP, ModelSpec, sha256_file, spec_from_onnx
from jetlink.transport.base import LinkError, LinkTimeout, Message, Transport

log = logging.getLogger('jetlink.server')

PROGRESS_MIN_INTERVAL = 0.25  # s; the comma only needs a progress bar, not every step


@dataclass
class Loaded:
  """An engine resident on the GPU, with the state that goes with it."""
  sha256: str
  spec: ModelSpec
  engine: object
  queues: object
  host_inputs: dict


@dataclass
class Job:
  """One build or load in flight, or its outcome."""
  sha256: str
  load_only: bool
  state: str = 'building'   # building|ready|failed
  detail: str = ''


@dataclass(frozen=True)
class Request:
  """What a client asked for: enough to identify the model without the file."""
  sha256: str
  nbytes: int
  frame_skip: int


class EngineHost:
  """Process-wide owner of the loaded engine and of the build in flight.

  Sessions come and go; this does not. Everything that touches `loaded` does
  so under `lock`: the job thread swaps engines in and out while the request
  loop is running frames, and freeing pinned memory a frame's numpy views are
  still laid over is a segfault, not an exception.
  """

  def __init__(self, cache: EngineCache, telemetry: Telemetry | None = None):
    self.cache = cache
    self.telemetry = telemetry or Telemetry()
    self.lock = threading.Lock()
    self.loaded: Loaded | None = None
    self.job: Job | None = None
    self.session: Session | None = None   # who hears about progress and completion
    self._last_progress = 0.0

  # -- what a client sees ---------------------------------------------------

  def status(self, sha256: str | None) -> dict:
    """The engine state for one model, in the shape ENGINE_RESP carries."""
    with self.lock:
      loaded, job = self.loaded, self.job
    if sha256 is None:
      return {'state': 'none', 'detail': '', 'sha256': None, 'chunk': CHUNK}
    if loaded is not None and loaded.sha256 == sha256:
      return {'state': 'ready', 'detail': '', 'sha256': sha256, 'chunk': CHUNK,
              'spec': loaded.spec.to_dict()}
    if job is not None and job.sha256 == sha256 and job.state != 'ready':
      return {'state': job.state, 'detail': job.detail, 'sha256': sha256, 'chunk': CHUNK}
    if job is not None and job.state == 'building':
      return {'state': 'building', 'sha256': sha256, 'chunk': CHUNK,
              'detail': f'another build is in progress ({job.sha256[:16]})'}
    have = self._model_bytes(sha256)
    return {'state': 'need_upload', 'sha256': sha256, 'chunk': CHUNK,
            'detail': f'have {have} of the model'}

  def loaded_sha(self) -> str | None:
    with self.lock:
      return self.loaded.sha256 if self.loaded else None

  # -- requests -------------------------------------------------------------

  def request(self, req: Request, session: Session) -> dict:
    """Make `req` the model being served, starting whatever that takes."""
    with self.lock:
      self.session = session
      if self.loaded is not None and self.loaded.sha256 == req.sha256:
        return self._ready(self.loaded)
      if self.job is not None and self.job.state == 'building':
        # Either it is this model, and the client simply attaches to the build
        # that is already running, or another build owns the GPU right now.
        return self.status(req.sha256)
    entry = self.cache.entry(req.sha256)
    model_path = self.cache.model_path(req.sha256)
    spec = self._spec_on_disk(entry, model_path, req.frame_skip)
    if entry.exists and spec is not None:
      self._start(Job(req.sha256, load_only=True), req, entry, model_path, spec)
    elif model_path.is_file() and model_path.stat().st_size == req.nbytes:
      self._start(Job(req.sha256, load_only=False), req, entry, model_path, spec)
    return self.status(req.sha256)

  def _ready(self, loaded: Loaded) -> dict:
    return {'state': 'ready', 'detail': '', 'sha256': loaded.sha256, 'chunk': CHUNK,
            'spec': loaded.spec.to_dict()}

  def _model_bytes(self, sha256: str) -> int:
    path = self.cache.model_path(sha256)
    return path.stat().st_size if path.exists() else 0

  def _spec_on_disk(self, entry: CacheEntry, model_path: Path, frame_skip: int) -> ModelSpec | None:
    """The spec for a cached plan, from its sidecar or, failing that, the ONNX.

    Plans built before the sidecar carried a spec still load if the model file
    is around to derive one from; otherwise the client is asked to upload,
    after which the existing plan is reused rather than rebuilt.
    """
    if entry.exists:
      try:
        d = entry.meta().get('spec')
        if d:
          return ModelSpec.from_dict({**d, 'frame_skip': frame_skip})
      except (OSError, ValueError, KeyError):
        log.warning("unreadable sidecar for %s", entry.plan_path.name)
    if model_path.is_file():
      try:
        return self._derive_spec(model_path, frame_skip)
      except Exception:
        log.exception("could not derive a spec from %s", model_path.name)
    return None

  # -- the worker -----------------------------------------------------------

  def _start(self, job: Job, req: Request, entry: CacheEntry, model_path: Path,
             spec: ModelSpec | None) -> None:
    job.detail = 'loading engine' if job.load_only else 'building engine'
    with self.lock:
      self.job = job
    threading.Thread(target=self._run, args=(job, req, entry, model_path, spec),
                     daemon=True, name='jetlink-build').start()

  def _run(self, job: Job, req: Request, entry: CacheEntry, model_path: Path,
           spec: ModelSpec | None) -> None:
    engine = None
    try:
      # One engine resident at a time. A build needs the memory, and a load of
      # a different model would otherwise hold two 1.7 GB engines for a moment
      # on a board with 8 GB.
      self._unload()
      if not job.load_only:
        if spec is None:
          self._progress('parse', 0.0, 'reading model metadata', force=True)
          spec = self._derive_spec(model_path, req.frame_skip)
        self._build(model_path, entry.plan_path, {'spec': spec.to_dict()})
        # Never let the sweep take the plan we just wrote: offroad the clock
        # can be behind every plan already on disk.
        self.cache.prune(protect=entry.plan_path)
        self.cache.sweep_temp()
      assert spec is not None
      try:
        meta = entry.meta()
      except (OSError, ValueError):
        # A plan from before sidecars carried specs, or one whose sidecar went
        # missing. Rewrite it rather than failing a build that already ran.
        meta = {}
      if 'spec' not in meta:
        entry.write_meta({**meta, 'spec': spec.to_dict()})

      self._progress('load', 0.0, 'deserializing engine', force=True)
      engine = self._load_engine(entry.plan_path)
      loaded = self._warm(engine, spec)
      engine = None   # owned by `loaded` from here
      with self.lock:
        self.loaded = loaded
        job.state, job.detail = 'ready', ''
      self._progress('load', 1.0, 'ready', force=True)
      log.info("engine ready: %s", entry.plan_path)
    except Exception as e:
      log.exception("engine preparation failed")
      if engine is not None:
        engine.close()
      with self.lock:
        job.state, job.detail = 'failed', f'{type(e).__name__}: {e}'
      self._progress('failed', 1.0, job.detail, force=True)
    finally:
      session = self.session
      if session is not None:
        session.engine_update()

  def _warm(self, engine, spec: ModelSpec) -> Loaded:
    from jetlink.queues import PolicyQueues
    queues = PolicyQueues(spec)
    _check_shapes(engine, spec)
    # One warm run so the first real frame is not the one that pays for
    # lazily-initialised CUDA state.
    host_inputs = {n: engine.host_input(n) for n in engine.inputs}
    warped = np.zeros(spec.warped_shape, np.uint8)
    packed = np.zeros(spec.packed_nelem, np.float32)
    queues.step_into(warped, packed, host_inputs)
    engine.run()
    if engine.capture_graph():
      engine.run()   # first replay, so the steady state is never the first
      log.info("cuda graph captured")
    else:
      log.info("cuda graph unavailable, enqueueing per frame")
    queues.reset()
    return Loaded(spec.sha256, spec, engine, queues, host_inputs)

  def _unload(self) -> None:
    with self.lock:
      loaded, self.loaded = self.loaded, None
    if loaded is not None:
      loaded.engine.close()
      log.info("engine %s unloaded", loaded.sha256[:16])

  def close(self) -> None:
    self._unload()

  # Seams the tests replace: everything below touches TensorRT or a real ONNX.

  def _load_engine(self, plan_path: Path):
    from jetlink.server.engine import TrtEngine
    return TrtEngine(str(plan_path))

  def _derive_spec(self, model_path: Path, frame_skip: int) -> ModelSpec:
    return spec_from_onnx(str(model_path), frame_skip=frame_skip)

  def _build(self, model_path: Path, plan_path: Path, meta_extra: dict) -> None:
    build_engine(model_path, plan_path, report=self._progress, meta_extra=meta_extra)

  # -- talking back ---------------------------------------------------------

  def _progress(self, stage: str, frac: float, msg: str = '', force: bool = False) -> None:
    now = time.monotonic()
    if not force and frac < 1.0 and now - self._last_progress < PROGRESS_MIN_INTERVAL:
      return
    self._last_progress = now
    session = self.session
    if session is not None:
      session.progress(stage, frac, msg)


def _check_shapes(engine, spec: ModelSpec) -> None:
  """The engine is ground truth for what will execute; the spec came from the
  file. Disagreement means the plan on disk is not this model's."""
  for name, shape in engine.input_shapes.items():
    want = spec.input_shapes.get(name)
    if want is None:
      raise ValueError(f"engine input {name!r} is not in the model spec")
    if int(np.prod(shape)) != int(np.prod(want)):
      raise ValueError(f"input {name}: engine {shape} vs spec {want}")
  out = next(iter(engine.output_shapes.values()))
  if int(np.prod(out)) != spec.output_nelem:
    raise ValueError(f"output: engine {out} vs spec {spec.output_nelem}")


class Session:
  def __init__(self, transport: Transport, host: EngineHost):
    self.t = transport
    self.host = host
    self.telemetry = host.telemetry
    self.request: Request | None = None
    self.send_lock = threading.Lock()
    self.frames = 0
    self.last_seq = 0
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

  def progress(self, stage: str, frac: float, msg: str) -> None:
    try:
      self._send_json(P.Msg.PROGRESS, 0, {'stage': stage, 'frac': round(frac, 4), 'msg': msg})
    except LinkError:
      pass  # the comma may have given up and fallen back; the build continues

  def engine_update(self) -> None:
    """The worker finished. Tell the client that is here now, whoever it is."""
    try:
      self._respond_engine(0)
    except LinkError:
      pass

  def close(self) -> None:
    """The connection is gone. The engine stays: see EngineHost."""
    with self.host.lock:
      if self.host.session is self:
        self.host.session = None

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
    # The comma's USB controller occasionally sends a request twice; see
    # FfsTransport.write_chunk. The client numbers requests from 1 and never
    # reuses one on a connection, so anything at or below the last seq is the
    # replay, already answered. Running a frame twice would push the same
    # image into the history queues twice, silently.
    if msg.seq <= self.last_seq:
      log.warning("dropping replayed message type=%d seq=%d (last %d)", msg.msg_type, msg.seq, self.last_seq)
      return
    self.last_seq = msg.seq
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

  def _wanted(self) -> str | None:
    return self.request.sha256 if self.request else None

  def on_hello(self, msg: Message) -> None:
    import tensorrt as trt
    from jetlink.server.builder import device_tag
    self._send_json(P.Msg.HELLO_RESP, msg.seq, {
      'protocol': P.VERSION,
      'trt_version': trt.__version__,
      'device': device_tag(),
      'engine_state': self.host.status(self._wanted())['state'],
      'loaded': self.host.loaded_sha(),
      'frames_served': self.frames,
      'telemetry': self.telemetry.read(),
    })

  def on_engine_req(self, msg: Message) -> None:
    d = json.loads(bytes(msg.payload))
    self.request = Request(str(d['sha256']), int(d['nbytes']),
                           int(d.get('frame_skip', DEFAULT_FRAME_SKIP)))
    self._send_json(P.Msg.ENGINE_RESP, msg.seq, self.host.request(self.request, self))

  def _respond_engine(self, seq: int) -> None:
    self._send_json(P.Msg.ENGINE_RESP, seq, self.host.status(self._wanted()))

  def on_upload_chunk(self, msg: Message) -> None:
    req = self.request
    if req is None:
      return self._error(msg.seq, 'no_model', 'send ENGINE_REQ first')
    offset = int.from_bytes(bytes(msg.payload[:8]), 'little')
    data = msg.payload[8:]
    path = self.host.cache.model_path(req.sha256)
    mode = 'r+b' if path.exists() and offset else 'wb'
    with open(path, mode) as f:
      f.seek(offset)
      f.write(data)
    # Deliberately silent. The client streams chunks without reading between
    # them, and over USB a gadget only accepts data while its peer has a read
    # posted - so a progress write here stalls for the whole transfer timeout
    # on every attempt. The client reports its own upload progress anyway.

  def on_upload_done(self, msg: Message) -> None:
    req = self.request
    if req is None:
      return self._error(msg.seq, 'no_model', 'send ENGINE_REQ first')
    path = self.host.cache.model_path(req.sha256)
    if sha256_file(str(path))[0] != req.sha256:
      path.unlink(missing_ok=True)
      self._send_json(P.Msg.ENGINE_RESP, msg.seq, {
        'state': 'failed', 'detail': 'sha256 mismatch after upload',
        'sha256': req.sha256, 'chunk': CHUNK})
      return
    self.progress('upload', 1.0, 'verified')
    self._send_json(P.Msg.ENGINE_RESP, msg.seq, self.host.request(req, self))

  # -- the hot path ---------------------------------------------------------

  def on_infer(self, msg: Message) -> None:
    host = self.host
    with host.lock:
      loaded = host.loaded
      if loaded is None or loaded.sha256 != self._wanted():
        self._send(P.Msg.INFER_RESP, msg.seq,
                   (P.pack_infer_resp(0, P.Status.NOT_READY, 0, 0, 0),))
        return
      self._infer(loaded, msg)

  def _infer(self, loaded: Loaded, msg: Message) -> None:
    t0 = time.perf_counter()
    spec = loaded.spec
    if msg.payload.nbytes != spec.infer_req_nbytes:
      # The offsets below come from our spec, not from the wire. A client with
      # a different model would otherwise have its scalars read out of the
      # middle of the image, and the result would look perfectly finite.
      self._send(P.Msg.INFER_RESP, msg.seq,
                 (P.pack_infer_resp(0, P.Status.BAD_SHAPE, 0, 0, 0),))
      return
    frame_id, flags = P.unpack_infer_req(msg.payload)
    if flags & P.Flag.RESET_QUEUES:
      loaded.queues.reset()

    off = P.INFER_REQ_SIZE
    warped = np.frombuffer(msg.payload, np.uint8, spec.warped_nbytes, off).reshape(spec.warped_shape)
    off += spec.warped_nbytes
    packed = np.frombuffer(msg.payload, np.float32, spec.packed_nelem, off)

    # Gathers land straight in TensorRT's pinned input buffers: between the
    # wire and the GPU there is exactly one copy.
    loaded.queues.step_into(warped, packed, loaded.host_inputs)
    queue_us = int((time.perf_counter() - t0) * 1e6)

    outputs = loaded.engine.run()
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
    parts = [P.pack_infer_resp(frame_id, status, loaded.engine.last_gpu_us, queue_us, total_us),
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
    st = self.host.status(self._wanted())
    self._send_json(P.Msg.STATE_RESP, msg.seq, {
      **self.telemetry.read(),
      'engine_state': st['state'],
      'detail': st.get('detail', ''),
      'loaded': self.host.loaded_sha(),
      'frames_served': self.frames,
    })
