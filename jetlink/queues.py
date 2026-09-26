"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The model's history buffers, reimplemented in numpy, or looped for a graph
that keeps its own.

openpilot folds these into the tinygrad JIT on the GPU. Shipping the history
across the link would cost ~10 MB a frame, so the queues live on the server and
the comma sends only the newest warped frame and the packed scalars.
tests/test_queues.py checks this against the tinygrad original.

Two value-preserving differences: ring buffers rather than openpilot's rolling
`cat(buf[1:], new)`, which copies 8.4 MB a frame; and float16 storage rather
than float32 cast at the model boundary, so only the new row is cast. Pass
dtype=np.float32 for openpilot's exact intermediates.

A stateful graph (openpilot #38916) does the queueing itself, so for one of
those there is nothing to reimplement: StateLoop feeds each next_state_ output
back as its state_ input. for_model() picks the one a spec needs.
"""
from __future__ import annotations

import numpy as np

from jetlink.spec import ModelSpec


# numpy 1.x has no vectorised float16 store on aarch64: casting one 393 KB frame
# of uint8 to float16 takes 2.25 ms on the Orin under 1.26, and a lookup of the
# 256 possible values gives the same bits in 0.69 ms. numpy 2.x casts it in
# 0.27 ms (2.5.3, same Orin) against the lookup's 0.62, and an M1 Pro in 0.12 ms
# against 0.86, so the lookup is kept only where numpy needs it: the JetPack 6
# image.
# Viewed as uint16 so np.take can share the dtype.
_U8_TO_F16_BITS = np.arange(256, dtype=np.uint8).astype(np.float16).view(np.uint16)
_LOOKUP = int(np.__version__.split('.')[0]) < 2


def store(dest: np.ndarray, value: np.ndarray) -> None:
  """Copy `value` into `dest`, casting; uint8 images into fp16 through the lookup
  on numpy 1.x."""
  if _LOOKUP and dest.dtype == np.float16 and value.dtype == np.uint8:
    # take-with-out, not `dest[:] = lut[src]`: the latter builds a 393 KB
    # temporary, 2.4 ms against 1.4 ms on the Orin (the other way on x86, so
    # measure there). clip skips a bounds check a uint8 index cannot fail.
    np.take(_U8_TO_F16_BITS, value.reshape(-1),
            out=dest.reshape(-1).view(np.uint16), mode='clip')
  else:
    np.copyto(dest, value.reshape(dest.shape), casting='unsafe')


def _strided_runs(head: int, n: int, step: int) -> list[tuple[slice, slice]]:
  """Rows head, head+step, ... (mod n) as at most two strided slices.

  Strided slices hit numpy's memcpy path; fancy indexing costs more than the
  copy it performs. A wrapped sequence is always two runs of constant stride.
  """
  m = -(-n // step)                        # number of sampled rows
  m1 = min(m, -(-(n - head) // step))      # rows before the wrap
  runs = [(slice(head, head + m1 * step, step), slice(0, m1))]
  if m1 < m:
    start = head + m1 * step - n
    runs.append((slice(start, start + (m - m1) * step, step), slice(m1, m)))
  return runs


class RingQueue:
  """Fixed-length FIFO over a preallocated array.

  Logical index i (0 = oldest) lives at physical (head + i) % n.
  """

  def __init__(self, shape: tuple[int, ...], dtype):
    self.buf = np.zeros(shape, dtype=dtype)
    self.n = shape[0]
    self.head = 0

  def reset(self) -> None:
    self.buf[:] = 0
    self.head = 0

  def push(self, value) -> None:
    # The slot the oldest element occupies becomes the newest once head moves.
    store(self.buf[self.head], value)
    self.head = (self.head + 1) % self.n

  def gather(self, step: int, out: np.ndarray) -> np.ndarray:
    """Write logical rows 0, step, 2*step, ... into `out`, oldest first."""
    for src, dst in _strided_runs(self.head, self.n, step):
      out[dst] = self.buf[src]
    return out

  def logical(self, step: int = 1) -> np.ndarray:
    """Same as gather(), allocating the destination."""
    m = -(-self.n // step)
    return self.gather(step, np.empty((m, *self.buf.shape[1:]), self.buf.dtype))


def sample_skip(q: RingQueue, frame_skip: int, out: np.ndarray | None = None) -> np.ndarray:
  """openpilot: buf[::frame_skip].contiguous().flatten(0, 1).unsqueeze(0)

  With `out`, shaped (k, *buf.shape[1:]), the gather lands straight in
  TensorRT's pinned input buffers.
  """
  if out is not None:
    return q.gather(frame_skip, out)
  s = q.logical(frame_skip)
  return s.reshape(1, s.shape[0] * s.shape[1], *s.shape[2:])


def sample_desire(q: RingQueue, frame_skip: int, out: np.ndarray | None = None) -> np.ndarray:
  """openpilot: buf.reshape(-1, frame_skip, *buf.shape[1:]).max(1).flatten(0, 1).unsqueeze(0)"""
  s = q.logical(1)
  m = s.reshape(-1, frame_skip, *s.shape[1:]).max(axis=1)
  if out is not None:
    out[...] = m.reshape(out.shape)
    return out
  return m.reshape(1, m.shape[0] * m.shape[1], *m.shape[2:])


def _check_frame(spec: ModelSpec, warped: np.ndarray, packed: np.ndarray) -> None:
  if warped.shape != spec.warped_shape:
    raise ValueError(f"warped {warped.shape} != {spec.warped_shape}")
  if packed.size != spec.packed_nelem:
    raise ValueError(f"packed {packed.size} != {spec.packed_nelem}")


class PolicyQueues:
  """Server-side state for one model. Everything `run_policy` owned in the JIT."""

  def __init__(self, spec: ModelSpec, dtype=np.float16):
    self.spec = spec
    self.dtype = dtype
    self.frame_skip = spec.frame_skip

    self._packed_layout = list(spec.packed_layout.values())

    self.img_q = RingQueue(spec.img_buf_shape, dtype)
    self.big_img_q = RingQueue(spec.img_buf_shape, dtype)
    self.feat_q = RingQueue(spec.feat_q_shape, dtype)
    self.desire_q = RingQueue(spec.desire_q_shape, dtype)

    # features_buffer is declared 4-D (1,32,32,512); the queue produces the flat
    # (1,32,16384) over the same memory, so a reshape covers it.
    self.model_shapes = dict(spec.input_shapes)

  def reset(self) -> None:
    for q in (self.img_q, self.big_img_q, self.feat_q, self.desire_q):
      q.reset()

  def after_run(self, outputs: dict[str, np.ndarray], dest: dict[str, np.ndarray]) -> None:
    """Nothing: the comma sends the hidden state back as prev_feat."""

  def _unpack(self, packed: np.ndarray):
    # slice views, not np.split: the offsets never change and split allocates
    return tuple(packed[s].reshape(shape) for s, shape in self._packed_layout)

  def _push(self, warped: np.ndarray, packed: np.ndarray):
    _check_frame(self.spec, warped, packed)
    desire, traffic_convention, action_t, prev_feat = self._unpack(packed)
    # push() casts into a typed buffer, so uint8 -> float16 costs one row here
    # rather than the whole sampled window later
    self.img_q.push(warped[0])
    self.big_img_q.push(warped[1])
    self.desire_q.push(desire.reshape(1, -1))
    self.feat_q.push(prev_feat.reshape(1, -1))
    return traffic_convention, action_t

  def step(self, warped: np.ndarray, packed: np.ndarray) -> dict[str, np.ndarray]:
    """Advance the queues one frame and return the model's inputs.

    warped: (2, 6, H, W) uint8 from openpilot's warp. packed: flat float32,
    laid out per ModelSpec.packed_shapes. Allocates; the server uses
    step_into(), so there is one implementation to keep correct.
    """
    dest = {n: np.empty(s, self.dtype) for n, s in self.model_shapes.items()}
    self.step_into(warped, packed, dest)
    return dest

  def step_into(self, warped: np.ndarray, packed: np.ndarray,
                dest: dict[str, np.ndarray]) -> None:
    """Same as step(), writing into caller-owned buffers.

    `dest` maps input name to an array of the declared shape. In the server
    those are TensorRT's pinned buffers, so the gather is the only copy.
    """
    traffic_convention, action_t = self._push(warped, packed)
    fs = self.frame_skip

    for name, q in (('img', self.img_q), ('big_img', self.big_img_q)):
      d = dest[name]
      sample_skip(q, fs, out=d.reshape(-1, *q.buf.shape[1:]))

    d = dest['features_buffer']
    sample_skip(self.feat_q, fs, out=d.reshape(-1, *self.feat_q.buf.shape[1:]))

    sample_desire(self.desire_q, fs, out=dest['desire_pulse'])
    dest['traffic_convention'][...] = traffic_convention.reshape(
      dest['traffic_convention'].shape)
    dest['action_t'][...] = action_t.reshape(dest['action_t'].shape)


class StateLoop:
  """Server-side state for a graph that keeps its own history.

  The frame goes into new_img and the scalars into their inputs as they are;
  the queues are the graph's, handed back each frame as next_state_<q> and fed
  in as state_<q> on the next. That is openpilot's ModelState since #38916,
  which aliases each next_ output onto its state_ input.

  An engine with `loop_state` keeps the loop in device memory: the queues are
  12 MB, which would otherwise cross to the host and back every frame. For any
  other engine the copy is done here, after the reply has gone.
  """

  def __init__(self, spec: ModelSpec, engine):
    self.spec = spec
    self.engine = engine
    self.pairs = spec.state_pairs
    if not self.pairs:
      raise ValueError("the graph takes new_img but returns no next_state_ outputs")
    self._packed_layout = spec.packed_layout
    loop = getattr(engine, 'loop_state', None)
    self.on_engine = bool(loop(self.pairs)) if callable(loop) else False

  def reset(self) -> None:
    """Empty queues, as openpilot's warmup leaves them."""
    if self.on_engine:
      self.engine.reset_state()
    else:
      for name in self.pairs:
        self.engine.host_input(name)[...] = 0

  def step_into(self, warped: np.ndarray, packed: np.ndarray,
                dest: dict[str, np.ndarray]) -> None:
    """Write one frame's inputs; the state inputs are already in place."""
    _check_frame(self.spec, warped, packed)
    store(dest['new_img'], warped)
    for name, (s, _) in self._packed_layout.items():
      store(dest[name], packed[s])

  def after_run(self, outputs: dict[str, np.ndarray], dest: dict[str, np.ndarray]) -> None:
    """Advance the queues: each next_state_ becomes next frame's state_."""
    if self.on_engine:
      return
    for name, nxt in self.pairs.items():
      store(dest[name], outputs[nxt])


def for_model(spec: ModelSpec, engine) -> PolicyQueues | StateLoop:
  """The server-side state a model needs: queues for a queued graph, the loop
  for a stateful one."""
  return StateLoop(spec, engine) if spec.stateful else PolicyQueues(spec)
