"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The model's history buffers, reimplemented in numpy.

openpilot keeps img_q / big_img_q / feat_q / desire_q on the GPU and folds the
shift-and-sample into the tinygrad JIT. Shipping that history across the link
every frame would cost ~10 MB/frame, so the queues live here on the server
instead and the comma sends only the newest warped frame plus the packed
scalars. That makes this file the one piece of openpilot logic jetlink has to
mirror, and `tests/test_queues.py` checks it against the tinygrad original.

Two deliberate differences from openpilot, both value-preserving:

  * Ring buffers, not a rolling concat. openpilot rebuilds the buffer with
    `cat(buf[1:], new)` each frame; that is a full copy (8.4 MB for feat_q).
    A ring writes one slot and gathers only the sampled rows.
  * The queues hold float16. openpilot holds float32 and casts at the model
    boundary, so the values reaching the network are identical either way,
    and casting one new row per frame beats casting the whole sampled window.
    Pass dtype=np.float32 to get openpilot's exact intermediate representation
    (the equivalence test does).
"""
from __future__ import annotations

import numpy as np

from jetlink.spec import ModelSpec


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
    self.buf[self.head] = value
    self.head = (self.head + 1) % self.n

  def logical(self, step: int = 1) -> np.ndarray:
    """Gather logical indices 0, step, 2*step, ... in order."""
    idx = (self.head + np.arange(0, self.n, step)) % self.n
    return self.buf[idx]


def sample_skip(q: RingQueue, frame_skip: int, out: np.ndarray | None = None) -> np.ndarray:
  """openpilot: buf[::frame_skip].contiguous().flatten(0, 1).unsqueeze(0)

  With `out` (shaped (k, *buf.shape[1:])) the gather lands straight in the
  destination - which is how the server writes into TensorRT's pinned input
  buffers without an intermediate array.
  """
  idx = (q.head + np.arange(0, q.n, frame_skip)) % q.n
  if out is not None:
    np.take(q.buf, idx, axis=0, out=out)
    return out
  s = q.buf[idx]
  return s.reshape(1, s.shape[0] * s.shape[1], *s.shape[2:])


def sample_desire(q: RingQueue, frame_skip: int, out: np.ndarray | None = None) -> np.ndarray:
  """openpilot: buf.reshape(-1, frame_skip, *buf.shape[1:]).max(1).flatten(0, 1).unsqueeze(0)"""
  s = q.logical(1)
  m = s.reshape(-1, frame_skip, *s.shape[1:]).max(axis=1)
  if out is not None:
    out[...] = m.reshape(out.shape)
    return out
  return m.reshape(1, m.shape[0] * m.shape[1], *m.shape[2:])


class PolicyQueues:
  """Server-side state for one model. Everything `run_policy` owned in the JIT."""

  def __init__(self, spec: ModelSpec, dtype=np.float16):
    self.spec = spec
    self.dtype = dtype
    self.frame_skip = spec.frame_skip

    self.img_q = RingQueue(spec.img_buf_shape, dtype)
    self.big_img_q = RingQueue(spec.img_buf_shape, dtype)
    self.feat_q = RingQueue(spec.feat_q_shape, dtype)
    self.desire_q = RingQueue(spec.desire_q_shape, dtype)

    # Model-facing shapes. features_buffer is declared 4-D (1,32,32,512) but the
    # queue produces the flattened (1,32,16384); same memory, so just reshape.
    self.model_shapes = dict(spec.input_shapes)

  def reset(self) -> None:
    for q in (self.img_q, self.big_img_q, self.feat_q, self.desire_q):
      q.reset()

  def _unpack(self, packed: np.ndarray):
    spec = self.spec
    parts = np.split(packed, np.cumsum(spec.packed_sizes[:-1]))
    return tuple(p.reshape(s) for p, s in zip(parts, spec.packed_shapes.values()))

  def _push(self, warped: np.ndarray, packed: np.ndarray):
    spec = self.spec
    if warped.shape != spec.warped_shape:
      raise ValueError(f"warped {warped.shape} != {spec.warped_shape}")
    if packed.size != spec.packed_nelem:
      raise ValueError(f"packed {packed.size} != {spec.packed_nelem}")
    desire, traffic_convention, action_t, prev_feat = self._unpack(packed)
    # push() assigns into a typed buffer, so the uint8 -> float16 cast happens
    # here, on one new row, rather than on the whole sampled window later.
    self.img_q.push(warped[0])
    self.big_img_q.push(warped[1])
    self.desire_q.push(desire.reshape(1, -1))
    self.feat_q.push(prev_feat.reshape(1, -1))
    return traffic_convention, action_t

  def step(self, warped: np.ndarray, packed: np.ndarray) -> dict[str, np.ndarray]:
    """Advance the queues one frame and return the model's inputs.

    warped: (2, 6, H, W) uint8, as produced by openpilot's warp on the comma.
    packed: flat float32, laid out per ModelSpec.packed_shapes.
    """
    traffic_convention, action_t = self._push(warped, packed)
    return {
      'img': sample_skip(self.img_q, self.frame_skip).reshape(self.model_shapes['img']),
      'big_img': sample_skip(self.big_img_q, self.frame_skip).reshape(self.model_shapes['big_img']),
      'features_buffer': sample_skip(self.feat_q, self.frame_skip).reshape(
        self.model_shapes['features_buffer']),
      'desire_pulse': sample_desire(self.desire_q, self.frame_skip).reshape(
        self.model_shapes['desire_pulse']),
      'traffic_convention': traffic_convention.astype(self.dtype, copy=False).reshape(
        self.model_shapes['traffic_convention']),
      'action_t': action_t.astype(self.dtype, copy=False).reshape(self.model_shapes['action_t']),
    }

  def step_into(self, warped: np.ndarray, packed: np.ndarray,
                dest: dict[str, np.ndarray]) -> None:
    """Same as step(), writing directly into caller-owned buffers.

    `dest` maps model input name to an array of the model's declared shape -
    in the server these are TensorRT's pinned host buffers, so the gather is
    the only copy between the wire and the GPU.
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
