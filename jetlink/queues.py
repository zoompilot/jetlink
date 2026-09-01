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


# uint8 -> float16 is 68% of the per-frame cost if you let numpy do it: on
# aarch64 numpy has no vectorised float16 *store* loop, so it converts at
# ~5.2 ns/element. The source here is uint8, which has only 256 possible
# values, so a lookup table indexed by the byte gives the exact same bits at
# memcpy speed. Stored as the fp16 bit patterns viewed as uint16, because
# np.take needs the table and the destination to share a dtype.
_U8_TO_F16_BITS = np.arange(256, dtype=np.uint8).astype(np.float16).view(np.uint16)


def _strided_runs(head: int, n: int, step: int) -> list[tuple[slice, slice]]:
  """Rows head, head+step, ... (mod n) as at most two strided slices.

  Fancy indexing (np.take / buf[idx]) costs more than the copy it performs;
  plain strided slices hit numpy's memcpy path. The wrapped sequence is always
  two runs of constant stride, so no generality is lost.
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
    # Only the fp16 rings can use the lookup table; a float32 ring (used by the
    # equivalence test) falls back to a plain assignment.
    self._lut = self.buf.dtype == np.float16

  def reset(self) -> None:
    self.buf[:] = 0
    self.head = 0

  def push(self, value) -> None:
    # The slot the oldest element occupies becomes the newest once head moves.
    dest = self.buf[self.head]
    if self._lut and getattr(value, 'dtype', None) == np.uint8:
      # np.take with out=, NOT `dest[:] = lut[src]`. Advanced indexing measures
      # faster on x86/newer numpy, but on the Orin's A78 cores with the numpy
      # in the JetPack image it is the other way round: take-with-out keeps
      # 1.4 ms of queue time where advanced indexing costs 2.4 ms, because it
      # writes straight into the destination instead of building a 393 KB
      # temporary. Measure on the device before changing this.
      # mode='clip' skips a bounds check that a uint8 index can never fail.
      np.take(_U8_TO_F16_BITS, value.reshape(-1),
              out=dest.reshape(-1).view(np.uint16), mode='clip')
    else:
      dest[...] = value
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

  With `out` (shaped (k, *buf.shape[1:])) the gather lands straight in the
  destination - which is how the server writes into TensorRT's pinned input
  buffers without an intermediate array.
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


class PolicyQueues:
  """Server-side state for one model. Everything `run_policy` owned in the JIT."""

  def __init__(self, spec: ModelSpec, dtype=np.float16):
    self.spec = spec
    self.dtype = dtype
    self.frame_skip = spec.frame_skip

    offset = 0
    self._packed_layout = []
    for size, shape in zip(spec.packed_sizes, spec.packed_shapes.values()):
      self._packed_layout.append((offset, offset + size, shape))
      offset += size

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
    # Slice views, not np.split: split builds a list of new array objects every
    # frame for no benefit, and the offsets never change.
    return tuple(packed[a:b].reshape(shape) for a, b, shape in self._packed_layout)

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

    Allocates its destinations; the server uses step_into() instead. Kept as a
    thin wrapper so there is only one implementation to keep correct.
    """
    dest = {n: np.empty(s, self.dtype) for n, s in self.model_shapes.items()}
    self.step_into(warped, packed, dest)
    return dest

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
