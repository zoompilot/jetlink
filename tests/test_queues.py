"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

jetlink's numpy history buffers must produce exactly what openpilot's tinygrad
ones do.

img_q / big_img_q / feat_q / desire_q are reimplemented in numpy on the Jetson,
and drift means the model silently sees the wrong history. Compared against the
real tinygrad functions over a run long enough for every ring to wrap twice.

Run with openpilot and tinygrad importable:
    PYTHONPATH=/path/to/sunnypilot:/path/to/sunnypilot/tinygrad_repo pytest tests/test_queues.py
"""
from __future__ import annotations

import numpy as np
import pytest

from jetlink.queues import PolicyQueues
from jetlink.spec import ModelSpec

tinygrad = pytest.importorskip('tinygrad', reason='needs tinygrad + openpilot on PYTHONPATH')
op_compile = pytest.importorskip('openpilot.selfdrive.modeld.compile_modeld',
                                 reason='needs openpilot on PYTHONPATH')

from tinygrad.tensor import Tensor  # noqa: E402

# The big (chestnut) model, as shipped: 4-D features_buffer, 33-step desire.
BIG_SHAPES = {
  'img': (1, 12, 128, 256),
  'big_img': (1, 12, 128, 256),
  'desire_pulse': (1, 33, 8),
  'traffic_convention': (1, 2),
  'action_t': (1, 2),
  'features_buffer': (1, 32, 32, 512),
}
SMALL_SHAPES = {
  'img': (1, 12, 128, 256),
  'big_img': (1, 12, 128, 256),
  'desire_pulse': (1, 25, 8),
  'traffic_convention': (1, 2),
  'action_t': (1, 2),
  'features_buffer': (1, 24, 512),
}


def make_spec(shapes, frame_skip=4) -> ModelSpec:
  return ModelSpec(sha256='0' * 64, nbytes=0, frame_skip=frame_skip,
                   input_shapes=shapes, output_shapes={'outputs': (1, 18452)},
                   output_slices={'hidden_state': slice(2066, 18450)}, checkpoint=None)


class TinygradReference:
  """openpilot's queues, driven exactly as run_policy drives them."""

  def __init__(self, spec: ModelSpec):
    import math
    fs = spec.frame_skip
    fb = spec.input_shapes['features_buffer']
    dp = spec.input_shapes['desire_pulse']
    img = spec.input_shapes['img']
    self.fs = fs

    # Derived from openpilot's own formulae rather than jetlink's properties, so
    # this is an independent reference. Mirrors upstream master's
    # get_policy_npy_shapes; this fork's checkout still has the 3-D fb[2] bug.
    feat_dim = math.prod(fb[2:])
    assert feat_dim == spec.feat_dim, f"spec.feat_dim {spec.feat_dim} != openpilot's {feat_dim}"
    n_frames = img[1] // 6
    img_buf_shape = (fs * (n_frames - 1) + 1, 6, img[2], img[3])
    assert img_buf_shape == spec.img_buf_shape

    self.img_q = Tensor(np.zeros(img_buf_shape, np.uint8)).contiguous().realize()
    self.big_img_q = Tensor(np.zeros(img_buf_shape, np.uint8)).contiguous().realize()
    self.feat_q = Tensor(np.zeros((fs * fb[1], fb[0], feat_dim), np.float32)).contiguous().realize()
    self.desire_q = Tensor(np.zeros((fs * dp[1], dp[0], dp[2]), np.float32)).contiguous().realize()

  def step(self, warped: np.ndarray, desire: np.ndarray, prev_feat: np.ndarray):
    w = Tensor(warped)
    img = op_compile.shift_and_sample(self.img_q, w[0:1],
                                      lambda b: op_compile.sample_skip(b, self.fs))
    big = op_compile.shift_and_sample(self.big_img_q, w[1:2],
                                      lambda b: op_compile.sample_skip(b, self.fs))
    des = op_compile.shift_and_sample(self.desire_q, Tensor(desire).reshape(1, 1, -1),
                                      lambda b: op_compile.sample_desire(b, self.fs))
    feat = op_compile.shift_and_sample(self.feat_q, Tensor(prev_feat).reshape(1, 1, -1),
                                       lambda b: op_compile.sample_skip(b, self.fs))
    return (img.numpy(), big.numpy(), des.numpy(), feat.numpy())


@pytest.mark.parametrize('shapes,name', [(BIG_SHAPES, 'big'), (SMALL_SHAPES, 'small')])
def test_matches_openpilot(shapes, name):
  spec = make_spec(shapes)
  # float32 queues so this compares representation as well as ordering; the
  # server runs float16, which only changes the cast point, not the values.
  ours = PolicyQueues(spec, dtype=np.float32)
  ref = TinygradReference(spec)
  rng = np.random.default_rng(0)

  # Long enough for every ring (the longest is desire_q at frame_skip*33=132)
  # to wrap more than once.
  for i in range(300):
    warped = rng.integers(0, 256, spec.warped_shape, dtype=np.uint8)
    desire = (rng.random(spec.packed_shapes['desire']) > 0.8).astype(np.float32)
    prev_feat = rng.standard_normal(spec.packed_shapes['prev_feat']).astype(np.float32)
    packed = np.concatenate([
      desire.ravel(),
      rng.standard_normal(2).astype(np.float32),
      rng.standard_normal(2).astype(np.float32),
      prev_feat.ravel(),
    ])

    got = ours.step(warped, packed)
    want_img, want_big, want_des, want_feat = ref.step(warped, desire, prev_feat)

    assert np.array_equal(got['img'].reshape(want_img.shape), want_img), f'{name} img @{i}'
    assert np.array_equal(got['big_img'].reshape(want_big.shape), want_big), f'{name} big_img @{i}'
    assert np.array_equal(got['desire_pulse'].reshape(want_des.shape), want_des), f'{name} desire @{i}'
    assert np.array_equal(got['features_buffer'].reshape(want_feat.shape), want_feat), f'{name} feat @{i}'


def test_reset_returns_to_initial_state():
  spec = make_spec(BIG_SHAPES)
  q = PolicyQueues(spec, dtype=np.float32)
  rng = np.random.default_rng(1)
  for _ in range(10):
    q.step(rng.integers(0, 256, spec.warped_shape, dtype=np.uint8),
           rng.standard_normal(spec.packed_nelem).astype(np.float32))
  q.reset()

  fresh = PolicyQueues(spec, dtype=np.float32)
  warped = np.zeros(spec.warped_shape, np.uint8)
  packed = np.zeros(spec.packed_nelem, np.float32)
  a, b = q.step(warped, packed), fresh.step(warped, packed)
  for k in a:
    assert np.array_equal(a[k], b[k]), k


def test_sampling_picks_oldest_first():
  """sample_skip must return history in order, oldest first, newest last."""
  spec = make_spec(BIG_SHAPES)
  q = PolicyQueues(spec, dtype=np.float32)
  # img_q holds frame_skip*(n_frames-1)+1 = 5 rows; sampling every 4 gives rows 0 and 4.
  for v in range(1, 8):
    warped = np.full(spec.warped_shape, v, np.uint8)
    out = q.step(warped, np.zeros(spec.packed_nelem, np.float32))
  img = out['img'].reshape(spec.img_shape)
  oldest, newest = img[0, 0, 0, 0], img[0, 6, 0, 0]
  assert newest == 7, f'newest should be the frame just pushed, got {newest}'
  assert oldest == 3, f'oldest should be 4 frames back, got {oldest}'
