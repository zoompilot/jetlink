"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

A queued graph's history buffers against openpilot's own, as golden vectors.

img_q, big_img_q, feat_q and desire_q live on the server in numpy
(jetlink.queues), where openpilot kept them in tinygrad inside its JIT, and
drift means the model silently sees the wrong history. The fixture holds a
digest of everything openpilot's functions fed the model over a run long
enough for every ring to wrap twice: shift_and_sample, sample_skip and
sample_desire, which compile_modeld had until openpilot moved to stateful
graphs and sunnypilot keeps frozen for the models from before
(openpilot/sunnypilot/modeld_v2/stock_dependencies.py). Run from a fork
checkout, this module writes the fixture again:

  DEV=CPU PYTHONPATH=<fork>:<fork>/tinygrad_repo:. python -m tests.test_queues
"""
from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path

import numpy as np

from jetlink.queues import PolicyQueues
from jetlink.spec import ModelSpec

FIXTURE = Path(__file__).parent / 'fixtures' / 'policy_queues_openpilot.json'
# the longest ring is desire_q at frame_skip * 33 = 132 rows
FRAMES = 300
# what the queues feed the model, and the dtype openpilot's queues hold it in
FED = {'img': np.uint8, 'big_img': np.uint8, 'desire_pulse': np.float32, 'features_buffer': np.float32}
# the big (chestnut) layout, 4-D features_buffer and 33 desire steps, and the
# small one; images a few pixels across, which the queues treat alike
CONFIGS = {
  'big': {'img': (1, 12, 2, 4), 'big_img': (1, 12, 2, 4), 'desire_pulse': (1, 33, 8), 'traffic_convention': (1, 2),
          'action_t': (1, 2), 'features_buffer': (1, 32, 2, 4)},
  'small': {'img': (1, 12, 2, 4), 'big_img': (1, 12, 2, 4), 'desire_pulse': (1, 25, 8), 'traffic_convention': (1, 2),
            'action_t': (1, 2), 'features_buffer': (1, 24, 8)},
}


def make_spec(shapes, frame_skip=4) -> ModelSpec:
  # the hidden state is the features queue's row, as the model returns it
  feat = math.prod(shapes['features_buffer'][2:])
  return ModelSpec(sha256='0' * 64, nbytes=0, frame_skip=frame_skip, input_shapes=shapes,
                   output_shapes={'outputs': (1, feat + 4)}, output_slices={'hidden_state': slice(2, 2 + feat)},
                   checkpoint=None)


def frame(spec: ModelSpec, i: int):
  """Frame i's warped frame, desire, packed floats and model output. Whole
  numbers, which float32 holds exactly on either side, and a desire that
  pulses about one frame in two."""
  warped = ((np.arange(math.prod(spec.warped_shape)) * 7 + i * 13) % 256).astype(np.uint8).reshape(spec.warped_shape)
  desire = ((np.arange(spec.packed_shapes['desire'][0]) * 3 + i * 7) % 11 == 0).astype(np.float32)
  packed = np.concatenate([desire, np.full(spec.packed_nelem - desire.size, i % 3, np.float32)])
  output = ((np.arange(spec.output_nelem) * 3 + i * 5) % 251).astype(np.float32)
  return warped, desire, packed, output


def fed_bytes(name: str, value: np.ndarray) -> bytes:
  return np.ascontiguousarray(value, dtype=FED[name]).tobytes()


def jetlink_digests(shapes) -> dict[str, str]:
  spec = make_spec(shapes)
  # float32, so this compares representation as well as ordering; the server
  # runs float16, which only moves the cast
  queues = PolicyQueues(spec, dtype=np.float32)
  digests = {name: hashlib.sha256() for name in FED}
  for i in range(FRAMES):
    warped, _, packed, output = frame(spec, i)
    fed = queues.step(warped, packed)
    for name, digest in digests.items():
      digest.update(fed_bytes(name, fed[name]))
    queues.after_run({'outputs': output})
  return {name: digest.hexdigest() for name, digest in digests.items()}


def openpilot_digests(shapes) -> dict[str, str]:
  """openpilot's queues, driven exactly as its run_policy drove them. Needs
  the fork's tinygrad and openpilot."""
  from tinygrad.tensor import Tensor

  from openpilot.sunnypilot.modeld_v2.stock_dependencies import sample_desire, sample_skip, shift_and_sample
  spec = make_spec(shapes)
  fs, img, fb, dp = spec.frame_skip, shapes['img'], shapes['features_buffer'], shapes['desire_pulse']

  # openpilot's own shapes for them (get_policy_npy_shapes), not jetlink's
  def queue(shape, dtype):
    return Tensor(np.zeros(shape, dtype)).contiguous().realize()
  img_q, big_img_q = (queue((fs * (img[1] // 6 - 1) + 1, 6, img[2], img[3]), np.uint8) for _ in range(2))
  feat_q = queue((fs * fb[1], fb[0], math.prod(fb[2:])), np.float32)
  desire_q = queue((fs * dp[1], dp[0], dp[2]), np.float32)

  def skip(buf):
    return sample_skip(buf, fs)

  def desires(buf):
    return sample_desire(buf, fs)

  digests = {name: hashlib.sha256() for name in FED}
  # modeld's prev_feat: zero at the start, then each frame's hidden state
  prev_feat = np.zeros(spec.prev_feat_shape, np.float32)
  for i in range(FRAMES):
    warped, desire, _, output = frame(spec, i)
    w = Tensor(warped)
    fed = {'img': shift_and_sample(img_q, w[0:1], skip), 'big_img': shift_and_sample(big_img_q, w[1:2], skip),
           'desire_pulse': shift_and_sample(desire_q, Tensor(desire).reshape(1, 1, -1), desires),
           'features_buffer': shift_and_sample(feat_q, Tensor(prev_feat).reshape(1, 1, -1), skip)}
    for name, digest in digests.items():
      digest.update(fed_bytes(name, fed[name].numpy()))
    prev_feat = output[spec.output_slices['hidden_state']].reshape(spec.prev_feat_shape)
  return {name: digest.hexdigest() for name, digest in digests.items()}


class TestPolicyQueues(unittest.TestCase):
  def test_every_frame_feeds_what_openpilot_fed(self):
    expected = json.loads(FIXTURE.read_text())
    for name, shapes in CONFIGS.items():
      with self.subTest(name):
        self.assertEqual(jetlink_digests(shapes), expected[name])

  def test_reset_returns_to_initial_state(self):
    spec = make_spec(CONFIGS['big'])
    q = PolicyQueues(spec, dtype=np.float32)
    for i in range(10):
      q.step(*frame(spec, i)[::2][:2])
    q.reset()

    fresh = PolicyQueues(spec, dtype=np.float32)
    warped = np.zeros(spec.warped_shape, np.uint8)
    packed = np.zeros(spec.packed_nelem, np.float32)
    a, b = q.step(warped, packed), fresh.step(warped, packed)
    for k in a:
      self.assertTrue(np.array_equal(a[k], b[k]), k)

  def test_sampling_picks_oldest_first(self):
    """sample_skip must return history in order, oldest first, newest last."""
    spec = make_spec(CONFIGS['big'])
    q = PolicyQueues(spec, dtype=np.float32)
    # img_q holds frame_skip*(n_frames-1)+1 = 5 rows; sampling every 4 gives rows 0 and 4.
    for v in range(1, 8):
      out = q.step(np.full(spec.warped_shape, v, np.uint8), np.zeros(spec.packed_nelem, np.float32))
    img = out['img'].reshape(spec.img_shape)
    oldest, newest = img[0, 0, 0, 0], img[0, 6, 0, 0]
    self.assertEqual(newest, 7, 'newest should be the frame just pushed')
    self.assertEqual(oldest, 3, 'oldest should be 4 frames back')


if __name__ == '__main__':
  FIXTURE.write_text(json.dumps({name: openpilot_digests(shapes) for name, shapes in CONFIGS.items()}, indent=2) + '\n')
  print(f"wrote {FIXTURE}")
