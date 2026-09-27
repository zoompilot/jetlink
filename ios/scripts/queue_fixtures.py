#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Writes what jetlink.queues.PolicyQueues produces, for the Swift queues to be
held to bit for bit (ios/JetlinkKit/Tests/JetlinkKitTests/QueueTests.swift).

Small shapes, so the fixtures stay a few hundred KB, run long enough for
every ring to wrap several times, with resets part way, NaN and values past
float16's range in the scalars, and rounding cases in between.

    python3 ios/scripts/queue_fixtures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jetlink.queues import PolicyQueues
from jetlink.spec import ModelSpec

OUT = Path(__file__).resolve().parents[1] / 'JetlinkKit' / 'Tests' / 'JetlinkKitTests' / 'Fixtures'
ORDER = ('img', 'big_img', 'desire_pulse', 'traffic_convention', 'action_t', 'features_buffer')

CASES = {
  # the chestnut layout at a toy size: 4-D features_buffer
  'queues_fs4': ({'img': (1, 12, 4, 8), 'big_img': (1, 12, 4, 8), 'desire_pulse': (1, 5, 3),
                  'traffic_convention': (1, 2), 'action_t': (1, 2), 'features_buffer': (1, 6, 2, 3)}, 4, 70, (0, 41)),
  # the small model's 3-D features_buffer, another frame_skip, three image frames
  'queues_fs2': ({'img': (1, 18, 2, 4), 'big_img': (1, 18, 2, 4), 'desire_pulse': (1, 7, 4),
                  'traffic_convention': (1, 2), 'action_t': (1, 2), 'features_buffer': (1, 5, 6)}, 2, 50, (0, 17)),
}


def make_spec(shapes, frame_skip) -> ModelSpec:
  return ModelSpec(sha256='0' * 64, nbytes=0, frame_skip=frame_skip, input_shapes=shapes,
                   output_shapes={'outputs': (1, 8)}, output_slices={'hidden_state': slice(0, 8)}, checkpoint=None)


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  for name, (shapes, fs, frames, resets) in CASES.items():
    spec = make_spec(shapes, fs)
    q = PolicyQueues(spec)
    rng = np.random.default_rng(1234)
    ins, outs = bytearray(), bytearray()
    for i in range(frames):
      reset = i in resets
      if reset:
        q.reset()
      warped = rng.integers(0, 256, spec.warped_shape, dtype=np.uint8)
      packed = (rng.standard_normal(spec.packed_nelem) * 3).astype(np.float32)
      # rounding ties and edges float16 has to decide
      packed[0] = np.float32(1.0009765625 + 2 ** -12)
      if i % 7 == 3:
        packed[1] = np.nan
      if i % 11 == 5:
        packed[-1] = np.float32(1e6)      # past float16's range: inf
        packed[-2] = np.float32(-7e4)
      if i % 5 == 0:
        packed[2] = np.float32(3e-8)       # a float16 subnormal
      ins += bytes([1 if reset else 0]) + warped.tobytes() + packed.tobytes()
      dest = q.step(warped, packed)
      for k in ORDER:
        outs += np.ascontiguousarray(dest[k], np.float16).tobytes()
    (OUT / f'{name}.json').write_text(json.dumps({**spec.to_dict(), 'frames': frames}, indent=1))
    (OUT / f'{name}.in.bin').write_bytes(bytes(ins))
    (OUT / f'{name}.out.bin').write_bytes(bytes(outs))
    print(f"{name}: {frames} frames, {len(ins)} bytes in, {len(outs)} bytes out")
  return 0


if __name__ == '__main__':
  sys.exit(main())
