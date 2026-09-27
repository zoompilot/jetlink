#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Fixtures for ServerTests.swift, which runs the Swift server over TCP with
tests/tiny_model.py's model on onnxruntime's CPU provider: the model (named
.onnx.bin so .gitignore's *.onnx leaves it tracked), a run of frames with
resets part way, and what jetlink.queues plus onnxruntime on the unmodified
model make of them, which is scripts/verify_parity.py's reference.

    python3 ios/scripts/server_fixtures.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

from jetlink.queues import PolicyQueues  # noqa: E402
from jetlink.spec import spec_from_onnx  # noqa: E402
from tiny_model import write  # noqa: E402

OUT = ROOT / 'ios' / 'JetlinkKit' / 'Tests' / 'JetlinkKitTests' / 'Fixtures'
FRAMES = 12
RESETS = (0, 7)
DTYPES = {'tensor(uint8)': np.uint8, 'tensor(float16)': np.float16, 'tensor(float)': np.float32}


def main() -> int:
  import onnxruntime as ort
  ort.disable_telemetry_events()
  with tempfile.TemporaryDirectory() as tmp:
    model = Path(tmp) / 'tiny.onnx'
    write(model)
    data = model.read_bytes()
    spec = spec_from_onnx(str(model))
    # onnxruntime cannot load the tinygrad layout hint, which the real models
    # define as a local function and this one does not. It is a passthrough,
    # so the reference drops it and nothing else: the uint8 images and their
    # casts stay as the model declares them.
    import onnx
    from jetlink.onnx_patch import strip_tinygrad_ops
    m = onnx.load(str(model))
    strip_tinygrad_ops(m)
    sess = ort.InferenceSession(m.SerializeToString(), providers=['CPUExecutionProvider'])
  dtypes = {i.name: DTYPES[i.type] for i in sess.get_inputs()}

  rng = np.random.default_rng(99)
  q = PolicyQueues(spec)
  ins, outs = bytearray(), bytearray()
  for i in range(FRAMES):
    if i in RESETS:
      q.reset()
    warped = rng.integers(0, 256, spec.warped_shape, dtype=np.uint8)
    packed = (rng.standard_normal(spec.packed_nelem) * 0.5).astype(np.float32)
    feed = {k: np.ascontiguousarray(v, dtype=dtypes[k]) for k, v in q.step(warped, packed).items()}
    out = np.asarray(sess.run(None, feed)[0], np.float32).reshape(-1)
    ins += bytes([1 if i in RESETS else 0]) + warped.tobytes() + packed.tobytes()
    outs += out.tobytes()

  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / 'tiny_model.onnx.bin').write_bytes(data)
  (OUT / 'tiny_spec.json').write_text(json.dumps({**spec.to_dict(), 'frames': FRAMES}, indent=1))
  (OUT / 'tiny_frames.in.bin').write_bytes(bytes(ins))
  (OUT / 'tiny_frames.out.bin').write_bytes(bytes(outs))
  print(f"tiny model {len(data)} bytes, sha256 {spec.sha256[:16]}, {FRAMES} frames, "
        f"{spec.infer_req_nbytes} bytes a request")
  return 0


if __name__ == '__main__':
  sys.exit(main())
