#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Check a built engine against a reference output.

Runs on the Jetson, inside the server image. Feeds fixed inputs straight to the
engine (bypassing the queues, which tests/test_queues.py covers separately) and
compares with a reference produced by onnxruntime on another machine.

    python3 scripts/verify_engine.py --engine <plan> --inputs <dir> --ref ref_out.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from jetlink.server.engine import TrtEngine


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument('--engine', required=True)
  p.add_argument('--inputs', required=True, help='dir with in_<name>.npy per model input')
  p.add_argument('--ref', help='reference output .npy')
  p.add_argument('--iters', type=int, default=50)
  args = p.parse_args()

  engine = TrtEngine(args.engine)
  print('engine inputs:')
  for n, b in engine.inputs.items():
    print(f'  {n:<20} {str(b.shape):<24} {b.dtype}')

  in_dir = Path(args.inputs)
  values = {}
  for name, b in engine.inputs.items():
    f = in_dir / f'in_{name}.npy'
    if not f.exists():
      print(f'missing {f}', file=sys.stderr)
      return 1
    values[name] = np.load(f).astype(b.dtype, copy=False).reshape(b.shape)

  out = next(iter(engine.infer(values).values())).astype(np.float32).reshape(-1)
  print(f'output: {out.shape} finite={np.all(np.isfinite(out))} '
        f'mean={out.mean():.5f} std={out.std():.5f}')

  if args.ref:
    ref = np.load(args.ref).astype(np.float32).reshape(-1)
    n = min(len(ref), len(out))
    a, b = out[:n], ref[:n]
    corr = float(np.corrcoef(a, b)[0, 1])
    print(f'vs reference: max abs {np.abs(a - b).max():.4f}  '
          f'mean abs {np.abs(a - b).mean():.5f}  corr {corr:.6f}')
    if corr < 0.999:
      print('CORRELATION TOO LOW', file=sys.stderr)
      return 2

  import time
  ts = []
  for _ in range(args.iters):
    t0 = time.perf_counter()
    engine.run()
    ts.append((time.perf_counter() - t0) * 1e3)
  ts = np.array(ts[5:])
  print(f'engine-only latency over {len(ts)}: mean {ts.mean():.2f} ms  '
        f'min {ts.min():.2f}  p50 {np.percentile(ts,50):.2f}  '
        f'p95 {np.percentile(ts,95):.2f}  max {ts.max():.2f}')
  return 0


if __name__ == '__main__':
  sys.exit(main())
