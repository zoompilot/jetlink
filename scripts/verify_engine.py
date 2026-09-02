#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Check a built engine against a reference output.

Runs on the Jetson, inside the server image. Feeds fixed inputs straight to the
engine (bypassing the queues, which tests/test_queues.py covers, and the link,
which verify_parity.py covers) and compares with a reference produced by
onnxruntime on another machine.

Without --spec the only correlation available is over the whole output vector,
and 16384 of its 18452 values are hidden_state, so a wrong head hides behind a
right recurrence. That mode proves load, shape and finiteness, nothing more.
With the model spec each output slice and each column within it is checked,
the same test verify_parity.py makes.

    python3 scripts/verify_engine.py --engine <plan> --inputs <dir> --ref ref_out.npy --spec spec.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from jetlink.server.engine import TrtEngine

# Sibling script: python puts this file's directory first on sys.path, and the
# image ships scripts/ whole.
from verify_parity import MIN_CORR, load_spec_file, report_slices


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument('--engine', required=True)
  p.add_argument('--inputs', required=True, help='dir with in_<name>.npy per model input')
  p.add_argument('--ref', help='reference output .npy')
  p.add_argument('--spec', help='model spec json: compare per output slice instead of one number')
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
    if args.spec:
      spec = load_spec_file(args.spec)
      if len(out) < spec.output_nelem or len(ref) < spec.output_nelem:
        print(f'spec expects {spec.output_nelem} outputs, got {len(out)} from the engine '
              f'and {len(ref)} in the reference', file=sys.stderr)
        return 2
      print('per output slice:')
      gate = report_slices(spec, out, ref)
      bad = {k: v for k, v in gate.items() if v < MIN_CORR}
      if bad:
        print(f'FAIL: {len(bad)} slice(s) below corr {MIN_CORR}, whole or in a column: '
              + ', '.join(f'{k} {v:.6f}' for k, v in sorted(bad.items(), key=lambda kv: kv[1])),
              file=sys.stderr)
        return 2
      print(f'OK: every slice and every column at or above corr {MIN_CORR}')
    else:
      # A low whole-vector number is a certain failure. A high one proves
      # nothing about the heads, which are the 2068 values that steer.
      print('no --spec: this checks load, shape and finiteness only. The whole-vector '
            'correlation is mostly hidden_state and passes with a wrong head; pass '
            '--spec to compare per slice.')
      if corr < MIN_CORR:
        print('CORRELATION TOO LOW', file=sys.stderr)
        return 2

  # The server captures the per-frame sequence into a CUDA graph and replays
  # it, so that is the path worth timing; run() enqueues per frame until then.
  # capture_graph wants a warm run first, which infer() above was.
  if engine.capture_graph():
    engine.run()  # first replay, so the loop below is the steady state
    print('cuda graph: captured, timing graph replay as the server runs it')
  else:
    print('cuda graph: unavailable, timing per-frame enqueue (the server would fall back to this too)')
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
