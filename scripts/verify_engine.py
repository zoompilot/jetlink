#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Check a built engine against a reference output.

Runs where the engine was built: inside the Jetson's server image, or on the
Mac or laptop that serves. Feeds fixed inputs straight to the engine, bypassing
the queues (tests/test_queues.py) and the link (verify_parity.py), and compares
with a reference from onnxruntime elsewhere.

Without --spec the correlation is over the whole vector, 16384 of whose 18452
values are hidden_state, so a wrong head hides behind a right recurrence: that
mode proves load, shape and finiteness only. --spec checks each output slice and
column, but on one frame; verify_parity's multi-frame capture is where columns
are really judged.

    python3 scripts/verify_engine.py --engine <plan> --inputs <dir> --ref ref_out.npy --spec spec.json

--capture asks instead whether the comma received what the engine computed. It
replays a verify_parity capture through the same PolicyQueues and staging
buffers the server uses and demands bit identity with out_link_*, which leaves
every remaining difference against onnxruntime as inference precision.

    python3 scripts/verify_engine.py --engine <plan> --capture <dir from verify_parity capture>

--backend names what built the artifact (auto picks the same way the server
does); --device is passed to it as the server's --device is.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# sibling script: python puts this file's directory first on sys.path
from verify_parity import MIN_CORR, load_spec_file, report_slices

from jetlink.server.backends import NAMES, select
from jetlink.server.backends.base import Engine, infer


def replay_capture(engine: Engine, d: Path) -> int:
  """Feed a verify_parity capture through the server's own path and compare with out_link_*."""
  from jetlink.queues import PolicyQueues

  spec = load_spec_file(d / 'spec.json')
  queues = PolicyQueues(spec)
  host_inputs = {n: engine.host_input(n) for n in engine.inputs}
  # same warm-up as EngineHost._warm, so the replay runs the kernels the server runs
  queues.step_into(np.zeros(spec.warped_shape, np.uint8), np.zeros(spec.packed_nelem, np.float32), host_inputs)
  print(engine.warm())
  queues.reset()

  n = len(list(d.glob('in_warped_*.npy')))
  if not n:
    print(f'no captured inputs in {d}', file=sys.stderr)
    return 1
  bad = 0
  for i in range(n):
    # in_packed_i already carries the hidden state the link returned for frame i-1
    queues.step_into(np.load(d / f'in_warped_{i}.npy'), np.load(d / f'in_packed_{i}.npy'), host_inputs)
    out = np.asarray(next(iter(engine.run().values())).reshape(-1), np.float32)
    link = np.load(d / f'out_link_{i}.npy').reshape(-1)
    same = np.array_equal(out, link)
    bad += not same
    print(f'frame {i:2d}: {"identical" if same else "DIFFERS"} to the link, '
          f'{np.count_nonzero(out != link)} values, max abs {np.abs(out - link).max():.6f}')
  if bad:
    print(f'{bad} of {n} frames differ: the link did not carry the engine output unchanged', file=sys.stderr)
    return 2
  print(f'OK: the comma received the engine output bit for bit on all {n} frames')
  return 0


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument('--engine', required=True, help='the artifact: a .plan, .pkl or .ortcache')
  p.add_argument('--backend', choices=('auto', *NAMES), default='auto')
  p.add_argument('--device', default='auto')
  p.add_argument('--inputs', help='dir with in_<name>.npy per model input')
  p.add_argument('--capture', help='replay a verify_parity capture dir and require bit-identity with out_link_*')
  p.add_argument('--ref', help='reference output .npy')
  p.add_argument('--spec', help='model spec json: compare per output slice instead of one number')
  p.add_argument('--iters', type=int, default=50)
  args = p.parse_args()

  if not args.capture and not args.inputs:
    p.error('one of --inputs or --capture is required')

  backend = select(args.backend, args.device)
  info = backend.describe()
  print(f"backend {info['backend']} {info['runtime_version']} on {info['device']}")
  t0 = time.perf_counter()
  engine = backend.load(Path(args.engine))
  print(f'loaded in {time.perf_counter() - t0:.1f} s')
  if args.capture:
    return replay_capture(engine, Path(args.capture))
  print('engine inputs:')
  for n, b in engine.inputs.items():
    print(f'  {n:<20} {str(tuple(b.shape)):<24} {b.dtype}')

  in_dir = Path(args.inputs)
  values = {}
  for name, b in engine.inputs.items():
    f = in_dir / f'in_{name}.npy'
    if not f.exists():
      print(f'missing {f}', file=sys.stderr)
      return 1
    values[name] = np.load(f).astype(b.dtype, copy=False).reshape(b.shape)

  out = next(iter(infer(engine, values).values())).astype(np.float32).reshape(-1)
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
      passed = report_slices(spec, out, ref)
      bad = sorted(k for k, ok in passed.items() if not ok)
      if bad:
        print(f'FAIL: {len(bad)} slice(s) below corr {MIN_CORR}, whole or in a column: ' + ', '.join(bad),
              file=sys.stderr)
        return 2
      print(f'OK: every slice and every column at or above corr {MIN_CORR}')
    else:
      # a low whole-vector number is a certain failure; a high one proves nothing
      # about the 2068 head values that steer
      print('no --spec: this checks load, shape and finiteness only. The whole-vector '
            'correlation is mostly hidden_state and passes with a wrong head; pass '
            '--spec to compare per slice.')
      if corr < MIN_CORR:
        print('CORRELATION TOO LOW', file=sys.stderr)
        return 2

  # the server runs the engine warmed (a CUDA graph replay on TensorRT), so
  # that is the path worth timing
  print(engine.warm())
  ts = []
  for _ in range(args.iters):
    t0 = time.perf_counter()
    engine.run()
    ts.append((time.perf_counter() - t0) * 1e3)
  ts = np.array(ts[5:])
  print(f'engine-only latency over {len(ts)}: mean {ts.mean():.2f} ms  '
        f'min {ts.min():.2f}  p50 {np.percentile(ts,50):.2f}  '
        f'p95 {np.percentile(ts,95):.2f}  max {ts.max():.2f}')
  engine.close()
  return 0


if __name__ == '__main__':
  sys.exit(main())
