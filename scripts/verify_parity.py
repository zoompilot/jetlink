#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Does the link return the same numbers the model would?

bench_link.py proves the round trip is fast enough. It says nothing about
whether the answer is right, and a TensorRT engine that runs at 20 Hz while
producing subtly wrong outputs steers the car with no error anywhere. This
compares what comes back over the cable against onnxruntime on the unmodified
ONNX, per output slice, so a regression lands on a named head rather than in an
18452-wide vector.

Everything between the two sides is covered: the queues, the UINT8->FP16 graph
surgery in onnx_patch, FP16 accumulation on the GPU, the wire format and the
output slicing.

    # 1. on the comma, over the cable (stop jetlinkd first, it owns the link)
    python3 scripts/verify_parity.py capture --spec spec.json --dir out

    # 2. anywhere with onnxruntime, on the ONNX the engine was built from
    python3 scripts/verify_parity.py reference --spec spec.json --onnx big.onnx --dir out

    # 3. either machine
    python3 scripts/verify_parity.py compare --spec spec.json --dir out
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from jetlink.spec import ModelSpec

# FP16 against FP32 on a 40-layer network, so exact equality is not the bar.
# Correlation is: it moves the moment a head is wired up wrong, transposed or
# fed a stale queue, all of which an absolute tolerance would wave through.
MIN_CORR = 0.999


def load_spec(path: str) -> ModelSpec:
  return ModelSpec.from_dict(json.loads(Path(path).read_text()))


def make_inputs(spec: ModelSpec, n: int, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
  """Deterministic stand-in frames, identical on both sides.

  Structured rather than uniform noise: white noise drives a vision network
  into activations no road ever produces, which is where FP16 and FP32 diverge
  most and least usefully. Gradients and moving blobs keep it in range while
  still exercising every channel.
  """
  rng = np.random.default_rng(seed)
  h, w = spec.model_hw
  yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
  frames = []
  for i in range(n):
    warped = np.empty(spec.warped_shape, np.uint8)
    for cam in range(warped.shape[0]):
      for ch in range(warped.shape[1]):
        phase = 0.7 * i + 1.3 * ch + 2.1 * cam
        base = (96 + 64 * np.sin(xx / 37.0 + phase) + 48 * np.cos(yy / 23.0 - phase)
                + 24 * np.sin((xx + yy) / 61.0))
        warped[cam, ch] = np.clip(base + rng.normal(0, 6, (h, w)), 0, 255).astype(np.uint8)
    packed = np.zeros(spec.packed_nelem, np.float32)
    off = 0
    for name, shape in spec.packed_shapes.items():
      size = int(np.prod(shape))
      if name == 'traffic_convention':
        packed[off:off + size] = np.array([1.0, 0.0][:size])
      elif name == 'action_t':
        packed[off:off + size] = 0.05 * (i + 1)
      elif name == 'desire':
        packed[off + (i % size)] = 1.0
      off += size
    frames.append((warped, packed))
  return frames


def hidden_slice(spec: ModelSpec) -> slice:
  return spec.output_slices['hidden_state']


# -- capture: what actually comes back over the link -------------------------

def capture(args) -> int:
  from jetlink.client import JetlinkClient

  spec = load_spec(args.spec)
  out = Path(args.dir)
  out.mkdir(parents=True, exist_ok=True)
  frames = make_inputs(spec, args.n, args.seed)

  if args.ffs:
    client = JetlinkClient.open_ffs(args.ffs_mount, gadget=args.gadget, deadline=args.deadline)
  elif args.host:
    client = JetlinkClient.open_tcp(args.host, args.port, deadline=args.deadline)
  else:
    client = JetlinkClient.open_usb(deadline=args.deadline)

  try:
    hello = client.hello(timeout=60.0)  # the jetson may still be re-enumerating
    print(f"server: trt {hello['trt_version']} on {hello['device']}")
    client.ensure_engine('/nonexistent', spec=spec, build_timeout=300.0)

    hid = hidden_slice(spec)
    for i, (warped, packed) in enumerate(frames):
      # The hidden state must carry between frames exactly as modeld carries
      # it, or frame 2 onward compares two different recurrences.
      result = client.infer(warped, packed, frame_id=i, reset=(i == 0))
      np.save(out / f'in_warped_{i}.npy', warped)
      np.save(out / f'in_packed_{i}.npy', packed)
      np.save(out / f'out_link_{i}.npy', np.asarray(result, np.float32))
      if i + 1 < len(frames):
        frames[i + 1][1][-(hid.stop - hid.start):] = result[hid]
      print(f"  frame {i}: {len(result)} values, "
            f"finite={bool(np.all(np.isfinite(result)))}")
  finally:
    client.close()
  print(f"wrote {args.n} frames to {out}")
  return 0


# -- reference: the same inputs through onnxruntime ---------------------------

def reference(args) -> int:
  import onnxruntime as ort

  from jetlink.queues import PolicyQueues

  spec = load_spec(args.spec)
  d = Path(args.dir)

  sess = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
  # The ONNX is untouched, so its image inputs are still UINT8 - that is the
  # whole point of comparing against it. The queues hand back the FP16 the
  # patched graph wants; 0..255 is exact in both, so the cast is lossless.
  dtypes = {i.name: i.type for i in sess.get_inputs()}
  queues = PolicyQueues(spec)
  queues.reset()

  n = len(sorted(d.glob('in_warped_*.npy')))
  if not n:
    raise SystemExit(f"no captured inputs in {d}; run capture first")

  hid = hidden_slice(spec)
  prev_hidden = None
  for i in range(n):
    warped = np.load(d / f'in_warped_{i}.npy')
    packed = np.load(d / f'in_packed_{i}.npy').copy()
    # Everything but the recurrence is the captured input, so both sides see
    # the same desire and action. The hidden state is our own previous output:
    # feeding the link's back would hide exactly the drift we are looking for.
    if prev_hidden is not None:
      packed[-(hid.stop - hid.start):] = prev_hidden
    feed = queues.step(warped, packed)
    feed = {k: (v.astype(np.uint8) if 'uint8' in dtypes[k] else v) for k, v in feed.items()}
    out = np.asarray(sess.run(None, feed)[0], np.float32).reshape(-1)
    np.save(d / f'out_ref_{i}.npy', out)
    print(f"  frame {i}: {out.shape[0]} values, finite={bool(np.all(np.isfinite(out)))}")
    prev_hidden = out[hid]
  return 0


# -- compare ------------------------------------------------------------------

def _corr(a: np.ndarray, b: np.ndarray) -> float:
  if a.std() == 0 or b.std() == 0:
    return 1.0 if np.allclose(a, b) else 0.0
  return float(np.corrcoef(a, b)[0, 1])


def compare(args) -> int:
  spec = load_spec(args.spec)
  d = Path(args.dir)
  n = len(sorted(d.glob('out_link_*.npy')))
  if not n:
    raise SystemExit(f"no captured outputs in {d}")

  worst = {}
  for i in range(n):
    link = np.load(d / f'out_link_{i}.npy').reshape(-1)
    ref_path = d / f'out_ref_{i}.npy'
    if not ref_path.exists():
      raise SystemExit(f"missing {ref_path}; run reference first")
    ref = np.load(ref_path).reshape(-1)
    m = min(len(link), len(ref))
    print(f"\nframe {i}: corr {_corr(link[:m], ref[:m]):.6f}  "
          f"max abs {np.abs(link[:m] - ref[:m]).max():.4f}")
    for name, sl in sorted(spec.output_slices.items()):
      a, b = link[sl], ref[sl]
      c = _corr(a, b)
      flag = '' if c >= MIN_CORR else '   <-- FAIL'
      print(f"    {name:24} corr {c:8.6f}  max abs {np.abs(a - b).max():8.4f}  "
            f"mean abs {np.abs(a - b).mean():7.5f}{flag}")
      if c < worst.get(name, 2.0):
        worst[name] = c

  bad = {k: v for k, v in worst.items() if v < MIN_CORR}
  if bad:
    print(f"\nFAIL: {len(bad)} slice(s) below corr {MIN_CORR}: "
          + ', '.join(f'{k} {v:.6f}' for k, v in sorted(bad.items(), key=lambda kv: kv[1])))
    return 1
  print(f"\nOK: every slice at or above corr {MIN_CORR} on all {n} frames")
  return 0


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('mode', choices=('capture', 'reference', 'compare'))
  p.add_argument('--spec', required=True)
  p.add_argument('--dir', default='parity')
  p.add_argument('--n', type=int, default=3)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--onnx', help='reference mode: the ONNX the engine was built from')
  p.add_argument('--ffs', action='store_true', help='capture mode: this end is the gadget')
  p.add_argument('--ffs-mount', default='/dev/ffs-jetlink')
  p.add_argument('--gadget', default='/sys/kernel/config/usb_gadget/jetlink')
  p.add_argument('--host', help='capture mode: TCP host instead of USB')
  p.add_argument('--port', type=int, default=5599)
  p.add_argument('--deadline', type=float, default=1.0)
  args = p.parse_args()

  if args.mode == 'reference' and not args.onnx:
    raise SystemExit('reference mode needs --onnx')
  return {'capture': capture, 'reference': reference, 'compare': compare}[args.mode](args)


if __name__ == '__main__':
  sys.exit(main())
