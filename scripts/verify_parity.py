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
ONNX, per output slice and per column within a slice, so a regression lands on
a named head rather than in an 18452-wide vector.

What is independently checked, and what is not. The reference is onnxruntime
on the ONNX the engine was built from, so everything the server does to that
graph and to its output is covered: the UINT8->FP16 surgery in onnx_patch, the
TensorRT build and its FP16 accumulation, the wire format and the output
slicing. The queues are not. reference() feeds onnxruntime through the same
jetlink.queues.PolicyQueues the server runs, so a queue bug is applied
identically on both sides and cannot show up here. tests/test_queues.py is the
queue check: it compares against openpilot's own tinygrad implementation and
passes on the comma, where both are importable.

What the numbers can resolve. The graph is float16 end to end (weights, the
output tensor, the reference's own output), so TensorRT and onnxruntime are
two float16 implementations that differ in accumulation order, not a half
precision engine against a full precision truth. Measured on Cinque Terre,
2026-09-05: the disagreement is roughly an absolute 0.005 to 0.03 across the
2068 head values, with lead x (~130 m) worst at 0.875, seven float16 steps. A
column whose values spread less than that cannot be judged by correlation,
and a slice of three values cannot be judged by correlation on one frame; see
MIN_SAMPLES and CONSTANT_FRACTION. The transport was ruled out the same day by
`verify_engine.py --capture`, which replays a capture through the plan on the
Jetson and demands the bytes the comma received, bit for bit.

    # 1. on the comma, over the cable (stop jetlinkd first, it owns the link).
    #    the server returns the spec of a model it already has, so only the
    #    model's identity is needed; --spec overrides it
    python3 scripts/verify_parity.py capture --ffs --dir out --sha256 <hex> --nbytes <n>

    # 2. anywhere with onnxruntime, on the ONNX the engine was built from
    python3 scripts/verify_parity.py reference --onnx big.onnx --dir out

    # 3. either machine
    python3 scripts/verify_parity.py compare --dir out
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from jetlink.spec import ModelSpec

# Two float16 implementations of a 40-layer network, so exact equality is not
# the bar. Correlation is: it moves the moment a head is wired up wrong,
# transposed or fed a stale queue, all of which an absolute tolerance would
# wave through.
MIN_CORR = 0.999

# Correlation needs samples. Per frame, lead_prob is three logits and every
# pose, euler and road_transform column is one value: three points correlate at
# 0.998 over float16 rounding, and one point correlates at exactly 1.0 with
# anything, which is how those columns went unchecked until 2026-09-05. Slices
# and columns are gated on all captured frames pooled, a column with fewer
# pooled values than this is reported but not gated, and only a slice at least
# this wide is gated per frame as well. Measured on the 2026-09-05 capture, the
# one-value-per-frame columns read 0.95 pooled over 4 frames and 0.9993 over
# 16, so capture's default is 32 frames and compare warns below 16.
MIN_SAMPLES = 16

# A column whose reference values spread less than this fraction of its slice
# is constant at float16 resolution (2^-11 relative). wide_from_device_euler's
# roll is ~1e-6 rad beside pitch and yaw of ~7, and correlating its rounding
# noise against the reference's read 0.9877. Such a column is held to an
# absolute error inside that same fraction of the slice spread instead.
CONSTANT_FRACTION = 1e-3

# How openpilot's Parser reads each regression head (parse_model_outputs.py):
# the raw slice is `hypotheses` blocks of [mu | std | selection], mu and std
# each a matrix whose last axis is `columns` wide. That last axis is where the
# units mix: a plan row is position in metres (up to ~200) beside velocity,
# acceleration, orientation and rate, and the stds beside all of them. A
# whole-slice correlation is set by whichever column is largest, so each is
# checked on its own. The spec only carries the flat slices, which is why the
# layout lives here. A head whose slice fits none of its listed layouts is
# compared whole, and says so.
MDN_LAYOUTS = {
  'plan': [(0, 0, 15), (5, 1, 15)],
  'lane_lines': [(0, 0, 2)],
  'road_edges': [(0, 0, 2)],
  'lead': [(0, 0, 4), (2, 3, 4)],
  'pose': [(0, 0, 6)],
  'wide_from_device_euler': [(0, 0, 3)],
  'road_transform': [(0, 0, 6)],
  'sim_pose': [(0, 0, 6)],
}


def load_spec_file(path: str | Path) -> ModelSpec:
  return ModelSpec.from_dict(json.loads(Path(path).read_text()))


def load_spec(args) -> ModelSpec | None:
  """--spec wins; otherwise the copy capture wrote next to the frames."""
  if args.spec:
    return load_spec_file(args.spec)
  path = Path(args.dir) / 'spec.json'
  return load_spec_file(path) if path.exists() else None


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
        # Lateral and longitudinal action horizons, in seconds: modeld sends
        # the actuator delays plus a frame, 0.2 to 0.4 s on a real car. This
        # used to ramp 0.05 s a frame, which at 32 frames asked for a 1.6 s
        # horizon; by frame 20 the plan ran backwards at -74 m and the two
        # float16 implementations disagreed by metres there. Bounded, so the
        # model stays in distribution however many frames are captured.
        packed[off:off + size] = np.array([0.25 + 0.1 * np.sin(0.5 * i), 0.35 + 0.1 * np.cos(0.5 * i)][:size])
      elif name == 'desire':
        # A pulse every eighth frame, cycling through the seven real desires;
        # index 0 is "none" and modeld zeroes it. One pulse a frame was not a
        # drive anyone takes.
        if i % 8 == 0:
          packed[off + 1 + (i // 8) % (size - 1)] = 1.0
      off += size
    frames.append((warped, packed))
  return frames


def hidden_slice(spec: ModelSpec) -> slice:
  return spec.output_slices['hidden_state']


# -- capture: what actually comes back over the link -------------------------

def capture(args) -> int:
  from jetlink.client import JetlinkClient

  spec = load_spec_file(args.spec) if args.spec else None
  if spec is not None:
    sha256, nbytes = spec.sha256, spec.nbytes
  elif args.sha256 and args.nbytes:
    sha256, nbytes = args.sha256, args.nbytes
  else:
    raise SystemExit("capture needs --spec, or --sha256 and --nbytes of a model the server already has")
  out = Path(args.dir)
  out.mkdir(parents=True, exist_ok=True)

  if args.ffs:
    client = JetlinkClient.open_ffs(args.ffs_mount, gadget=args.gadget)
  elif args.host:
    client = JetlinkClient.open_tcp(args.host, args.port)
  else:
    client = JetlinkClient.open_usb()

  try:
    hello = client.hello(timeout=60.0)  # the jetson may still be re-enumerating
    print(f"server: trt {hello['trt_version']} on {hello['device']}")
    # No ONNX on this end: a model the server does not have is a build job,
    # not a parity check.
    spec = client.ensure_engine(sha256, nbytes, build_timeout=300.0)
    # reference and compare need the same slices this capture was made with.
    (out / 'spec.json').write_text(json.dumps(spec.to_dict()))
    frames = make_inputs(spec, args.n, args.seed)

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

_ORT_DTYPES = {
  'tensor(uint8)': np.uint8,
  'tensor(int32)': np.int32,
  'tensor(int64)': np.int64,
  'tensor(float16)': np.float16,
  'tensor(float)': np.float32,
  'tensor(double)': np.float64,
  'tensor(bool)': np.bool_,
}


def _ort_feed_dtypes(sess) -> dict[str, np.dtype]:
  """What the graph declares for each input, not a guess about which need casting."""
  dtypes = {}
  for i in sess.get_inputs():
    if i.type not in _ORT_DTYPES:
      raise SystemExit(f"input {i.name} is {i.type}; add it to _ORT_DTYPES")
    dtypes[i.name] = _ORT_DTYPES[i.type]
  return dtypes


def reference(args) -> int:
  import onnxruntime as ort

  from jetlink.queues import PolicyQueues

  spec = load_spec(args)
  if spec is None:
    from jetlink.spec import spec_from_onnx
    spec = spec_from_onnx(args.onnx)
  d = Path(args.dir)

  sess = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
  # The ONNX is untouched, so its image inputs are still UINT8 - that is the
  # whole point of comparing against it - while the queues hand back the FP16
  # the patched graph wants. 0..255 is exact in both, so the cast is lossless.
  # The other inputs are whatever the graph says they are.
  dtypes = _ort_feed_dtypes(sess)
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
    missing = set(dtypes) - set(feed)
    if missing:
      raise SystemExit(f"the queues produced no value for graph input(s) {sorted(missing)}")
    feed = {k: np.ascontiguousarray(v, dtype=dtypes[k]) for k, v in feed.items()}
    out = np.asarray(sess.run(None, feed)[0], np.float32).reshape(-1)
    np.save(d / f'out_ref_{i}.npy', out)
    print(f"  frame {i}: {out.shape[0]} values, finite={bool(np.all(np.isfinite(out)))}")
    prev_hidden = out[hid]
  return 0


# -- compare ------------------------------------------------------------------

def _corr(a: np.ndarray, b: np.ndarray) -> float:
  if a.std() == 0 and b.std() == 0:
    # Nothing varies on either side. A head the model holds constant, or a
    # one-value column, is a pass, not a division by zero.
    return 1.0
  if a.std() == 0 or b.std() == 0:
    return 1.0 if np.allclose(a, b) else 0.0
  return float(np.corrcoef(a, b)[0, 1])


def columns(name: str, a: np.ndarray) -> dict[str, np.ndarray] | None:
  """Split a raw head into unit-homogeneous columns, or None if its layout is unknown."""
  for hyp, sel, width in MDN_LAYOUTS.get(name, ()):
    rows = max(hyp, 1)
    if a.size % rows:
      continue
    raw = a.reshape(rows, -1)
    n = (raw.shape[1] - sel) // 2
    if n <= 0 or (raw.shape[1] - sel) % 2 or n % width:
      continue
    mu = raw[:, :n].reshape(-1, width)
    std = raw[:, n:2 * n].reshape(-1, width)
    cols = {f'mu[{j}]': mu[:, j] for j in range(width)}
    cols.update({f'std[{j}]': std[:, j] for j in range(width)})
    if sel:
      cols['sel'] = raw[:, 2 * n:].reshape(-1)
    return cols
  return None


def _frames(x) -> list[np.ndarray]:
  """One array or a list of per-frame arrays, as flat float32 frames."""
  seq = x if isinstance(x, (list, tuple)) else [x]
  return [np.asarray(f, np.float32).reshape(-1) for f in seq]


def report_slices(spec: ModelSpec, links, refs) -> dict[str, bool]:
  """One line per output slice with every frame pooled. Returns whether each passed.

  A slice passes when its pooled correlation and every column's clear MIN_CORR.
  A column flatter than CONSTANT_FRACTION of its slice is judged on absolute
  error within that fraction instead, because correlation over what is left of
  it is rounding noise against rounding noise.
  """
  links, refs = _frames(links), _frames(refs)
  passed = {}
  for name, sl in sorted(spec.output_slices.items()):
    a = np.concatenate([x[sl] for x in links])
    b = np.concatenate([y[sl] for y in refs])
    whole = _corr(a, b)
    ok = whole >= MIN_CORR
    detail = f"{'(compared whole)':38}"
    cols_a = [columns(name, x[sl]) for x in links]
    if cols_a[0]:
      cols_b = [columns(name, y[sl]) for y in refs]
      bound = CONSTANT_FRACTION * b.std()
      worst_c, worst_k, failed, flat, thin = 2.0, '', [], 0, 0
      for k in cols_a[0]:
        ca = np.concatenate([c[k] for c in cols_a])
        cb = np.concatenate([c[k] for c in cols_b])
        if ca.size < MIN_SAMPLES:
          thin += 1
          continue
        c = _corr(ca, cb)
        if cb.std() < bound:
          flat += 1
          if np.abs(ca - cb).max() > bound:
            failed.append(f'{k} flat, max abs {np.abs(ca - cb).max():.4g} > {bound:.4g}')
          continue
        if c < worst_c:
          worst_c, worst_k = c, k
        if c < MIN_CORR:
          failed.append(f'{k} {c:.6f}')
      ok &= not failed
      notes = ([f'{flat} flat'] if flat else []) + ([f'{thin} under {MIN_SAMPLES} samples, not gated'] if thin else [])
      worst = f"worst col {worst_c:8.6f} {worst_k:8}" if worst_k else f"{'no column gated':27}"
      detail = f"{worst} {f'({', '.join(notes)})' if notes else '':9}"
      if failed:
        detail += '  cols: ' + ', '.join(failed)
    passed[name] = ok
    flag = '' if ok else '   <-- FAIL'
    print(f"    {name:24} corr {whole:8.6f}  {detail}  max abs {np.abs(a - b).max():8.4f}  "
          f"mean abs {np.abs(a - b).mean():7.5f}{flag}")
  return passed


def compare(args) -> int:
  spec = load_spec(args)
  if spec is None:
    raise SystemExit(f"no spec: pass --spec, or run capture first (it writes {Path(args.dir) / 'spec.json'})")
  d = Path(args.dir)
  n = len(sorted(d.glob('out_link_*.npy')))
  if not n:
    raise SystemExit(f"no captured outputs in {d}")

  links, refs = [], []
  for i in range(n):
    link = np.load(d / f'out_link_{i}.npy').reshape(-1)
    ref_path = d / f'out_ref_{i}.npy'
    if not ref_path.exists():
      raise SystemExit(f"missing {ref_path}; run reference first")
    ref = np.load(ref_path).reshape(-1)
    m = min(len(link), len(ref))
    links.append(link[:m])
    refs.append(ref[:m])

  # Per frame, the whole-slice correlation: a stale queue or a dropped reset
  # shows on the frame it happens to. Slices too small to correlate on one
  # frame are only gated pooled, below.
  frame_fail: dict[str, float] = {}
  for i, (link, ref) in enumerate(zip(links, refs)):
    print(f"\nframe {i}: corr {_corr(link, ref):.6f}  max abs {np.abs(link - ref).max():.4f}")
    for name, sl in sorted(spec.output_slices.items()):
      a, b = link[sl], ref[sl]
      c = _corr(a, b)
      gated = sl.stop - sl.start >= MIN_SAMPLES
      if gated and c < MIN_CORR:
        frame_fail[name] = min(frame_fail.get(name, 2.0), c)
      flag = '   <-- FAIL' if gated and c < MIN_CORR else ('' if gated else '   (pooled only)')
      print(f"    {name:24} corr {c:8.6f}  max abs {np.abs(a - b).max():8.4f}  "
            f"mean abs {np.abs(a - b).mean():7.5f}{flag}")

  print(f"\npooled over {n} frames, per slice and per column:")
  if n < MIN_SAMPLES:
    print(f"    only {n} frames: columns with one value per frame (pose, euler, road_transform) "
          f"have too few samples to gate; capture at least {MIN_SAMPLES}")
  passed = report_slices(spec, links, refs)

  bad = [f'{k} {v:.6f} on one frame' for k, v in sorted(frame_fail.items(), key=lambda kv: kv[1])]
  bad += [f'{k} pooled' for k, ok in passed.items() if not ok and k not in frame_fail]
  if bad:
    print(f"\nFAIL: {len(bad)} slice(s) below corr {MIN_CORR}, whole or in a column: " + ', '.join(bad))
    return 1
  print(f"\nOK: every slice and every column at or above corr {MIN_CORR}, "
        f"per frame and pooled over all {n} frames")
  return 0


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('mode', choices=('capture', 'reference', 'compare'))
  p.add_argument('--spec', help='model spec json; overrides the one capture writes to --dir')
  p.add_argument('--sha256', help='capture mode: model identity, when the server already has it')
  p.add_argument('--nbytes', type=int, help='capture mode: ONNX size in bytes, with --sha256')
  p.add_argument('--dir', default='parity')
  p.add_argument('--n', type=int, default=32,
                 help=f'capture mode: frames; pose-like columns have one value per frame and need {MIN_SAMPLES}+')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--onnx', help='reference mode: the ONNX the engine was built from')
  p.add_argument('--ffs', action='store_true', help='capture mode: this end is the gadget')
  p.add_argument('--ffs-mount', default='/dev/ffs-jetlink')
  p.add_argument('--gadget', default='/sys/kernel/config/usb_gadget/jetlink')
  p.add_argument('--host', help='capture mode: TCP host instead of USB')
  p.add_argument('--port', type=int, default=5599)
  args = p.parse_args()

  if args.mode == 'reference' and not args.onnx:
    raise SystemExit('reference mode needs --onnx')
  return {'capture': capture, 'reference': reference, 'compare': compare}[args.mode](args)


if __name__ == '__main__':
  sys.exit(main())
