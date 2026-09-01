#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Measure the round trip the car actually depends on.

Bandwidth was never the question - 520 KB/frame at 20 Hz is 85 Mbit/s against
USB 3's 5 Gbit/s. Jitter is. modeld has a 50 ms budget per frame and the model
alone takes ~21 ms, so what matters is the whole distribution, not the mean.
This sends real-sized payloads at the real rate and reports the tail.

    # against a Jetson on the LAN
    python3 scripts/bench_link.py --host 192.168.1.87 --sha256 <hex> --n 400

    # over the USB gadget, from the comma
    python3 scripts/bench_link.py --usb --sha256 <hex>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from jetlink.client import JetlinkClient
from jetlink.spec import DEFAULT_FRAME_SKIP, ModelSpec

# The big (chestnut) model as shipped. Override with --spec if it changes.
BIG_SHAPES = {
  'img': (1, 12, 128, 256),
  'big_img': (1, 12, 128, 256),
  'desire_pulse': (1, 33, 8),
  'traffic_convention': (1, 2),
  'action_t': (1, 2),
  'features_buffer': (1, 32, 32, 512),
}
BIG_OUTPUT = {'outputs': (1, 18452)}
BIG_SLICES = {'hidden_state': slice(2066, 18450)}


def load_spec(args) -> ModelSpec:
  if args.spec:
    d = json.loads(Path(args.spec).read_text())
    return ModelSpec(
      sha256=d['sha256'], nbytes=d['nbytes'], frame_skip=d.get('frame_skip', DEFAULT_FRAME_SKIP),
      input_shapes={k: tuple(v) for k, v in d['input_shapes'].items()},
      output_shapes={k: tuple(v) for k, v in d['output_shapes'].items()},
      output_slices={k: slice(*v) for k, v in d['output_slices'].items()},
      checkpoint=d.get('checkpoint'))
  return ModelSpec(sha256=args.sha256, nbytes=args.nbytes, frame_skip=DEFAULT_FRAME_SKIP,
                   input_shapes=BIG_SHAPES, output_shapes=BIG_OUTPUT,
                   output_slices=BIG_SLICES, checkpoint=None)


def pct(a: np.ndarray, q: float) -> float:
  return float(np.percentile(a, q))


def main() -> int:
  p = argparse.ArgumentParser()
  g = p.add_mutually_exclusive_group(required=True)
  g.add_argument('--host', help='TCP host of the Jetson')
  g.add_argument('--usb', action='store_true', help='use the USB gadget')
  p.add_argument('--port', type=int, default=5599)
  p.add_argument('--sha256', help='sha256 of an already-cached model')
  p.add_argument('--nbytes', type=int, default=765953504)
  p.add_argument('--spec', help='json spec file (overrides --sha256/--nbytes)')
  p.add_argument('--onnx', help='upload and build this model if the server lacks it')
  p.add_argument('--n', type=int, default=400)
  p.add_argument('--rate', type=float, default=20.0, help='Hz; 0 = as fast as possible')
  p.add_argument('--deadline', type=float, default=0.2, help='per-frame timeout, seconds')
  args = p.parse_args()

  client = (JetlinkClient.open_usb(deadline=args.deadline) if args.usb
            else JetlinkClient.open_tcp(args.host, args.port, deadline=args.deadline))
  hello = client.hello()
  print(f"server: trt {hello['trt_version']} on {hello['device']}, "
        f"engine {hello['engine_state']}")

  spec = load_spec(args)
  t0 = time.time()
  client.ensure_engine(args.onnx or '/nonexistent', spec=spec,
                       progress=lambda s, f, m: print(f"  {s:<7} {f*100:5.1f}%  {m}"))
  print(f"engine ready in {time.time() - t0:.1f}s")

  rng = np.random.default_rng(0)
  warped = rng.integers(0, 256, spec.warped_shape, dtype=np.uint8)
  packed = np.zeros(spec.packed_nelem, np.float32)
  hidden = spec.output_slices['hidden_state']

  lat, gpu, queue, srv = [], [], [], []
  period = 1.0 / args.rate if args.rate > 0 else 0.0
  next_t = time.perf_counter()
  for i in range(args.n):
    if period:
      now = time.perf_counter()
      if next_t > now:
        time.sleep(next_t - now)
      next_t += period
    t = time.perf_counter()
    out = client.infer(warped, packed, frame_id=i, reset=(i == 0), deadline=args.deadline)
    lat.append((time.perf_counter() - t) * 1e3)
    # Feed the hidden state back exactly as modeld does, so the queues see a
    # realistic sequence rather than a constant.
    packed[-(hidden.stop - hidden.start):] = out[hidden]
    g_us, q_us, t_us = client.last_timings
    gpu.append(g_us / 1e3)
    queue.append(q_us / 1e3)
    srv.append(t_us / 1e3)

  a = np.array(lat[10:])
  print(f"\npayload: {spec.infer_req_nbytes/1e3:.0f} KB up, "
        f"{spec.infer_resp_nbytes/1e3:.0f} KB down, "
        f"{(spec.infer_req_nbytes+spec.infer_resp_nbytes)*args.rate*8/1e6:.0f} Mbit/s at {args.rate:g} Hz")
  print(f"round trip over {len(a)} frames (ms):")
  print(f"  mean {a.mean():6.2f}  min {a.min():6.2f}  p50 {pct(a,50):6.2f}  "
        f"p90 {pct(a,90):6.2f}  p99 {pct(a,99):6.2f}  max {a.max():6.2f}")
  print(f"  jitter: p99-p50 {pct(a,99)-pct(a,50):5.2f}  stdev {a.std():5.2f}")
  s = np.array(srv[10:])
  print(f"server-side total {np.mean(s):6.2f} ms  (gpu {np.mean(gpu[10:]):5.2f}, "
        f"queues {np.mean(queue[10:]):5.2f})")
  print(f"transport overhead: {a.mean() - s.mean():.2f} ms mean")
  over = int((a > 50).sum())
  print(f"frames over the 50 ms budget: {over}/{len(a)} ({100*over/len(a):.1f}%)")
  client.close()
  return 0


if __name__ == '__main__':
  sys.exit(main())
