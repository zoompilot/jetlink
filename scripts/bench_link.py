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
    python3 scripts/bench_link.py --host 192.168.1.87 --onnx big_model.onnx --n 400

    # over the cable, from the comma (the comma is the gadget)
    python3 scripts/bench_link.py --ffs --spec spec.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from jetlink.client import JetlinkClient
from jetlink.spec import ModelSpec, spec_from_onnx

def load_spec(args) -> ModelSpec:
  """Take the spec from a file, or from the model itself."""
  if args.spec:
    return ModelSpec.from_dict(json.loads(Path(args.spec).read_text()))
  if args.onnx:
    return spec_from_onnx(args.onnx)
  raise SystemExit("need --spec or --onnx (the shapes come from the model)")


def _wait_for_host(timeout: float) -> None:
  """Block until a USB host has configured us.

  The gadget has to be bound before anything can enumerate it, and binding is
  what opening the transport does - so the wait belongs after the open, not
  before it. Until a host attaches, the UDC reports "not attached" and every
  read would simply time out.
  """
  udcs = list(Path('/sys/class/udc').glob('*/state'))
  deadline = time.monotonic() + timeout
  reported = False
  while time.monotonic() < deadline:
    for udc in udcs:
      try:
        if udc.read_text().strip() == 'configured':
          print(f"host attached ({udc.parent.name})")
          return
      except OSError:
        pass
    if not reported:
      print(f"waiting up to {timeout:.0f}s for a USB host to enumerate the gadget...")
      reported = True
    time.sleep(0.25)
  raise SystemExit("no USB host attached: check the cable")


def pct(a: np.ndarray, q: float) -> float:
  return float(np.percentile(a, q))


def main() -> int:
  p = argparse.ArgumentParser()
  g = p.add_mutually_exclusive_group(required=True)
  g.add_argument('--host', help='TCP host of the Jetson')
  g.add_argument('--usb', action='store_true',
                 help='this end is the USB host (libusb)')
  g.add_argument('--ffs', action='store_true',
                 help='this end is the USB gadget (FunctionFS) - use this on a comma')
  p.add_argument('--port', type=int, default=5599)
  p.add_argument('--ffs-mount', default='/dev/ffs-jetlink')
  p.add_argument('--gadget', default='/sys/kernel/config/usb_gadget/jetlink')
  p.add_argument('--wait-host', type=float, default=0.0, metavar='SECONDS',
                 help='gadget mode: wait for a host to enumerate us before starting')
  p.add_argument('--spec', help='json spec file, as written by --dump-spec')
  p.add_argument('--onnx', help='read the spec from this model, uploading it if the server lacks it')
  p.add_argument('--n', type=int, default=400)
  p.add_argument('--rate', type=float, default=20.0, help='Hz; 0 = as fast as possible')
  p.add_argument('--deadline', type=float, default=0.2, help='per-frame timeout, seconds')
  args = p.parse_args()

  if args.usb:
    client = JetlinkClient.open_usb(deadline=args.deadline)
  elif args.ffs:
    # On a comma the comma is the gadget: opening this writes the descriptors
    # and binds the UDC, so the Jetson can enumerate us.
    client = JetlinkClient.open_ffs(args.ffs_mount, gadget=args.gadget,
                                    deadline=args.deadline)
  else:
    client = JetlinkClient.open_tcp(args.host, args.port, deadline=args.deadline)
  if args.wait_host:
    _wait_for_host(args.wait_host)

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
