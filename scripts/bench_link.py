#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Measure the round trip the car depends on.

Jitter is the question, not bandwidth: modeld has 50 ms a frame and the model
takes ~21 ms of it. Sends real-sized payloads at the real rate and reports the tail.

    # against a Jetson on the LAN
    python3 scripts/bench_link.py --host 192.168.1.87 --onnx big_model.onnx --n 400

    # over the cable, from the comma (the comma is the gadget). the server
    # returns the spec of a model it already has, so its identity is enough
    python3 scripts/bench_link.py --ffs --sha256 <hex> --nbytes <n>
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

# modeld's per-frame budget; a frame past it is dropped, and frameDropPerc > 1 soft-disables
FRAME_BUDGET_MS = 50.0


def load_spec(args) -> ModelSpec | None:
  """Take the spec from a file or the model itself; None leaves it to the server."""
  if args.spec:
    return ModelSpec.from_dict(json.loads(Path(args.spec).read_text()))
  if args.onnx:
    return spec_from_onnx(args.onnx)
  return None


def _wait_for_host(timeout: float) -> None:
  """Block until a USB host has configured us.

  Opening the transport is what binds the gadget, so the wait belongs after the
  open: until a host attaches every read just times out.
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
  p.add_argument('--sha256', help='model identity, for a model the server already has')
  p.add_argument('--nbytes', type=int, help='ONNX size in bytes, with --sha256')
  p.add_argument('--n', type=int, default=400)
  p.add_argument('--rate', type=float, default=20.0, help='Hz; 0 = as fast as possible')
  args = p.parse_args()

  if args.usb:
    client = JetlinkClient.open_usb()
  elif args.ffs:
    # opening this writes the descriptors and binds the UDC, so the Jetson can enumerate us
    client = JetlinkClient.open_ffs(args.ffs_mount, gadget=args.gadget)
  else:
    client = JetlinkClient.open_tcp(args.host, args.port)
  try:
    return _run(args, client)
  finally:
    # a FunctionFS owner that dies leaves the gadget bound with nothing servicing it
    client.close()


def _run(args, client) -> int:
  if args.wait_host:
    _wait_for_host(args.wait_host)

  hello = client.hello()
  print(f"server: trt {hello['trt_version']} on {hello['device']}, "
        f"engine {hello['engine_state']}")

  spec = load_spec(args)
  if spec is not None:
    sha256, nbytes = spec.sha256, spec.nbytes
  elif args.sha256 and args.nbytes:
    sha256, nbytes = args.sha256, args.nbytes
  else:
    raise SystemExit("need --spec, --onnx, or --sha256 and --nbytes of a model the server already has")
  t0 = time.time()
  spec = client.ensure_engine(sha256, nbytes, onnx_path=args.onnx,
                              progress=lambda s, f, m: print(f"  {s:<7} {f*100:5.1f}%  {m}"))
  print(f"engine ready in {time.time() - t0:.1f}s")

  rng = np.random.default_rng(0)
  warped = rng.integers(0, 256, spec.warped_shape, dtype=np.uint8)
  packed = np.zeros(spec.packed_nelem, np.float32)
  hidden = spec.output_slices['hidden_state']

  lat, gpu, queue, srv = [], [], [], []
  send_ms, recv_ms = [], []
  period = 1.0 / args.rate if args.rate > 0 else 0.0
  next_t = time.perf_counter()
  for i in range(args.n):
    if period:
      now = time.perf_counter()
      if next_t > now:
        time.sleep(next_t - now)
      next_t += period
    t = time.perf_counter()
    seq = client.infer_begin(warped, packed, frame_id=i, reset=(i == 0))
    t_sent = time.perf_counter()
    out = client.infer_end(seq)
    t_done = time.perf_counter()
    lat.append((t_done - t) * 1e3)
    send_ms.append((t_sent - t) * 1e3)
    recv_ms.append((t_done - t_sent) * 1e3)
    # feed the hidden state back as modeld does, so the queues see a real sequence
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
  # a USB write returns once the host has taken the data, so send is wire time there;
  # a TCP send is a memcpy and the whole wire cost lands in recv
  snd, rcv = np.array(send_ms[10:]), np.array(recv_ms[10:])
  if args.host:
    print("  split (TCP): send is the copy into the socket buffer; both directions of wire time are in recv")
  else:
    print("  split (USB): the write blocks until the host has read, so send is request wire time, "
          "recv is server + reply")
  print(f"  send ({spec.infer_req_nbytes/1e3:.0f} KB up):   mean {snd.mean():6.2f}  p50 {pct(snd,50):6.2f}  max {snd.max():6.2f}")
  print(f"  recv ({spec.infer_resp_nbytes/1e3:.0f} KB down): mean {rcv.mean():6.2f}  p50 {pct(rcv,50):6.2f}  max {rcv.max():6.2f}")
  s = np.array(srv[10:])
  print(f"server-side total {np.mean(s):6.2f} ms  (gpu {np.mean(gpu[10:]):5.2f}, "
        f"queues {np.mean(queue[10:]):5.2f})")
  print(f"transport overhead: {a.mean() - s.mean():.2f} ms mean")
  over = int((a > FRAME_BUDGET_MS).sum())
  print(f"frames over the {FRAME_BUDGET_MS:.0f} ms budget: {over}/{len(a)} ({100*over/len(a):.1f}%)")
  return 0


if __name__ == '__main__':
  sys.exit(main())
