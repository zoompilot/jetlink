#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Inference server entrypoint: a Jetson in the car, or any machine with a GPU.

    # as the USB host, which is how the Jetson runs in the car
    python3 -m jetlink.server.main --transport usb

    # over ethernet, for development and benchmarking
    python3 -m jetlink.server.main --transport tcp --port 5599

    # on a Mac, tinygrad on Metal; scripts/run-mac.sh wraps this
    python3 -m jetlink.server.main --backend tinygrad --device METAL --transport tcp

    # build an engine ahead of time, no client needed
    python3 -m jetlink.server.main --build /path/to/big_driving_supercombo.onnx

--backend auto picks TensorRT where it imports, then tinygrad, then
onnxruntime; docs/platforms.md has the measured frame times and start-up
costs behind that order.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from jetlink.server import platform
from jetlink.server.backends import NAMES, available, select
from jetlink.server.cache import EngineCache
from jetlink.server.session import EngineHost, Session
from jetlink.server.sleep import SLEEP_AFTER, Sleeper
from jetlink.server.telemetry import pick_source
from jetlink.transport.base import LinkError

log = logging.getLogger('jetlink.server')

# Longer than the client's FRAME_TIMEOUT, so it is the one that gives up.
DRAIN_TIMEOUT = 5.0


def _serve(cache: EngineCache, open_transport, sleeper: Sleeper | None = None) -> None:
  """Serve one client at a time forever.

  `open_transport()` returns a transport, or None to wait and retry; the three
  transports differ only in how they open. The engine host is shared across
  sessions: the comma reconnects at every handover and the engine must not
  reload. With a `sleeper`, a long run of None suspends the box; see sleep.py.
  """
  host = EngineHost(cache, pick_source(cache.backend.name))
  # before accepting anything, so two callers cannot race to start GPU loads
  host.preload()
  try:
    while True:
      transport = open_transport()
      if transport is None:
        if sleeper is not None and sleeper.idle():
          continue  # just woke up; look for the gadget right away
        time.sleep(2.0)
        continue
      if sleeper is not None:
        sleeper.touch()
      session = Session(transport, host)
      try:
        session.serve_forever()
      except LinkError as e:
        log.info("session ended: %s", e)
      finally:
        session.close()
        if getattr(transport, '_desynced', False):
          # The client is still mid-message. Let it finish and time out rather
          # than reopening under it; see StreamTransport.drain.
          transport.drain(DRAIN_TIMEOUT)
        transport.close()
        if sleeper is not None:
          sleeper.touch()
        log.info("client disconnected")
  finally:
    # Ctrl-C or a stop: release the engine on the thread that owns it rather
    # than leaving it to interpreter teardown, which some runtimes survive
    # less well than others (backends/tinygrad/owner.py).
    host.close()


def _tcp_opener(args):
  from jetlink.transport.tcp import TcpTransport
  srv = TcpTransport.listen(args.host, args.port)
  log.info("listening on %s:%d", args.host, args.port)

  def open_transport():
    transport, addr = TcpTransport.accept(srv)
    log.info("client connected from %s", addr)
    return transport
  return open_transport


def _usb_opener(args, sleeper: Sleeper | None = None):
  """This end is the USB host. Needs no kernel driver: libusb uses usbfs."""
  from jetlink.transport.usbbulk import UsbBulkTransport

  def open_transport():
    if not UsbBulkTransport.present(args.vid, args.pid):
      log.warning("waiting for a jetlink gadget at %04x:%04x", args.vid, args.pid)
      return None
    if sleeper is not None:
      # Present but not yet openable still means the comma is there; sleeping
      # now needs it to bounce the gadget before anything wakes the box.
      sleeper.touch()
    try:
      transport = UsbBulkTransport.open(args.vid, args.pid, timeout_ms=args.usb_timeout_ms)
      log.info("client connected over usb")
      return transport
    except Exception as e:
      # Broad on purpose: this loop is the only supervisor, and anything that
      # escapes it turns a retry into a container crash loop.
      log.warning("could not open the gadget: %s", e)
      return None
  return open_transport


def _ffs_opener(args):
  """This end is the USB gadget."""
  from jetlink.transport.ffs import FfsTransport
  mount = Path(args.ffs_mount)

  def open_transport():
    if not (mount / 'ep0').exists():
      log.warning("waiting for functionfs at %s (run scripts/setup_gadget.sh)", mount)
      return None
    try:
      # This writes the descriptors and binds the UDC; either can fail
      # transiently, and returning None just retries.
      return FfsTransport(str(mount), gadget=args.gadget, udc=args.udc)
    except Exception as e:
      log.warning("could not open the gadget: %s", e)
      return None
  return open_transport


OPENERS = {'tcp': _tcp_opener, 'usb': _usb_opener, 'ffs': _ffs_opener}


def main(argv=None) -> int:
  p = argparse.ArgumentParser(description='jetlink inference server')
  p.add_argument('--backend', choices=('auto', *NAMES), default='auto',
                 help='what runs the model: trt (TensorRT), ort (onnxruntime: CoreML, CUDA or '
                      'CPU), tinygrad. auto takes the first that comes up, in that order')
  p.add_argument('--device', default='auto',
                 help='backend-specific: a CUDA device index for trt; METAL, CUDA, NV, AMD or '
                      'CPU for tinygrad; coreml, cuda or cpu for ort')
  p.add_argument('--list-backends', action='store_true',
                 help='print the backends whose runtime is installed here, and exit')
  p.add_argument('--transport', choices=('tcp', 'usb', 'ffs'), default='tcp',
                 help='usb = this end is the USB host (the usual case for a Jetson); '
                      'ffs = this end is the USB gadget')
  p.add_argument('--host', default='0.0.0.0')
  p.add_argument('--port', type=int, default=5599)
  p.add_argument('--ffs-mount', default='/dev/ffs-jetlink')
  p.add_argument('--gadget', default='/sys/kernel/config/usb_gadget/jetlink',
                 help='configfs gadget to bind once descriptors are written')
  p.add_argument('--udc', default=None, help='UDC name (default: the first one)')
  p.add_argument('--vid', type=lambda x: int(x, 0), default=0x1209)
  p.add_argument('--pid', type=lambda x: int(x, 0), default=0x0001)
  p.add_argument('--usb-timeout-ms', type=int, default=2000)
  p.add_argument('--sleep-after', type=float, default=0.0, metavar='SECONDS',
                 help='suspend the box (deep, USB wakes it) after this long with no '
                      f'gadget; 0 = never. In the car use {SLEEP_AFTER:.0f}. '
                      'Needs /sys/power writable in the container.')
  p.add_argument('--cache', default=str(platform.default_cache_dir()),
                 help='engines and uploaded models; JETLINK_CACHE sets the default')
  p.add_argument('--build', metavar='ONNX', help='build an engine and exit')
  p.add_argument('--dump-spec', metavar='ONNX',
                 help='write this model\'s spec as json to stdout and exit')
  p.add_argument('--log-level', default='INFO')
  args = p.parse_args(argv)

  logging.basicConfig(
    level=getattr(logging, args.log_level.upper(), logging.INFO),
    format='%(asctime)s %(levelname)-7s %(name)s: %(message)s')

  if args.dump_spec:
    import json

    from jetlink.spec import spec_from_onnx
    print(json.dumps(spec_from_onnx(args.dump_spec).to_dict()))
    return 0

  if args.list_backends:
    found = available()
    print('\n'.join(found) if found else 'none: pip install jetlink[trt], jetlink[ort] or jetlink[tinygrad]')
    return 0

  backend = select(args.backend, args.device)
  info = backend.describe()
  log.info("backend %s %s on %s, cache %s", info['backend'], info['runtime_version'],
           info['device'], args.cache)
  cache = EngineCache(Path(args.cache), backend)

  if args.build:
    from jetlink.spec import sha256_file, spec_from_onnx
    sha, nbytes = sha256_file(args.build)
    entry = cache.entry(sha)
    log.info("model %s (%d MB) -> %s", sha[:16], nbytes >> 20, entry.path)
    if entry.exists:
      log.info("already built: %s", entry.meta())
      return 0

    last = [0.0]

    def report(stage, frac, msg):
      if frac - last[0] >= 0.02 or frac >= 1.0 or frac == 0.0:
        last[0] = frac
        log.info("%-6s %5.1f%%  %s", stage, frac * 100, msg)

    # Carry the spec into the sidecar like a served build does, so the first
    # client to connect loads the artifact instead of reparsing the ONNX for it.
    backend.build(Path(args.build), entry.path, report=report,
                  meta_extra={'spec': spec_from_onnx(args.build).to_dict()})
    log.info("built: %s", entry.meta())
    return 0

  from jetlink.server.power import clear_stale_flag
  clear_stale_flag(cache.root)

  sleeper = None
  if args.sleep_after > 0:
    if args.transport != 'usb':
      # tcp blocks in accept and never sees an absent client; ffs is the
      # Jetson-as-gadget inversion, where the host end is the one that sleeps.
      p.error('--sleep-after only makes sense with --transport usb')
    if not platform.can_suspend():
      # macOS and Windows own their own sleep, and a laptop lid is not a USB
      # edge that wakes anything; the flag is the Jetson's.
      p.error('--sleep-after needs /sys/power, which this host has not got')
    sleeper = Sleeper(args.sleep_after)
    log.info("will suspend after %.0f s without a gadget", args.sleep_after)
  opener = _usb_opener(args, sleeper) if args.transport == 'usb' else OPENERS[args.transport](args)
  try:
    _serve(cache, opener, sleeper)
  except KeyboardInterrupt:
    log.info("stopped")
  return 0


if __name__ == '__main__':
  sys.exit(main())
