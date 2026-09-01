#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Jetson inference server entrypoint.

    # over USB, as it runs in the car (gadget must already be up)
    python3 -m jetlink.server.main --transport ffs

    # over ethernet, for development and benchmarking
    python3 -m jetlink.server.main --transport tcp --port 5599

    # build an engine ahead of time, no client needed
    python3 -m jetlink.server.main --build /path/to/big_driving_supercombo.onnx
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from jetlink.server.builder import DEFAULT_CACHE, EngineCache, build_engine
from jetlink.server.session import Session
from jetlink.server.telemetry import Telemetry
from jetlink.transport.base import LinkError

log = logging.getLogger('jetlink.server')


def _serve_tcp(args, cache: EngineCache) -> None:
  from jetlink.transport.tcp import TcpTransport
  srv = TcpTransport.listen(args.host, args.port)
  log.info("listening on %s:%d", args.host, args.port)
  while True:
    transport, addr = TcpTransport.accept(srv)
    log.info("client connected from %s", addr)
    try:
      Session(transport, cache, Telemetry()).serve_forever()
    except LinkError as e:
      log.info("session ended: %s", e)
    finally:
      transport.close()
      log.info("client disconnected")


def _serve_usb(args, cache: EngineCache) -> None:
  """This end is the USB host. Needs no kernel driver: libusb uses usbfs."""
  from jetlink.transport.usbbulk import UsbBulkTransport
  while True:
    if not UsbBulkTransport.present(args.vid, args.pid):
      log.warning("waiting for a jetlink gadget at %04x:%04x", args.vid, args.pid)
      time.sleep(2.0)
      continue
    log.info("opening usb gadget %04x:%04x", args.vid, args.pid)
    try:
      transport = UsbBulkTransport.open(args.vid, args.pid, timeout_ms=args.usb_timeout_ms)
    except LinkError as e:
      log.warning("could not open gadget: %s", e)
      time.sleep(2.0)
      continue
    try:
      Session(transport, cache, Telemetry()).serve_forever()
    except LinkError as e:
      log.info("session ended: %s", e)
    finally:
      transport.close()
    time.sleep(1.0)


def _serve_ffs(args, cache: EngineCache) -> None:
  from jetlink.transport.ffs import FfsTransport
  mount = Path(args.ffs_mount)
  while not (mount / 'ep0').exists():
    log.warning("waiting for functionfs at %s (run scripts/setup_gadget.sh)", mount)
    time.sleep(2.0)
  while True:
    log.info("opening functionfs at %s", mount)
    try:
      transport = FfsTransport(str(mount), gadget=args.gadget, udc=args.udc)
    except OSError as e:
      # Constructing it writes descriptors and binds the UDC, either of which
      # can fail transiently. Retry; exiting here would defeat this loop.
      log.warning("could not open the gadget: %s", e)
      time.sleep(2.0)
      continue
    try:
      Session(transport, cache, Telemetry()).serve_forever()
    except LinkError as e:
      log.info("session ended: %s", e)
    finally:
      transport.close()
    time.sleep(1.0)  # host went away; re-open and wait for it to come back


def main(argv=None) -> int:
  p = argparse.ArgumentParser(description='jetlink Jetson inference server')
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
  p.add_argument('--cache', default=str(DEFAULT_CACHE))
  p.add_argument('--build', metavar='ONNX', help='build an engine and exit')
  p.add_argument('--log-level', default='INFO')
  args = p.parse_args(argv)

  logging.basicConfig(
    level=getattr(logging, args.log_level.upper(), logging.INFO),
    format='%(asctime)s %(levelname)-7s %(name)s: %(message)s')

  cache = EngineCache(Path(args.cache))

  if args.build:
    from jetlink.spec import sha256_file
    sha, nbytes = sha256_file(args.build)
    entry = cache.entry(sha)
    log.info("model %s (%d MB) -> %s", sha[:16], nbytes >> 20, entry.plan_path)
    if entry.exists:
      log.info("already built: %s", entry.meta())
      return 0

    last = [0.0]

    def report(stage, frac, msg):
      if frac - last[0] >= 0.02 or frac >= 1.0:
        last[0] = frac
        log.info("%-6s %5.1f%%  %s", stage, frac * 100, msg)

    build_engine(args.build, entry.plan_path, report=report)
    log.info("built: %s", entry.meta())
    return 0

  {'tcp': _serve_tcp, 'usb': _serve_usb, 'ffs': _serve_ffs}[args.transport](args, cache)
  return 0


if __name__ == '__main__':
  sys.exit(main())
