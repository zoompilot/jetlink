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
from jetlink.server.session import EngineHost, Session
from jetlink.server.telemetry import Telemetry
from jetlink.transport.base import LinkError

log = logging.getLogger('jetlink.server')

# Longer than the client's FRAME_TIMEOUT, so it is the one that gives up.
DRAIN_TIMEOUT = 5.0


def _serve(cache: EngineCache, open_transport) -> None:
  """Serve one client at a time forever.

  `open_transport()` returns a transport, or None to wait and retry. The three
  transports differ only in how they are opened, so the session lifecycle lives
  here once. The engine host is shared across sessions on purpose: the comma
  reconnects at every jetlinkd/modeld handover and the engine must not be
  reloaded each time.
  """
  host = EngineHost(cache, Telemetry())
  while True:
    transport = open_transport()
    if transport is None:
      time.sleep(2.0)
      continue
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
      log.info("client disconnected")


def _tcp_opener(args):
  from jetlink.transport.tcp import TcpTransport
  srv = TcpTransport.listen(args.host, args.port)
  log.info("listening on %s:%d", args.host, args.port)

  def open_transport():
    transport, addr = TcpTransport.accept(srv)
    log.info("client connected from %s", addr)
    return transport
  return open_transport


def _usb_opener(args):
  """This end is the USB host. Needs no kernel driver: libusb uses usbfs."""
  from jetlink.transport.usbbulk import UsbBulkTransport

  def open_transport():
    if not UsbBulkTransport.present(args.vid, args.pid):
      log.warning("waiting for a jetlink gadget at %04x:%04x", args.vid, args.pid)
      return None
    try:
      return UsbBulkTransport.open(args.vid, args.pid, timeout_ms=args.usb_timeout_ms)
    except Exception as e:
      # Broad on purpose: this loop is the server's only supervisor. Anything
      # that escapes here exits the process, and under a restart policy that is
      # a crash loop rather than a retry.
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

  cache = EngineCache(Path(args.cache))

  if args.build:
    from jetlink.spec import sha256_file, spec_from_onnx
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

    # Carry the spec into the sidecar like a served build does, so the first
    # client to connect loads the plan instead of reparsing the ONNX for it.
    build_engine(args.build, entry.plan_path, report=report,
                 meta_extra={'spec': spec_from_onnx(args.build).to_dict()})
    log.info("built: %s", entry.meta())
    return 0

  _serve(cache, OPENERS[args.transport](args))
  return 0


if __name__ == '__main__':
  sys.exit(main())
