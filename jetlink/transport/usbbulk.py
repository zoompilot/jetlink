"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

USB host side of the link: libusb bulk transfers to a FunctionFS gadget.

On a comma+Jetson pair this is the *Jetson* end. A host needs no kernel driver
at all - libusb goes through usbfs - which is what lets this work on a stripped
L4T rootfs with no gadget modules. openpilot already drives the panda and
chestnut the same way, so `usb1` is not a new dependency.
"""
from __future__ import annotations

import logging
from pathlib import Path

from jetlink.transport.base import LinkError, StreamTransport

# pid.codes test allocation. Get a real PID before distributing this.
JETLINK_VID = 0x1209
JETLINK_PID = 0x0001

EP_OUT = 0x01
EP_IN = 0x82
MAX_PACKET = 1024   # SuperSpeed bulk
READ_CHUNK = 256 * MAX_PACKET
DEFAULT_TIMEOUT_MS = 2000

log = logging.getLogger('jetlink.usb')


class UsbBulkTransport(StreamTransport):
  packet_size = MAX_PACKET
  read_chunk = READ_CHUNK

  def __init__(self, handle, context=None, timeout_ms: int = DEFAULT_TIMEOUT_MS,
               interface: int = 0):
    super().__init__(rx_size=2 << 20)
    self.handle = handle
    self.context = context
    self.timeout_ms = timeout_ms
    self.interface = interface
    self._zero_copy_reads = True
    # libusb has no vectored bulk write, so messages are gathered here. Reused
    # so the steady state does not allocate half a megabyte per frame.
    self._tx = bytearray(1 << 20)

  @classmethod
  def open(cls, vid: int = JETLINK_VID, pid: int = JETLINK_PID,
           timeout_ms: int = DEFAULT_TIMEOUT_MS, interface: int = 0) -> UsbBulkTransport:
    import usb1
    context = usb1.USBContext()
    context.open()
    handle = context.openByVendorIDAndProductID(vid, pid, skip_on_error=True)
    if handle is None:
      context.close()
      raise LinkError(f"no jetlink gadget at {vid:04x}:{pid:04x}")
    try:
      handle.claimInterface(interface)
    except Exception as e:
      handle.close()
      context.close()
      raise LinkError(f"could not claim interface {interface}: {e}") from e
    return cls(handle, context, timeout_ms, interface)

  @staticmethod
  def present(vid: int = JETLINK_VID, pid: int = JETLINK_PID) -> bool:
    """Cheap presence check that does not open the device."""
    for d in Path('/sys/bus/usb/devices').glob('*'):
      try:
        if (int((d / 'idVendor').read_text(), 16) == vid
            and int((d / 'idProduct').read_text(), 16) == pid):
          return True
      except (OSError, ValueError):
        pass
    return False

  def _ms(self, timeout: float | None) -> int:
    return self.timeout_ms if timeout is None else max(1, int(timeout * 1000))

  def _write(self, bufs: list[memoryview]) -> int:
    import usb1
    total = sum(b.nbytes for b in bufs)
    if len(self._tx) < total:
      self._tx = bytearray(max(total, len(self._tx) * 2))
    off = 0
    for b in bufs:
      self._tx[off:off + b.nbytes] = b
      off += b.nbytes
    try:
      # One transfer, not several: multiple writes would let the host scheduler
      # interleave and show up as jitter.
      return self.handle.bulkWrite(EP_OUT, memoryview(self._tx)[:total],
                                   timeout=self.timeout_ms)
    except usb1.USBErrorTimeout as e:
      # Report what actually went out so the caller resends only the remainder;
      # claiming zero would duplicate bytes the device already has.
      return getattr(e, 'transferred', 0)
    except usb1.USBError as e:
      raise LinkError(f"usb bulk write failed: {e}") from e

  def _read_into(self, dest: memoryview, timeout: float | None) -> int:
    import usb1
    # _clamp_read rounds to whole packets: a bulk IN whose buffer is not a
    # packet multiple can overflow when the device delivers a full final packet.
    n = self._clamp_read(dest)
    if n == 0:
      return 0
    if self._zero_copy_reads:
      try:
        # bulkRead() allocates a 256 KB buffer, slices it, and hands back a
        # copy - about 768 KB of allocation and 918 KB of memcpy per frame on
        # the receive path, which is exactly what RxBuffer exists to avoid.
        # create_binary_buffer over `dest` writes straight into it.
        buf, _ = usb1.create_binary_buffer(dest[:n])
        return self.handle._bulkTransfer(EP_IN, buf, n, self._ms(timeout))
      except usb1.USBErrorTimeout as e:
        # libusb attaches whatever did arrive to the exception. Dropping it
        # would desync the stream, far worse than a late frame.
        return getattr(e, 'transferred', 0)
      except usb1.USBError as e:
        raise LinkError(f"usb bulk read failed: {e}") from e
      except (AttributeError, TypeError) as e:
        # A python-libusb1 without the private transfer helper. Fall back for
        # good rather than paying the exception on every read.
        self._zero_copy_reads = False
        log.warning("jetlink: no zero-copy bulk read (%s), using bulkRead", e)

    try:
      data = self.handle.bulkRead(EP_IN, n, timeout=self._ms(timeout))
    except usb1.USBErrorTimeout as e:
      data = getattr(e, 'received', b'')
    except usb1.USBError as e:
      raise LinkError(f"usb bulk read failed: {e}") from e
    if not data:
      return 0  # zero-length packet: a transfer terminator, not an error
    dest[:len(data)] = data
    return len(data)

  def close(self) -> None:
    for fn in (lambda: self.handle.releaseInterface(self.interface), self.handle.close,
               (self.context.close if self.context is not None else None)):
      if fn is None:
        continue
      try:
        fn()
      except Exception:
        pass
