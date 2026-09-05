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

from jetlink import protocol as P
from jetlink.transport.base import LinkError, StreamTransport

# pid.codes test allocation. Get a real PID before distributing this.
JETLINK_VID = 0x1209
JETLINK_PID = 0x0001

# NB the endpoint addresses are discovered, never assumed. FunctionFS treats
# the addresses in the gadget's descriptors as logical and renumbers them when
# it binds, so a gadget that declares 0x01/0x82 can appear to the host as
# 0x01/0x81. Reading a hardcoded address that does not exist fails with a bare
# LIBUSB_ERROR_IO and looks exactly like a broken cable.
USB_ENDPOINT_DIR_IN = 0x80
USB_TRANSFER_TYPE_BULK = 0x02
MAX_PACKET = 1024   # SuperSpeed bulk
READ_CHUNK = 256 * MAX_PACKET
DEFAULT_TIMEOUT_MS = 2000

log = logging.getLogger('jetlink.usb')


class UsbBulkTransport(StreamTransport):
  packet_size = MAX_PACKET
  read_chunk = READ_CHUNK
  rx_align = P.GADGET_TX_ALIGN
  # A bulk IN read has to be posted for a whole packet, so the buffer needs a
  # packet of headroom beyond the message itself. Without it a message whose
  # length is not a packet multiple *and* big enough to resize the buffer ends
  # with a few bytes of room, which rounds down to zero packets and stalls the
  # read for good. Only the model upload is ever that big.
  read_slack = MAX_PACKET

  def __init__(self, handle, context=None, timeout_ms: int = DEFAULT_TIMEOUT_MS,
               interface: int = 0, ep_in: int = 0x81, ep_out: int = 0x01):
    super().__init__(rx_size=2 << 20)
    self.handle = handle
    self.context = context
    self.timeout_ms = timeout_ms
    self.interface = interface
    self.ep_in = ep_in
    self.ep_out = ep_out
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
    handle = None
    try:
      device = next((d for d in context.getDeviceIterator(skip_on_error=True)
                     if (d.getVendorID(), d.getProductID()) == (vid, pid)), None)
      if device is None:
        raise LinkError(f"no jetlink gadget at {vid:04x}:{pid:04x}")
      ep_in, ep_out = _find_bulk_endpoints(device, interface)
      # libusb_open itself can fail with EIO on a device that is enumerated but
      # not answering - which is exactly what a FunctionFS gadget looks like
      # when the process owning its endpoints has exited. Everything in here
      # has to come back as LinkError, or it escapes the server's accept loop
      # and takes the process down instead of retrying.
      handle = device.open()
      handle.claimInterface(interface)
    except LinkError:
      _close_quietly(handle, context)
      raise
    except Exception as e:
      _close_quietly(handle, context)
      raise LinkError(f"could not open {vid:04x}:{pid:04x}: {e}") from e
    return cls(handle, context, timeout_ms, interface, ep_in, ep_out)

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
      return self.handle.bulkWrite(self.ep_out, memoryview(self._tx)[:total],
                                   timeout=self._ms(self._write_timeout()))
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
        return self.handle._bulkTransfer(self.ep_in, buf, n, self._ms(timeout))
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
      data = self.handle.bulkRead(self.ep_in, n, timeout=self._ms(timeout))
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


def _close_quietly(handle, context) -> None:
  for closer in (getattr(handle, 'close', None), getattr(context, 'close', None)):
    if closer is None:
      continue
    try:
      closer()
    except Exception:
      pass


def _find_bulk_endpoints(device, interface: int) -> tuple[int, int]:
  """(IN, OUT) bulk endpoint addresses for `interface`, from its descriptors."""
  for cfg in device.iterConfigurations():
    for iface in cfg:
      for setting in iface:
        if setting.getNumber() != interface:
          continue
        ep_in = ep_out = None
        for ep in setting:
          if ep.getAttributes() & 0x03 != USB_TRANSFER_TYPE_BULK:
            continue
          if ep.getAddress() & USB_ENDPOINT_DIR_IN:
            ep_in = ep.getAddress()
          else:
            ep_out = ep.getAddress()
        if ep_in is not None and ep_out is not None:
          return ep_in, ep_out
  raise LinkError(f"interface {interface} has no bulk IN/OUT endpoint pair")
