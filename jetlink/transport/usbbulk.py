"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

USB host side of the link: libusb bulk transfers to the Jetson's FunctionFS
gadget. This is the comma end.

No kernel driver is involved on either side. openpilot already carries `usb1`
and already drives the panda and chestnut through usbfs the same way, so this
adds no dependency and needs no AGNOS kernel change - which is the point, since
AGNOS has no host-side USB-ethernet driver to fall back on.
"""
from __future__ import annotations

from jetlink import protocol as P
from jetlink.transport.base import FramedBuffer, LinkError, LinkTimeout, Message, Transport, now_ns

# pid.codes test allocation. Get a real PID before shipping this widely.
JETLINK_VID = 0x1209
JETLINK_PID = 0x0001
JETLINK_PRODUCT = 'jetlink'

EP_OUT = 0x01
EP_IN = 0x82
READ_CHUNK = 256 * 1024

DEFAULT_TIMEOUT_MS = 100


class UsbBulkTransport(Transport):
  def __init__(self, handle, context=None, timeout_ms: int = DEFAULT_TIMEOUT_MS):
    self.handle = handle
    self.context = context
    self.timeout_ms = timeout_ms
    self.fb = FramedBuffer()
    self._pending = bytearray()
    self._read_buf = bytearray(READ_CHUNK)

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
    return cls(handle, context, timeout_ms)

  @staticmethod
  def present(vid: int = JETLINK_VID, pid: int = JETLINK_PID) -> bool:
    """Cheap presence check that does not open the device."""
    from pathlib import Path
    for d in Path('/sys/bus/usb/devices').glob('*'):
      try:
        if (int((d / 'idVendor').read_text(), 16) == vid
            and int((d / 'idProduct').read_text(), 16) == pid):
          return True
      except (OSError, ValueError):
        pass
    return False

  def send(self, msg_type: int, seq: int, parts=(), flags: int = 0) -> None:
    import usb1
    parts = [memoryview(p).cast('B') for p in parts]
    length = sum(p.nbytes for p in parts)
    header = P.pack_header(msg_type, seq, length, flags, now_ns())
    # libusb has no vectored bulk write, so build one contiguous buffer. The
    # copy costs ~0.1 ms for a 460 KB request and buys a single transfer, which
    # matters more: several transfers would let the host scheduler interleave.
    out = bytearray(P.HEADER_SIZE + length)
    out[:P.HEADER_SIZE] = header
    off = P.HEADER_SIZE
    for p in parts:
      out[off:off + p.nbytes] = p
      off += p.nbytes
    try:
      written = self.handle.bulkWrite(EP_OUT, out, timeout=self.timeout_ms)
    except usb1.USBErrorTimeout as e:
      raise LinkTimeout("usb bulk write timed out") from e
    except usb1.USBError as e:
      raise LinkError(f"usb bulk write failed: {e}") from e
    if written != len(out):
      raise LinkError(f"short usb write: {written}/{len(out)}")

  def _fill(self, need: int, timeout_ms: int) -> None:
    import usb1
    while len(self._pending) < need:
      try:
        chunk = self.handle.bulkRead(EP_IN, READ_CHUNK, timeout=timeout_ms)
      except usb1.USBErrorTimeout as e:
        raise LinkTimeout("usb bulk read timed out") from e
      except usb1.USBError as e:
        raise LinkError(f"usb bulk read failed: {e}") from e
      if not chunk:
        continue  # zero-length packet: a transfer terminator, not an error
      self._pending += chunk

  def recv(self, timeout: float | None = None) -> Message:
    timeout_ms = self.timeout_ms if timeout is None else max(1, int(timeout * 1000))
    self._fill(P.HEADER_SIZE, timeout_ms)
    _, _, msg_type, seq, flags, length, t_mono_ns = P.unpack_header(self._pending)
    self._fill(P.HEADER_SIZE + length, timeout_ms)
    view = self.fb.ensure(length)
    view[:length] = self._pending[P.HEADER_SIZE:P.HEADER_SIZE + length]
    del self._pending[:P.HEADER_SIZE + length]
    return Message(msg_type, seq, flags, t_mono_ns, view[:length])

  def close(self) -> None:
    try:
      self.handle.releaseInterface(0)
    except Exception:
      pass
    try:
      self.handle.close()
    except Exception:
      pass
    if self.context is not None:
      try:
        self.context.close()
      except Exception:
        pass
