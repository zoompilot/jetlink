"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

USB gadget side of the link: a FunctionFS vendor-specific bulk function.

This is the Jetson end. The comma end is `transport.usbbulk`, which talks to it
with libusb and needs no kernel driver on the host.

Why raw bulk rather than a USB ethernet gadget: AGNOS's kernel has no host-side
CDC-NCM, CDC-ECM or RNDIS driver, so an ethernet gadget simply will not
enumerate on a comma. Raw bulk also removes the IP stack, DHCP and NCM's
aggregation timer from the latency path, and openpilot already speaks to both
the panda and chestnut this way.

Bring-up (see scripts/setup_gadget.sh, which does all of this):
    configfs gadget -> functions/ffs.jetlink -> mount -t functionfs
    write descriptors + strings to ep0 -> ep1 (OUT) and ep2 (IN) appear
    echo <udc> > UDC
"""
from __future__ import annotations

import os
import struct

from jetlink import protocol as P
from jetlink.transport.base import FramedBuffer, LinkError, LinkTimeout, Message, Transport, now_ns

# --- FunctionFS ABI -------------------------------------------------------

FUNCTIONFS_DESCRIPTORS_MAGIC_V2 = 3
FUNCTIONFS_STRINGS_MAGIC = 2

FLAG_HAS_FS = 1 << 0
FLAG_HAS_HS = 1 << 1
FLAG_HAS_SS = 1 << 2

USB_DT_INTERFACE = 0x04
USB_DT_ENDPOINT = 0x05
USB_DT_SS_ENDPOINT_COMP = 0x30

USB_ENDPOINT_XFER_BULK = 0x02
EP_OUT = 0x01  # host -> device, we read()
EP_IN = 0x82   # device -> host, we write()

# libusb/xHCI will split anything larger; must stay a multiple of the SS max
# packet size (1024) so FunctionFS accepts OUT reads.
SS_MAX_PACKET = 1024
READ_CHUNK = 256 * SS_MAX_PACKET  # 256 KB


def _interface_desc(n_endpoints: int = 2, i_interface: int = 1) -> bytes:
  return struct.pack('<BBBBBBBBB', 9, USB_DT_INTERFACE, 0, 0, n_endpoints,
                     0xFF, 0xFF, 0xFF, i_interface)


def _endpoint_desc(addr: int, max_packet: int) -> bytes:
  return struct.pack('<BBBBHB', 7, USB_DT_ENDPOINT, addr, USB_ENDPOINT_XFER_BULK, max_packet, 0)


def _ss_companion(max_burst: int = 15) -> bytes:
  return struct.pack('<BBBBH', 6, USB_DT_SS_ENDPOINT_COMP, max_burst, 0, 0)


def build_descriptors() -> bytes:
  fs = _interface_desc() + _endpoint_desc(EP_OUT, 64) + _endpoint_desc(EP_IN, 64)
  hs = _interface_desc() + _endpoint_desc(EP_OUT, 512) + _endpoint_desc(EP_IN, 512)
  ss = (_interface_desc()
        + _endpoint_desc(EP_OUT, SS_MAX_PACKET) + _ss_companion()
        + _endpoint_desc(EP_IN, SS_MAX_PACKET) + _ss_companion())

  # struct usb_functionfs_descs_head_v2: magic, length, flags, then ALL the
  # per-speed counts together (one __le32 per flag set, in flag order), and only
  # then the descriptor blocks. Interleaving count/block per speed gets EINVAL.
  counts = struct.pack('<III', 3, 3, 5)   # fs, hs, ss descriptor counts
  body = counts + fs + hs + ss
  flags = FLAG_HAS_FS | FLAG_HAS_HS | FLAG_HAS_SS
  length = 12 + len(body)
  return struct.pack('<III', FUNCTIONFS_DESCRIPTORS_MAGIC_V2, length, flags) + body


def build_strings(name: str = 'jetlink') -> bytes:
  s = name.encode() + b'\0'
  length = 16 + 2 + len(s)
  return (struct.pack('<IIII', FUNCTIONFS_STRINGS_MAGIC, length, 1, 1)
          + struct.pack('<H', 0x0409) + s)


class FfsTransport(Transport):
  """Server-side transport over a mounted FunctionFS instance."""

  def __init__(self, mount: str = '/dev/ffs-jetlink', gadget: str | None = None,
               udc: str | None = None):
    self.mount = mount
    self.gadget = gadget
    self.bound_udc: str | None = None
    self.ep0 = os.open(os.path.join(mount, 'ep0'), os.O_RDWR)
    os.write(self.ep0, build_descriptors())
    os.write(self.ep0, build_strings())
    # The endpoint files only exist once ep0 has accepted the descriptors.
    self.ep_out = os.open(os.path.join(mount, 'ep1'), os.O_RDWR)
    self.ep_in = os.open(os.path.join(mount, 'ep2'), os.O_RDWR)
    self.fb = FramedBuffer()
    self._pending = bytearray()
    if gadget is not None:
      # Bind last. A FunctionFS gadget cannot be attached to a UDC until its
      # descriptors have been written, so the setup script deliberately leaves
      # UDC empty and we finish the job here.
      self.bind(udc)

  def bind(self, udc: str | None = None) -> None:
    if self.gadget is None:
      raise LinkError("no gadget path given")
    if udc is None:
      udcs = sorted(os.listdir('/sys/class/udc'))
      if not udcs:
        raise LinkError("no USB device controller found")
      udc = udcs[0]
    with open(os.path.join(self.gadget, 'UDC'), 'w') as f:
      f.write(udc + '\n')
    self.bound_udc = udc

  def unbind(self) -> None:
    if self.gadget is None or self.bound_udc is None:
      return
    try:
      with open(os.path.join(self.gadget, 'UDC'), 'w') as f:
        f.write('\n')
    except OSError:
      pass
    self.bound_udc = None

  @classmethod
  def attach(cls, mount: str = '/dev/ffs-jetlink') -> FfsTransport:
    """Attach to an instance whose descriptors another process already wrote."""
    self = cls.__new__(cls)
    self.mount = mount
    self.gadget = None
    self.bound_udc = None
    self.ep0 = -1
    self.ep_out = os.open(os.path.join(mount, 'ep1'), os.O_RDWR)
    self.ep_in = os.open(os.path.join(mount, 'ep2'), os.O_RDWR)
    self.fb = FramedBuffer()
    self._pending = bytearray()
    return self

  def send(self, msg_type: int, seq: int, parts=(), flags: int = 0) -> None:
    parts = [memoryview(p).cast('B') for p in parts]
    length = sum(p.nbytes for p in parts)
    header = P.pack_header(msg_type, seq, length, flags, now_ns())
    # os.writev keeps header + payload in one transfer, so the host sees one
    # bulk stream with no interleaving risk.
    bufs = [memoryview(header), *parts]
    try:
      while bufs:
        n = os.writev(self.ep_in, bufs)
        if n <= 0:
          raise LinkError("gadget write returned 0 (host gone?)")
        bufs = _advance(bufs, n)
    except OSError as e:
      raise LinkError(f"gadget write failed: {e}") from e

  def _fill(self, need: int) -> None:
    while len(self._pending) < need:
      try:
        chunk = os.read(self.ep_out, READ_CHUNK)
      except OSError as e:
        raise LinkError(f"gadget read failed: {e}") from e
      if not chunk:
        raise LinkError("gadget read returned EOF (host disconnected)")
      self._pending += chunk

  def recv(self, timeout: float | None = None) -> Message:
    # FunctionFS blocking reads have no timeout; the caller supervises liveness.
    self._fill(P.HEADER_SIZE)
    _, _, msg_type, seq, flags, length, t_mono_ns = P.unpack_header(self._pending)
    self._fill(P.HEADER_SIZE + length)
    view = self.fb.ensure(length)
    view[:length] = self._pending[P.HEADER_SIZE:P.HEADER_SIZE + length]
    del self._pending[:P.HEADER_SIZE + length]
    return Message(msg_type, seq, flags, t_mono_ns, view[:length])

  def close(self) -> None:
    self.unbind()
    for fd in (self.ep_in, self.ep_out, self.ep0):
      if fd is not None and fd >= 0:
        try:
          os.close(fd)
        except OSError:
          pass


def _advance(bufs: list[memoryview], n: int) -> list[memoryview]:
  out: list[memoryview] = []
  for mv in bufs:
    if n:
      if n >= mv.nbytes:
        n -= mv.nbytes
        continue
      mv = mv[n:]
      n = 0
    out.append(mv)
  return out
