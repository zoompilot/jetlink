"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

USB gadget side of the link: a FunctionFS vendor-specific bulk function.

On a comma this is the *comma* end. The roles look backwards and are not:
AGNOS has CONFIG_USB_F_FS and libcomposite built in, while a USB host needs no
kernel driver at all, which matters because L4T rootfs images are often
stripped of the gadget modules. See docs/transport.md.

Bring-up (scripts/setup_gadget.sh does all of this):
    configfs gadget -> functions/ffs.jetlink -> mount -t functionfs
    write descriptors + strings to ep0 -> ep1 (OUT) and ep2 (IN) appear
    echo <udc> > UDC
"""
from __future__ import annotations

import errno
import fcntl
import os
import struct
import time

from jetlink.transport.base import LinkError, StreamTransport

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
EP_OUT = 0x01  # host -> device
EP_IN = 0x82   # device -> host

SS_MAX_PACKET = 1024
READ_CHUNK = 256 * SS_MAX_PACKET  # 256 KB
# Consecutive EAGAINs *before a single successful read* that mean this kernel
# has no working non-blocking path. Only counted before the first success:
# an idle link produces EAGAIN exactly like a broken one, so a running counter
# would trip on any quiet second and lose the deadline for good.
EAGAIN_FALLBACK = 2000

# A host enumerating us is not the same as a host being ready to talk: it still
# has to open the device and claim the interface, and until it does the gadget's
# endpoints return EIO/ESHUTDOWN. The server polls sysfs every couple of
# seconds to notice us, so that gap is easily a second or two. Wait it out
# rather than reporting a dead link on the first frame after connect.
EP_READY_TIMEOUT = 10.0
_NOT_READY = (errno.EIO, errno.ESHUTDOWN, errno.ENODEV)


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
  body = struct.pack('<III', 3, 3, 5) + fs + hs + ss
  flags = FLAG_HAS_FS | FLAG_HAS_HS | FLAG_HAS_SS
  return struct.pack('<III', FUNCTIONFS_DESCRIPTORS_MAGIC_V2, 12 + len(body), flags) + body


def build_strings(name: str = 'jetlink') -> bytes:
  s = name.encode() + b'\0'
  return (struct.pack('<IIII', FUNCTIONFS_STRINGS_MAGIC, 16 + 2 + len(s), 1, 1)
          + struct.pack('<H', 0x0409) + s)


class FfsTransport(StreamTransport):
  # FunctionFS rejects an OUT read whose buffer is not a multiple of the max
  # packet size, so always keep a full chunk of room available.
  read_slack = READ_CHUNK
  packet_size = SS_MAX_PACKET
  read_chunk = READ_CHUNK

  def __init__(self, mount: str = '/dev/ffs-jetlink', gadget: str | None = None,
               udc: str | None = None):
    super().__init__(rx_size=2 << 20)
    self.mount = mount
    self.gadget = gadget
    self.bound_udc: str | None = None
    self.ep0 = self.ep_out = self.ep_in = -1
    self._blocking_reads = False
    self._eagain_streak = 0
    self._ever_read = False
    self._ready_deadline: float | None = None
    try:
      self.ep0 = os.open(os.path.join(mount, 'ep0'), os.O_RDWR)
      os.write(self.ep0, build_descriptors())
      os.write(self.ep0, build_strings())
      # The endpoint files only exist once ep0 has accepted the descriptors.
      # O_NONBLOCK so a dead host cannot wedge the control path forever; if the
      # kernel ignores it for FunctionFS the read simply blocks, as before.
      self.ep_out = os.open(os.path.join(mount, 'ep1'), os.O_RDWR | os.O_NONBLOCK)
      self.ep_in = os.open(os.path.join(mount, 'ep2'), os.O_RDWR)
      if gadget is not None:
        # Bind last: a FunctionFS gadget cannot attach to a controller until its
        # descriptors have been written, which is why setup_gadget.sh leaves UDC
        # empty and we finish the job here.
        self.bind(udc)
    except BaseException:
      self.close()   # otherwise a failed bring-up leaks the descriptors it did open
      raise

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

  def _write(self, bufs: list[memoryview]) -> int:
    while True:
      try:
        return os.writev(self.ep_in, bufs)
      except OSError as e:
        # FunctionFS submits a write as one request, so a failed writev put
        # nothing on the wire and is safe to retry.
        if e.errno in _NOT_READY and self._wait_for_host_ready():
          continue
        raise LinkError(f"gadget write failed: {e}") from e

  def _read_into(self, dest: memoryview, timeout: float | None) -> int:
    n = self._clamp_read(dest)
    if n == 0:
      return 0
    try:
      got = os.readv(self.ep_out, [dest[:n]])
    except BlockingIOError:
      # Nothing queued yet. A short sleep keeps the deadline enforceable
      # without spinning a core; _fill decides when to give up.
      if not self._ever_read:
        self._eagain_streak += 1
        if self._eagain_streak > EAGAIN_FALLBACK:
          self._use_blocking_reads()
      time.sleep(0.0005)
      return 0
    except OSError as e:
      # NB BlockingIOError subclasses OSError and is handled above.
      if e.errno in _NOT_READY and self._wait_for_host_ready():
        return 0
      raise LinkError(f"gadget read failed: {e}") from e
    if got == 0:
      raise LinkError("gadget read returned EOF (host disconnected)")
    self._ever_read = True
    self._ready_deadline = None
    return got

  def _wait_for_host_ready(self) -> bool:
    """True while we are still inside the grace period for the host to claim us.

    Enumeration and readiness are different things: the host has to open the
    device and claim the interface before our endpoints work, and until then
    they return EIO. Starting the clock on the first such error rather than at
    open means the wait covers a re-enumeration too.
    """
    now = time.monotonic()
    if self._ready_deadline is None:
      self._ready_deadline = now + EP_READY_TIMEOUT
    if now < self._ready_deadline:
      time.sleep(0.005)
      return True
    return False

  def _use_blocking_reads(self) -> None:
    """Give up on O_NONBLOCK if this kernel never delivers data through it.

    Some FunctionFS builds have no non-blocking read path and return EAGAIN
    forever, which would spin instead of receiving. Blocking reads cost us the
    per-frame deadline on this side, but they do work.
    """
    if self._blocking_reads:
      return
    flags = fcntl.fcntl(self.ep_out, fcntl.F_GETFL)
    fcntl.fcntl(self.ep_out, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
    self._blocking_reads = True
    self._eagain_streak = 0

  def close(self) -> None:
    self.unbind()
    # Clear each fd as it is closed: a second close() would otherwise shut
    # whatever those descriptor numbers had been recycled into.
    for name in ('ep_in', 'ep_out', 'ep0'):
      fd = getattr(self, name, -1)
      setattr(self, name, -1)
      if fd is not None and fd >= 0:
        try:
          os.close(fd)
        except OSError:
          pass
