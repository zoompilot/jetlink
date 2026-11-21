"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Extra FunctionFS interfaces on the comma's gadget, for a fork's own add-ons:
models jetlink-server serves beside the driving model (docs/addons.md), or a
server of the fork's own on a Jetson running jetlink-server --no-addons. A fork declares them on
its adapter (Stock.GADGET_FUNCTIONS); jetlink only presents them.

jetlink-root.sh gadget links each as ffs.<name> after ffs.jetlink, so Jetlink
stays interface 0, and mounts it at /dev/ffs-<name>. A linked function with no
descriptors fails the bind for the whole gadget, and closing its ep0 drops the
whole gadget, so the owner writes each one's descriptors before its own bind
and holds its ep0 for the owner's life. One that cannot be described is left
out of the next build, and Jetlink comes up alone.

The add-on opens only ep1 (OUT) and ep2 (IN), and only once a host has
configured the gadget (gadget.host_attached): opening one before costs the
gadget until a reboot. A host finds the interface by its class triple, not its
number. USB only: an iOS gadget has none.

Standard library and jetlink.protocol only at the top: the resident owner
imports this.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from jetlink import protocol as P

NAME = re.compile(r'[a-z0-9]{1,16}\Z')
# ffs.jetlink is Jetlink's own
RESERVED = frozenset({'jetlink'})
MOUNT_PREFIX = '/dev/ffs-'


@dataclass(frozen=True)
class GadgetFunction:
  """One extra vendor interface with a bulk OUT and a bulk IN endpoint."""
  name: str                          # the FunctionFS instance, ffs.<name>, and its mount, /dev/ffs-<name>
  iface_class: tuple[int, int, int]  # (class, subclass, protocol) a host finds it by

  def __post_init__(self):
    if not isinstance(self.name, str) or not NAME.match(self.name) or self.name in RESERVED:
      raise ValueError(f"gadget function name {self.name!r}: 1-16 of a-z and 0-9, and not {sorted(RESERVED)}")
    triple = tuple(self.iface_class)
    if len(triple) != 3 or not all(isinstance(b, int) and 0 <= b <= 0xFF for b in triple):
      raise ValueError(f"gadget function {self.name}: iface_class must be three bytes, not {self.iface_class!r}")
    if triple == P.USB_VENDOR_CLASS:
      raise ValueError(f"gadget function {self.name}: {triple} is Jetlink's own interface class")
    object.__setattr__(self, 'iface_class', triple)

  @property
  def mount(self) -> Path:
    return Path(MOUNT_PREFIX + self.name)

  @property
  def ep0(self) -> Path:
    return self.mount / 'ep0'

  @property
  def ep_out(self) -> Path:
    """Host to comma; the add-on opens it once the host has configured the gadget."""
    return self.mount / 'ep1'

  @property
  def ep_in(self) -> Path:
    """Comma to host; the add-on opens it once the host has configured the gadget."""
    return self.mount / 'ep2'


def present(fn: GadgetFunction) -> bool:
  """Was the gadget built with this function, mounted and ready to describe?"""
  try:
    return fn.ep0.exists()
  except OSError:
    return False


def describe(fn: GadgetFunction) -> int:
  """Open its ep0 and write its descriptors and strings; the caller holds the
  fd for as long as the gadget is presented, as closing it drops the gadget.
  Raises OSError: EBUSY while an add-on from before still has ep1 or ep2."""
  from jetlink.transport import ffs
  fd = os.open(fn.ep0, os.O_RDWR)
  try:
    os.write(fd, ffs.build_descriptors(fn.iface_class))
    os.write(fd, ffs.build_strings(fn.name))
  except BaseException:
    os.close(fd)
    raise
  return fd
