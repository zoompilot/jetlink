"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Powering the Jetson off on the comma's say-so.

The comma has a battery policy of its own: hardwared shuts the device down
below 11.8 V or after 30 hours parked. A Jetson on an always-on feed does not
know a battery exists, and even asleep it draws something, so when the comma
decides the battery needs protecting it tells the Jetson to go too. Off is
off: nothing but a DC cycle or the power button brings it back, which is why
this is a separate request from sleeping (see sleep.py) and why the comma
only sends it from its own shutdown path.

The server runs in a container and cannot power the host off. It drops a
flag file on the shared cache volume and a host-side path unit
(scripts/jetlink-poweroff.path) does the real work. Two things keep that
from becoming a boot loop: the host script deletes the flag before it powers
off, and it ignores a flag older than the current boot.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

log = logging.getLogger('jetlink.power')

FLAG_NAME = 'poweroff'


def flag_path(cache_root: str | Path) -> Path:
  return Path(cache_root) / FLAG_NAME


def request_poweroff(cache_root: str | Path, reason: str = '') -> bool:
  """Leave the flag for the host. True if it was written."""
  path = flag_path(cache_root)
  try:
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps({'reason': reason, 'time': time.time()}))
    tmp.replace(path)
  except OSError as e:
    log.error("could not write %s: %s", path, e)
    return False
  log.warning("poweroff flag written to %s", path)
  return True


def clear_stale_flag(cache_root: str | Path) -> None:
  """At startup. A flag that survived a boot means the host unit is not
  installed, and it must not be honoured by one installed later."""
  path = flag_path(cache_root)
  try:
    if path.exists():
      path.unlink()
      log.warning("removed a stale poweroff flag; is jetlink-poweroff.path installed on the host?")
  except OSError as e:
    log.error("could not remove %s: %s", path, e)
