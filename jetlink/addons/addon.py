"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

A fork's add-on, declared once (docs/addons.md, "Offload a model"):

    VASM = Addon('vasm', (0xFF, 0x56, 0x41), setting='MyForkVasmOnJetson',
                 models=(('V-ASM', MODELS / 'v_asm.onnx', 'sha256 of that file'),))

    class Adapter(Stock):
      ADDONS = (VASM,)                     # the gadget, the provisioning run and the setting's watch

    net = VASM.wrap(cv2.dnn.readNetFromONNX(path), sha256, [('output0', (1, 3))], params=Params(), scaled_u8=True)
    run = VASM.model(lambda blob: session.run(None, {'images': blob}), sha256, [('output0', (1, 3))], params=Params())

From then on the setting turns it on and off live, a parked provisioning run
sends the Jetson its models, the Jetson serves them, and the comma runs the
net itself whenever the Jetson does not answer. report() and state() are for
a fork's panel.

The standard library only: the owner reads ADDONS off the adapter.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from jetlink.comma.functions import GadgetFunction, present
from jetlink.openpilot.interface import AddonModel

SETTING_REFRESH_S = 2.0
# the link reports every 30 s while it is used
REPORT_FRESH_S = 75.0
REPORT_DIR = Path('/tmp')
PREPARING_CODES = ('addon_upload', 'addon_build')
# the worst of a model's states (provision.addon_state) names the add-on's
WORST_FIRST = ('unsupported', 'failed', 'missing', None)


def live_setting(params, key: str | None, refresh_s: float = SETTING_REFRESH_S):
  """A BOOL param as a cheap callable, read at most every refresh_s; always on without a key.
  params is anything with get_bool: openpilot's Params, or jetlink's FileParams."""
  if key is None:
    return lambda: True
  read_at: float | None = None
  value = False

  def on() -> bool:
    nonlocal read_at, value
    now = time.monotonic()
    if read_at is None or now - read_at >= refresh_s:
      read_at, value = now, bool(params.get_bool(key))
    return value
  return on


@dataclass(frozen=True)
class Addon:
  """One add-on: its interface on the gadget, the BOOL param that turns it on
  (None: always on), and its models as (name, path, sha256), each the ONNX the
  comma runs and the hash it is named by on the wire."""
  name: str
  iface_class: tuple[int, int, int]
  setting: str | None = None
  models: tuple[tuple[str, Path, str], ...] = ()
  report_path: Path | None = field(default=None, compare=False)

  @property
  def function(self) -> GadgetFunction:
    return GadgetFunction(self.name, self.iface_class)

  @property
  def report_file(self) -> Path:
    return self.report_path or REPORT_DIR / f'jetlink-addon-{self.name}.json'

  def enabled(self, params) -> bool:
    return self.setting is None or params.get_bool(self.setting) is True

  def addon_models(self, params) -> tuple[AddonModel, ...]:
    """Its models while it is on, for the provisioning run (Stock.addon_models)."""
    if not self.enabled(params):
      return ()
    return tuple(AddonModel(self.function, name, Path(path), sha256) for name, path, sha256 in self.models)

  def present(self) -> bool:
    """Does the gadget carry its interface?"""
    return present(self.function)

  # -- the comma's model process ------------------------------------------

  def wrap(self, net, sha256: str, outputs, *, params, scaled_u8: bool = False):
    """net, asking the Jetson first while the setting is on, its link reported
    to report_file; net itself if the wrapper cannot be made
    (jetlink.addons.link.with_host)."""
    from jetlink.addons.link import with_host
    return with_host(net, self.function, sha256, outputs, scaled_u8=scaled_u8, report=self.write_report,
                     enabled=live_setting(params, self.setting))

  def model(self, local, sha256: str, outputs, *, params, scaled_u8: bool = False):
    """local(blob), any runtime's model returning every output in `outputs`
    order, as a call asking the Jetson first while the setting is on; local
    itself if the wrapper cannot be made (jetlink.addons.link.host_model)."""
    from jetlink.addons.link import host_model
    return host_model(local, self.function, sha256, outputs, scaled_u8=scaled_u8, report=self.write_report,
                      enabled=live_setting(params, self.setting))

  def write_report(self, report: dict) -> None:
    path = self.report_file
    tmp = path.with_name(f'.{path.name}.{os.getpid()}')
    try:
      tmp.write_text(json.dumps(report))
      os.replace(tmp, path)
    except OSError:
      tmp.unlink(missing_ok=True)

  # -- a fork's panel ---------------------------------------------------------

  def report(self, now: float | None = None) -> dict | None:
    """The link's last report while fresh: connected, yielding, missing, counts and round trips."""
    path = self.report_file
    try:
      if (time.time() if now is None else now) - path.stat().st_mtime > REPORT_FRESH_S:
        return None
      report = json.loads(path.read_text())
    except (OSError, ValueError):
      return None
    return report if isinstance(report, dict) else None

  def state(self, server: dict | None, progress: dict | None) -> dict:
    """Its models on the Jetson the comma last spoke to, as the owner keeps its
    hello (jetlink.comma.gadget.SERVER): `jetson` the worst of ready, missing,
    failed and unsupported (None before any hello), and `preparing`, the
    provisioning run's progress text (the fork's progress param) while it
    sends or builds one of them."""
    from jetlink.openpilot.provision import addon_state
    states = {addon_state(sha256, server) for _, _, sha256 in self.models}
    jetson = next((s for s in WORST_FIRST if s in states), 'ready') if states else None
    progress = progress if isinstance(progress, dict) else {}
    args = progress.get('args') if isinstance(progress.get('args'), dict) else {}
    mine = progress.get('code') in PREPARING_CODES and args.get('name') in {name for name, _, _ in self.models}
    return {'jetson': jetson, 'preparing': str(progress.get('msg', '')) if mine else ''}
