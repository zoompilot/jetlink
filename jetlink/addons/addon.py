"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

A fork's add-on, declared once (docs/addons.md, "Offload a model"):

    def vasm():
      return Addon('vasm', (0xFF, 0x56, 0x41), setting='MyForkVasmOnJetson',
                   models=(('V-ASM', MODELS / 'v_asm.onnx', 'sha256 of that file'),))

    class Adapter(Stock):
      ADDONS = (vasm(),)                   # the gadget, the provisioning run and the setting's watch

    net = vasm().wrap(cv2.dnn.readNetFromONNX(path), sha256, [('output0', (1, 3))], params=Params(), scaled_u8=True)
    run = vasm().model(lambda blob: session.run(None, {'images': blob}), sha256, [('output0', (1, 3))], params=Params())

From then on the setting turns it on and off live, a parked provisioning run
sends the Jetson its models, the Jetson serves them, and the comma runs the
net itself whenever the Jetson does not answer. report() and state() are for
a fork's panel.

The standard library only: the owner reads ADDONS off the adapter.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from jetlink.comma.functions import GadgetFunction, cable_network, present
from jetlink.openpilot.interface import AddonModel

SETTING_REFRESH_S = 2.0
# the link reports every 30 s while it is used
REPORT_FRESH_S = 75.0
# beside the owner's records (jetlink.comma.gadget)
REPORT_DIR = Path('/dev/shm/jetlink')
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

  @property
  def function(self) -> GadgetFunction:
    return GadgetFunction(self.name, self.iface_class)

  @property
  def report_file(self) -> Path:
    return REPORT_DIR / f'jetlink-addon-{self.name}.json'

  def enabled(self, params) -> bool:
    return self.setting is None or params.get_bool(self.setting) is True

  def addon_models(self, params) -> tuple[AddonModel, ...]:
    """Its models while it is on, for the provisioning run (Stock.addon_models)."""
    if not self.enabled(params):
      return ()
    return tuple(AddonModel(self.function, name, Path(path), sha256) for name, path, sha256 in self.models)

  def present(self) -> bool:
    """Does the gadget carry its interface, or the cable network that stands for it on an iPhone?"""
    return present(self.function) or cable_network()

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
    from jetlink.comma.gadget import write_record
    try:
      write_record(self.report_file, report)
    except OSError:
      pass

  # -- a fork's panel ---------------------------------------------------------

  def report(self, now: float | None = None) -> dict | None:
    """The link's last report while fresh: connected, yielding, missing, counts and round trips."""
    path = self.report_file
    try:
      # wall clock, as st_mtime is
      if (time.time() if now is None else now) - path.stat().st_mtime > REPORT_FRESH_S:
        return None
      report = json.loads(path.read_text())
    except (OSError, ValueError):
      return None
    return report if isinstance(report, dict) else None

  def state(self, server: dict | None, progress: dict | None, run: dict | None = None) -> dict:
    """Its models on the server the comma last spoke to (a Jetson, a Mac, a
    phone), as the owner keeps its hello (jetlink.comma.gadget.SERVER): `host`
    the worst of ready on its interface, missing, failed and unsupported (None
    before any hello), and `preparing`, the provisioning run's progress text
    (the fork's progress param) while it sends or builds one of them, each
    model known by its interface and sha256, never its name. With
    the last provisioning run's record (jetlink.comma.gadget.ADDON_RUN),
    `last_check` says what came of it for this add-on and `last_check_at` when,
    in epoch seconds; '' and None without one."""
    from jetlink.addons.wire import interface_triple
    from jetlink.openpilot.provision import addon_state
    states = {addon_state(sha256, server, self.iface_class) for _, _, sha256 in self.models}
    host = next((s for s in WORST_FIRST if s in states), 'ready') if states else None
    progress = progress if isinstance(progress, dict) else {}
    args = progress.get('args') if isinstance(progress.get('args'), dict) else {}
    mine = (progress.get('code') in PREPARING_CODES and args.get('interface') == interface_triple(self.iface_class)
            and args.get('sha256') in {sha256 for _, _, sha256 in self.models})
    return {'host': host, 'preparing': str(progress.get('msg', '')) if mine else '', **self.last_check(run)}

  def last_check(self, run: dict | None) -> dict:
    """What the last provisioning run made of this add-on's models: each it
    sent on its interface, else the run's own result."""
    from jetlink.addons.wire import interface_triple
    if not isinstance(run, dict) or not isinstance(run.get('at'), (int, float)):
      return {'last_check': '', 'last_check_at': None}
    models = run.get('models') if isinstance(run.get('models'), dict) else {}
    models = models.get(interface_triple(self.iface_class))
    models = models if isinstance(models, dict) else {}
    words = {'ready': 'sent and built', 'failed': "JetLink can't run it", 'left': 'left for the next park'}
    mine = []
    for name, _, sha256 in self.models:
      r = models.get(sha256)
      if isinstance(r, dict) and r.get('state') in words:
        detail = str(r.get('detail') or '')
        mine.append(f"{name} {words[r['state']]}" + (f": {detail}" if detail and r['state'] != 'ready' else ''))
    return {'last_check': '; '.join(mine) or str(run.get('result', '')), 'last_check_at': float(run['at'])}
