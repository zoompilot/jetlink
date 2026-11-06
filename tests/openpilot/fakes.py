"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The openpilot a jetlink test sees: the fork's adapter, faked.

No Params, no cloudlog, no modeld: a dict holds the JSON params, files in a
temporary directory hold the two settings jetlink reads as files, and a list
holds the log. Nothing here can reach a live params store, which is what
cleared JetlinkSpec on a comma once when a suite ran on the device.

The module is also an adapter module, adapter() and owner_config(), for the
entry points that take --adapter; JETLINK_FAKE_ROOT says where its files go.
Its top level is the standard library and jetlink's light half, as a real
one's must be (testing.check_adapter holds it to that): numpy is imported
where it is used (ones()).
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from jetlink.openpilot.interface import MODES, Keys, ModelFace, OwnerConfig
# one copy, published for forks; their names here are the ones the tests use
from jetlink.openpilot.testing import FakeDevice, FakeTensor, RecordingLog, fake_jit, fake_tinygrad  # noqa: F401

# the fork's names, so a log line or an assertion reads as it would on a comma
KEYS = Keys(link='JetlinkLink', offroad='IsOffroad', progress='AcceleratorProgress', spec='JetlinkSpec',
            big_model='ModelManager_ActiveBundleChestnut',
            catalog='ModelManager_ModelsCache_Chestnut', charge_phone='JetlinkChargePhone')
CHESTNUT_IDS = frozenset({(0xADD1, 0x0001), (0x3801, 0x0001), (0x174C, 0x2464), (0x174C, 0x2463)})
# (cam_w, cam_h, model_w, model_h): a comma 3X, and a comma four
TICI = (1928, 1208, 512, 256)
MICI = (1344, 760, 512, 256)


class FakeParser:
  """modeld's Parser, as far as jetlink uses it: the outputs as they came."""

  def parse_outputs(self, outputs: dict) -> dict:
    return dict(outputs)


def frame_size(width: int, height: int) -> int:
  """An NV12 buffer's size, as get_nv12_info(w, h)[3] gives it (without the stride padding)."""
  return width * height * 3 // 2


def get_action_from_model(*args):
  return ('action', args)


FACE = ModelFace(parser=FakeParser, frame_size=frame_size, desire_len=8,
                 constants=SimpleNamespace(MODEL_FREQ=20, DESIRE_LEN=8), lat_smooth_seconds=0.0,
                 long_smooth_seconds=0.3, get_action_from_model=get_action_from_model)


class FakeOpenpilot:
  """Every side of the interface, over a temporary directory."""

  def __init__(self, root: Path | None = None, *, camera=TICI, chestnut: bool = False,
               keys: Keys = KEYS):
    self.root = Path(root if root is not None else tempfile.mkdtemp())
    self.keys = keys
    self.log = RecordingLog()
    self.basedir = self.root / 'openpilot'
    self.basedir.mkdir(parents=True, exist_ok=True)
    # the params store, laid out as openpilot's under <root>/params
    self.store_dir = self.root / 'params' / 'd'
    self.store_dir.mkdir(parents=True, exist_ok=True)
    self.store: dict[str, object] = {}
    self.events: list[tuple[str, dict]] = []
    self.chestnut = chestnut
    self.geometry = camera
    self.put_error: Exception | None = None
    self.face = FACE
    # what in_control() answers: openpilot or MADS in control
    self.engaged = True

  # -- the readers ------------------------------------------------------------

  def get(self, key: str):
    # a round trip through JSON, as Params hands back a fresh decoded value
    value = self.store.get(key)
    return None if value is None else json.loads(json.dumps(value))

  def params_dir(self) -> Path:
    return self.store_dir

  def chestnut_present(self) -> bool:
    return self.chestnut

  def camera(self) -> tuple[int, int, int, int]:
    return self.geometry

  def warp_path(self, cam_w: int, cam_h: int, model_w: int, model_h: int) -> Path:
    return self.root / 'warps' / f'warp_{cam_w}x{cam_h}_{model_w}x{model_h}_tinygrad.pkl'

  # -- the provisioning run ----------------------------------------------------

  def put(self, key: str, value, *, block: bool = False) -> None:
    if self.put_error is not None:
      raise self.put_error
    self.store[key] = json.loads(json.dumps(value))

  def remove(self, key: str) -> None:
    self.store.pop(key, None)

  def model_root(self) -> Path:
    return self.root / 'models'

  # -- modeld ------------------------------------------------------------------

  def model_face(self) -> ModelFace:
    return self.face

  def in_control(self) -> bool:
    return self.engaged

  def event(self, name: str, **fields) -> None:
    self.events.append((name, fields))

  # -- the adapter module's, for jetlinkd --------------------------------------

  def owner_config(self) -> OwnerConfig:
    return OwnerConfig(params_dir=self.store_dir, keys=self.keys, chestnut_ids=CHESTNUT_IDS, adapter=__name__,
                       cwd=self.basedir, env={'PYTHONPATH': str(self.basedir)}, log_file=self.root / 'jetlink-owner.log')

  # -- what a test sets --------------------------------------------------------

  def set_mode(self, mode: str | None) -> None:
    """The Jetlink setting as the panels write it; None unsets it."""
    path = self.store_dir / self.keys.link
    if mode is None:
      path.unlink(missing_ok=True)
    else:
      path.write_bytes(str(MODES.index(mode)).encode())

  def set_offroad(self, parked: bool | None) -> None:
    path = self.store_dir / self.keys.offroad
    if parked is None:
      path.unlink(missing_ok=True)
    else:
      path.write_bytes(b'1' if parked else b'0')


def _root() -> Path:
  return Path(os.environ.get('JETLINK_FAKE_ROOT') or tempfile.mkdtemp())


def adapter() -> FakeOpenpilot:
  root = _root()
  if os.environ.get('JETLINK_FAKE_ISOLATE'):
    # an entry point run as its own process by a test: nothing of the
    # machine's own in reach for the life of it
    redirect(root)
  return FakeOpenpilot(root)


def owner_config() -> OwnerConfig:
  return adapter().owner_config()


# every file of the comma layer's that a jetlink.openpilot test could reach:
# its records in /dev/shm, the gadget in configfs and FunctionFS, the UDC and
# the USB-C port in sysfs, and the lend socket
GADGET_FILES = ('LINK', 'NET_STATUS', 'GADGET_STATUS', 'LENDER_STATUS', 'DORMANT', 'SHUTDOWN_REQUEST', 'STATUS',
                'STARTS', 'OWNER_LOCK', 'SERVER', 'CC_ORIENTATION', 'MODELD')
# the USB-C port's sysfs, by the name of its fake under a test's root
PORT_FILES = {'POWER_ROLE': 'current_pr', 'DATA_ROLE': 'current_dr', 'CONTRACT': 'contract',
              'TYPEC_MODE': 'typec_mode', 'CHARGER': 'real_type', 'UDC_MODE': 'udc-mode', 'USB_DEVICES': 'usb-devices',
              'THERMAL': 'thermal'}


def redirect(root: Path) -> list:
  """Point every file of the comma layer's that jetlink could read or write
  under `root`. The patchers, started; whoever called stops them."""
  from jetlink.comma import gadget, lending, port
  from jetlink.transport import base
  dev = root / 'dev'
  dev.mkdir(parents=True, exist_ok=True)
  patchers = [mock.patch.object(gadget, name, dev / name.lower()) for name in GADGET_FILES]
  patchers += [mock.patch.object(gadget, 'GADGET_PATH', root / 'configfs' / 'jetlink'),
               mock.patch.object(gadget, 'FFS_MOUNT', root / 'ffs-jetlink'),
               mock.patch.object(gadget, 'UDC_PATH', root / 'udc'),
               mock.patch.object(base, 'UDC_SYSFS', str(root / 'udc')),
               mock.patch.object(lending, 'SOCKET', dev / 'jetlink-lend.sock')]
  patchers += [mock.patch.object(port, name, root / f) for name, f in PORT_FILES.items()]
  for p in patchers:
    p.start()
  return patchers


def isolate(test: unittest.TestCase, root: Path) -> None:
  """redirect() for the length of `test`. conftest.py fails a test that still
  reaches the machine's own files."""
  for p in redirect(root):
    test.addCleanup(p.stop)


class OpenpilotTest(unittest.TestCase):
  """A test with a fresh fake adapter and jetlink bound to it, and nothing of
  the machine's own in reach."""

  def setUp(self):
    from jetlink.comma import gadget
    from jetlink.openpilot import bind
    self.tmp = Path(tempfile.mkdtemp())
    isolate(self, self.tmp)
    self.op = FakeOpenpilot(self.tmp)
    # bind points jetlink.comma's log at the fake's; put it back after
    p = mock.patch.object(gadget, 'log', gadget.log)
    p.start()
    self.addCleanup(p.stop)
    p = mock.patch.object(gadget.root, 'log', gadget.root.log)
    p.start()
    self.addCleanup(p.stop)
    self.jl = bind(self.op)
    # what jetlink keeps behind the API, for the tests of its internals
    self.parts = self.jl._parts

  def patch(self, target, name, *args, **kwargs):
    """mock.patch.object, undone after the test; the mock it made."""
    p = mock.patch.object(target, name, *args, **kwargs)
    self.addCleanup(p.stop)
    return p.start()


# -- a fork's small model, of each shape jetlink resets ---------------------------
# Each builder returns the model, the buffers its reset must zero in place,
# and the ones it must leave alone; all start as ones.

def ones(*shape: int):
  """A float32 buffer of ones; numpy imported here, so the top level stays light."""
  import numpy as np
  return np.ones(shape, np.float32)


class SmallModel:
  """A fork's ModelState of some shape: its attributes, and a run()."""

  def __init__(self, **attrs):
    self.__dict__.update(attrs)

  def run(self, bufs, transforms, inputs, after_enqueue=None):
    return {'from': 'small'}


def queue(device='CPU'):
  return FakeTensor(ones(4, 8), device=device)


def starpilot():
  """StarPilot's CatalogModelState: GPU and NPY tensors and numpy views, and
  none of state_pairs, packed_input, lat_delay, desire_key or numpy_inputs."""
  m = SmallModel(input_queues={'img_q': queue(), 'feat_q': queue(), 'npy_inputs': queue('NPY')},
                 npy={'desire': ones(8), 'traffic_convention': ones(2)},
                 prev_desire=ones(8))
  q = m.input_queues
  return m, [q['img_q'].array, q['feat_q'].array, *m.npy.values(), m.prev_desire], [q['npy_inputs'].array]


def stock():
  """Upstream modeld's ModelState: recurrent state and one packed upload, no lat_delay."""
  packed = ones(16)
  m = SmallModel(input_queues={'state_img_q': queue(), 'state_feat_q': queue(), 'desire': queue()},
                 state_pairs={'state_img_q': 'next_state_img_q', 'state_feat_q': 'next_state_feat_q'},
                 packed_input=packed, npy={'desire': packed[:8], 'traffic_convention': packed[8:10]},
                 prev_desire=ones(8))
  q = m.input_queues
  return m, [q['state_img_q'].array, q['state_feat_q'].array, packed, m.prev_desire], [q['desire'].array]


def v2_native():
  """A modeld_v2 native bundle: its adapter keeps stock modeld's layout."""
  packed = ones(16)
  adapter = SimpleNamespace(is_native=True, input_queues={'state_img_q': queue(), 'desire': queue()},
                            state_pairs={'state_img_q': 'next_state_img_q'}, packed_input=packed)
  m = SmallModel(adapter=adapter, prev_desire=ones(8), desire_key='desire', numpy_inputs={},
                 lat_delay=0.1, PLANPLUS_CONTROL=1.0)
  q = adapter.input_queues
  return m, [q['state_img_q'].array, packed, m.prev_desire], [q['desire'].array]


def v2_legacy():
  """A modeld_v2 legacy bundle: GPU history queues, an NPY tensor, numpy inputs."""
  packed = ones(16)
  adapter = SimpleNamespace(is_native=False, input_queues={'img_q': queue(), 'packed_npy_inputs': queue('NPY')},
                            numpy_inputs={'desire': packed[:8], 'rest': packed[8:]})
  m = SmallModel(adapter=adapter, prev_desire=ones(8), desire_key='desire',
                 numpy_inputs=adapter.numpy_inputs, lat_delay=0.1, PLANPLUS_CONTROL=1.0)
  q = adapter.input_queues
  return m, [q['img_q'].array, packed, m.prev_desire], [q['packed_npy_inputs'].array]


SHAPES = (starpilot, stock, v2_native, v2_legacy)
