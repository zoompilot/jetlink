"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Stock, the base adapter, over a fake openpilot.

No openpilot here, and none needed: the fake puts a stub module in
sys.modules for exactly the names stock.py imports, which a test below holds
to stock.py's own imports, so a new import cannot slip past it. Whether those
names exist upstream is scripts/upstream's question (the upstream-paths job);
these tests say what Stock does with them.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import numpy as np

from jetlink.comma import port, root
from jetlink.openpilot import interface, stock, testing
from jetlink.openpilot.interface import Openpilot
from jetlink.openpilot.stock import Stock
from tests import load_script
from tests.openpilot import fakes

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'jetlink' / 'openpilot' / 'stock.py'


class UnknownKeyName(Exception):
  pass


class FakeParams:
  """openpilot's Params over a dict, declaring only `declared`."""
  made = 0

  def __init__(self, world):
    self.world = world
    type(self).made += 1

  def _check(self, key):
    if self.world.params_error is not None:
      raise self.world.params_error
    if key not in self.world.declared:
      raise UnknownKeyName(key)

  def get(self, key, block=False, return_default=False):
    self._check(key)
    return self.world.params.get(key)

  def put(self, key, dat, block=False):
    self._check(key)
    self.world.params[key] = dat
    self.world.puts.append((key, dat, block))

  def remove(self, key):
    self._check(key)
    self.world.params.pop(key, None)


class FakeEvent:
  """cereal's log.Event over JSON: {which: {field: value}}, as calibrationd saves one in CalibrationParams."""

  @staticmethod
  def from_bytes(raw):
    which, fields = next(iter(json.loads(raw).items()))
    msg = SimpleNamespace(which=lambda: which, **{which: SimpleNamespace(**fields)})
    return mock.MagicMock(__enter__=lambda self: msg, __exit__=lambda *a: False)


class FakeSubMaster(testing.FakeSubMaster):
  """The kit's, over the world's messages, failing as the world says."""

  def __init__(self, world, services):
    if world.sm_error is not None:
      raise world.sm_error
    super().__init__(services, **{s: world.messages[s] for s in services if s in world.messages})
    self.world = world
    world.submasters.append(self)

  def update(self, timeout=100):
    if self.world.update_error is not None:
      raise self.world.update_error
    super().update(timeout)


class FakeLog(fakes.RecordingLog):
  def __init__(self):
    super().__init__()
    self.events: list[tuple[str, dict]] = []

  def event(self, event, *args, **kwargs):
    self.events.append((event, kwargs))


def nv12_info(width, height):
  return width, height, height // 2, width * height * 3 // 2 + 7   # the +7 shows [3] is what Stock takes


class World:
  """The fake openpilot's state, and its modules by name."""

  def __init__(self, root: Path, *, device='tici', model_root: bool = False):
    self.root = root
    self.declared = {'JetlinkLink', 'JetlinkSpec', 'IsOffroad'}
    self.params: dict = {}
    self.puts: list = []
    self.params_error: Exception | None = None
    self.device = device
    self.log = FakeLog()
    self.submasters: list[FakeSubMaster] = []
    self.sm_error: Exception | None = None
    self.update_error: Exception | None = None
    self.messages = {
      'carState': SimpleNamespace(),
      'carControl': SimpleNamespace(enabled=False, latActive=False, longActive=False),
      'selfdriveState': SimpleNamespace(enabled=False, active=False),
    }
    world = self

    class Paths:
      @staticmethod
      def comma_home():
        return str(root / 'home' / '.comma')

    if model_root:
      Paths.model_root = staticmethod(lambda: str(root / 'sp-models'))

    class Parser:
      def parse_outputs(self, outs):
        return dict(outs)

    self.constants = SimpleNamespace(DESIRE_LEN=8, MODEL_FREQ=20)
    self.action = mock.Mock(name='get_action_from_model')
    # (module, name) -> value: stock.py's openpilot imports, and nothing else
    self.names = {
      ('openpilot.common.basedir', 'BASEDIR'): str(root / 'checkout'),
      ('openpilot.common.swaglog', 'cloudlog'): self.log,
      ('openpilot.common.params', 'Params'): lambda: FakeParams(world),
      ('openpilot.common.hardware', 'HARDWARE'): SimpleNamespace(get_device_type=lambda: world.device),
      ('openpilot.common.hardware.hw', 'Paths'): Paths,
      ('openpilot.common.transformations.camera', '_ar_ox_fisheye'): SimpleNamespace(width=1928, height=1208),
      ('openpilot.common.transformations.camera', '_os_fisheye'): SimpleNamespace(width=1344, height=760),
      ('openpilot.common.transformations.model', 'MEDMODEL_INPUT_SIZE'): (512, 256),
      ('openpilot.selfdrive.modeld.constants', 'ModelConstants'): self.constants,
      ('openpilot.selfdrive.modeld.modeld', 'LAT_SMOOTH_SECONDS'): 0.0,
      ('openpilot.selfdrive.modeld.modeld', 'LONG_SMOOTH_SECONDS'): 0.3,
      ('openpilot.selfdrive.modeld.modeld', 'get_action_from_model'): self.action,
      ('openpilot.selfdrive.modeld.parse_model_outputs', 'Parser'): Parser,
      ('openpilot.system.camerad.cameras.nv12_info', 'get_nv12_info'): nv12_info,
      ('openpilot.cereal', 'messaging'): None,   # the module below
      ('openpilot.cereal', 'log'): SimpleNamespace(Event=FakeEvent),
    }

  def modules(self) -> dict[str, ModuleType]:
    found: dict[str, ModuleType] = {}

    def module(name):
      if name not in found:
        found[name] = ModuleType(name)
        if '.' in name:
          parent, _, child = name.rpartition('.')
          setattr(module(parent), child, found[name])
      return found[name]

    for (name, attr), value in self.names.items():
      setattr(module(name), attr, value)
    module('openpilot.cereal.messaging').SubMaster = lambda services: FakeSubMaster(self, services)
    return found


class StockTest(unittest.TestCase):
  """A fake openpilot in sys.modules, and nothing of the machine's own in reach."""
  model_root = False

  def setUp(self):
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    self.tmp = Path(tmp.name)
    fakes.isolate(self, self.tmp)
    self.world = World(self.tmp, model_root=self.model_root)
    for p in (mock.patch.dict(sys.modules, self.world.modules()),
              mock.patch.dict(os.environ, {'HOME': str(self.tmp / 'home')}),
              mock.patch.object(FakeParams, 'made', 0)):
      p.start()
      self.addCleanup(p.stop)
    os.environ.pop('PARAMS_ROOT', None)
    os.environ.pop('OPENPILOT_PREFIX', None)
    self.op = Stock()


class TestShape(StockTest):
  def test_it_is_the_whole_interface(self):
    assert interface.conformance(self.op, Openpilot) == []

  def test_its_keys_are_the_canonical_names_with_jetlinks_default_model(self):
    keys = self.op.keys
    assert (keys.link, keys.offroad, keys.progress, keys.spec, keys.pointers, keys.charge_phone) == \
      ('JetlinkLink', 'IsOffroad', 'JetlinkProgress', 'JetlinkSpec', 'JetlinkModelPointers', 'JetlinkChargePhone')
    assert keys.big_model is None and keys.catalog is None
    assert self.op.catalog_selector == 0

  def test_a_subclass_overrides_the_data(self):
    class Fork(Stock):
      KEYS = fakes.KEYS
      CATALOG_SELECTOR = 19
    assert Fork().keys is fakes.KEYS and Fork().catalog_selector == 19

  def test_the_chestnut_ids_are_the_running_and_rom_ones(self):
    assert Stock.CHESTNUT_IDS == fakes.CHESTNUT_IDS

  def test_making_one_imports_nothing(self):
    with mock.patch.dict(sys.modules, {'openpilot': None}):
      Stock()


class TestParams(StockTest):
  def test_get_reads_the_store(self):
    self.world.params['JetlinkSpec'] = {'built': True}
    assert self.op.get('JetlinkSpec') == {'built': True}
    assert self.op.get('JetlinkLink') is None

  def test_get_never_raises(self):
    assert self.op.get('NotDeclared') is None
    self.world.params_error = RuntimeError('params_last_error')
    assert self.op.get('JetlinkSpec') is None

  def test_get_is_none_when_params_cannot_be_made_and_backs_off(self):
    made = self.world.names[('openpilot.common.params', 'Params')] = mock.Mock(side_effect=OSError('no library'))
    with mock.patch.dict(sys.modules, self.world.modules()), mock.patch.object(stock.time, 'monotonic') as now:
      now.return_value = 100.0
      op = Stock()
      assert [op.get('JetlinkSpec') for _ in range(3)] == [None] * 3
      assert made.call_count == 1
      now.return_value += stock.RETRY_S
      assert op.get('JetlinkSpec') is None
      assert made.call_count == 2

  def test_put_and_remove_go_through_and_may_raise(self):
    self.op.put('JetlinkSpec', {'a': 1}, block=True)
    assert self.world.puts == [('JetlinkSpec', {'a': 1}, True)]
    self.op.remove('JetlinkSpec')
    assert 'JetlinkSpec' not in self.world.params
    with self.assertRaises(UnknownKeyName):
      self.op.put('NotDeclared', 1)

  def test_one_params_per_store(self):
    for _ in range(3):
      self.op.get('JetlinkLink')
    assert FakeParams.made == 1
    os.environ['OPENPILOT_PREFIX'] = 'bench'
    self.op.get('JetlinkLink')
    assert FakeParams.made == 2

  def test_params_dir_is_params_ccs_rule(self):
    home = self.tmp / 'home'
    assert self.op.params_dir() == home / '.comma' / 'params' / 'd'
    os.environ['OPENPILOT_PREFIX'] = 'p1'
    assert self.op.params_dir() == home / '.commap1' / 'params' / 'p1'
    os.environ['PARAMS_ROOT'] = str(self.tmp / 'pr')
    assert self.op.params_dir() == self.tmp / 'pr' / 'p1'
    del os.environ['PARAMS_ROOT'], os.environ['OPENPILOT_PREFIX']
    with mock.patch.object(root, 'AGNOS', True):
      assert self.op.params_dir() == Path('/data/params/d')

  def test_params_dir_without_home_is_the_c_plus_pluses(self):
    # getenv("HOME") + "/.comma": an absolute /.comma, never a relative one
    del os.environ['HOME']
    assert self.op.params_dir() == Path('/.comma/params/d')


class TestDevice(StockTest):
  def test_camera_calibration_is_calibrationds_saved_one(self):
    self.world.declared.add('CalibrationParams')
    assert self.op.camera_calibration() is None, 'not yet calibrated: nothing saved'
    for which in ('liveCalibration', 'extrinsicsCalibration'):  # upstream's name, and a fork's
      self.world.params['CalibrationParams'] = json.dumps(
        {which: {'calStatus': 'calibrated', 'wideFromDeviceEuler': [0.0, 0.0217, -0.0161], 'height': [1.374]}}).encode()
      assert self.op.camera_calibration() == {'wide_from_device_euler': [0.0, 0.0217, -0.0161], 'height': 1.374}
    self.world.params['CalibrationParams'] = json.dumps(
      {'liveCalibration': {'calStatus': 'uncalibrated', 'wideFromDeviceEuler': [0.0, 0.0, 0.0], 'height': [1.22]}}).encode()
    assert self.op.camera_calibration() is None, 'only a calibrated one'
    self.world.params['CalibrationParams'] = json.dumps(
      {'liveCalibration': {'calStatus': 'calibrated', 'wideFromDeviceEuler': [0.0, 0.0217, -0.0161], 'height': []}}).encode()
    assert self.op.camera_calibration() is None, 'only a whole one'
    self.world.params['CalibrationParams'] = b'not an event'
    assert self.op.camera_calibration() is None, 'never raises'

  def test_camera_by_device(self):
    assert self.op.camera() == fakes.TICI
    self.world.device = 'mici'
    assert self.op.camera() == fakes.MICI

  def test_warp_path_is_modelds_driving_warp(self):
    assert self.op.warp_path(1344, 760, 512, 256) == \
      self.tmp / 'checkout' / 'openpilot/selfdrive/modeld/models/driving_warp_1344x760_tinygrad.pkl'

  def test_model_root_without_paths_model_root(self):
    assert self.op.model_root() == self.tmp / 'home' / '.comma' / 'media' / '0' / 'models'
    with mock.patch.object(root, 'AGNOS', True):
      assert self.op.model_root() == Path('/data/media/0/models')

  def usb(self, *devices):
    tree = self.tmp / 'usb-devices'
    tree.mkdir(exist_ok=True)
    for name, vid, pid in devices:
      (tree / name).mkdir()
      (tree / name / 'idVendor').write_text(f'{vid:04x}\n')
      (tree / name / 'idProduct').write_text(f'{pid:04x}\n')
    assert port.USB_DEVICES == tree   # fakes.isolate's

  def test_no_chestnut(self):
    assert not self.op.chestnut_present()
    self.usb(('usb1', 0x1D6B, 0x0002), ('1-1', 0x0955, 0x7020))
    assert not self.op.chestnut_present()

  def test_a_chestnut_in_its_rom_counts(self):
    # modeld's own test ignores one until its firmware matches; jetlink never takes the port beside it
    self.usb(('usb1', 0x1D6B, 0x0002), ('1-1', 0x174C, 0x2464))
    assert self.op.chestnut_present()

  def test_a_running_chestnut_counts_and_interfaces_do_not(self):
    self.usb(('1-1:1.0', 0xADD1, 0x0001))
    assert not self.op.chestnut_present()
    self.usb(('2-1', 0x3801, 0x0001))
    assert self.op.chestnut_present()


class TestPathsModelRoot(StockTest):
  model_root = True

  def test_model_root_is_paths_model_root_where_there_is_one(self):
    assert self.op.model_root() == self.tmp / 'sp-models'


class TestModeld(StockTest):
  def test_model_face(self):
    face = self.op.model_face()
    assert face.parser().parse_outputs({'a': 1}) == {'a': 1}
    assert face.parser() is not face.parser()
    assert face.frame_size(1928, 1208) == 1928 * 1208 * 3 // 2 + 7
    assert (face.desire_len, face.lat_smooth_seconds, face.long_smooth_seconds) == (8, 0.0, 0.3)
    assert face.constants is self.world.constants and face.get_action_from_model is self.world.action

  def test_model_face_from_the_overrides(self):
    v2 = SimpleNamespace(DESIRE_LEN=8, MODEL_FREQ=20, V2=True)

    class Fork(Stock):
      def model_constants(self):
        return v2
    assert Fork().model_face().constants is v2

  def test_event_is_cloudlogs(self):
    self.op.event('jetlink_swap', drops=2)
    assert self.world.log.events == [('jetlink_swap', {'drops': 2})]


class TestInControl(StockTest):
  def test_nothing_engaged(self):
    assert self.op.in_control() is False
    assert self.world.submasters[0].timeouts == [0]
    assert self.world.submasters[0].services == ['carState', 'carControl', 'selfdriveState']

  def test_each_engagement_counts(self):
    for msg, field in (('carControl', 'enabled'), ('carControl', 'latActive'), ('carControl', 'longActive'),
                       ('selfdriveState', 'enabled'), ('selfdriveState', 'active')):
      setattr(self.world.messages[msg], field, True)
      assert self.op.in_control() is True, (msg, field)
      setattr(self.world.messages[msg], field, False)
    assert self.op.in_control() is False
    assert len(self.world.submasters) == 1

  def test_stale_or_invalid_counts_as_in_control(self):
    assert self.op.in_control() is False
    [sm] = self.world.submasters
    sm.alive['carState'] = False
    assert self.op.in_control() is True
    sm.alive['carState'], sm.valid['carControl'] = True, False
    assert self.op.in_control() is True

  def test_a_failure_counts_as_in_control_and_logs_once(self):
    self.world.update_error = RuntimeError('msgq')
    assert [self.op.in_control() for _ in range(3)] == [True] * 3
    assert len(self.world.log.lines('error')) == 1
    self.world.update_error = None
    assert self.op.in_control() is False
    self.world.update_error = RuntimeError('msgq')
    assert self.op.in_control() is True
    assert len(self.world.log.lines('error')) == 2

  def test_no_submaster_counts_as_in_control_and_backs_off(self):
    self.world.sm_error = OSError('no msgq')
    with mock.patch.object(stock.time, 'monotonic') as now:
      now.return_value = 100.0
      assert [self.op.in_control() for _ in range(20)] == [True] * 20
      assert len(self.world.log.lines('error')) == 1   # tried once
      self.world.sm_error = None
      assert self.op.in_control() is True and self.world.submasters == []
      now.return_value += stock.RETRY_S
      assert self.op.in_control() is False and len(self.world.submasters) == 1

  def test_the_base_rule_reads_only_the_listed_services(self):
    # a fork that lists another service and keeps the base engaged()
    class Fork(Stock):
      IN_CONTROL_SERVICES = ('carState', 'carControl', 'carControlSP')
    del self.world.messages['selfdriveState']
    self.world.messages['carControlSP'] = SimpleNamespace()
    fork = Fork()
    assert fork.in_control() is False and self.world.log.lines('error') == []
    self.world.messages['carControl'].latActive = True
    assert fork.in_control() is True

  def test_a_forks_engagement_rule(self):
    class Fork(Stock):
      IN_CONTROL_SERVICES = ('carState', 'carControl', 'carControlSP')

      def engaged(self, sm):
        return sm['carControlSP'].mads.enabled
    self.world.messages['carControlSP'] = SimpleNamespace(mads=SimpleNamespace(enabled=True))
    fork = Fork()
    assert fork.in_control() is True
    self.world.messages['carControlSP'].mads.enabled = False
    assert fork.in_control() is False
    assert self.world.submasters[0].services == ['carState', 'carControl', 'carControlSP']

  def test_a_rule_that_raises_counts_as_in_control(self):
    class Fork(Stock):
      def engaged(self, sm):
        return sm['carControlSP']
    assert Fork().in_control() is True


class TestSmallReset(StockTest):
  def setUp(self):
    super().setUp()
    p = mock.patch.dict(sys.modules, fakes.fake_tinygrad())
    p.start()
    self.addCleanup(p.stop)

  def test_stock_modelstate_is_zeroed_in_place(self):
    model, zeroed, kept = fakes.stock()
    queues = dict(model.input_queues)
    reset = self.op.small_reset(model)
    assert reset is not None
    reset()
    assert all(not a.any() for a in zeroed)
    assert all(a.all() for a in kept)
    # the buffers modeld's JIT is linked to, every one of them
    assert {k: id(v) for k, v in model.input_queues.items()} == {k: id(v) for k, v in queues.items()}
    assert all(model.input_queues[k].array is queues[k].array for k in queues)
    assert np.shares_memory(model.npy['desire'], model.packed_input)

  def test_another_shape_is_left_to_jetlink(self):
    for shape in (fakes.starpilot, fakes.v2_native, fakes.v2_legacy):
      assert self.op.small_reset(shape()[0]) is None, shape.__name__


def _adapter_class():
  # the pattern a fork's adapter module follows: the subclass defined lazily
  class Adapter(Stock):
    KEYS = fakes.KEYS
  return Adapter


class TestOwner(StockTest):
  def test_owner_config(self):
    config = _adapter_class().owner_config()
    assert config.adapter == __name__
    assert config.keys is fakes.KEYS and config.chestnut_ids == fakes.CHESTNUT_IDS
    assert config.params_dir == self.tmp / 'home' / '.comma' / 'params' / 'd'
    assert config.cwd == self.tmp / 'checkout' and config.env == {'PYTHONPATH': str(self.tmp / 'checkout')}
    assert config.log_file == Path('/data/log/jetlink-owner.log')

  def test_adapter_module_names_another(self):
    class Elsewhere(Stock):
      ADAPTER_MODULE = 'starpilot.jetlink.adapter'
    assert Elsewhere.owner_config().adapter == 'starpilot.jetlink.adapter'

  def test_main_runs_the_owner(self):
    with mock.patch('jetlink.openpilot.owner.main') as run:
      _adapter_class().main()
    run.assert_called_once_with(_adapter_class().owner_config())


OWNER = '''
import json, sys
before = set(sys.modules)
from jetlink.openpilot.stock import Stock
config = Stock.owner_config()
print(json.dumps(sorted(set(sys.modules) - before)))
'''


def test_owner_config_imports_only_basedir(tmp_path):
  # in a fresh interpreter, over a checkout holding only common/basedir.py:
  # the owner pays for nothing of openpilot's past os
  (tmp_path / 'openpilot' / 'common').mkdir(parents=True)
  for f in ('openpilot/__init__.py', 'openpilot/common/__init__.py'):
    (tmp_path / f).write_text('')
  (tmp_path / 'openpilot/common/basedir.py').write_text("BASEDIR = '/x'\n")
  env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(ROOT), str(tmp_path)])}
  run = subprocess.run([sys.executable, '-c', OWNER], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
  assert run.returncode == 0, run.stderr
  loaded = json.loads(run.stdout)
  assert [m for m in loaded if m.startswith('openpilot')] == ['openpilot', 'openpilot.common', 'openpilot.common.basedir']
  foreign = sorted({m.split('.')[0] for m in loaded} - set(sys.stdlib_module_names) - {'jetlink', 'openpilot'})
  assert foreign == []



def _upstream(name: str):
  # the checker as CI runs it, with check_imports' own path edit kept to this test
  with mock.patch.object(sys, 'path', list(sys.path)), mock.patch.dict(sys.modules):
    return load_script(ROOT / 'scripts' / 'upstream' / f'{name}.py')

def test_the_upstream_checker_reads_stocks_literals():
  # what check_paths holds the tree to is what Stock holds
  values = _upstream('check_paths').stock_values()
  for name in ('CHESTNUT_IDS', 'IN_CONTROL_SERVICES', 'WARP_DIR', 'WARP_NAME', 'CATALOG_SELECTOR'):
    assert values[name] == getattr(Stock, name), name
  assert values['KEYS']['offroad'] == Stock.KEYS.offroad
  assert values['_ENGAGED'] == stock._ENGAGED


def test_the_fake_stubs_exactly_what_stock_imports(tmp_path):
  # read off stock.py as the upstream check reads it, so a new import cannot slip past either
  found = set(_upstream('check_imports').openpilot_imports(SOURCE))
  assert found == set(World(tmp_path).names)
