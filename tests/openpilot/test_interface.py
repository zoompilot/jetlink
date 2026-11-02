"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The interface the fork implements, frozen, and the checker both repos run.

A change to anything pinned here fails this test on purpose: the fork's
adapter implements these members, and a pin bump that changes one breaks it.
Decide whether the change is additive (a member jetlink reads with getattr and
a fallback) or breaking (API goes up), then change the table.
"""
from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

import jetlink.openpilot as jo
from jetlink.openpilot import interface
from jetlink.openpilot.settings import FileParams, Settings
from tests.openpilot import fakes

READERS = {
  'keys': None,
  'log': None,
  'catalog_selector': None,
  'params_dir': '() -> Path',
  'get': '(key: str) -> Any',
  'chestnut_present': '() -> bool',
  'camera': '() -> tuple[int, int, int, int]',
  'warp_path': '(cam_w: int, cam_h: int, model_w: int, model_h: int) -> Path',
}
WORKER = {
  **READERS,
  'basedir': None,
  'put': '(key: str, value: Any, *, block: bool = False) -> None',
  'remove': '(key: str) -> None',
  'model_root': '() -> Path',
}
MODELD = {
  **WORKER,
  'model_face': '() -> ModelFace',
  'in_control': '() -> bool',
  'event': '(name: str, **fields: Any) -> None',
  'small_reset': '(small: Any) -> Callable[[], None] | None',
  'model_warp': '(small: Any, cam_w: int, cam_h: int, model_w: int, model_h: int) -> FrameWarp | None',
  'camera_calibration': '() -> dict | None',
}
SIDES = {'StatusSide': READERS, 'WorkerSide': WORKER, 'ModelSide': MODELD, 'Openpilot': MODELD}
# what modeld hands attach, and what it reads off and writes onto the result
RUN = '(bufs: Any, transforms: Any, inputs: Any, after_enqueue: Callable[[], None] | None = None) -> Any'
SMALL = {'desire_key': None, 'numpy_inputs': None, 'lat_delay': None, 'PLANPLUS_CONTROL': None, 'run': RUN}
JOINED = {'handovers': None, 'big_model_state': None, 'chestnut': None, 'get_action_from_model': None,
          'constants': None, 'LAT_SMOOTH_SECONDS': None, 'LONG_SMOOTH_SECONDS': None, 'frame_drop_ratio': None,
          'lat_delay': None, 'run': RUN, 'close': '() -> None'}
# what an adapter's model_warp returns
WARP = {'output': None, 'start': '(frame: int, big_frame: int, tfm: Any, big_tfm: Any) -> None', 'wait': '() -> None'}
MODELS = {'SmallModel': SMALL, 'JoinedModel': JOINED, 'FrameWarp': WARP}
# the members an implementation may lack: jetlink reads them with getattr and a fallback
OPTIONAL = {'ModelSide': {'small_reset', 'model_warp', 'camera_calibration'}, 'Openpilot': {'small_reset', 'model_warp', 'camera_calibration'},
            'SmallModel': {'desire_key', 'numpy_inputs', 'lat_delay', 'PLANPLUS_CONTROL'},
            'JoinedModel': {'get_action_from_model', 'constants', 'LAT_SMOOTH_SECONDS', 'LONG_SMOOTH_SECONDS', 'lat_delay'}}

FIELDS = {
  'Keys': ['link', 'offroad', 'progress', 'spec', 'pointers', 'big_model', 'catalog', 'charge_phone', 'select'],
  'OwnerConfig': ['params_dir', 'keys', 'chestnut_ids', 'adapter', 'cwd', 'env', 'log_file'],
  'ModelFace': ['parser', 'frame_size', 'desire_len', 'constants', 'lat_smooth_seconds', 'long_smooth_seconds',
                'get_action_from_model'],
}


def described(protocol: type) -> dict[str, str | None]:
  # the annotations are strings here (postponed evaluation); drop their quotes
  return {n: None if s is None else str(s).replace("'", '') for n, s in interface.members(protocol).items()}


class TestTheContract(unittest.TestCase):
  def test_the_api_version(self):
    self.assertEqual(jo.API, 3)

  def test_every_side_is_as_pinned(self):
    for name, expected in SIDES.items():
      with self.subTest(name):
        self.assertEqual(described(getattr(interface, name)), expected)

  def test_the_models_either_side_of_attach_are_as_pinned(self):
    for name, expected in MODELS.items():
      with self.subTest(name):
        self.assertEqual(described(getattr(interface, name)), expected)

  def test_the_optional_members_are_as_pinned(self):
    # each of these is additive: an adapter or a model without it still conforms
    for name in (*SIDES, *MODELS):
      with self.subTest(name):
        self.assertEqual(interface.optional(getattr(interface, name)), OPTIONAL.get(name, set()))

  def test_the_data_it_hands_over_is_as_pinned(self):
    for name, expected in FIELDS.items():
      with self.subTest(name):
        cls = getattr(interface, name)
        self.assertEqual([f.name for f in dataclasses.fields(cls)], expected)
        self.assertTrue(cls.__dataclass_params__.frozen, 'handed between processes and threads; nobody may change it')

  def test_the_settings_and_the_states_are_the_forks_names(self):
    # the panels store an index into MODES; selfdrived reads STATES as capnp enum names
    self.assertEqual(jo.MODES, ('off', 'usb', 'ios'))
    self.assertEqual(jo.STATES, ('none', 'joining', 'retrying', 'ready', 'running', 'unavailable'))

  def test_keys_without_a_model_manager(self):
    # upstream openpilot has no model manager: no pick to watch, no catalog
    keys = jo.Keys(link='JetlinkLink', offroad='IsOffroad', progress='P', spec='S', pointers='Q')
    self.assertIsNone(keys.big_model)
    self.assertIsNone(keys.catalog)


class TestConformance(unittest.TestCase):
  def setUp(self):
    self.op = fakes.FakeOpenpilot(Path(tempfile.mkdtemp()))

  def test_the_fake_is_a_whole_adapter(self):
    self.assertEqual(jo.conformance(self.op, jo.Openpilot), [])

  def test_a_side_needs_only_its_own_members(self):
    class Reader:
      keys = fakes.KEYS
      log = fakes.RecordingLog()
      catalog_selector = 0

      def params_dir(self): ...
      def get(self, key): ...
      def chestnut_present(self): ...
      def camera(self): ...
      def warp_path(self, cam_w, cam_h, model_w, model_h): ...

    self.assertEqual(jo.conformance(Reader(), interface.StatusSide), [])
    self.assertIn('put: missing', jo.conformance(Reader(), interface.WorkerSide))

  def test_an_adapter_without_the_optional_hooks_conforms(self):
    # zoompilot's adapter has neither, and runs this against itself
    self.assertFalse(hasattr(self.op, 'small_reset'))
    self.assertFalse(hasattr(self.op, 'model_warp'))
    self.assertEqual(jo.conformance(self.op, jo.Openpilot), [])

  def test_an_optional_member_that_is_there_is_held_to_its_signature(self):
    self.op.small_reset = lambda small: None
    self.op.model_warp = lambda small, cam_w, cam_h, model_w, model_h: None
    self.assertEqual(jo.conformance(self.op, jo.Openpilot), [])
    self.op.small_reset = lambda: None
    self.op.model_warp = lambda small, cam_w, cam_h, geometry: None
    self.assertEqual(jo.conformance(self.op, jo.Openpilot), [
      'small_reset(): expected small_reset(small)',
      'model_warp(small, cam_w, cam_h, geometry): expected model_warp(small, cam_w, cam_h, model_w, model_h)'])

  def test_a_missing_member_is_named(self):
    del self.op.catalog_selector
    self.assertEqual(jo.conformance(self.op, jo.Openpilot), ['catalog_selector: missing'])

  def test_other_parameters_are_named(self):
    self.op.put = lambda key, value: None   # no block
    [problem] = jo.conformance(self.op, jo.Openpilot)
    self.assertEqual(problem, 'put(key, value): expected put(key, value, *, block=False)')

  def test_a_positional_parameter_is_the_adapters_to_name(self):
    # jetlink passes it by position; a caller cannot trip on its name
    self.op.remove = lambda name: None
    self.op.get = lambda param: None
    self.assertEqual(jo.conformance(self.op, jo.Openpilot), [])

  def test_a_keyword_parameter_must_keep_its_name(self):
    # jetlink passes block by keyword
    self.op.put = lambda key, value, *, blocking=False: None
    self.assertEqual(len(jo.conformance(self.op, jo.Openpilot)), 1)

  def test_one_parameter_too_many_or_too_few_is_a_difference(self):
    self.op.warp_path = lambda cam_w, cam_h, model_w: None
    self.op.camera = lambda which: None
    self.assertEqual(len(jo.conformance(self.op, jo.Openpilot)), 2)

  def test_a_default_where_none_is_expected_is_a_difference(self):
    # a caller relying on it being required never passes it; the other way round breaks
    self.op.put = lambda key, value=None, *, block=False: None
    self.assertEqual(len(jo.conformance(self.op, jo.Openpilot)), 1)

  def test_a_member_that_is_not_callable_is_named(self):
    self.op.camera = (1928, 1208, 512, 256)
    self.assertEqual(jo.conformance(self.op, jo.Openpilot), ['camera: not callable'])


class TestFileParams(unittest.TestCase):
  """params.cc's put format, read back off the files."""

  def setUp(self):
    self.dir = Path(tempfile.mkdtemp())
    self.params = FileParams(self.dir)
    self.settings = Settings(self.params, fakes.KEYS)

  def write(self, key: str, value: bytes) -> None:
    (self.dir / key).write_bytes(value)

  def test_a_bool_is_true_and_nothing_else(self):
    for raw, expected in ((b'1', True), (b'0', False), (b'', False), (b'true', True), (b'1\n', True)):
      self.write('IsOffroad', raw)
      self.assertIs(self.params.get_bool('IsOffroad'), expected, raw)
      self.assertIs(self.settings.offroad(), expected, raw)

  def test_a_missing_param_is_not_a_false(self):
    # None and False are different answers: offroad treats an unwritten param
    # as parked, and the link setting treats it as off
    self.assertIsNone(self.params.get_bool('IsOffroad'))
    self.assertIsNone(self.params.get_int('JetlinkLink'))
    self.assertTrue(self.settings.offroad())
    self.assertEqual(self.settings.mode(), 'off')

  def test_the_link_setting_is_an_index(self):
    for raw, expected in ((b'0', 'off'), (b'1', 'usb'), (b'2', 'ios'), (b'3', 'off'), (b'-1', 'off'),
                          (b'usb', 'off'), (b'', 'off')):
      self.write('JetlinkLink', raw)
      self.assertEqual(self.settings.mode(), expected, raw)

  def test_a_directory_that_is_not_there_reads_as_unset(self):
    settings = Settings(FileParams(self.dir / 'nowhere'), fakes.KEYS)
    self.assertEqual((settings.mode(), settings.offroad()), ('off', True))
    self.assertEqual(settings.marks(), {'ModelManager_ActiveBundleChestnut': 0, 'JetlinkSpec': 0})

  def test_marks_follow_the_pick_and_the_spec(self):
    self.assertEqual(self.settings.marks(), {'ModelManager_ActiveBundleChestnut': 0, 'JetlinkSpec': 0})
    self.write('JetlinkSpec', b'{}')
    marks = self.settings.marks()
    self.assertEqual(marks['JetlinkSpec'], (self.dir / 'JetlinkSpec').stat().st_mtime_ns)
    self.assertGreater(marks['JetlinkSpec'], 0)
    self.assertEqual(marks['ModelManager_ActiveBundleChestnut'], 0)

  def test_without_a_model_manager_only_the_spec_is_watched(self):
    keys = dataclasses.replace(fakes.KEYS, big_model=None, catalog=None)
    self.assertEqual(list(Settings(self.params, keys).marks()), ['JetlinkSpec'])


if __name__ == '__main__':
  unittest.main()
