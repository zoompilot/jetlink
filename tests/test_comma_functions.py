"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

A fork's extra interfaces on the gadget: how one is declared and described,
what the owner does with it around its own bind, and what the root script
refuses. Linking and mounting need a real configfs and are not tested here.
"""
import errno
import os
import struct
from pathlib import Path
from unittest import mock

import pytest

from jetlink import protocol as P
from jetlink.addons.addon import Addon
from jetlink.comma import functions, gadget, owner
from jetlink.openpilot.interface import OwnerConfig
from jetlink.openpilot.stock import Stock
from jetlink.transport import ffs
from tests.comma_fakes import run_script
from tests.test_comma_owner import OwnerTest

VASM = functions.GadgetFunction('vasm', (0xFF, 0x56, 0x41))
SLV = functions.GadgetFunction('slv', (0xFF, 0x53, 0x4C))


def interface_classes(descriptors: bytes) -> list[tuple[int, int, int]]:
  """Every interface descriptor's class triple, at each speed."""
  found, i = [], 12 + 12   # the v2 head, then the three per-speed counts
  while i < len(descriptors):
    length, kind = descriptors[i], descriptors[i + 1]
    if kind == ffs.USB_DT_INTERFACE:
      found.append(tuple(descriptors[i + 5:i + 8]))
    i += length
  return found


# -- declaring one ----------------------------------------------------------

@pytest.mark.parametrize('name', ['', 'Vasm', 'v-asm', 'a' * 17, '../x', 'jetlink', None])
def test_a_bad_name_is_refused(name):
  with pytest.raises(ValueError):
    functions.GadgetFunction(name, (0xFF, 0x56, 0x41))


@pytest.mark.parametrize('iface_class', [(0xFF, 0x56), (0xFF, 0x56, 0x100), (0xFF, -1, 0), P.USB_VENDOR_CLASS])
def test_a_bad_class_is_refused(iface_class):
  # Jetlink's own triple too: a host finds each interface by it
  with pytest.raises(ValueError):
    functions.GadgetFunction('vasm', iface_class)


def test_its_port_follows_its_class_and_two_on_one_port_are_named():
  assert VASM.port == functions.PORT_BASE + 0x5641 % functions.PORTS
  assert VASM.port != SLV.port
  assert functions.shared_ports((VASM, SLV)) == []
  twin = functions.GadgetFunction('twin', (0xFF, 0x57, 0xD1))   # 0x5641 + 400
  assert functions.shared_ports((VASM, twin)) == [f"gadget functions vasm and twin share port {VASM.port}; pick another class for one"]


def test_its_files_are_under_its_own_mount():
  assert VASM.mount == Path('/dev/ffs-vasm')
  assert (VASM.ep0.name, VASM.ep_out.name, VASM.ep_in.name) == ('ep0', 'ep1', 'ep2')


def test_a_fork_declares_them_on_its_adapter():
  class Adapter(Stock):
    ADDONS = (Addon(VASM.name, VASM.iface_class), Addon(SLV.name, SLV.iface_class))
  with mock.patch.dict('sys.modules', {'openpilot': mock.Mock(), 'openpilot.common': mock.Mock(),
                                       'openpilot.common.basedir': mock.Mock(BASEDIR='/data/openpilot')}):
    assert Adapter.owner_config().functions == (VASM, SLV)
    assert Stock.owner_config().functions == ()


def test_a_fork_picks_them_from_its_settings(tmp_path):
  (tmp_path / 'd').mkdir()
  (tmp_path / 'd' / 'ForkVasm').write_text('1')

  class Adapter(Stock):
    @classmethod
    def gadget_functions(cls, params):
      return (VASM,) if params.get_bool('ForkVasm') else ()
  with mock.patch.dict('os.environ', {'PARAMS_ROOT': str(tmp_path)}), \
       mock.patch.dict('sys.modules', {'openpilot': mock.Mock(), 'openpilot.common': mock.Mock(),
                                       'openpilot.common.basedir': mock.Mock(BASEDIR='/data/openpilot')}):
    os.environ.pop('OPENPILOT_PREFIX', None)
    assert Adapter.owner_config().functions == (VASM,)
    (tmp_path / 'd' / 'ForkVasm').write_text('0')
    assert Adapter.owner_config().functions == ()


def test_a_broken_pick_builds_the_gadget_without_extras():
  class Adapter(Stock):
    @classmethod
    def gadget_functions(cls, params):
      raise RuntimeError('fork bug')
  with mock.patch.dict('sys.modules', {'openpilot': mock.Mock(), 'openpilot.common': mock.Mock(),
                                       'openpilot.common.basedir': mock.Mock(BASEDIR='/data/openpilot')}):
    assert Adapter.owner_config().functions == ()


def test_the_config_has_none_by_default():
  config = OwnerConfig(Path('params'), None, frozenset(), 'adapter', Path('.'), {}, Path('log'))
  assert config.functions == ()


# -- describing one ---------------------------------------------------------

def test_jetlinks_own_descriptors_are_unchanged():
  assert ffs.build_descriptors() == ffs.build_descriptors(P.USB_VENDOR_CLASS)
  assert interface_classes(ffs.build_descriptors()) == [P.USB_VENDOR_CLASS] * 3


def test_an_extra_interface_carries_its_class_at_every_speed():
  assert interface_classes(ffs.build_descriptors(VASM.iface_class)) == [VASM.iface_class] * 3


def test_describe_writes_the_descriptors_then_the_strings(tmp_path, monkeypatch):
  monkeypatch.setattr(functions, 'MOUNT_PREFIX', str(tmp_path / 'ffs-'))
  VASM.mount.mkdir()
  VASM.ep0.write_bytes(b'')
  assert functions.present(VASM)
  fd = functions.describe(VASM)
  os.close(fd)
  written = VASM.ep0.read_bytes()
  descriptors = ffs.build_descriptors(VASM.iface_class)
  assert written == descriptors + ffs.build_strings('vasm')
  assert struct.unpack_from('<I', written)[0] == ffs.FUNCTIONFS_DESCRIPTORS_MAGIC_V2


def test_a_gadget_built_without_it_has_none_to_describe(tmp_path, monkeypatch):
  monkeypatch.setattr(functions, 'MOUNT_PREFIX', str(tmp_path / 'ffs-'))
  assert not functions.present(VASM)


# -- the gadget -------------------------------------------------------------

def test_setup_names_each_function_to_the_script():
  with mock.patch.object(gadget.root, 'run', return_value=True) as run, \
       mock.patch.object(gadget, 'link_configured', return_value=True):
    assert gadget.setup_gadget(False, (VASM, SLV))
    assert gadget.setup_gadget(False)
  assert run.call_args_list == [mock.call('gadget', '--function', 'vasm', '--function', 'slv'), mock.call('gadget')]


def test_the_linked_functions_are_read_off_the_config(tmp_path, monkeypatch):
  monkeypatch.setattr(gadget, 'GADGET_PATH', tmp_path)
  config = tmp_path / 'configs' / 'c.1'
  config.mkdir(parents=True)
  assert gadget.linked_functions() == frozenset()
  for name in ('ffs.jetlink', 'ffs.vasm', 'ncm.usb0'):
    (config / name).symlink_to(tmp_path)
  assert gadget.linked_functions() == {'vasm'}


@pytest.mark.parametrize('args', [('--function',), ('--function', 'Bad'), ('--function', 'jetlink'),
                                  ('--function', 'a' * 17), ('--function', '../x')])
def test_the_script_refuses_a_bad_name(tmp_path, args):
  result = run_script(tmp_path, 'gadget', *args)
  assert result.returncode == 2, result.stderr
  assert 'usage' in result.stderr


# -- the owner --------------------------------------------------------------

class FunctionsTest(OwnerTest):
  def setUp(self):
    super().setUp()
    self.ffs = self.tmp / 'ffs-'
    p = mock.patch.object(functions, 'MOUNT_PREFIX', str(self.ffs))
    self.addCleanup(p.stop)
    p.start()
    self.linked = set()
    for name, value in (('linked_functions', mock.Mock(side_effect=lambda: frozenset(self.linked))),
                        ('bound_udc', mock.Mock(return_value=None)),
                        ('setup_gadget', mock.Mock(side_effect=self.setup))):
      p = mock.patch.object(gadget, name, value)
      self.addCleanup(p.stop)
      setattr(self, name, p.start())

  def setup(self, ios, extra=()):
    """The script: links what it was given, and none for iOS."""
    self.linked = set() if ios else {fn.name for fn in extra}
    for fn in extra:
      fn.mount.mkdir(parents=True, exist_ok=True)
      fn.ep0.write_bytes(b'')
    return True

  def fresh(self, *declared):
    o = self.make(functions=declared)
    o.built_ios = False
    self.addCleanup(o.close_functions)
    return o


class TestDescribing(FunctionsTest):
  def test_each_is_described_before_jetlinks_bind_and_held(self):
    o = self.fresh(VASM, SLV)
    self.assertTrue(o.ensure_gadget(False))
    order = []
    real = functions.describe

    def describe(fn):
      order.append(fn.name)
      return real(fn)

    def transport(*args, **kwargs):
      order.append('jetlink')
      return mock.Mock()
    with mock.patch.object(owner, 'describe_function', side_effect=describe), \
         mock.patch('jetlink.transport.ffs.FfsTransport', side_effect=transport):
      self.assertTrue(o.open_link())
    self.assertEqual(order, ['vasm', 'slv', 'jetlink'])
    self.assertEqual(set(o.function_fds), {'vasm', 'slv'})

  def test_they_are_closed_after_jetlinks_transport(self):
    o = self.fresh(VASM)
    o.ensure_gadget(False)
    order = []
    with mock.patch('jetlink.transport.ffs.FfsTransport') as made:
      made.return_value.close.side_effect = lambda: order.append('jetlink')
      o.open_link()
    fd = o.function_fds['vasm']
    real = os.close
    with mock.patch.object(owner.os, 'close', side_effect=lambda f: (order.append(f), real(f))):
      o.close_link()
    self.assertEqual(order, ['jetlink', fd])
    self.assertEqual(o.function_fds, {})

  def test_one_that_fails_is_left_out_and_the_gadget_built_without_it(self):
    o = self.fresh(VASM, SLV)
    o.ensure_gadget(False)
    real = functions.describe

    def describe(fn):
      if fn.name == 'slv':
        raise OSError(errno.EINVAL, 'Invalid argument')
      return real(fn)
    with mock.patch.object(owner, 'describe_function', side_effect=describe), \
         mock.patch('jetlink.transport.ffs.FfsTransport') as made, mock.patch.object(gadget, 'log'):
      self.assertFalse(o.open_link())
      made.assert_not_called()
      self.assertEqual(o.function_fds, {}, 'the ones already described were left open')
      self.assertEqual(o.left_out, {'slv'})
      self.assertTrue(o.open_link())   # rebuilt without it, then presented
    self.assertEqual(self.setup_gadget.call_args_list[-1], mock.call(False, (VASM,)))
    self.assertEqual(self.linked, {'vasm'})
    self.assertEqual(set(o.function_fds), {'vasm'})

  def test_a_busy_one_waits_and_is_not_left_out(self):
    o = self.fresh(VASM)
    o.ensure_gadget(False)
    busy = OSError(errno.EBUSY, 'Device or resource busy', str(VASM.ep0))
    with mock.patch.object(owner, 'describe_function', side_effect=busy), \
         mock.patch('jetlink.transport.ffs.FfsTransport') as made, mock.patch.object(gadget, 'log') as log:
      self.assertFalse(o.open_link())
    made.assert_not_called()
    self.assertEqual(o.left_out, set())
    self.assertIn('vasm', log.warning.call_args.args[1])
    log.exception.assert_not_called()

  def test_an_add_on_that_never_lets_go_does_not_keep_jetlink_down(self):
    o = self.fresh(VASM)
    o.ensure_gadget(False)
    busy = OSError(errno.EBUSY, 'Device or resource busy', str(VASM.ep0))
    now = [1000.0]
    with mock.patch.object(owner.time, 'monotonic', side_effect=lambda: now[0]), \
         mock.patch.object(owner, 'describe_function', side_effect=busy), \
         mock.patch('jetlink.transport.ffs.FfsTransport') as made, mock.patch.object(gadget, 'log') as log:
      while now[0] < 1000.0 + owner.FUNCTION_BUSY_LIMIT:
        self.assertFalse(o.open_link())
        now[0] += owner.EP0_BUSY_RETRY
      self.assertEqual(log.warning.call_count, 1, 'the wait was said more than once')
      self.assertFalse(o.open_link())   # left out
      self.assertEqual(o.left_out, {'vasm'})
      self.assertTrue(o.open_link())    # rebuilt without it, and Jetlink presented
    made.assert_called_once()
    self.assertEqual(self.linked, set())

  def test_a_failed_jetlink_bind_lets_go_of_them(self):
    o = self.fresh(VASM)
    o.ensure_gadget(False)
    with mock.patch('jetlink.transport.ffs.FfsTransport', side_effect=OSError(errno.ENODEV, 'No such device')), \
         mock.patch.object(gadget, 'log'):
      self.assertFalse(o.open_link())
    self.assertEqual(o.function_fds, {})


class TestRebuilding(FunctionsTest):
  def test_a_gadget_built_without_them_is_built_again_while_unbound(self):
    o = self.fresh(VASM)
    self.assertTrue(o.ensure_gadget(False))
    self.setup_gadget.assert_called_once_with(False, (VASM,))
    self.assertTrue(o.ensure_gadget(False))
    self.setup_gadget.assert_called_once()   # settled

  def test_not_while_something_holds_the_bind(self):
    o = self.fresh(VASM)
    self.bound_udc.return_value = 'a600000.dwc3'
    self.assertTrue(o.ensure_gadget(False))
    self.setup_gadget.assert_not_called()

  def test_not_while_the_link_is_presented(self):
    o = self.fresh(VASM)
    o.transport = mock.Mock()
    self.assertTrue(o.ensure_gadget(False))
    self.setup_gadget.assert_not_called()

  def test_one_the_build_did_not_link_is_left_out(self):
    o = self.fresh(VASM)
    self.setup_gadget.side_effect = lambda ios, extra=(): True   # links nothing
    with mock.patch.object(gadget, 'log'):
      self.assertTrue(o.ensure_gadget(False))
      self.assertTrue(o.ensure_gadget(False))
    self.setup_gadget.assert_called_once()
    self.assertEqual(o.left_out, {'vasm'})

  def test_one_no_longer_declared_is_unlinked(self):
    o = self.fresh()
    self.linked = {'vasm'}
    self.assertTrue(o.ensure_gadget(False))
    self.setup_gadget.assert_called_once_with(False)
    self.assertEqual(self.linked, set())

  def test_an_ios_gadget_has_none(self):
    o = self.fresh(VASM)
    o.built_ios = True
    self.assertTrue(o.ensure_gadget(True))
    self.setup_gadget.assert_not_called()
    with mock.patch.object(owner, 'describe_function') as describe, mock.patch('jetlink.transport.ffs.FfsTransport'):
      self.assertTrue(o.open_link())
    describe.assert_not_called()

  def test_without_any_nothing_changes(self):
    o = self.fresh()
    self.assertTrue(o.ensure_gadget(False))
    self.setup_gadget.assert_not_called()
