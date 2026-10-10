"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

What the fork calls, and the answers it gets.

modeld, manager, hardwared, the UI and the model manager reach jetlink
through bind() and the Jetlink it returns, so what is pinned here is the
surface itself and selection: what a device with the link off, on but not
ready, and ready gets told, and that a request that hangs costs the large
model and nothing else.
"""
import inspect
import sys
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import jetlink.openpilot as jo
from jetlink.openpilot import interface, status
from jetlink.comma import gadget
from jetlink.openpilot import joining
from jetlink.spec import ModelSpec
from tests.openpilot import fakes
from tests.openpilot.fakes import OpenpilotTest

# what the fork calls on a Jetlink, as it calls it. Additive only: a changed
# one, or a removed one an adapter still calls, is an API bump (see
# jetlink/openpilot/__init__.py)
JETLINK = {
  'enabled': '()',
  'status': '()',
  'reason': '()',
  'model_state': '(ref)',
  'prepare': '()',
  'attach': '(small, cam_w, cam_h)',
  'request_shutdown': "(reason='')",
  'shutdown_pending': '()',
  'should_extend_catalog': '()',
  'extend_catalog': '(catalog)',
}


def plain(fn) -> str:
  sig = inspect.signature(fn)
  return str(sig.replace(parameters=[p.replace(annotation=p.empty) for p in sig.parameters.values()],
                         return_annotation=sig.empty))


# what the package exports: the contract, and nothing jetlink keeps to itself
EXPORTS = ['API', 'MODES', 'STATES', 'Jetlink', 'Keys', 'ModelFace', 'Openpilot', 'OwnerConfig', 'Status', 'bind',
           'conformance']


class TestTheSurface(OpenpilotTest):
  def test_the_methods_the_fork_calls(self):
    for name, expected in JETLINK.items():
      with self.subTest(name):
        self.assertEqual(plain(getattr(self.jl, name)), expected)

  def test_a_jetlink_shows_the_fork_nothing_else(self):
    # a part the fork could reach would become API without a bump noticing
    self.assertEqual({n for n in dir(self.jl) if not n.startswith('_')}, set(JETLINK))

  def test_the_package_exports_the_contract(self):
    self.assertEqual(sorted(jo.__all__), sorted(EXPORTS))
    for name in EXPORTS:
      self.assertTrue(hasattr(jo, name), name)

  def test_what_the_ui_calls_on_a_status(self):
    self.assertEqual(plain(jo.Status.icon), '(self, started, model_seen, running_big, state)')
    self.assertIsInstance(jo.Status.active_model, property)

  def test_the_entry_points_the_fork_names(self):
    from jetlink.openpilot import owner, provision
    self.assertEqual(plain(owner.main), '(config)')
    self.assertEqual(plain(provision.main), '(argv=None)')
    self.assertEqual(plain(jo.bind), '(op)')
    self.assertEqual(plain(interface.load_adapter), '(module)')

  def test_bind_points_the_comma_layers_log_at_the_adapters(self):
    # a heavy process's lines belong in the drive's log
    self.assertIs(gadget.log, self.op.log)
    self.assertIs(gadget.root.log, self.op.log)
    self.assertIsInstance(self.jl, jo.Jetlink)
    self.assertIs(self.parts.op, self.op)

  def test_an_adapter_module_makes_its_adapter(self):
    with mock.patch.dict('os.environ', {'JETLINK_FAKE_ROOT': str(self.tmp)}):
      op = interface.load_adapter('tests.openpilot.fakes')
    self.assertIsInstance(op, fakes.FakeOpenpilot)
    self.assertEqual(op.root, self.tmp)


class LenderFailureTest(OpenpilotTest):
  """Only the owner holds ep0. When its lender cannot listen it keeps the
  gadget, retries, and records why: that line is the offroad alert. modeld
  still prepares, so its join picks the link up once the lender listens."""

  def test_the_owners_lender_error_is_the_alert_not_a_no(self):
    built = self.tmp / 'jetlink-gadget'
    built.write_text('ok\n')
    self.op.set_mode('usb')
    self.patch(gadget, 'GADGET_STATUS', built)
    self.patch(gadget, 'LENDER_STATUS', self.tmp / 'jetlink-lender')
    self.patch(gadget, 'link_configured', return_value=True)
    self.patch(self.parts.warps, 'built', return_value=True)
    with mock.patch('jetlink.openpilot.warp.init_device'):
      gadget.note_lender_error('address in use')
      self.assertEqual(self.jl.status().reason, 'the lender could not listen: address in use')
      self.assertFalse(self.jl.status().ready)
      self.assertTrue(self.jl.prepare())
      # cleared once the lender listens again
      gadget.note_lender_error(None)
      self.assertIsNone(self.jl.status().reason)


class LoadTest(OpenpilotTest):
  """modeld's two calls: prepare() before it goes realtime, attach() once the camera is up."""

  def setUp(self):
    super().setUp()
    self.small = SimpleNamespace(name='small', client=None)
    self.init_device = self.patch(sys.modules['jetlink.openpilot.warp'], 'init_device')

  def prepared(self):
    """prepare() down its yes path, with nothing real behind it."""
    with mock.patch.object(self.jl, 'enabled', return_value=True), \
         mock.patch.object(gadget, 'link_configured', return_value=True), \
         mock.patch.object(self.parts.warps, 'built', return_value=True):
      self.assertTrue(self.jl.prepare())
    self.init_device.assert_called_once_with(self.op.log)

  def test_no_warp_is_no_before_the_gpu_comes_up(self):
    with mock.patch.object(self.jl, 'enabled', return_value=True), \
         mock.patch.object(gadget, 'link_configured', return_value=True), \
         mock.patch.object(self.parts.warps, 'built', return_value=False):
      self.assertFalse(self.jl.prepare())
    self.init_device.assert_not_called()
    self.assertTrue(self.op.log.has('no warp built for this camera, staying on the small model'))

  def test_no_usable_gadget_is_no_and_says_why(self):
    with mock.patch.object(self.jl, 'enabled', return_value=True), \
         mock.patch.object(gadget, 'link_configured', return_value=False), \
         mock.patch.object(gadget, 'gadget_error', return_value='no configfs'):
      self.assertFalse(self.jl.prepare())
    self.init_device.assert_not_called()
    self.assertTrue(self.op.log.has('no usable gadget (no configfs), staying on the small model'))

  def test_the_link_off_says_no_before_any_setup(self):
    with mock.patch.object(self.jl, 'enabled', return_value=False), \
         mock.patch.object(gadget, 'link_configured') as link_configured:
      self.assertFalse(self.jl.prepare())
    link_configured.assert_not_called()

  def test_nothing_joins_without_prepare(self):
    # the GPU's thread would start on modeld's realtime core
    with mock.patch.object(joining, 'join') as join:
      self.assertIsNone(self.jl.attach(self.small, 1928, 1208))
    join.assert_not_called()

  def test_a_later_no_takes_the_yes_back(self):
    self.prepared()
    with mock.patch.object(self.jl, 'enabled', return_value=False):
      self.assertFalse(self.jl.prepare())
    with mock.patch.object(joining, 'join') as join:
      self.assertIsNone(self.jl.attach(self.small, 1928, 1208))
    join.assert_not_called()

  def test_prepared_joins_modeld(self):
    self.prepared()
    joined = SimpleNamespace(client=object())
    with mock.patch.object(joining, 'join', return_value=joined) as join:
      model = self.jl.attach(self.small, 1928, 1208)
    join.assert_called_once_with(self.parts, 1928, 1208, self.small)
    self.assertIs(model, joined)

  def test_a_failed_build_drives_the_small_model_and_says_so(self):
    self.prepared()
    with mock.patch.object(joining, 'join', side_effect=RuntimeError('no warp')):
      model = self.jl.attach(self.small, 1928, 1208)
    self.assertEqual(self.op.log.lines('exception'), ["jetlink load failed"])
    self.assertIs(model, self.small)


def spec(model_hw=(128, 256)) -> ModelSpec:
  inputs = {'new_img': (2, 6, *model_hw), 'desire': (8,), 'traffic_convention': (1, 2), 'action_t': (1, 2)}
  return ModelSpec(sha256='a' * 64, nbytes=1, frame_skip=4, input_shapes=inputs, output_shapes={'outputs': (1, 16)},
                   output_slices={'plan': slice(0, 16)}, checkpoint=None)


class TestTheJoinFactory(OpenpilotTest):
  """What attach() builds: the joining model over the small one, with the warp
  loaded and warm before the frame loop exists."""

  def setUp(self):
    super().setUp()
    self.small = SimpleNamespace(name='small')
    p = mock.patch.dict(sys.modules, fakes.fake_tinygrad())
    p.start()
    self.addCleanup(p.stop)
    from jetlink.openpilot import link, warp
    self.present = self.patch(link, 'present_early')
    self.reset = self.patch(warp, 'prepare_reset')
    self.warp = self.patch(warp, 'Warp')
    self.loaded = self.patch(self.parts.warps, 'load')
    # the join thread is the joining state's; not here
    self.patch(joining.JoiningModelState, '_join_loop', lambda s: None)

  def join(self):
    s = joining.join(self.parts, 1928, 1208, self.small)
    self.addCleanup(s.close)
    return s

  def test_the_warp_is_sized_from_the_record(self):
    self.parts.spec.store(spec(model_hw=(64, 128)))
    self.join()
    self.loaded.assert_called_once_with(1928, 1208, 256, 128)
    self.warp.assert_called_once_with(self.loaded.return_value, fakes.frame_size(1928, 1208))
    self.reset.assert_called_once_with(self.small)
    self.present.assert_called_once()

  def test_the_early_present_leaves_modelds_realtime_core(self):
    from jetlink.transport.priority import background_thread
    self.join()
    link, background = self.present.call_args.args
    self.assertIs(background, background_thread)

  def test_without_a_record_it_is_this_devices_geometry(self):
    self.join()
    self.loaded.assert_called_once_with(1928, 1208, 512, 256)

  def test_a_warp_that_will_not_load_lets_the_link_go_and_raises(self):
    from jetlink.openpilot import link
    self.loaded.side_effect = RuntimeError('stale warp')
    with mock.patch.object(link, 'Link') as made, self.assertRaisesRegex(RuntimeError, 'stale warp'):
      joining.join(self.parts, 1928, 1208, self.small)
    made.return_value.close.assert_called_once_with()

  def test_the_join_can_be_stopped_through_the_link_open(self):
    from jetlink.openpilot import link
    s = self.join()
    stop = object()
    with mock.patch.object(link, 'open_link', return_value=('client', 'spec')) as opened:
      self.assertEqual(s._connect(stop), ('client', 'spec'))
    parts, held, should_stop = opened.call_args.args
    self.assertIs(parts, self.parts)
    self.assertIsInstance(held, link.Link)
    self.assertIs(should_stop, stop)

  def test_the_small_models_reset_is_the_one_prepared(self):
    s = self.join()
    s._reset_small()
    self.reset.return_value.assert_called_once_with()

  def test_the_model_logs_its_telemetry_to_the_adapters_event(self):
    s = self.join()
    client = mock.Mock()
    big = s._build(client, spec())
    self.assertEqual(big._event, self.op.event)
    self.assertIs(big._log, self.op.log)

  def test_it_runs_the_adapters_face_and_reports_through_jetlink(self):
    s = self.join()
    self.assertIs(s._progress, self.parts.progress)
    client = mock.Mock()
    big = s._build(client, spec())
    self.assertIs(big.face, self.op.face)
    self.assertIs(big.warp, self.warp.return_value)

  def test_a_server_model_of_another_geometry_is_refused(self):
    s = self.join()
    with self.assertRaisesRegex(RuntimeError, 'no prepared warp'):
      s._build(mock.Mock(), spec(model_hw=(64, 128)))


class ShuttingTheJetsonDown(OpenpilotTest):
  """hardwared hands the request to the owner only when a Jetson is there to take it."""

  def shutdown(self, mode='usb', present=True, requested=True):
    self.op.set_mode(mode)
    with mock.patch.object(status.Presence, 'present', return_value=present), \
         mock.patch.object(gadget, 'request_shutdown', return_value=requested) as request:
      asked = self.jl.request_shutdown('car battery')
    return request, asked

  def test_the_link_off_asks_nothing(self):
    request, asked = self.shutdown(mode='off')
    request.assert_not_called()
    self.assertFalse(asked)

  def test_no_jetson_there_asks_nothing(self):
    request, asked = self.shutdown(present=False)
    request.assert_not_called()
    self.assertFalse(asked)

  def test_a_jetson_there_is_asked(self):
    request, asked = self.shutdown()
    request.assert_called_once_with('car battery')
    self.assertTrue(asked)

  def test_a_request_that_could_not_be_written_is_not_pending(self):
    request, asked = self.shutdown(requested=False)
    request.assert_called_once_with('car battery')
    self.assertFalse(asked)

  def test_a_jetson_that_just_left_is_not_asked(self):
    # hardwared's own readers keep this process's presence fresh; a host seen
    # moments before the power-off would have the run wait 20 s for nobody
    self.op.set_mode('usb')
    with mock.patch.object(gadget, 'host_attached', return_value=True):
      self.assertTrue(self.parts.presence.present())
    with mock.patch.object(gadget, 'host_attached', return_value=False), \
         mock.patch.object(gadget, 'dormant', return_value=False), \
         mock.patch.object(gadget, 'request_shutdown') as request:
      self.assertTrue(self.parts.presence.present(), 'the hold this test is about')
      self.assertFalse(self.jl.request_shutdown('car battery'))
    request.assert_not_called()


class PoweringOffWithoutWaiting(OpenpilotTest):
  """hardwared's power-off: it asks once and goes on publishing deviceState,
  and puts DoShutdown once the request is taken or its 25 s have passed. The
  request is the file the owner has always looked for."""

  def setUp(self):
    super().setUp()
    self.op.set_mode('usb')
    self.patch(gadget, 'host_attached', return_value=True)

  def test_a_jetson_there_is_asked_and_the_answer_is_at_once(self):
    t0 = time.monotonic()
    self.assertTrue(self.jl.request_shutdown('car battery'))
    self.assertLess(time.monotonic() - t0, 0.5)
    self.assertEqual(gadget.pending_shutdown(), 'car battery')
    self.assertTrue(self.jl.shutdown_pending())
    self.assertTrue(self.op.log.has('asking the jetson to power off: car battery'))

  def test_the_owners_run_taking_it_clears_it(self):
    self.jl.request_shutdown('car battery')
    gadget.finish_shutdown()   # what the owner's run does, whatever the jetson answered
    self.assertFalse(self.jl.shutdown_pending())
    self.assertTrue(self.op.log.has('the owner took the shutdown request after'))
    self.assertFalse(self.jl.shutdown_pending())
    self.assertEqual(len(self.op.log.lines('warning')), 2)   # asked, taken: said once

  def test_nobody_taking_it_by_the_deadline_withdraws_it(self):
    # a comma that outlives its DoShutdown must not have the Jetson powered
    # off later by an owner that finds the request still there
    self.jl.request_shutdown('car battery')
    later = time.monotonic() + jo.SHUTDOWN_TIMEOUT - 1.0
    with mock.patch.object(jo.time, 'monotonic', return_value=later):
      self.assertTrue(self.jl.shutdown_pending())
    with mock.patch.object(jo.time, 'monotonic', return_value=later + 1.0):
      self.assertFalse(self.jl.shutdown_pending())
    self.assertFalse(gadget.SHUTDOWN_REQUEST.exists())
    self.assertTrue(self.op.log.has('nobody took the shutdown request within 25 s'))
    self.assertFalse(self.jl.shutdown_pending())

  def test_nothing_to_ask_is_nothing_to_wait_for(self):
    self.op.set_mode('off')
    self.assertFalse(self.jl.request_shutdown('car battery'))
    self.op.set_mode('usb')
    self.op.chestnut = True
    self.parts._chestnut = None
    self.assertFalse(self.jl.request_shutdown('car battery'))
    self.op.chestnut = False
    self.parts._chestnut = None
    with mock.patch.object(gadget, 'host_attached', return_value=False), \
         mock.patch.object(gadget, 'dormant', return_value=False):
      self.assertFalse(self.jl.request_shutdown('car battery'))
    self.assertFalse(self.jl.shutdown_pending())
    self.assertFalse(gadget.SHUTDOWN_REQUEST.exists())

  def test_a_sleeping_jetson_is_asked_too(self):
    # the owner's bind wakes it, and the run asks it
    with mock.patch.object(gadget, 'host_attached', return_value=False), \
         mock.patch.object(gadget, 'dormant', return_value=True), \
         mock.patch.object(gadget, 'port_has_host', return_value=True):
      self.assertTrue(self.jl.request_shutdown('car battery'))

  def test_it_never_raises(self):
    with mock.patch.object(gadget, 'request_shutdown', side_effect=RuntimeError('boom')):
      self.assertFalse(self.jl.request_shutdown('car battery'))
    self.assertEqual(self.op.log.lines('exception'), ['jetlink: shutdown request failed'])
    with mock.patch.object(gadget, 'request_shutdown', return_value=False):
      self.assertFalse(self.jl.request_shutdown('car battery'))
    with mock.patch.object(gadget, 'SHUTDOWN_REQUEST', mock.Mock(**{'exists.side_effect': PermissionError('no')})):
      self.assertFalse(self.jl.shutdown_pending())

  def test_the_owner_takes_it_as_it_always_has(self):
    # the Jetson on the user's supply is armed with --poweroff: what reaches
    # it has to be the owner's shutdown run, as before
    from jetlink.comma.owner import Owner
    o = Owner((), settings=self.parts.settings, chestnut_ids=())
    self.addCleanup(o.cable.close)
    o.lender = mock.Mock(lent=False, listening=True)
    self.patch(o, 'open_link', return_value=True)
    spawn = self.patch(o, 'spawn_worker')
    self.assertTrue(self.jl.request_shutdown('comma shutting down, offroad since 12.0'))
    o.step()
    spawn.assert_called_once_with('the jetson has to be shut down: comma shutting down, offroad since 12.0')
    self.assertTrue(o.shutting_down)


class TestExtendsCatalog(OpenpilotTest):
  """Hardware, not the link setting: the model manager drops a pick its catalog
  does not list, so a catalog that followed the setting lost one on a boot with
  it off."""

  def extends(self, chestnut=False, mode='off'):
    self.op.chestnut = chestnut
    self.op.set_mode(mode)
    self.parts._chestnut = None
    return self.jl.should_extend_catalog()

  def test_without_a_chestnut_it_is_extended_whatever_the_setting(self):
    self.assertTrue(self.extends(mode='off'))
    self.assertTrue(self.extends(mode='usb'))

  def test_a_chestnut_leaves_it_as_fetched(self):
    self.assertFalse(self.extends(chestnut=True, mode='usb'))
    self.assertFalse(self.extends(chestnut=True, mode='off'))


if __name__ == '__main__':
  unittest.main()
