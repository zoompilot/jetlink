"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

What the readers are told: whether the link is on, whether it can run, what
is on the other end, the progress the panels show, and which icon that is.
Every answer follows from the setting, the pick, the spec record, the gadget's
files and whether the build made a warp for this camera; no link IO.
"""
import ast
import os
import re
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from jetlink.comma import gadget
from jetlink.openpilot import link, status
from jetlink.openpilot.status import Status
from tests.openpilot.fakes import OpenpilotTest


class SelectionTest(OpenpilotTest):
  """ready is params only, and every answer follows from the link setting,
  the pick, the spec record and whether the build made a warp for this camera."""

  def configure(self, mode='usb', model=None, spec_sha=None, ready=False, gadget_error=None, warp=True):
    self.op.set_mode(mode)
    # the spec record carries whether the engine for the sha it names is built
    if spec_sha:
      self.op.store['JetlinkSpec'] = {'sha256': spec_sha, 'ready': ready}
    else:
      self.op.store.pop('JetlinkSpec', None)
    for target, name, value in ((gadget, 'gadget_error', gadget_error),
                                (self.parts.warps, 'built', warp),
                                (gadget, 'host_attached', False),
                                (gadget, 'dormant', False),
                                (self.parts.models, 'selected_model', {'name': model, 'oid': 'a' * 64} if model else None),
                                (self.parts.spec, 'load', SimpleNamespace(sha256=spec_sha) if spec_sha else None)):
      self.patch(target, name, return_value=value)

  def test_the_last_model_built_stands_in_while_the_pick_is_not_ready(self):
    # by design: it drives until the pick is fetched and built
    self.configure(model='ResAction', spec_sha='b' * 64, ready=True)
    self.patch(self.parts.models, 'name_for', return_value='Cinque Terre V3')
    s = self.jl.status()
    self.assertEqual((s.ready, s.model, s.standin), (False, 'ResAction', 'Cinque Terre V3'))

  def test_no_stand_in_once_the_pick_is_built_or_with_nothing_built(self):
    for spec_sha, ready in (('a' * 64, True), ('b' * 64, False), (None, False)):
      with self.subTest(spec_sha=spec_sha, ready=ready):
        self.configure(model='ResAction', spec_sha=spec_sha, ready=ready)
        self.assertIsNone(self.jl.status().standin)

  def test_a_model_the_catalog_does_not_name_stands_in_by_its_sha(self):
    self.configure(model='ResAction', spec_sha='b' * 64, ready=True)
    self.assertEqual(self.jl.status().standin, 'b' * 16)

  def test_a_stand_in_is_what_runs_and_lights_the_icon(self):
    # not "uncompiled": the last model built drives, so the big model is there
    self.configure(model='ResAction', spec_sha='b' * 64, ready=True)
    self.patch(self.parts.models, 'name_for', return_value='Cinque Terre V3')
    s = self.jl.status()
    self.assertEqual(s.active_model, 'Cinque Terre V3')
    self.assertEqual(s._replace(present=True).icon(False, False, False, 'none'), 'ready')
    self.assertEqual(s._replace(present=True).icon(True, True, False, 'ready'), 'waiting')

  def test_the_link_off_or_unset_is_disabled(self):
    for mode in (None, 'off'):
      with self.subTest(mode=mode):
        self.configure(mode=mode, model='m', spec_sha='a' * 64, ready=True, gadget_error='no gadget')
        s = self.jl.status()
        self.assertFalse(s.present)
        self.assertFalse(s.ready)
        # A device with the feature off is never nagged about its kernel.
        self.assertIsNone(s.reason)
        self.assertFalse(s.enabled)
        self.assertFalse(self.jl.enabled())
        self.assertEqual(s.mode, 'off')

  def test_enabled_but_not_provisioned(self):
    self.configure(model='m')
    s = self.jl.status()
    self.assertFalse(s.ready)
    self.assertIsNone(s.reason)
    self.assertIsNone(s.active_model)
    self.assertEqual(s.model, 'm')

  def test_enabled_with_a_broken_gadget_says_why(self):
    self.configure(model='m', spec_sha='a' * 64, ready=True, gadget_error='no gadget')
    s = self.jl.status()
    self.assertFalse(s.ready)
    self.assertEqual(s.reason, 'no gadget')

  def test_no_warp_for_this_camera_says_why(self):
    # the build made none, and nothing compiles one at runtime: every drive
    # would be the small model while the panel said ready
    self.configure(model='m', spec_sha='a' * 64, ready=True, warp=False)
    s = self.jl.status()
    self.assertFalse(s.ready)
    self.assertEqual(s.reason, status.NO_WARP)
    self.op.set_mode('off')
    self.assertIsNone(self.jl.status().reason)

  def test_ready(self):
    self.configure(model='m', spec_sha='a' * 64, ready=True)
    s = self.jl.status()
    self.assertTrue(s.ready)
    self.assertIsNone(s.reason)
    self.assertTrue(s.enabled)
    self.assertEqual(s.active_model, 'm')
    self.assertIsNone(self.jl.reason())

  def test_an_old_engine_is_not_the_new_selection(self):
    self.configure(model='m', spec_sha='b' * 64, ready=True)
    self.assertFalse(self.jl.status().ready)

  def test_a_spec_whose_engine_is_not_built_is_not_ready(self):
    self.configure(model='m', spec_sha='a' * 64, ready=False)
    self.assertFalse(self.jl.status().ready)

  def test_enabled_is_the_setting_alone(self):
    # configuration only, never link state or readiness. The model defaults
    # through selected_model(), so it is not part of it either
    self.configure(model=None)
    self.assertTrue(self.jl.enabled())
    self.configure(model='m')
    with mock.patch.object(gadget, 'link_configured', return_value=False):
      self.assertTrue(self.jl.enabled())
    self.configure(mode=None, model='m')
    with mock.patch.object(gadget, 'link_configured', return_value=True):
      self.assertFalse(self.jl.enabled())

  def test_a_chestnut_turns_it_off_whatever_the_setting(self):
    # it runs the big model natively; jetlinkd never takes the USB controller from it
    self.configure(model='m', spec_sha='a' * 64, ready=True)
    self.op.chestnut = True
    self.parts._chestnut = None
    s = self.jl.status()
    self.assertEqual((s.enabled, s.ready, s.reason, s.mode), (False, False, None, 'usb'))

  def test_the_chestnut_is_looked_for_every_two_seconds_at_most(self):
    self.configure(model='m')
    with mock.patch.object(self.op, 'chestnut_present', return_value=False) as walk:
      for _ in range(10):
        self.jl.enabled()
      self.assertEqual(walk.call_count, 1)
      from jetlink.openpilot import parts
      with mock.patch.object(parts.time, 'monotonic', return_value=parts.time.monotonic() + parts.CHESTNUT_TTL + 1):
        self.jl.enabled()
      self.assertEqual(walk.call_count, 2)

  def test_enabled_never_raises(self):
    # manager asks it on every device; an exception would take manager down
    self.configure(model='m')
    with mock.patch.object(self.op, 'chestnut_present', side_effect=RuntimeError('usb walk failed')):
      self.parts._chestnut = None
      for _ in range(3):
        self.assertFalse(self.jl.enabled())
    self.assertEqual(self.op.log.lines('exception'), ['jetlink: could not read whether the link is on'])
    self.assertTrue(self.jl.enabled())

  def test_a_directory_named_as_text_is_not_a_new_store_every_read(self):
    self.configure(model='m')
    self.op.params_dir = lambda: str(self.op.store_dir)
    first = self.parts.settings
    self.assertIs(self.parts.settings, first)
    self.assertTrue(self.jl.enabled())

  def test_the_setting_follows_the_store_the_adapter_names(self):
    # a shell under its own OPENPILOT_PREFIX, as Params follows it
    self.configure(model='m')
    self.assertTrue(self.jl.enabled())
    elsewhere = self.tmp / 'params' / 'other'
    elsewhere.mkdir(parents=True)
    self.op.store_dir = elsewhere
    self.assertFalse(self.jl.enabled())


class TestReason(OpenpilotTest):
  """hardwared's offroad alert, twice a second on every device: the gadget's
  and the build's files, and nothing that parses the catalog."""

  def test_the_link_off_is_no_reason_and_reads_no_model(self):
    with mock.patch.object(gadget, 'gadget_error', return_value='no gadget'):
      self.assertIsNone(self.jl.reason())
    self.assertNotIn('models', vars(self.parts))
    self.assertNotIn('spec', vars(self.parts))

  def test_the_link_on_says_why(self):
    self.op.set_mode('usb')
    with mock.patch.object(gadget, 'gadget_error', return_value='no gadget'):
      self.assertEqual(self.jl.reason(), 'no gadget')
    self.assertEqual(self.jl.reason(), status.NO_WARP)
    self.assertNotIn('models', vars(self.parts))

  def test_beside_a_chestnut_there_is_nothing_to_say(self):
    self.op.set_mode('usb')
    self.op.chestnut = True
    self.assertIsNone(self.jl.reason())

  def test_a_failure_is_the_reason_only_for_someone_who_asked_for_the_link(self):
    with mock.patch.object(gadget, 'gadget_error', side_effect=RuntimeError('boom')):
      self.op.set_mode('usb')
      self.assertEqual(self.jl.reason(), 'jetlink status failed: RuntimeError: boom')
      self.op.set_mode('off')
      self.assertIsNone(self.jl.reason())


class TestPresence(OpenpilotTest):
  """What the panels are told is on the other end. The markers are jetlink.comma's."""

  def setUp(self):
    super().setUp()
    for name in ('DORMANT', 'SHUTDOWN_REQUEST'):
      self.patch(gadget, name, self.tmp / name.lower())
    self.cc = self.tmp / 'cc'
    self.patch(gadget, 'CC_ORIENTATION', self.cc)

  def test_dormant_counts_as_present_without_a_host(self):
    with mock.patch.object(gadget, 'host_attached', return_value=False):
      self.cc.write_text('1')
      assert not self.parts.presence.present()
      gadget.set_dormant(True)
      self.addCleanup(gadget.set_dormant, False)
      assert self.parts.presence.present()
      self.cc.write_text('0')
      assert not self.parts.presence.present()

  def test_a_host_is_held_through_a_blink(self):
    # a USB3 link recovery passes through "addressed"
    with mock.patch.object(gadget, 'host_attached', return_value=True):
      assert self.parts.presence.present()
    with mock.patch.object(gadget, 'host_attached', return_value=False):
      assert self.parts.presence.present()
      with mock.patch.object(status.time, 'monotonic', return_value=status.time.monotonic() + gadget.PRESENCE_HOLD):
        assert not self.parts.presence.present()

  def test_the_port_is_the_cc_pin(self):
    self.assertIsNone(status.usb_port())
    self.cc.write_text('0\n')
    self.assertEqual(status.usb_port(), 'empty')
    for raw in ('1', '2'):
      self.cc.write_text(raw)
      self.assertEqual(status.usb_port(), 'host')


class TestTheOwnersRecord(OpenpilotTest):
  """With a live owner the readers take its record, and read none of the
  gadget's files: the UI and hardwared then agree with it, and with each
  other. Its heartbeat going stale is the offroad alert."""

  def setUp(self):
    super().setUp()
    self.op.set_mode('usb')
    self.patch(self.parts.warps, 'built', return_value=True)

  def record(self, age: float = 0.0, **fields) -> None:
    gadget.write_record(gadget.STATUS, {'pid': 1, 'at': time.monotonic() - age, 'mode': 'usb', 'link': 'usb',
                                        'peer': None, 'error': None, 'net': None, 'dormant': False, 'udc': 'configured',
                                        'speed': 'super-speed', 'present': True, 'worker': False, **fields})

  def files_left_alone(self) -> None:
    for name in ('gadget_error', 'host_attached', 'dormant', 'link_kind', 'link_peer'):
      self.patch(gadget, name, side_effect=AssertionError(f'{name}() read with a live record'))

  def test_a_live_record_is_the_answer(self):
    self.record(link='cable', peer='192.168.60.3')
    self.files_left_alone()
    s = self.jl.status()
    self.assertEqual((s.present, s.transport, s.reason), (True, 'iOS over USB (192.168.60.3)', None))
    self.assertIsNone(self.jl.reason())
    self.record(present=False, link='usb')
    s = self.jl.status()
    self.assertEqual((s.present, s.transport), (False, 'USB'))

  def test_its_error_is_the_reason(self):
    self.record(error='the lender could not listen: address in use')
    self.files_left_alone()
    self.assertEqual(self.jl.reason(), 'the lender could not listen: address in use')
    self.assertEqual(self.jl.status().reason, 'the lender could not listen: address in use')

  def test_a_heartbeat_older_than_the_timeout_is_a_stopped_service(self):
    self.record(age=gadget.HEARTBEAT_TIMEOUT - 1.0)
    self.assertIsNone(self.jl.reason())
    self.record(age=gadget.HEARTBEAT_TIMEOUT + 0.1)
    self.assertEqual(self.jl.reason(), status.STOPPED)
    s = self.jl.status()
    self.assertEqual(s.reason, status.STOPPED)
    self.assertFalse(s.ready)
    self.record()   # manager started it again
    self.assertIsNone(self.jl.reason())

  def test_an_owner_stopped_on_purpose_and_killed_in_its_teardown_is_no_alert(self):
    # manager SIGKILLs an owner still stopping 5 s after its SIGINT; the next
    # owner writes its own record a moment after it starts
    self.record(age=60.0, stopping=True)
    with mock.patch.object(gadget, 'gadget_error', return_value=None):
      self.assertIsNone(self.jl.reason())
      self.assertIsNone(self.jl.status().reason)

  def test_a_stopped_service_nags_only_someone_who_turned_the_link_on(self):
    self.record(age=60.0)
    self.op.set_mode('off')
    self.assertIsNone(self.jl.reason())
    self.assertIsNone(self.jl.status().reason)
    self.op.chestnut = True
    self.parts._chestnut = None
    self.op.set_mode('usb')
    self.assertIsNone(self.jl.reason())

  def test_with_a_stale_record_the_rest_is_read_from_the_files(self):
    self.record(age=60.0, present=True, link='cable', peer='192.168.60.3')
    with mock.patch.object(gadget, 'host_attached', return_value=False), \
         mock.patch.object(gadget, 'dormant', return_value=False):
      s = self.jl.status()
    self.assertEqual((s.present, s.transport, s.reason), (False, 'USB', status.STOPPED))

  def test_without_one_it_is_the_files_as_before(self):
    # an owner too old to write it, or none since boot
    with mock.patch.object(gadget, 'gadget_error', return_value='no gadget'):
      self.assertEqual(self.jl.reason(), 'no gadget')
    self.assertIsNone(self.jl.reason())

  def test_a_record_that_is_not_one_is_none(self):
    gadget.STATUS.parent.mkdir(parents=True, exist_ok=True)
    for text in ('', 'not json', '[1, 2]'):
      gadget.STATUS.write_text(text)
      self.assertIsNone(gadget.owner_status(), text)
    self.assertFalse(gadget.owner_alive({'at': 'yesterday'}))
    self.assertFalse(gadget.owner_alive({}))

  def test_what_an_owner_writes_is_what_the_readers_read(self):
    from jetlink.comma.owner import Owner
    o = Owner((), settings=self.parts.settings, chestnut_ids=())
    self.addCleanup(o.cable.close)
    o.mode, o.built_ios, o._peer = 'ios', True, '192.168.60.3'
    self.op.set_mode('ios')
    with mock.patch.object(gadget, 'udc_state', return_value='configured'), \
         mock.patch.object(gadget, 'usb_speed', return_value='high-speed'):
      o.publish_status()
    self.files_left_alone()
    s = self.jl.status()
    self.assertEqual((s.present, s.transport, s.reason), (True, 'iOS over USB (192.168.60.3)', None))


class TestTransport(OpenpilotTest):
  def setUp(self):
    super().setUp()
    self.patch(gadget, 'LINK', self.tmp / 'link')

  def test_usb_unless_the_owner_built_for_a_phone(self):
    self.assertEqual(status.link_transport(), 'USB')
    gadget.note_link('usb')
    self.assertEqual(status.link_transport(), 'USB')
    gadget.note_link('cable')
    self.assertEqual(status.link_transport(), 'iOS over USB')
    gadget.note_link('cable', '192.168.60.3')
    self.assertEqual(status.link_transport(), 'iOS over USB (192.168.60.3)')

  def test_wifi_names_the_phone_it_dials_whatever_the_records_say(self):
    gadget.note_link('cable', '192.168.60.3')
    self.assertEqual(status.link_transport('wifi'), 'Wi-Fi (172.20.10.1:5599)')

  def test_it_never_raises(self):
    with mock.patch.object(gadget, 'link_kind', side_effect=OSError('gone')):
      self.assertEqual(status.link_transport(), 'USB')


class TestProgress(OpenpilotTest):
  def test_nothing_recorded_is_no_progress(self):
    self.assertIsNone(self.parts.progress.read())

  def test_a_dict_comes_through(self):
    payload = {'stage': 'build', 'frac': 0.5, 'msg': ''}
    self.op.store['AcceleratorProgress'] = payload
    self.assertEqual(self.parts.progress.read(), payload)

  def test_a_non_dict_is_ignored(self):
    self.op.store['AcceleratorProgress'] = "build 50%"
    self.assertIsNone(self.parts.progress.read())

  def test_a_store_that_raises_does_not_take_down_the_ui(self):
    # the adapter's get never should; this is the reader's own guard
    with mock.patch.object(self.op, 'get', side_effect=RuntimeError("UnknownKeyName")):
      self.assertIsNone(self.parts.progress.read())

  def test_reporting_never_raises(self):
    # called from except handlers in the run and the join
    self.op.put_error = RuntimeError("params gone")
    self.parts.progress.report('build', 0.5)
    with mock.patch.object(self.op, 'remove', side_effect=RuntimeError("params gone")):
      self.parts.progress.clear()
    self.assertEqual(len(self.op.log.lines('exception')), 2)

  def test_held_to_four_hertz_within_a_stage(self):
    with mock.patch.object(self.op, 'put', wraps=self.op.put) as put:
      for frac in (0.1, 0.2, 0.3):
        self.parts.progress.report('upload', frac)
      self.assertEqual(put.call_count, 1)
      # a new stage, and the end of one, always go through
      self.parts.progress.report('build', 0.0)
      self.parts.progress.report('build', 1.0)
      self.assertEqual(put.call_count, 3)
    self.assertEqual(self.parts.progress.read(), {'stage': 'build', 'frac': 1.0, 'msg': '', 'drops': 0})

  def test_a_report_with_no_fraction_always_goes_through(self):
    # a rejoin's 're-engage to switch' can come 50 ms after its 'waiting for
    # jetlink'; holding it would leave the panel saying waiting for the drive
    self.parts.progress.report('connect', 0.0, 'waiting for jetlink')
    self.parts.progress.report('connect', 0.0, 're-engage to switch')
    self.assertEqual(self.parts.progress.read()['msg'], 're-engage to switch')

  def test_clearing_removes_it(self):
    self.parts.progress.report('build', 1.0)
    self.parts.progress.clear()
    self.assertIsNone(self.parts.progress.read())


class BuildEtaTest(OpenpilotTest):
  """The estimate is what tells a driver watching "build 12%" whether that is
  five minutes or thirty."""

  def report(self, *args, size=1_850_000_000):
    seen = []
    with mock.patch.object(self.parts.models, 'selected_model', return_value={'size': size}), \
         mock.patch.object(self.parts.progress, 'report', lambda *a, **kw: seen.append(a + (kw,))):
      for call in args:
        self.parts.progress.report_with_eta(*call)
    return seen

  def test_the_build_stage_gets_a_time_remaining(self):
    seen = self.report(('build', 0.0, 'building the engine'), ('build', 0.8, 'building the engine'))
    self.assertEqual(seen[0][2], "about 5 min left")
    self.assertEqual(seen[1][2], "about 60s left")

  def test_other_stages_keep_their_own_message(self):
    # The upload already counts MB of MB, and a connect has nothing to predict.
    self.assertEqual(self.report(('upload', 0.5, '380/766 MB'))[0][2], '380/766 MB')

  def test_the_estimate_follows_the_measurements(self):
    self.assertTrue(100 <= status.estimated_build_seconds(766_000_000) <= 200)     # built in 102 to 166 s
    self.assertTrue(230 <= status.estimated_build_seconds(1_757_000_000) <= 320)   # 230 to 294 s

  def test_a_model_not_resolved_yet_is_survived(self):
    seen = self.report(('build', 0.3, 'building the engine'), size=None)
    self.assertEqual(seen[0][2], 'building the engine')


def snapshot(**fields) -> Status:
  """A snapshot as zoompilot's tests/helpers.jetlink_status() builds it: by
  keyword, from the fields Status had before running and active."""
  base = {'enabled': False, 'mode': 'usb' if fields.get('enabled') else 'off', 'transport': 'USB', 'present': False,
          'port': None, 'ready': False, 'reason': None, 'progress': None, 'model': None, 'default_model': None}
  return Status(**{**base, **fields})


class TestIcon(unittest.TestCase):
  """The chestnut icon's state for the link, as the fork's UI drew it from
  its own mapping (SP ui_state._accelerator_state)."""

  def test_offroad(self):
    cases = [
      ({'present': False}, 'disconnected'),
      ({'present': False, 'ready': True, 'progress': {'stage': 'build'}}, 'disconnected'),
      ({'progress': {'stage': 'upload', 'frac': 0.5}}, 'loading'),
      ({'progress': {'stage': 'connect'}}, 'loading'),
      ({'progress': {'stage': 'failed'}}, 'failed'),
      ({'progress': {'stage': 'ready'}}, 'uncompiled'),
      ({'progress': {'stage': 'ready'}, 'ready': True}, 'ready'),
      ({'ready': True}, 'ready'),
      ({}, 'uncompiled'),
    ]
    for fields, expected in cases:
      with self.subTest(fields):
        # onroad inputs are ignored offroad
        self.assertEqual(snapshot(**{'present': True, **fields}).icon(False, True, True, 'running'), expected)

  def test_onroad(self):
    cases = [
      # (present, ready, model_seen, running_big, state) -> icon
      ((True, True, True, True, 'running'), 'active'),
      ((False, False, True, True, 'none'), 'active'),       # a live big model outranks everything
      ((True, True, False, True, 'running'), 'loading'),    # no modelV2 yet this drive
      ((False, True, True, False, 'running'), 'disconnected'),
      ((True, True, True, False, 'joining'), 'loading'),
      ((True, False, True, False, 'retrying'), 'loading'),
      ((True, True, True, False, 'ready'), 'waiting'),
      ((True, False, True, False, 'ready'), 'waiting'),
      ((True, False, True, False, 'none'), 'uncompiled'),
      ((True, False, True, False, 'running'), 'uncompiled'),
      ((True, True, True, False, 'running'), 'active'),
      ((True, True, True, False, 'unavailable'), 'failed'),
      ((True, True, True, False, 'none'), 'failed'),
    ]
    for (present, ready, seen, big, state), expected in cases:
      with self.subTest((present, ready, seen, big, state)):
        self.assertEqual(snapshot(present=present, ready=ready).icon(True, seen, big, state), expected)

  def test_every_icon_is_a_chestnut_state(self):
    icons = {'disconnected', 'uncompiled', 'ready', 'loading', 'active', 'failed', 'waiting'}
    seen = {snapshot(present=p, ready=r, progress=g).icon(st, ms, rb, state)
            for p in (False, True) for r in (False, True) for g in (None, {'stage': 'build'}, {'stage': 'failed'})
            for st in (False, True) for ms in (False, True) for rb in (False, True)
            for state in ('none', 'joining', 'retrying', 'ready', 'running', 'unavailable')}
    self.assertEqual(seen, icons)


class TestTheSnapshot(OpenpilotTest):
  def test_its_fields_are_only_ever_added(self):
    self.assertEqual(Status._fields, ('enabled', 'mode', 'transport', 'present', 'port', 'ready', 'reason', 'progress',
                                      'model', 'default_model', 'standin', 'running', 'active', 'running_sha256',
                                      'running_checkpoint'))

  def test_a_model_in_the_list_says_whether_it_is_built_or_downloaded(self):
    models = [{'name': 'Built', 'ref': 'r1', 'oid': 'b' * 64, 'size': 10},
              {'name': 'Fetched', 'ref': 'r2', 'oid': 'c' * 64, 'size': 10},
              {'name': 'Neither', 'ref': 'r3', 'oid': 'd' * 64, 'size': 10},
              {'name': 'Unresolved', 'ref': 'r4', 'oid': None, 'size': None}]
    self.op.store['JetlinkSpec'] = {'sha256': 'b' * 64, 'ready': True}
    with mock.patch.object(self.parts.models, 'model_index', return_value=models), \
         mock.patch.object(self.parts.models, 'has_file', side_effect=lambda m: m['ref'] == 'r2'):
      self.assertEqual([self.jl.model_state(r) for r in ('r1', 'r2', 'r3', 'r4', 'unlisted')],
                       ['ready', 'downloaded', None, None, None])

  def test_it_reads_each_file_once(self):
    self.op.set_mode('usb')
    settings = self.parts.settings
    with mock.patch.object(self.parts.warps, 'built', return_value=True), \
         mock.patch.object(settings, 'mode', wraps=settings.mode) as mode, \
         mock.patch.object(gadget, 'link_kind', wraps=gadget.link_kind) as kind, \
         mock.patch.object(gadget, 'gadget_error', return_value=None) as error, \
         mock.patch.object(self.parts.spec, '_raw', wraps=self.parts.spec._raw) as spec:
      self.jl.status()
    self.assertEqual((mode.call_count, error.call_count, spec.call_count), (1, 1, 1))
    # the transport's fallback is the setting already read, not a second read
    kind.assert_called_once_with('usb')

  def test_a_failure_with_the_link_on_says_so(self):
    self.op.set_mode('usb')
    with mock.patch.object(status, 'link_transport', side_effect=RuntimeError('boom')):
      for _ in range(3):
        s = self.jl.status()
    self.assertEqual((s.enabled, s.mode), (False, 'usb'))
    self.assertEqual(s.reason, 'jetlink status failed: RuntimeError: boom')
    # once, not five times a second
    self.assertEqual(self.op.log.lines('exception'), ['jetlink: could not read the status'])

  def test_a_failure_with_the_link_off_nags_nobody(self):
    # the snapshot reads more than the fork's hardwared ever did: a bad
    # catalog must not raise an alert on a device that never turned the link on
    with mock.patch.object(self.parts.models, 'selected_model_name', side_effect=ValueError('bad pointers')):
      s = self.jl.status()
    self.assertEqual((s.enabled, s.mode, s.reason), (False, 'off', None))

  def test_a_failure_that_clears_and_comes_back_is_logged_again(self):
    self.op.set_mode('usb')
    boom = mock.patch.object(status, 'link_transport', side_effect=RuntimeError('boom'))
    with boom:
      self.jl.status()
    self.assertTrue(self.jl.status().enabled)
    with boom:
      self.jl.status()
    self.assertEqual(len(self.op.log.lines('exception')), 2)

  def test_a_setting_that_cannot_be_read_is_off(self):
    with mock.patch.object(self.op, 'params_dir', side_effect=RuntimeError('adapter bug')):
      s = self.jl.status()
      self.assertFalse(self.jl.enabled())
    self.assertEqual((s.mode, s.enabled, s.reason), ('off', False, None))
    self.assertEqual(self.op.log.lines('exception'), ['jetlink: could not read the link setting'])

  def test_names_come_from_the_models(self):
    self.op.set_mode('usb')
    with mock.patch.object(self.parts.models, 'selected_model_name', return_value='Picked'), \
         mock.patch.object(self.parts.models, 'default_model_name', return_value='Default'):
      s = self.jl.status()
    self.assertEqual((s.model, s.default_model, s.active_model), ('Picked', 'Default', None))


class TestTheForksFixtures(unittest.TestCase):
  def test_the_old_fields_alone_still_make_one(self):
    # positionally, and by keyword as zoompilot's helpers do
    s = Status(True, 'usb', 'USB', True, 'host', True, None, None, 'm', 'd')
    self.assertEqual((s.standin, s.running, s.active), (None, None, False))
    self.assertEqual(snapshot(enabled=True, ready=True).phase, 'prepared')
    failed = status.failed('boom', 'usb')
    self.assertEqual((failed.reason_code, failed.phase), ('status_failed', 'unavailable'))


class TestWhatThePanelsDerive(unittest.TestCase):
  """zoompilot's accelerator_link.link_status and ui_state_zp's jetlink_view
  and _enforce_usb_port, ported, on the same inputs as its tests."""

  @staticmethod
  def link_status(s) -> str:
    if s.present:
      return f"Jetlink connected: {s.transport}."
    if s.port is None:
      return ""
    return "Nothing on the USB port." if s.port == "empty" else "A device is on the USB port."

  def test_port_state_is_the_status_line(self):
    line = {'connected': lambda s: f"Jetlink connected: {s.transport}.", 'empty': lambda s: "Nothing on the USB port.",
            'device': lambda s: "A device is on the USB port.", None: lambda s: ""}
    cases = [({'port': 'empty'}, 'empty'), ({'port': 'host'}, 'device'), ({'present': True, 'port': 'host'}, 'connected'),
             ({'port': None}, None), ({'present': True, 'port': None}, 'connected'),
             ({'present': True, 'transport': 'iOS over USB (192.168.60.3)'}, 'connected')]
    for fields, expected in cases:
      with self.subTest(fields):
        s = snapshot(**fields)
        self.assertEqual(s.port_state, expected)
        self.assertEqual(line[s.port_state](s), self.link_status(s))

  def test_shown_on_zoompilots_view_cases(self):
    cases = [({}, False),                                                     # nothing to show
             ({'present': True, 'ready': True, 'enabled': True}, True),        # an attached accelerator
             ({'present': True, 'progress': {'stage': 'upload', 'frac': 0.5, 'msg': ''}}, True),   # progress alone
             ({'progress': {'stage': 'upload'}}, True), ({'enabled': True}, True), ({'ready': True, 'port': 'host'}, False)]
    for fields, expected in cases:
      with self.subTest(fields):
        self.assertEqual(snapshot(**fields).shown, expected)

  def test_holds_port_is_the_setting(self):
    # ADB off while the link is on; beside a chestnut enabled is already False
    for enabled in (False, True):
      self.assertEqual(snapshot(enabled=enabled, present=not enabled).holds_port, enabled)

  def test_wifi_leaves_the_port_to_adb(self):
    self.assertFalse(snapshot(enabled=True, mode='wifi').holds_port)
    self.assertTrue(snapshot(enabled=True, mode='ios').holds_port)


class TestPhase(unittest.TestCase):
  def test_each_phase_and_what_outranks_what(self):
    on = {'enabled': True}
    cases = [
      ({'reason': 'jetlink status failed: boom'}, 'unavailable'),   # a failed snapshot: the reason, link off
      ({**on, 'reason': status.NO_WARP, 'running': 'running', 'active': True}, 'unavailable'),
      ({}, 'off'),
      ({'running': 'running', 'active': True}, 'off'),
      ({**on, 'running': 'running', 'active': True, 'progress': {'stage': 'connect'}}, 'active'),
      ({**on, 'running': 'ready', 'ready': True, 'progress': {'stage': 'connect'}}, 'waiting'),
      ({**on, 'running': 'retrying', 'progress': {'stage': 'connect'}}, 'reconnecting'),
      ({**on, 'running': 'retrying', 'progress': {'stage': 'build', 'frac': 0.2}}, 'reconnecting'),
      ({**on, 'running': 'running', 'ready': True}, 'reconnecting'),  # stopped answering: about to hand back
      ({**on, 'progress': {'stage': 'failed'}, 'ready': True}, 'failed'),
      ({**on, 'running': 'joining', 'progress': {'stage': 'build', 'frac': 0.2}}, 'preparing'),
      ({**on, 'progress': {'stage': 'download', 'frac': 0.3}, 'ready': True}, 'preparing'),
      ({**on, 'progress': {'stage': 'waiting'}}, 'preparing'),
      ({**on, 'running': 'joining'}, 'connecting'),
      ({**on, 'running': 'joining', 'ready': True, 'progress': {'stage': 'connect'}}, 'connecting'),
      ({**on, 'progress': {'stage': 'connect', 'msg': 'connecting'}}, 'connecting'),   # a provisioning run's
      ({**on, 'ready': True}, 'prepared'),
      ({**on, 'ready': True, 'progress': {'stage': 'ready'}}, 'prepared'),
      ({**on, 'standin': 'Cinque Terre V3'}, 'prepared'),
      ({**on}, 'preparing'),                             # nothing built and nothing working on it yet
    ]
    for fields, expected in cases:
      with self.subTest(fields):
        self.assertEqual(snapshot(**fields).phase, expected)
    self.assertEqual({p for _, p in cases}, set(status.PHASES))

  def test_every_combination_is_a_phase(self):
    seen = {snapshot(enabled=e, reason=r, ready=rd, running=st, active=a, progress=g).phase
            for e in (False, True) for r in (None, status.STOPPED) for rd in (False, True)
            for st in (None, 'joining', 'retrying', 'ready', 'running') for a in (False, True)
            for g in (None, {'stage': 'connect'}, {'stage': 'build'}, {'stage': 'ready'}, {'stage': 'failed'}, {})}
    self.assertEqual(seen, set(status.PHASES))


# what jetlink's own servers report as a stage (JetlinkKit), by the calls that report one
SWIFT_SOURCES = Path(__file__).parents[2] / 'JetlinkKit' / 'Sources'
SWIFT_STAGE = re.compile(r'\b(?:report|progress|paced)\("([a-z_]+)"')


class TestCodes(unittest.TestCase):
  def test_each_reason_has_its_code(self):
    cases = [(None, None), (status.NO_WARP, 'no_warp'), (status.STOPPED, 'service_stopped'),
             (f"{gadget.BACKING_OFF} (5 times in 10 min), waiting 60 s before starting it again", 'backing_off'),
             (status.failure('RuntimeError: boom', 'usb'), 'status_failed'),
             ('the lender could not listen: address in use', 'gadget')]
    for reason, code in cases:
      with self.subTest(reason):
        self.assertEqual(snapshot(enabled=True, reason=reason).reason_code, code)
    self.assertEqual({c for _, c in cases} - {None}, set(status.REASON_CODES))

  def test_every_progress_code_jetlink_and_its_servers_report_is_listed(self):
    reported = set()
    for path in Path(status.__file__).parents[1].rglob('*.py'):
      for node in ast.walk(ast.parse(path.read_text())):
        name = getattr(getattr(node, 'func', None), 'attr', None)
        if not isinstance(node, ast.Call) or name not in ('report', '_report', 'progress_cb'):
          continue
        given = [k.value for k in node.keywords if k.arg == 'code']
        if name == '_report' and len(node.args) > 2:
          given.append(node.args[2])
        if name == 'progress_cb':   # the client's own stage, which report_with_eta passes on as the code
          given.append(node.args[0])
        reported |= {g.value for g in given if isinstance(g, ast.Constant) and g.value}
    reported.add(link.ModelMissing.code)
    if not SWIFT_SOURCES.is_dir():
      self.assertLessEqual(reported, set(status.PROGRESS_CODES))
      self.skipTest('no JetlinkKit checkout: its stages are not checked')
    for path in SWIFT_SOURCES.rglob('*.swift'):
      reported |= set(SWIFT_STAGE.findall(path.read_text()))
    self.assertEqual(reported, set(status.PROGRESS_CODES))

  def test_a_code_and_its_args_travel_beside_the_message(self):
    progress = status.Progress(SimpleNamespace(keys=SimpleNamespace(progress='p'), put=mock.Mock()))
    progress.report('download', 0.5, 'downloading 1 of 2 MB', code='downloading', args={'got_mb': 1, 'total_mb': 2})
    progress.op.put.assert_called_once_with('p', {'stage': 'download', 'frac': 0.5, 'msg': 'downloading 1 of 2 MB',
                                                  'drops': 0, 'code': 'downloading', 'args': {'got_mb': 1, 'total_mb': 2}})
    progress.report('connect', 0.0, 'waiting for jetlink', code='waiting_for_host', args={})
    self.assertEqual(progress.op.put.call_args.args[1], {'stage': 'connect', 'frac': 0.0, 'msg': 'waiting for jetlink',
                                                         'drops': 0, 'code': 'waiting_for_host'})

  def test_the_server_stages_are_the_codes_with_their_numbers(self):
    seen = []
    progress = status.Progress(SimpleNamespace(), size=lambda: 1_850_000_000)
    with mock.patch.object(progress, 'report', lambda *a, **kw: seen.append(kw)):
      progress.report_with_eta('build', 0.8, 'building the engine')
      progress.report_with_eta('upload', 0.5, '882/1764 MB')
      progress.report_with_eta('compile', 0.5, 'compiling')
      progress.report_with_eta('', 0.0, 'a message with no stage')
    self.assertEqual(seen, [{'code': 'build', 'args': {'eta_s': 60}},
                            {'code': 'upload', 'args': {'got_mb': 882, 'total_mb': 1764}},
                            {'code': 'compile', 'args': None}, {'code': None, 'args': None}])


class TestTheRuntimeRecord(OpenpilotTest):
  """modeld's record (gadget.MODELD) as the snapshot reads it: fresh, from a
  live process that did not stop, and active only on a recent answer."""

  def write(self, **fields):
    now = time.monotonic()
    gadget.write_record(gadget.MODELD, {'pid': os.getpid(), 'at': now, 'stopping': False, 'state': 'running',
                                        'active': True, 'last_output_at': now - 0.05, **fields})

  def test_a_live_record_with_a_fresh_answer_is_active(self):
    self.write()
    self.assertEqual(status.runtime()[:2], ('running', True))

  def test_which_model_drives_is_told_only_while_it_does(self):
    # what a fork's companion shows of the model on the road (StarPilot's Galaxy)
    self.write(sha256='ab' * 32, checkpoint='cinque-v3')
    self.assertEqual(status.runtime(), status.Runtime('running', True, 'ab' * 32, 'cinque-v3'))
    self.write(sha256='ab' * 32, checkpoint='cinque-v3', active=False)
    self.assertEqual(status.runtime(), status.Runtime('running', False, None, None))
    self.write(sha256=7, checkpoint=['x'])
    self.assertEqual(status.runtime(), status.Runtime('running', True, None, None))

  def test_an_answer_too_old_when_written_is_not_active(self):
    now = time.monotonic()
    self.write(at=now, last_output_at=now - status.OUTPUT_FRESH - 0.05)
    self.assertEqual(status.runtime()[:2], ('running', False))
    self.write(state='ready', active=False)
    self.assertEqual(status.runtime()[:2], ('ready', False))

  def test_a_record_too_old_to_drive_still_says_what_runs(self):
    # a modeld frozen for two writer periods: not driving, but not gone for HEARTBEAT_TIMEOUT
    self.assertLess(status.ACTIVE_FRESH + 0.5, gadget.HEARTBEAT_TIMEOUT)
    at = time.monotonic() - status.ACTIVE_FRESH - 0.5
    self.write(at=at, last_output_at=at - 0.05)
    self.assertEqual(status.runtime()[:2], ('running', False))

  def test_a_stale_stopping_or_dead_record_is_none(self):
    self.write(at=time.monotonic() - gadget.HEARTBEAT_TIMEOUT - 0.5)
    self.assertEqual(status.runtime()[:2], (None, False))
    self.write(stopping=True, state='unavailable')
    self.assertEqual(status.runtime()[:2], (None, False))
    self.write()
    with mock.patch.object(gadget.os, 'kill', side_effect=ProcessLookupError):
      self.assertEqual(status.runtime()[:2], (None, False))
    for pid in (0, -1, None, '12', True, 2 ** 80):
      self.write(pid=pid)
      self.assertEqual(status.runtime()[:2], (None, False), pid)

  def test_no_record_or_one_that_is_not_one_is_none(self):
    self.assertEqual(status.runtime()[:2], (None, False))
    for text in ('nope', '[]', '{"pid": 1}', '{"at": "x"}'):
      gadget.MODELD.write_text(text)
      self.assertEqual(status.runtime()[:2], (None, False), text)

  def test_a_process_that_is_not_ours_to_signal_is_alive(self):
    with mock.patch.object(gadget.os, 'kill', side_effect=PermissionError):
      self.assertTrue(gadget.pid_alive(1))

  def test_the_snapshot_carries_it_and_reads_it_only_with_the_link_on(self):
    self.op.set_mode('usb')
    self.patch(self.parts.warps, 'built', return_value=True)
    self.patch(gadget, 'gadget_error', return_value=None)
    self.write()
    s = self.jl.status()
    self.assertEqual((s.running, s.active, s.phase), ('running', True, 'active'))
    self.op.set_mode('off')
    with mock.patch.object(status, 'runtime') as runtime:
      s = self.jl.status()
    runtime.assert_not_called()
    self.assertEqual((s.running, s.active, s.phase), (None, False, 'off'))

  def test_runtime_is_the_snapshots_running_alone(self):
    # Jetlink.runtime(), for selfdrived's loop: what status() says of modeld
    # in every case, and no param read for it (spec, progress, the models)
    self.patch(self.parts.warps, 'built', return_value=True)
    self.patch(gadget, 'gadget_error', return_value=None)
    for record in (False, True):
      for mode, chestnut in (('usb', False), ('off', False), ('usb', True), ('ios', False)):
        with self.subTest(record=record, mode=mode, chestnut=chestnut):
          if record:   # fresh for each, so a slow run never ages it past ACTIVE_FRESH between the two reads
            self.write(sha256='ab' * 32, checkpoint='cinque-v3')
          self.op.set_mode(mode)
          self.op.chestnut, self.parts._chestnut = chestnut, None
          s = self.jl.status()
          with mock.patch.object(self.op, 'get', side_effect=AssertionError('a param read')), \
               mock.patch.object(status, 'read', side_effect=AssertionError('the snapshot')):
            found = self.jl.runtime()
          self.assertEqual(tuple(found or status.Runtime()), (s.running, s.active, s.running_sha256, s.running_checkpoint))
          self.assertEqual(found is None, not s.enabled)


if __name__ == '__main__':
  unittest.main()
