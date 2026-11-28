"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The process that holds the gadget: what it keeps, what it lets go of, and when
it starts the heavy half. Its settings are jetlink.openpilot's Settings over
the directory and keys the fork's adapter names, as jetlink.openpilot.owner
hands them over.
"""
import errno
import json
import logging
import os
import select
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from jetlink.comma import gadget, lending, owner, root
from jetlink.openpilot import owner as openpilot_owner
from jetlink.openpilot.settings import FileParams, Settings
from tests import comma_fakes
from tests.openpilot.fakes import CHESTNUT_IDS, KEYS


class OwnerTest(unittest.TestCase):
  def setUp(self):
    self.tmp = Path(tempfile.mkdtemp())
    self.params = self.tmp / 'params'
    self.params.mkdir()
    self.write('JetlinkLink', b'1')   # USB
    self.write('IsOffroad', b'1')
    for name, value in (('DORMANT', self.tmp / 'dormant'),
                        ('SHUTDOWN_REQUEST', self.tmp / 'shutdown'),
                        ('GADGET_STATUS', self.tmp / 'gadget-status'),
                        ('LENDER_STATUS', self.tmp / 'lender-status'),
                        ('STATUS', self.tmp / 'run' / 'status.json'),
                        ('STARTS', self.tmp / 'run' / 'starts.json'),
                        ('OWNER_LOCK', self.tmp / 'run' / 'owner.lock'),
                        ('SERVER', self.tmp / 'run' / 'server.json'),
                        ('CC_ORIENTATION', self.tmp / 'cc'),
                        ('link_configured', mock.Mock(return_value=True)),
                        ('host_attached', mock.Mock(return_value=True)),
                        ('udc_state', mock.Mock(return_value='configured')),
                        ('wait_for_host', mock.Mock(return_value=True)),
                        # the owner's own records, and a loopback stand-in for usb0
                        ('LINK', self.tmp / 'link'),
                        ('CABLE_ADDR', ('127.0.0.1', 0)),
                        ('net_up', mock.Mock(return_value=True)),
                        ('net_status', mock.Mock(return_value='ok 192.168.60.1')),
                        ('usb_speed', mock.Mock(return_value='super-speed'))):
      p = mock.patch.object(gadget, name, value)
      self.addCleanup(p.stop)
      p.start()
    # every root step: the real one is sudo on a comma
    p = mock.patch.object(root, 'run', mock.Mock(return_value=True))
    self.addCleanup(p.stop)
    self.root_run = p.start()
    # the real one runs sudo on a comma, and these run there too
    p = mock.patch.object(owner.port, 'Port', mock.Mock())
    self.addCleanup(p.stop)
    p.start()
    # no policy engine, as off a comma; TestDraw sets what is on the port
    p = mock.patch.object(owner.port, 'power_role', mock.Mock(return_value=None))
    self.addCleanup(p.stop)
    self.power_role = p.start()
    # no chestnut on the port; the tests that want one say so
    p = mock.patch.object(owner.port, 'chestnut_attached', mock.Mock(return_value=False))
    self.addCleanup(p.stop)
    self.chestnut = p.start()

  def write(self, key: str, value: bytes) -> None:
    path = self.params / key
    path.write_bytes(value)
    # The owner sees a param move by its mtime, and Linux stamps files from a
    # clock that ticks every few ms: two writes in one tick look like none.
    self.stamp = max(getattr(self, 'stamp', 0), time.time_ns()) + 10_000_000
    os.utime(path, ns=(self.stamp, self.stamp))

  def make(self, *args, **kwargs) -> owner.Owner:
    """An owner over jetlink.openpilot's Settings, as jetlink.openpilot.owner
    hands it over, on this test's params."""
    kwargs.setdefault('settings', Settings(FileParams(self.params), KEYS))
    kwargs.setdefault('chestnut_ids', CHESTNUT_IDS)
    return owner.Owner(*(args or ((),)), **kwargs)

  def stopped(self, o) -> None:
    """manager stops the owner, without another step: it reads the same
    setting, and finds a chestnut on its own."""
    o.stop = True
    o.run()

  def root_calls(self, command: str) -> list[str]:
    """What the owner asked jetlink-root.sh `command` to do, in order."""
    return [c.args[1] for c in self.root_run.call_args_list if c.args[0] == command]

  def heard(self, o, sleep_after=1.0) -> None:
    """A borrower passed on the server's hello (lending.Loan.note_server)."""
    o.note_server('modeld', {'device': 'orin', 'sleep_after': sleep_after})

  def owner(self, presented=True, lendable=False):
    o = self.make()
    o.lender = mock.Mock(lent=False, listening=True)
    o.transport = mock.Mock(lendable=lendable) if presented else None
    for name in ('open_link', 'spawn_worker'):
      p = mock.patch.object(o, name, mock.Mock(return_value=True))
      self.addCleanup(p.stop)
      p.start()
    p = mock.patch.object(o, 'close_link', mock.Mock(side_effect=lambda: setattr(o, 'transport', None)))
    self.addCleanup(p.stop)
    p.start()
    self.addCleanup(o.cable.close)
    # a hello has been passed on, and nothing is outstanding: the far end sleeps
    self.heard(o, sleep_after=1.0)
    o.seen = o.settings.marks()
    o.had_host = True
    return o

  def dial(self, o) -> socket.socket:
    """A phone: connects to the listener the owner opened."""
    self.assertTrue(o.cable.listening, 'the owner is not listening for a phone')
    phone = socket.create_connection(o.cable.bound[:2], timeout=3.0)
    self.addCleanup(phone.close)
    # the loopback handshake can still be finishing when connect returns; the
    # owner's next step takes the dial once the listener has it to accept
    readable, _, _ = select.select([o.cable._srv], [], [], 3.0)
    self.assertTrue(readable, 'the dial never reached the listener')
    return phone

  def hang_up(self, o, phone: socket.socket) -> None:
    """The phone closes its end, and the owner's next step can see it: the
    loopback FIN can still be on its way when close returns."""
    phone.close()
    readable, _, _ = select.select([o.cable._sock], [], [], 3.0)
    self.assertTrue(readable, 'the hang-up never reached the owner')


class TestOnroad(OwnerTest):
  """Once the car is moving the owner holds the gadget and stays off the bus."""

  def onroad(self) -> None:
    self.write('IsOffroad', b'0')

  def test_it_holds_the_gadget_and_does_nothing_else(self):
    o = self.owner(lendable=True)
    self.onroad()
    o.step()
    o.spawn_worker.assert_not_called()
    o.close_link.assert_not_called()
    o.transport.release_endpoints.assert_not_called()

  def test_a_run_still_going_is_stopped_so_modeld_can_borrow(self):
    # a build started while parked can still be running when the driver pulls
    # away; the lease it holds would keep modeld out for the whole drive
    o = self.owner(lendable=True)
    worker = o.worker = mock.Mock(**{'poll.return_value': None})
    self.onroad()
    o.step()
    worker.terminate.assert_called_once()

  def test_a_borrower_keeps_the_gadget_on_the_bus(self):
    o = self.owner(lendable=True)
    o.lender.lent = True
    o.step()
    o.close_link.assert_not_called()
    o.spawn_worker.assert_not_called()

  def test_endpoints_left_open_are_put_down_without_letting_go_of_ep0(self):
    o = self.owner(lendable=False)
    self.onroad()
    o.step()
    o.transport.release_endpoints.assert_called_once()
    o.close_link.assert_not_called()

  def test_a_borrower_wakes_a_dormant_owner(self):
    o = self.owner(presented=False)
    o.dormant = True
    o.lender.lent = True
    o.step()
    self.assertFalse(o.dormant)
    o.open_link.assert_called_once()


class TestParked(OwnerTest):
  """Letting the Jetson sleep, and taking the gadget back when there is work."""

  def test_the_gadget_is_held_until_the_hold_has_passed(self):
    o = self.owner()
    o.step()
    self.assertFalse(o.dormant)

  def test_it_releases_once_there_is_nothing_to_do_and_the_far_end_sleeps(self):
    o = self.owner()
    o.idle_since = time.monotonic() - owner.DORMANT_HOLD
    o.step()
    self.assertTrue(o.dormant)
    o.close_link.assert_called_once()
    self.assertTrue(gadget.dormant())

  def test_a_far_end_that_never_sleeps_keeps_the_gadget(self):
    # on ignition power the Jetson stays up, and letting go would leave a
    # powered awake box unenumerated for the whole parked period
    o = self.owner()
    self.heard(o, sleep_after=0.0)
    o.idle_since = time.monotonic() - owner.DORMANT_HOLD
    o.step()
    self.assertFalse(o.dormant)
    o.close_link.assert_not_called()

  def test_a_far_end_too_old_to_say_keeps_the_release_it_always_had(self):
    o = self.owner()
    o.server = None
    o.idle_since = time.monotonic() - owner.DORMANT_HOLD
    o.step()
    self.assertTrue(o.dormant)

  def test_a_run_that_wakes_the_jetson_gets_the_hold_before_letting_go(self):
    # a run can finish seconds after the wake, with the Jetson still
    # enumerating; the hold runs from the wake, not from process start
    o = self.owner()
    o.idle_since = time.monotonic() - owner.DORMANT_HOLD
    o.worker = mock.Mock(**{'poll.return_value': 0, 'returncode': 0})
    o.step()
    self.assertFalse(o.dormant, 'let the gadget go while the jetson was waking')
    o.idle_since = time.monotonic() - owner.DORMANT_HOLD
    o.step()
    self.assertTrue(o.dormant)

  def test_a_borrower_holds_the_gadget_past_the_drive(self):
    o = self.owner()
    o.idle_since = time.monotonic() - owner.DORMANT_HOLD
    o.lender.lent = True
    o.step()
    o.lender.lent = False
    o.step()
    self.assertFalse(o.dormant, 'let the gadget go the moment the drive ended')


class TestStartingTheHeavyHalf(OwnerTest):
  """The owner cannot tell whether there is work: that needs the catalog, the
  spec and the Jetson. It notices what could have changed the answer."""

  def test_the_first_look_of_the_boot_always_runs(self):
    o = self.owner()
    o.seen = {}
    o.step()
    o.spawn_worker.assert_called_once()

  def test_a_new_pick_starts_a_run(self):
    o = self.owner()
    o.step()
    o.spawn_worker.assert_not_called()
    self.write('ModelManager_ActiveBundleChestnut', b'{"ref": "b" * 40}')
    o.step()
    o.spawn_worker.assert_called_once()

  def test_a_jetson_turning_up_starts_a_run(self):
    o = self.owner()
    o.had_host = False
    o.step()
    o.spawn_worker.assert_called_once()

  def test_a_shutdown_request_starts_a_run(self):
    o = self.owner()
    gadget.SHUTDOWN_REQUEST.write_text(json.dumps({'reason': 'car battery'}))
    o.step()
    o.spawn_worker.assert_called_once()

  def test_an_unfinished_run_is_tried_again_on_its_own_timer(self):
    o = self.owner()
    o.unfinished = True
    o.next_worker = time.monotonic() + owner.WORKER_BACKOFF
    o.step()
    o.spawn_worker.assert_not_called()
    o.next_worker = 0.0
    o.step()
    o.spawn_worker.assert_called_once()

  def test_only_one_run_at_a_time(self):
    o = self.owner()
    o.seen = {}
    o.worker = mock.Mock(**{'poll.return_value': None})
    o.step()
    o.spawn_worker.assert_not_called()

  def test_a_dormant_owner_wakes_before_starting_one(self):
    o = self.owner(presented=False)
    o.dormant = True
    o.seen = {}
    o.step()
    self.assertFalse(o.dormant)
    o.spawn_worker.assert_called_once()

  def test_nothing_is_started_over_the_servers_teardown(self):
    # the borrower let go a moment ago and the server is still reopening the
    # gadget it lost; a hello inside that window costs a re-enumeration
    o = self.owner()
    o.lender.lent = True
    o.step()
    o.lender.lent = False
    o.seen = {}
    o.step()
    o.spawn_worker.assert_not_called()
    o.lease_settled = 0.0
    o.step()
    o.spawn_worker.assert_called_once()


class TestTheRunThatFinishes(OwnerTest):
  def finished(self, o):
    """A worker that has just exited."""
    o.worker = mock.Mock(**{'poll.return_value': 0, 'returncode': 0})

  def test_a_successful_provision_does_not_start_a_second_run(self):
    # a run writes JetlinkSpec itself, so a mark taken when it was spawned
    # always differs by the time it exits
    o = self.owner()
    self.finished(o)
    self.write('JetlinkSpec', b'{}')
    o.step()
    o.spawn_worker.assert_not_called()

  def test_a_pick_changed_while_the_run_was_going_is_still_seen(self):
    o = self.owner()
    o.worker = mock.Mock(**{'poll.return_value': None})
    o.step()
    self.write('ModelManager_ActiveBundleChestnut', b'{"ref": "c"}')
    o.worker.poll.return_value = o.worker.returncode = 0
    o.step()
    # the mark is retaken when the run exits, so this looks unchanged...
    o.spawn_worker.assert_not_called()
    self.write('ModelManager_ActiveBundleChestnut', b'{"ref": "d"}')
    o.step()
    o.spawn_worker.assert_called_once()   # ...and a later pick still starts one


class TestWhatItIsTold(OwnerTest):
  """What the owner never speaks the protocol to learn: the server's hello, as
  every borrower passes it on after every hello, and whether a run left work,
  which is the run's exit status."""

  def test_every_hello_passed_on_refreshes_the_far_end(self):
    # a run with no work never refreshes the record, so without this a Jetson
    # moved to always-on power would still be let go every drive
    o = self.owner()
    self.heard(o, sleep_after=0.0)
    o.idle_since = time.monotonic() - owner.DORMANT_HOLD
    o.step()
    self.assertFalse(o.dormant, 'let an always-on jetson go')
    self.heard(o, sleep_after=60.0)   # the next drive's hello, with no model change
    o.step()
    self.assertTrue(o.dormant)

  def test_a_server_that_does_not_say_sleeps(self):
    for said in ({}, {'sleep_after': None}, {'sleep_after': 'soon'}, {'sleep_after': 120}):
      self.assertTrue(owner.server_sleeps(said), said)
    self.assertTrue(owner.server_sleeps(None))
    self.assertFalse(owner.server_sleeps({'sleep_after': 0}))

  def test_it_is_in_the_record_and_said_when_it_changes(self):
    o = self.owner()
    with mock.patch.object(gadget, 'log') as log:
      for _ in range(3):
        o.note_server('modeld', {'device': 'orin', 'sleep_after': 0.0, 'protocol': 3})
      o.note_server('provision', {'device': 'orin', 'sleep_after': 60.0, 'protocol': 3})
    self.assertEqual(log.warning.call_count, 2)
    self.assertIn('stays up', log.warning.call_args_list[0].args[3])
    o.publish_status()
    self.assertEqual(gadget.owner_status()['server'], {'device': 'orin', 'sleep_after': 60.0, 'protocol': 3})

  def test_an_owner_started_again_keeps_what_the_last_one_heard(self):
    # parked, with nobody to say hello again until the next drive: after a
    # crash, and after the link turned Off and On, whose clean stop removes
    # the status record
    for clean in (False, True):
      first = self.owner()
      self.heard(first, sleep_after=0.0)
      if clean:
        self.stopped(first)
        self.assertFalse(gadget.STATUS.exists())
      second = self.make()   # not self.owner(), which passes on a hello of its own
      self.addCleanup(second.cable.close)
      second.adopt()
      self.assertFalse(second.far_end_sleeps(), 'clean' if clean else 'crash')
      gadget.SERVER.unlink()

  def test_what_it_heard_is_kept_when_it_changes(self):
    o = self.owner()   # heard sleep_after 1.0 already
    with mock.patch.object(gadget, 'write_record', wraps=gadget.write_record) as write:
      self.heard(o, sleep_after=1.0)
      write.assert_not_called()
      self.heard(o, sleep_after=0.0)
    write.assert_called_once_with(gadget.SERVER, {'device': 'orin', 'sleep_after': 0.0})
    self.assertEqual(json.loads(gadget.SERVER.read_text()), {'device': 'orin', 'sleep_after': 0.0})

  def test_a_server_record_that_is_not_one_is_none(self):
    gadget.SERVER.parent.mkdir(parents=True, exist_ok=True)
    for text in ('', 'nope', '[0]'):
      gadget.SERVER.write_text(text)
      o = self.make()
      self.addCleanup(o.cable.close)
      o.adopt()
      self.assertIsNone(o.server, text)

  def test_the_runs_exit_status_says_whether_it_left_work(self):
    o = self.owner()
    for code, unfinished in ((0, False), (1, True), (-9, True), (0, False)):
      o.worker = mock.Mock(**{'poll.return_value': code, 'returncode': code})
      self.assertFalse(o.worker_running())
      self.assertEqual(o.unfinished, unfinished, code)
    o.publish_status()
    self.assertIs(gadget.owner_status()['unfinished'], False)

  def test_a_run_stopped_on_the_way_says_so_too(self):
    o = self.owner()
    o.worker = mock.Mock(**{'wait.return_value': 1})
    o.stop_worker()
    self.assertTrue(o.unfinished)
    o.worker = mock.Mock(**{'wait.return_value': 0})
    o.stop_worker()
    self.assertFalse(o.unfinished, 'a run that finished its round as it was stopped')
    o.worker = mock.Mock(**{'wait.side_effect': subprocess.TimeoutExpired('run', owner.POLL)})
    with mock.patch.object(owner, 'WORKER_GRACE', 0.0):
      o.stop_worker()
    self.assertTrue(o.unfinished, 'killed')

  def test_a_borrowers_note_reaches_it_through_the_lender(self):
    o = self.make()
    self.addCleanup(o.cable.close)
    with mock.patch.object(lending, 'SOCKET', self.tmp / 'lend.sock'):
      o.lender = lending.Lender(o.lendable, o.bounce_gadget, server=o.note_server)
      self.assertTrue(o.lender.start())
      self.addCleanup(o.lender.stop)
      conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
      conn.settimeout(lending.POLL)
      conn.connect(str(o.lender.path))
      loan = lending.Loan(conn, bytearray(), '', '', name='modeld')
      self.addCleanup(loan.close)
      self.assertTrue(loan.note_server({'device': 'orin', 'sleep_after': 0.0, 'loaded': 'x' * 64, 'engine_state': 'ready'}))
    self.assertEqual(o.server, {'device': 'orin', 'sleep_after': 0.0})


class TestShutdown(OwnerTest):
  """hardwared waits 25 s and a build takes minutes, so the request cannot
  queue behind a provisioning run."""

  def request(self) -> None:
    gadget.SHUTDOWN_REQUEST.write_text(json.dumps({'reason': 'car battery'}))

  def test_it_does_not_wait_for_a_run_in_flight(self):
    o = self.owner()
    worker = o.worker = mock.Mock(**{'poll.return_value': None})
    self.request()
    o.step()
    worker.terminate.assert_called_once()
    o.spawn_worker.assert_called_once()
    assert 'shut down' in o.spawn_worker.call_args.args[0]

  def test_the_run_it_starts_is_left_to_ask(self):
    # a step is half a second, less than a run takes to start: restarting it
    # every step would mean nothing ever asks
    o = self.owner()
    self.request()
    o.step()
    o.spawn_worker.assert_called_once()
    asking = o.worker = mock.Mock(**{'poll.return_value': None})
    for _ in range(5):
      o.step()
    asking.terminate.assert_not_called()
    o.spawn_worker.assert_called_once()

  def test_a_run_that_exits_with_the_request_there_is_tried_again_shortly(self):
    o = self.owner()
    self.request()
    o.step()
    o.worker = mock.Mock(**{'poll.return_value': 0, 'returncode': 0})
    o.step()                                  # it finished, within the retry wait
    o.spawn_worker.assert_called_once()
    o.next_shutdown_run = 0.0                 # the wait is over
    o.step()
    assert o.spawn_worker.call_count == 2

  def test_a_borrower_is_left_alone(self):
    # modeld has the endpoints: this cannot talk over it, and hardwared only
    # shuts a parked car down anyway
    o = self.owner()
    o.lender.lent = True
    self.request()
    o.step()
    o.spawn_worker.assert_not_called()


class TestNobodyCanBorrow(OwnerTest):
  """Only the owner ever holds ep0. A lender that cannot listen leaves nobody
  a way to the link, so the owner keeps the gadget, says why where the panels
  look, and tries again."""

  REASON = 'the lender could not listen: [Errno 30] Read-only file system'

  def owner(self, **kw):
    o = super().owner(**kw)
    o.lender.listening = False
    o.lender.start.return_value = False
    o.lender.error = '[Errno 30] Read-only file system'
    return o

  def test_the_gadget_is_held_through_the_drive(self):
    o = self.owner(lendable=True)
    self.write('IsOffroad', b'0')
    o.step()
    o.close_link.assert_not_called()
    self.assertIsNotNone(o.transport)
    self.assertEqual(gadget.gadget_error(), self.REASON)
    self.assertEqual(gadget.LENDER_STATUS.read_text(), f'error: {self.REASON}\n')

  def test_it_is_not_a_build_failure(self):
    # the gadget exists; rebuilding it would not help, and would unplug the host
    o = self.owner()
    o.step()
    self.assertIsNone(gadget.build_error())

  def test_said_once_and_retried_on_a_backoff(self):
    o = self.owner()
    with mock.patch.object(gadget, 'log') as log:
      for _ in range(3):
        o.step()
      o.lender.start.assert_called_once()
      self.assertEqual(log.error.call_count, 1)
      o.next_lender = 0.0
      o.step()
      self.assertEqual(o.lender.start.call_count, 2)
      self.assertEqual(log.error.call_count, 1, 'said it again every retry')

  def test_listening_again_clears_the_error(self):
    o = self.owner()
    o.step()
    o.lender.start.return_value = True
    o.next_lender = 0.0
    o.step()
    self.assertIsNone(gadget.gadget_error())
    self.assertFalse(o.lender_failed)

  def test_a_stop_clears_the_error(self):
    # a chestnut turning up stops the owner with the link still on
    o = self.owner()
    o.step()
    self.stopped(o)
    self.assertIsNone(gadget.gadget_error())

  def test_parked_it_still_provisions(self):
    o = self.owner(lendable=True)
    o.seen = {}
    o.step()
    o.close_link.assert_not_called()
    o.spawn_worker.assert_called_once()

  def test_a_listening_lender_is_left_alone(self):
    o = super().owner()
    o.step()
    o.lender.start.assert_not_called()
    self.assertIsNone(gadget.gadget_error())


class TestTheToggle(OwnerTest):
  def test_turning_it_off_lets_everything_go(self):
    o = self.owner()
    worker = o.worker = mock.Mock(**{'poll.return_value': None})
    self.write('JetlinkLink', b'0')
    o.step()
    o.close_link.assert_called_once()
    worker.terminate.assert_called_once()

  def test_the_first_gadget_is_not_held_up_by_the_sysctls(self):
    self.write('IsOffroad', b'0')
    o = self.owner(presented=False)
    with mock.patch.object(gadget, 'link_configured', return_value=False):
      o.step()
    self.assertEqual([c.args[0] for c in self.root_run.call_args_list], ['draw', 'gadget', 'vm', 'udc'])

  def test_the_port_is_kept_a_device_while_the_link_is_on(self):
    o = self.owner()
    o.step()
    o.port.update.assert_called_once_with(configured=False, charge=False)

  def test_on_ios_the_phone_is_not_charged_without_its_param(self):
    self.write('JetlinkLink', b'2')
    o = self.owner()
    o.step()
    o.port.update.assert_called_once_with(configured=False, charge=False)

  def test_charging_the_phone_waits_for_its_param(self):
    self.write('JetlinkLink', b'2')
    self.write('JetlinkChargePhone', b'1')
    o = self.owner()
    o.step()
    o.port.update.assert_called_once_with(configured=False, charge=True)

  def test_charging_is_only_for_ios(self):
    self.write('JetlinkChargePhone', b'1')
    o = self.owner()
    o.step()
    o.port.update.assert_called_once_with(configured=False, charge=False)

  def test_turning_it_off_gives_the_port_back(self):
    o = self.owner()
    self.write('JetlinkLink', b'0')
    o.step()
    o.port.off.assert_called_once()

  def test_on_wifi_the_port_and_the_gadget_are_let_go(self):
    # modeld dials the phone itself: nothing of the owner's is on the link
    o = self.owner()
    worker = o.worker = mock.Mock(**{'poll.return_value': None})
    self.write('JetlinkLink', b'3')
    o.step()
    o.close_link.assert_called_once()
    worker.terminate.assert_called_once()
    o.port.off.assert_called_once()
    o.port.update.assert_not_called()
    self.assertNotIn('gadget', [c.args[0] for c in self.root_run.call_args_list])

  def test_on_wifi_the_tuning_is_not_wanted_at_a_stop(self):
    o = self.owner()
    self.write('JetlinkLink', b'3')
    self.assertFalse(o.link_wanted())
    self.write('JetlinkLink', b'2')
    self.assertTrue(o.link_wanted())
    o.port.update.assert_not_called()

  def test_stopping_gives_the_port_back(self):
    o = self.owner()
    o.lender.start.return_value = True
    self.stopped(o)
    o.port.off.assert_called_once()


class TestVmTuning(OwnerTest):
  """When the owner applies and restores the VM tuning, against the real
  jetlink-root.sh vm on a fake /proc/sys, and the USB device side's knobs
  (udc) with it. A device with the link off runs stock values, one that turns
  it off or has a chestnut take the port gets them back, at a step or at the
  stop manager makes of it, and any other exit keeps them for the owner that
  comes next. The values and the ratio-mode restore are test_comma_root.py's."""

  def setUp(self):
    super().setUp()
    comma_fakes.proc_sys(self.tmp, comma_fakes.STOCK)
    # udc's knobs, so a restore that ran reads as one that worked
    comma_fakes.pe_params(self.tmp).mkdir()
    comma_fakes.udc_glue(self.tmp).mkdir()
    self.root_run.side_effect = self.run_script

  def run_script(self, *args: str, timeout: float = root.TIMEOUT) -> bool:
    """root.run, without sudo, on the fake /proc/sys."""
    return comma_fakes.run_script(self.tmp, *args, timeout=timeout).returncode == 0

  def tuned(self) -> bool:
    return comma_fakes.read_all(self.tmp, comma_fakes.TUNED) == comma_fakes.TUNED

  def assertRestored(self, vm: list[str], udc: list[str] | None = None) -> None:
    self.assertEqual(self.root_calls('vm'), vm)
    self.assertEqual(self.root_calls('udc'), vm if udc is None else udc)
    self.assertFalse(self.tuned())
    self.assertFalse(comma_fakes.record(self.tmp).exists(), 'the restore never ran')

  def test_applied_once_on_start_and_kept_on_exit(self):
    o = self.owner()
    o.step()
    o.step()
    self.stopped(o)
    self.assertEqual(self.root_calls('vm'), ['apply'], "an exit with the link on is a restart; the next owner keeps them")
    self.assertEqual(self.root_calls('udc'), ['apply'])
    self.assertTrue(self.tuned())
    self.assertTrue(comma_fakes.record(self.tmp).exists(), "the record is what a later disable restores to")

  def test_the_next_start_reapplies_without_touching_the_record(self):
    self.owner().step()
    stock = comma_fakes.record(self.tmp).read_text()
    self.owner().step()
    self.assertEqual(self.root_calls('vm'), ['apply', 'apply'])
    self.assertEqual(comma_fakes.record(self.tmp).read_text(), stock)

  def test_disabled_nothing_is_applied_and_the_restore_runs_once(self):
    # for a tuning an owner before this one left; without its record a no-op
    self.write('JetlinkLink', b'0')
    o = self.owner()
    o.step()
    o.step()
    self.assertRestored(['restore'])

  def test_disabling_mid_run_restores(self):
    o = self.owner()
    o.step()
    self.write('JetlinkLink', b'0')
    o.step()
    self.assertRestored(['apply', 'restore'])

  def test_turned_off_it_is_restored_once_not_every_step(self):
    o = self.owner()
    o.step()
    self.write('JetlinkLink', b'0')
    o.step()
    o.step()
    self.stopped(o)
    self.assertRestored(['apply', 'restore'])

  def test_a_failed_restore_is_tried_again_at_the_stop(self):
    o = self.owner()
    o.step()
    self.root_run.side_effect = lambda *a, **k: a != ('vm', 'restore') and self.run_script(*a, **k)
    self.write('JetlinkLink', b'0')
    o.step()
    o.step()
    self.assertEqual(self.root_calls('vm'), ['apply', 'restore'], 'a failure is not retried every step')
    self.root_run.side_effect = self.run_script
    self.stopped(o)
    self.assertRestored(['apply', 'restore', 'restore'])

  def test_a_stop_after_turning_it_off_restores(self):
    o = self.owner()
    o.step()
    self.write('JetlinkLink', b'0')
    self.stopped(o)
    self.assertRestored(['apply', 'restore'])

  def test_a_stop_restores_what_an_owner_before_applied(self):
    # restarted with the link on, which keeps it; turned off before the next
    # owner has stepped, so this one never applied it
    first = self.owner()
    first.step()
    self.stopped(first)
    self.assertTrue(self.tuned())
    second = self.owner()
    self.write('JetlinkLink', b'0')
    self.stopped(second)
    self.assertRestored(['apply', 'restore'])

  def test_a_stop_for_a_chestnut_restores(self):
    # the link stays on, but manager stops the owner and the comma hosts the chestnut
    o = self.owner()
    o.step()
    self.chestnut.return_value = True
    self.stopped(o)
    self.assertRestored(['apply', 'restore'])
    o.port.off.assert_called_once()

  def test_a_stop_that_cannot_tell_keeps_them(self):
    o = self.owner()
    o.step()
    self.chestnut.side_effect = Exception('a bug')
    with mock.patch.object(gadget.log, 'exception') as log:
      self.stopped(o)
    log.assert_called_once()
    self.assertEqual(self.root_calls('vm'), ['apply'])
    self.assertTrue(self.tuned())


class TestDraw(OwnerTest):
  """No current drawn from the port while the link is on: cut at once, given
  back only on an empty port. A stop with the link still wanted keeps it, like
  the tuning; one with the link turned off or a chestnut on the port gives it
  back, on an empty port."""

  def empty(self, empty: bool = True) -> None:
    self.power_role.return_value = 'none' if empty else 'sink'

  def test_an_exit_with_the_link_on_keeps_it(self):
    o = self.owner()
    o.step()
    o.step()
    self.stopped(o)
    self.assertEqual(self.root_calls('draw'), ['off'])

  def test_a_stop_after_turning_it_off_gives_it_back_on_an_empty_port(self):
    self.empty()
    o = self.owner()
    o.step()
    self.write('JetlinkLink', b'0')
    self.stopped(o)
    self.assertEqual(self.root_calls('draw'), ['off', 'on'])

  def test_a_stop_after_turning_it_off_keeps_it_with_a_host_in(self):
    self.empty()
    o = self.owner()
    o.step()
    self.empty(False)
    self.write('JetlinkLink', b'0')
    with mock.patch.object(gadget.log, 'warning') as log:
      self.stopped(o)
    self.assertEqual(self.root_calls('draw'), ['off'])
    said = [c for c in log.call_args_list if 'draws nothing' in c.args[0]]
    self.assertEqual(len(said), 1, 'no later step gives it back, so it is said once')

  def test_the_draw_is_judged_before_the_hold_goes(self):
    # letting go of a hold can drop the far end for a DRP toggle; the port
    # reading empty for that moment must not give the draw back
    self.empty()
    o = self.owner()
    o.step()
    self.empty(False)
    o.port.off.side_effect = self.empty
    self.write('JetlinkLink', b'0')
    self.stopped(o)
    self.assertEqual(self.root_calls('draw'), ['off'])

  def test_usb_and_ios_cut_it_at_once_with_a_host_in(self):
    # the comma runs off its own power port, never this one
    for link in (b'1', b'2'):
      with self.subTest(link):
        self.root_run.reset_mock()
        self.empty(False)
        self.write('JetlinkLink', link)
        o = self.owner()
        o.step()
        o.step()
        self.assertEqual(self.root_calls('draw'), ['off'])

  def test_without_a_policy_engine_it_is_cut_and_never_given_back(self):
    o = self.owner()
    o.step()
    self.write('JetlinkLink', b'0')
    self.stopped(o)
    self.assertEqual(self.root_calls('draw'), ['off'])

  def test_turning_the_link_off_gives_it_back_once_the_port_is_empty(self):
    self.empty()
    o = self.owner()
    o.step()
    self.empty(False)
    self.write('JetlinkLink', b'0')
    o.step()
    o.step()
    self.assertEqual(self.root_calls('draw'), ['off'])
    self.empty()
    o.step()
    o.step()
    self.assertEqual(self.root_calls('draw'), ['off', 'on'])

  def test_turned_back_on_before_the_port_empties_nothing_moves(self):
    self.empty()
    o = self.owner()
    o.step()
    self.empty(False)
    self.write('JetlinkLink', b'0')
    o.step()
    self.write('JetlinkLink', b'1')
    o.step()
    self.empty()
    o.step()
    self.assertEqual(self.root_calls('draw'), ['off'])

  def test_disabled_from_the_start_it_is_given_back_once(self):
    # for a cut an owner before this one left; letting go of the voter is safe to repeat
    self.empty()
    self.write('JetlinkLink', b'0')
    o = self.owner()
    o.step()
    o.step()
    self.assertEqual(self.root_calls('draw'), ['on'])

  def test_a_stop_for_a_chestnut_keeps_it(self):
    # the comma powers the chestnut, so the cut does nothing; giving it back
    # needs an empty port
    self.empty()
    o = self.owner()
    o.step()
    self.power_role.return_value = 'source'
    self.chestnut.return_value = True
    with mock.patch.object(gadget.log, 'warning') as warning, mock.patch.object(gadget.log, 'info') as info:
      self.stopped(o)
    self.assertEqual(self.root_calls('draw'), ['off'])
    o.port.off.assert_called_once()
    self.assertFalse([c for c in warning.call_args_list if 'USB-C' in c.args[0]], 'a chestnut is not a host')
    self.assertEqual(len([c for c in info.call_args_list if 'powers what is on' in c.args[0]]), 1)

  def test_a_failure_is_not_retried_every_step(self):
    self.empty()
    self.root_run.return_value = False
    o = self.owner()
    o.step()
    o.step()
    self.assertEqual(self.root_calls('draw'), ['off'])


class TestUsb(OwnerTest):
  """Jetlink USB: a Jetson or a Mac. The plain gadget, lent at once;
  nothing waits for a phone and nothing of the phone's runs."""

  def test_borrowers_are_lent_the_endpoint_files(self):
    o = self.owner()
    o.step()
    self.assertFalse(o.cable_mode())
    self.assertEqual(gadget.link_kind(), 'usb')

  def test_no_network_and_no_listener(self):
    o = self.owner()
    o.step()
    o.step()
    gadget.net_up.assert_not_called()
    self.assertFalse(o.cable.listening)


class IosTest(OwnerTest):
  """Jetlink iOS: an iPhone on the gadget's network interface."""

  def setUp(self):
    super().setUp()
    self.write('JetlinkLink', b'2')   # iOS

  def owner(self, **kw):
    o = super().owner(**kw)
    # built for iOS and said so, as the first step of a real owner does:
    # without the record a reader falls back to the setting it was handed
    o.built_ios = True
    o.publish()
    return o


class TestCable(IosTest):
  """The phone dials the comma: the owner while nobody holds the loan, the
  borrower while it does; nothing is ever lent the endpoint files, which a
  phone does not read."""

  def test_borrowers_listen_for_the_phone_themselves(self):
    o = self.owner()
    o.step()
    self.assertTrue(o.cable_mode())
    self.assertEqual(gadget.link_kind(), 'cable')

  def test_a_borrower_takes_the_port_and_the_owner_listens_again_after(self):
    o = self.owner()
    o.step()
    phone = self.dial(o)
    o.step()
    self.assertTrue(o.cable.held)
    # the lender, on a borrow: the owner's listener and the dial it holds go
    o.cable.vacate()
    o.lender.lent = True
    phone.settimeout(3.0)
    self.assertEqual(phone.recv(1), b'')
    o.step()
    self.assertFalse(o.cable.listening, 'listened while the borrower held the port')
    # the loan ends: the owner listens again, and the phone coming back is not news
    o.lender.lent = False
    o.spawn_worker.reset_mock()
    o.step()
    self.assertTrue(o.cable.listening)
    self.dial(o)
    o.step()
    o.spawn_worker.assert_not_called()

  def test_a_dial_is_the_link(self):
    o = self.owner()
    o.step()
    self.dial(o)
    o.step()
    self.assertEqual(gadget.link_peer(), '127.0.0.1')
    self.assertTrue(o.cable.held)
    o.spawn_worker.assert_called_once()
    self.assertIn('phone', o.spawn_worker.call_args.args[0])

  def test_the_phone_coming_back_after_a_run_is_not_another_run(self):
    # the owner hangs up when the borrower finishes and the phone dials
    # again: that must not start a run, which would end the same way, forever
    o = self.owner()
    o.step()
    o.cable.redial_expected = True
    self.dial(o)
    o.step()
    o.spawn_worker.assert_not_called()
    self.assertFalse(o.dialed)

  def test_a_dial_during_a_run_is_left_to_that_run(self):
    # the run spawned at the configured edge is still borrowing, and its
    # borrow takes the dial; a second run after it would be the same work
    o = self.owner()
    o.step()
    o.worker = mock.Mock(**{'poll.return_value': None})
    self.dial(o)
    o.step()
    self.assertFalse(o.dialed)
    o.worker.poll.return_value = 0
    o.worker.returncode = 0
    o.step()
    o.spawn_worker.assert_not_called()

  def test_the_owner_never_sleeps_or_settles(self):
    # every unbind takes the phone's network interface down with it
    o = self.owner(lendable=False)
    o.step()
    self.dial(o)
    o.step()
    o.idle_since = time.monotonic() - owner.DORMANT_HOLD
    o.step()
    self.assertFalse(o.dormant)
    o.close_link.assert_not_called()
    o.transport.release_endpoints.assert_not_called()

  def test_a_phone_that_hung_up_is_let_go_and_waited_for(self):
    o = self.owner()
    o.step()
    phone = self.dial(o)
    o.step()
    self.hang_up(o, phone)
    o.step()
    self.assertFalse(o.cable.held)
    self.assertIsNone(gadget.link_peer())
    self.assertTrue(o.cable_mode(), 'lent the endpoint files to a phone between its dials')

  def test_a_phone_that_hung_up_does_not_send_the_owner_dormant(self):
    # the record says the far end sleeps (a run with nothing to do never
    # asks), but for iOS the gadget stays up: letting go would take the
    # network interface the phone dials back over
    o = self.owner()
    o.step()
    phone = self.dial(o)
    o.step()
    self.hang_up(o, phone)
    o.step()
    o.idle_since = time.monotonic() - owner.DORMANT_HOLD
    o.step()
    self.assertFalse(o.dormant)

  def test_the_host_going_away_clears_the_cable_link(self):
    o = self.owner()
    o.step()
    phone = self.dial(o)
    o.step()
    gadget.host_attached.return_value = False
    o.step()
    self.assertIsNone(gadget.link_peer())
    self.assertFalse(o.cable.held)
    self.assertEqual(phone.recv(1), b'', 'the phone was left talking to nobody')

  def test_a_newer_dial_replaces_an_older_one(self):
    # the app restarted: its old connection must not keep the new one out
    o = self.owner()
    o.step()
    first = self.dial(o)
    o.step()
    self.dial(o)
    o.step()
    self.assertTrue(o.cable.held)
    first.settimeout(3.0)
    self.assertEqual(first.recv(1), b'')

  def test_the_shutdown_path_still_presents_the_gadget(self):
    # the phone is on the gadget's network interface: no gadget, no phone
    o = self.owner(presented=False)
    gadget.SHUTDOWN_REQUEST.write_text(json.dumps({'reason': 'car battery'}))
    o.step()
    o.open_link.assert_called_once()
    o.spawn_worker.assert_called_once()

  def test_closing_the_link_takes_the_listener_and_the_record_with_it(self):
    o = self.make()
    o.lender = mock.Mock(lent=False, listening=True)
    o.transport = mock.Mock()
    self.assertTrue(o.cable.open())
    gadget.note_link('cable', '192.168.60.3')
    o.close_link()
    self.assertFalse(o.cable.listening)
    self.assertIsNone(gadget.link_peer())
    o.transport = None


class TestTheGadgetNetwork(IosTest):
  """usb0 exists only once the UDC is bound, so the owner brings it up after
  its bind, and listens for a phone only once it is there."""

  def test_the_network_comes_up_once_per_bind(self):
    o = self.owner()
    o.step()
    o.step()
    gadget.net_up.assert_called_once()
    self.assertTrue(o.cable.listening)

  def test_a_failed_bring_up_is_retried_after_a_backoff_and_nothing_listens(self):
    gadget.net_up.return_value = False
    o = self.owner()
    for _ in range(3):
      o.step()
    gadget.net_up.assert_called_once()
    self.assertFalse(o.cable.listening, 'the bind to 192.168.60.1 fails without usb0')
    gadget.net_up.return_value = True
    o.next_net_attempt = 0.0
    o.step()
    self.assertEqual(gadget.net_up.call_count, 2)
    self.assertTrue(o.cable.listening)

  def steers(self) -> int:
    return sum(1 for c in self.root_run.call_args_list if c.args == ('rps',))

  def test_the_receive_work_is_steered_once_the_drive_starts(self):
    # net's own steer at the bind reads 00 while the big cores are parked
    o = self.owner()
    o.step()
    self.assertEqual(self.steers(), 0)
    self.write('IsOffroad', b'0')
    for _ in range(3):
      o.step()
    self.assertEqual(self.steers(), 1)

  def test_a_steer_that_does_not_read_back_is_retried_after_a_backoff(self):
    self.root_run.side_effect = lambda *args, **kw: args != ('rps',)
    o = self.owner()
    self.write('IsOffroad', b'0')
    for _ in range(3):
      o.step()
    self.assertEqual(self.steers(), 1)
    self.root_run.side_effect = None
    o.next_steer = 0.0
    o.step()
    o.step()
    self.assertEqual(self.steers(), 2)

  def test_a_steer_that_never_reads_back_is_given_up_until_the_next_bind(self):
    self.root_run.side_effect = lambda *args, **kw: args != ('rps',)
    o = self.owner()
    self.write('IsOffroad', b'0')
    for _ in range(owner.STEER_TRIES + 3):
      o.next_steer = 0.0
      o.step()
    self.assertEqual(self.steers(), owner.STEER_TRIES)

  def test_a_new_bind_is_steered_again(self):
    o = self.owner()
    self.write('IsOffroad', b'0')
    o.step()
    o.transport.rebind.return_value = True
    self.assertTrue(o.bounce_gadget())
    o.step()
    self.assertEqual(self.steers(), 2)

  def test_nothing_presented_brings_nothing_up(self):
    o = self.owner(presented=False)
    o.step()
    gadget.net_up.assert_not_called()

  def test_a_bounce_brings_the_network_up_again(self):
    # the unbind took usb0 with it and the rebind made a bare one
    o = self.owner()
    o.step()
    o.transport.rebind.return_value = True
    self.assertTrue(o.bounce_gadget())
    o.step()
    self.assertEqual(gadget.net_up.call_count, 2)

  def test_an_address_that_is_not_there_yet_does_not_stop_the_owner(self):
    # the network said ok but the address is not local (a race with the
    # script): the bind fails, is noted, and is tried again later
    gadget.CABLE_ADDR = ('192.0.2.1', 0)
    o = self.owner()
    o.step()
    self.assertFalse(o.cable.listening)
    self.assertGreater(o.cable.next_open, time.monotonic())
    gadget.CABLE_ADDR = ('127.0.0.1', 0)
    o.step()
    self.assertFalse(o.cable.listening, 'retried inside the backoff')
    o.cable.next_open = 0.0
    o.step()
    self.assertTrue(o.cable.listening)


class TestSwitchingMode(OwnerTest):
  """USB and iOS are different gadgets; moving the setting rebuilds it, only
  while parked: the rebuild is an unplug. These enter iOS from USB, which also
  waits for nobody to be on the link; TestLeavingIos goes the other way."""

  def setUp(self):
    super().setUp()
    p = mock.patch.object(gadget, 'setup_gadget', mock.Mock(return_value=True))
    self.addCleanup(p.stop)
    self.setup_gadget = p.start()

  def switched(self, **kw):
    o = self.owner(**kw)
    o.built_ios = False
    self.write('JetlinkLink', b'2')   # iOS
    return o

  def test_parked_it_rebuilds_for_the_new_host(self):
    o = self.switched()
    o.step()
    o.close_link.assert_called_once()
    self.setup_gadget.assert_called_once()
    self.assertTrue(o.built_ios)
    o.step()
    self.setup_gadget.assert_called_once()

  def test_entering_ios_a_dead_owners_borrower_holds_the_rebuild_back(self):
    # the owner died parked during a run, and the run is still on the
    # endpoints of the gadget the dead owner bound; the rebuild's unbind
    # would pull it out from under the run
    o = self.switched(presented=False)
    with mock.patch.object(gadget, 'bound_udc', return_value='a600000.dwc3'), mock.patch.object(gadget, 'log') as log:
      o.step()
      o.step()
    self.setup_gadget.assert_not_called()
    self.assertFalse(o.built_ios)
    self.assertEqual(sum('rebuilding it once that lets go' in c.args[0] for c in log.warning.call_args_list), 1)
    # the run exits, its files close, the kernel unbinds
    with mock.patch.object(gadget, 'bound_udc', return_value=None):
      o.step()
    self.setup_gadget.assert_called_once()
    self.assertTrue(o.built_ios)

  def test_our_own_bind_is_no_reason_to_wait(self):
    o = self.switched()   # presented: the bind is ours
    with mock.patch.object(gadget, 'bound_udc', return_value='a600000.dwc3'):
      o.step()
    self.setup_gadget.assert_called_once()

  def test_entering_ios_onroad_waits_for_the_car_to_park(self):
    self.write('IsOffroad', b'0')
    o = self.switched()
    o.step()
    self.setup_gadget.assert_not_called()
    self.write('IsOffroad', b'1')
    o.step()
    self.setup_gadget.assert_called_once()

  def test_entering_ios_a_borrower_on_the_link_is_not_unplugged(self):
    o = self.switched()
    o.lender.lent = True
    o.step()
    self.setup_gadget.assert_not_called()

  def test_what_is_built_is_learned_onroad_too(self):
    # an owner starting mid-drive on an iOS gadget must not take it for USB:
    # that would lend a phone the endpoint files
    self.write('IsOffroad', b'0')
    o = self.owner()
    o.built_ios = None
    with mock.patch.object(gadget, 'built_for_ios', return_value=True):
      o.step()
    self.assertTrue(o.built_ios)
    self.assertTrue(o.cable_mode())
    self.assertEqual(gadget.link_kind(), 'cable')
    self.setup_gadget.assert_not_called()

  def test_a_switch_just_after_a_build_is_not_held_back(self):
    # only a failed build sets the backoff
    o = self.switched()
    o.step()
    self.assertTrue(o.built_ios)
    self.write('JetlinkLink', b'1')
    o.step()
    self.assertFalse(o.built_ios)
    self.assertEqual(self.setup_gadget.call_count, 2)

  def test_a_failed_rebuild_is_retried_after_a_backoff(self):
    self.setup_gadget.return_value = False
    o = self.switched()
    for _ in range(3):
      o.step()
    self.setup_gadget.assert_called_once()
    self.assertFalse(o.built_ios)
    o.next_gadget_attempt = 0.0
    self.setup_gadget.return_value = True
    o.step()
    self.assertEqual(self.setup_gadget.call_count, 2)
    self.assertTrue(o.built_ios)

  def test_a_controller_another_gadget_holds_is_tried_again_in_seconds(self):
    # ADB on: AGNOS's gadget holds the controller until openpilot turns it off
    self.setup_gadget.return_value = False
    o = self.switched()
    with mock.patch.object(owner, 'GADGET_SETUP_BACKOFF', 1e9):
      with mock.patch.object(gadget, 'other_gadget', return_value='g1'):
        o.step()
      self.assertLessEqual(o.next_gadget_attempt - time.monotonic(), owner.GADGET_HELD_RETRY)
      with mock.patch.object(gadget, 'other_gadget', return_value=None):
        o.next_gadget_attempt = 0.0
        o.step()
      self.assertGreater(o.next_gadget_attempt - time.monotonic(), owner.GADGET_HELD_RETRY)

  def test_entering_ios_parked_a_run_of_ours_is_stopped_not_waited_for(self):
    # a run can sit in a build for half an hour; it is ours to stop, and its
    # loan ends as it exits
    o = self.switched()
    o.worker = worker = mock.Mock(**{'poll.return_value': None, 'wait.return_value': 0})
    o.step()
    worker.terminate.assert_called_once()
    self.assertIsNone(o.worker)
    self.setup_gadget.assert_not_called()
    o.step()
    self.setup_gadget.assert_called_once_with(True)
    self.assertTrue(o.built_ios)

  def test_entering_ios_a_run_asking_the_jetson_to_power_off_finishes_first(self):
    o = self.switched()
    o.worker = mock.Mock(**{'poll.return_value': None})
    o.shutting_down = True
    with mock.patch.object(gadget, 'pending_shutdown', return_value='comma shutting down'):
      o.step()
    o.worker.terminate.assert_not_called()
    self.setup_gadget.assert_not_called()


class TestLeavingIos(IosTest):
  """Out of iOS, parked, nothing on the comma's side holds the switch back: an
  iOS gadget lends nobody the endpoint files, so the rebuild pulls nothing out
  from under a borrower, even one waiting for a phone that never dials or a
  run stuck with a silent one."""

  def setUp(self):
    super().setUp()
    p = mock.patch.object(gadget, 'setup_gadget', mock.Mock(return_value=True))
    self.addCleanup(p.stop)
    self.setup_gadget = p.start()

  def assert_usb(self, o):
    self.setup_gadget.assert_called_once_with(False)
    self.assertFalse(o.built_ios)
    self.assertFalse(o.cable_mode())
    self.assertEqual(gadget.link_kind(), 'usb')

  def test_no_phone_and_a_borrower_waiting_for_one(self):
    o = self.owner()
    o.lender.lent = True   # modeld's join, told "retry, waiting for a phone to dial"
    o.step()
    self.write('JetlinkLink', b'1')
    o.step()
    o.close_link.assert_called_once()
    self.assert_usb(o)

  def test_a_real_borrower_is_told_the_cable_at_once_and_the_usb_gadget_after_the_switch(self):
    # nobody waits at the lender for a phone: a borrower in iOS is
    # told the cable and listens for the dial itself, so a switch out of iOS
    # finds nothing of a borrower's under the rebuild
    o = self.make()
    sock = self.tmp / 'lend.sock'
    o.lender = lending.Lender(o.lendable, o.bounce_gadget, path=sock, cable=o.cable_mode, vacate=o.cable.vacate,
                              server=o.note_server)
    self.addCleanup(o.lender.stop)
    self.addCleanup(o.cable.close)
    for name in ('open_link', 'spawn_worker'):
      p = mock.patch.object(o, name, mock.Mock(return_value=True))
      self.addCleanup(p.stop)
      p.start()
    o.built_ios = True
    o.publish()
    o.seen = o.settings.marks()
    o.had_host = True
    self.assertTrue(o.lender.start())
    loan = lending.borrow('modeld', timeout=5.0, path=sock)
    self.assertIsNotNone(loan, 'a borrower in iOS was kept waiting')
    self.assertTrue(loan.cable)
    loan.close()
    deadline = time.monotonic() + 2.0
    while o.lender.lent and time.monotonic() < deadline:
      time.sleep(0.01)
    self.assertFalse(o.lender.lent)
    self.write('JetlinkLink', b'1')
    with mock.patch.object(gadget, 'bound_udc', return_value='a600000.dwc3'):
      o.step()
      self.assert_usb(o)
      # the USB gadget presented, as open_link does on the next step
      o.transport = mock.Mock(lendable=True)
      loan = lending.borrow('modeld', timeout=5.0, path=sock)
    self.assertIsNotNone(loan, 'the next borrower was not lent the USB gadget')
    self.addCleanup(loan.close)
    self.assertFalse(loan.cable)   # the endpoint files, not a phone's dial
    self.assertEqual(loan.udc, 'a600000.dwc3')

  def test_a_silent_phone_with_a_stuck_run_is_left_inside_the_grace(self):
    o = self.owner()
    o.close_link.side_effect = lambda: owner.Owner.close_link(o)   # the real one: it hangs up on the phone
    o.step()
    phone = self.dial(o)   # connects and never sends a byte
    o.step()
    self.assertTrue(o.cable.held)
    # the run the dial started, stuck in the phone's hello
    worker = mock.Mock(**{'poll.return_value': None, 'wait.side_effect': subprocess.TimeoutExpired('run', 0.1)})
    o.worker = worker
    o.lender.lent = True
    self.write('JetlinkLink', b'1')
    with mock.patch.object(owner, 'WORKER_GRACE', 0.2):
      t0 = time.monotonic()
      o.step()
      took = time.monotonic() - t0
    self.assertLess(took, 0.2 + 1.0)
    worker.terminate.assert_called_once()
    worker.kill.assert_called_once()
    self.assertIsNone(o.worker)
    self.assert_usb(o)
    self.assertFalse(o.cable.listening)
    phone.settimeout(2.0)
    self.assertEqual(phone.recv(1), b'', 'the phone was not hung up on')

  def test_a_failed_ios_build_does_not_delay_the_way_back(self):
    # a half-built composite gadget: iOS failed, and the setting goes back to
    # USB inside the 60 s backoff
    o = self.owner()
    o.built_ios = False
    o.publish()
    self.setup_gadget.return_value = False
    o.step()   # iOS wanted: the build fails and backs off
    self.setup_gadget.assert_called_once_with(True)
    self.assertGreater(o.next_gadget_attempt, time.monotonic() + 30)
    self.setup_gadget.reset_mock(return_value=True)
    self.setup_gadget.return_value = True
    self.write('JetlinkLink', b'1')
    o.step()
    self.assert_usb(o)

  def test_a_failed_usb_build_is_retried_onroad_never_back_to_ios(self):
    # the USB build on the way out failed and the car started inside its backoff
    o = self.owner()
    self.setup_gadget.return_value = False
    self.write('JetlinkLink', b'1')
    o.step()
    self.setup_gadget.assert_called_once_with(False)
    self.write('IsOffroad', b'0')
    for _ in range(3):
      o.step()
    self.setup_gadget.assert_called_once_with(False)
    o.next_gadget_attempt = 0.0   # the backoff has passed
    self.setup_gadget.return_value = True
    o.step()
    self.assertEqual(self.setup_gadget.call_args_list, [mock.call(False)] * 2)
    self.assertFalse(o.built_ios)
    self.assertFalse(o.cable_mode())

  def test_off_lets_everything_go(self):
    o = self.owner()
    o.worker = worker = mock.Mock(**{'poll.return_value': None, 'wait.return_value': 0})
    o.lender.lent = True
    self.write('JetlinkLink', b'0')
    o.step()
    o.close_link.assert_called_once()
    worker.terminate.assert_called_once()
    self.setup_gadget.assert_not_called()

  def test_onroad_a_setting_written_another_way_waits_for_park(self):
    # the panels refuse the change onroad; the owner is the second line
    self.write('IsOffroad', b'0')
    o = self.owner()
    o.lender.lent = True
    self.write('JetlinkLink', b'1')
    for _ in range(3):
      o.step()
    self.setup_gadget.assert_not_called()
    self.assertTrue(o.built_ios)
    self.write('IsOffroad', b'1')
    o.step()
    self.assert_usb(o)


class TestTheSetModeOnroad(OwnerTest):
  """The lock while driving is on changing the setting, never on the link:
  the mode that is set is built, and built again, onroad as parked, so a
  Jetson on USB joins mid-drive after a boot, a failed build or a replug."""

  def setUp(self):
    super().setUp()
    self.write('IsOffroad', b'0')
    self.configured = False
    p = mock.patch.object(gadget, 'link_configured', lambda: self.configured)
    self.addCleanup(p.stop)
    p.start()
    p = mock.patch.object(gadget, 'setup_gadget', mock.Mock(side_effect=self.setup))
    self.addCleanup(p.stop)
    self.setup_gadget = p.start()
    self.works = True

  def setup(self, ios):
    self.configured = self.configured or self.works
    return self.works

  def test_no_gadget_yet_one_step_builds_usb_and_presents_it(self):
    o = self.owner(presented=False)
    o.built_ios = None
    o.step()
    self.setup_gadget.assert_called_once_with(False)
    self.assertFalse(o.built_ios)
    o.open_link.assert_called()

  def test_a_failed_build_of_the_set_mode_is_retried_onroad(self):
    self.works = False
    o = self.owner(presented=False)
    o.built_ios = None
    o.step()
    o.step()
    self.setup_gadget.assert_called_once_with(False)
    self.works = True
    o.next_gadget_attempt = 0.0   # the backoff has passed
    o.step()
    self.assertEqual(self.setup_gadget.call_count, 2)
    self.assertFalse(o.built_ios)
    o.open_link.assert_called()

  def test_a_half_built_ios_gadget_is_built_as_usb_onroad(self):
    # parked, a switch to iOS failed half way; the setting went back to USB and
    # the car started before the next step
    self.configured = True
    o = self.owner(presented=False)
    o.built_ios = False
    o.failed_ios = True
    o.next_gadget_attempt = time.monotonic() + 60
    o.step()
    self.setup_gadget.assert_called_once_with(False)
    self.assertIsNone(o.failed_ios)
    self.assertFalse(o.built_ios)

  def test_a_replug_keeps_the_gadget(self):
    self.configured = True
    o = self.owner()
    o.built_ios = False
    with mock.patch.object(gadget, 'host_attached', return_value=False):
      o.step()
    o.step()
    self.setup_gadget.assert_not_called()
    o.close_link.assert_not_called()


class TestTheLoop(OwnerTest):
  """What run() does around each step: nothing may escape it, since manager
  restarting the owner in a loop is worse than sitting out a cycle."""

  def run_steps(self, o, step) -> None:
    """run(), with `step` for the owner's step and no wait between cycles."""
    with mock.patch.object(o, 'step', side_effect=step), mock.patch.object(owner, 'POLL', 0.0):
      o.run()

  def test_an_error_lets_the_link_go_backs_off_and_carries_on(self):
    o = self.owner()
    seen = []

    def step():
      seen.append((o.next_attempt, o.close_link.call_count))
      if len(seen) == 1:
        raise RuntimeError('a step that fails')
      o.stop = True

    started = time.monotonic()
    with mock.patch.object(gadget, 'log') as log:
      self.run_steps(o, step)
    self.assertEqual(len(seen), 2, 'the loop ended at the error')
    next_attempt, closed = seen[1]
    self.assertEqual(closed, 1, 'the link was not let go after the error')
    self.assertGreaterEqual(next_attempt, started + owner.RECONNECT_BACKOFF)
    log.exception.assert_called_once()

  def test_a_previous_owners_record_is_cleared_before_the_first_step(self):
    # the record is this process's to write; one a killed owner left says
    # nothing about the gadget now
    gadget.note_link('cable', '192.168.60.3')
    o = self.owner()
    seen = []

    def step():
      seen.append(gadget.link_peer())
      o.stop = True

    self.run_steps(o, step)
    self.assertEqual(seen, [None])
    self.assertFalse(gadget.LINK.exists())


class TestTheStatusRecord(OwnerTest):
  """What the owner writes down for the UI and hardwared every step, whole,
  which is its heartbeat too (jetlink.openpilot.status reads it)."""

  def record(self, o) -> dict:
    o.publish_status()
    return gadget.owner_status()

  def test_it_is_written_before_the_first_step(self):
    # the first step builds the gadget, which can take a while
    o = self.owner()
    seen = []

    def step():
      seen.append(gadget.owner_status())
      o.stop = True
    with mock.patch.object(owner, 'POLL', 0.0), mock.patch.object(o, 'step', side_effect=step):
      o.run()
    self.assertEqual(seen[0]['pid'], os.getpid())
    self.assertTrue(gadget.owner_alive(seen[0]))

  def test_it_holds_what_the_readers_need(self):
    o = self.owner()
    o.step()
    r = self.record(o)
    self.assertEqual({k: r[k] for k in ('mode', 'link', 'peer', 'error', 'dormant', 'udc', 'speed', 'present', 'worker')},
                     {'mode': 'usb', 'link': 'usb', 'peer': None, 'error': None, 'dormant': False, 'udc': 'configured',
                      'speed': 'super-speed', 'present': True, 'worker': False})
    # a whole record replaced, never a half written one beside it
    self.assertEqual([n for n in os.listdir(gadget.STATUS.parent) if n.startswith('.')], [])

  def test_a_step_is_followed_by_a_record(self):
    o = self.owner()
    stamps = []

    def step():
      stamps.append(o.published)
      if len(stamps) == 2:
        o.stop = True
    with mock.patch.object(owner, 'POLL', 0.0), mock.patch.object(o, 'step', side_effect=step):
      o.run()
    self.assertLess(stamps[0], stamps[1], 'no record between the two steps')

  def test_a_clean_stop_leaves_none(self):
    # only an owner that died leaves a heartbeat behind to go stale
    o = self.owner()
    self.record(o)
    self.stopped(o)
    self.assertIsNone(gadget.owner_status())
    self.assertFalse(gadget.STATUS.exists())

  def test_presence_and_its_hold_are_worked_out_here(self):
    o = self.owner()
    self.assertTrue(self.record(o)['present'])
    gadget.udc_state.return_value = 'addressed'   # a USB3 link recovery passes through it
    self.assertTrue(self.record(o)['present'])
    self.assertIsNone(self.record(o)['speed'])
    o.last_configured -= gadget.PRESENCE_HOLD
    self.assertFalse(self.record(o)['present'])

  def test_a_sleeping_host_is_present_while_the_cable_says_so(self):
    o = self.owner(presented=False)
    o.dormant = True
    gadget.udc_state.return_value = None
    gadget.CC_ORIENTATION.write_text('1')
    r = self.record(o)
    self.assertEqual((r['present'], r['dormant']), (True, True))
    gadget.CC_ORIENTATION.write_text('0')
    self.assertFalse(self.record(o)['present'])

  def test_the_gadgets_files_are_folded_in(self):
    o = self.owner()
    o.built_ios, o._peer = True, '192.168.60.3'
    gadget.GADGET_STATUS.write_text('error: no configfs here\n')
    o.worker = mock.Mock(**{'poll.return_value': None})
    r = self.record(o)
    self.assertEqual((r['error'], r['net'], r['link'], r['peer'], r['worker']),
                     ('no configfs here', 'ok 192.168.60.1', 'cable', '192.168.60.3', True))
    gadget.GADGET_STATUS.unlink()
    gadget.note_lender_error('address in use')
    self.assertEqual(self.record(o)['error'], 'the lender could not listen: address in use')

  def test_a_wait_inside_a_step_keeps_the_heartbeat(self):
    o = self.owner()
    o.published = time.monotonic() - owner.POLL
    with mock.patch.object(o, 'publish_status') as publish:
      self.assertFalse(o.waiting())
      o.stop = True
      o.published = time.monotonic()
      self.assertTrue(o.waiting())
    publish.assert_called_once()   # at most once a POLL

  def test_settling_waits_with_the_heartbeat(self):
    o = self.owner(lendable=False)
    o.transport.release_endpoints.return_value = True
    o.settle()
    self.assertEqual(gadget.wait_for_host.call_args.kwargs['should_stop'], o.waiting)

  def test_stopping_a_slow_run_keeps_the_heartbeat(self):
    o = self.owner()
    worker = o.worker = mock.Mock()
    worker.wait.side_effect = [subprocess.TimeoutExpired('run', owner.POLL)] * 2 + [0]
    with mock.patch.object(o, 'beat') as beat:
      o.stop_worker()
    self.assertEqual(beat.call_count, 2)
    worker.kill.assert_not_called()
    self.assertIsNone(o.worker)

  def test_a_run_that_outlasts_the_grace_is_killed(self):
    o = self.owner()
    worker = o.worker = mock.Mock()
    worker.wait.side_effect = subprocess.TimeoutExpired('run', owner.POLL)
    with mock.patch.object(owner, 'WORKER_GRACE', 0.0):
      o.stop_worker()
    worker.kill.assert_called_once()

  def test_a_record_that_cannot_be_written_is_said_once_and_not_left_stale(self):
    # a stale record under a live owner reads as "service stopped"
    o = self.owner()
    o.publish_status()
    published = o.published
    with mock.patch.object(gadget, 'write_record', side_effect=OSError(28, 'No space left on device')), \
         mock.patch.object(gadget, 'log') as log:
      for _ in range(3):
        o.publish_status()
    self.assertEqual(log.error.call_count, 1)
    self.assertEqual(o.published, published)
    self.assertIsNone(gadget.owner_status(), 'the last record was left to go stale')
    o.publish_status()
    self.assertIsNone(o.status_error)
    self.assertIsNotNone(gadget.owner_status())

  def test_a_failed_write_leaves_no_temporary(self):
    o = self.owner()
    with mock.patch.object(gadget.os, 'replace', side_effect=OSError(28, 'No space left on device')), \
         mock.patch.object(gadget, 'log'):
      o.publish_status()
    self.assertEqual([n for n in os.listdir(gadget.STATUS.parent) if n.startswith('.')], [])

  def test_a_start_clears_what_a_killed_writer_left(self):
    gadget.STATUS.parent.mkdir(parents=True, exist_ok=True)
    (gadget.STATUS.parent / '.status.json.4242').write_text('{"half')
    (gadget.STATUS.parent / '.server.json.4242').write_text('{')
    # a live modeld's runtime record, mid-write, is not a leftover
    (gadget.STATUS.parent / f'.modeld.json.{os.getpid()}').write_text('{')
    o = self.make()
    self.addCleanup(o.cable.close)
    with mock.patch.object(gadget, 'pid_alive', side_effect=lambda pid: pid == os.getpid()):
      o.adopt()
    self.assertEqual([n for n in os.listdir(gadget.STATUS.parent) if n.startswith('.')], [f'.modeld.json.{os.getpid()}'])


class TestACrashLoop(OwnerTest):
  """manager starts a dead owner again, and every start re-enumerates the
  Jetson. Starts are written down and a clean stop takes its own back, so
  only owners that died count, and past CRASH_FREE of them a start waits."""

  def test_only_owners_that_died_lately_count(self):
    t = 1000.0
    for n in range(owner.CRASH_FREE):
      self.assertEqual(owner.note_start(t + n), (0.0, n))
    self.assertEqual(owner.note_start(t + 3), (owner.CRASH_BACKOFF, 3))
    self.assertEqual(owner.note_start(t + 4), (2 * owner.CRASH_BACKOFF, 4))
    for n in range(5, 15):
      wait, _ = owner.note_start(t + n)
    self.assertEqual(wait, owner.CRASH_BACKOFF_MAX)
    # a window later, none of them count
    self.assertEqual(owner.note_start(t + 15 + owner.CRASH_WINDOW), (0.0, 0))

  def run_two_steps(self) -> None:
    """An owner that steps twice and is stopped, as manager stops it."""
    o = self.owner()
    steps = []

    def step():
      steps.append(1)
      o.stop = len(steps) == 2
    with mock.patch.object(owner, 'POLL', 0.0), mock.patch.object(o, 'step', side_effect=step):
      o.run()
    self.assertEqual(len(steps), 2)

  def test_owners_that_stop_cleanly_never_wait(self):
    # turning the link off and on again any number of times never counts
    for _ in range(owner.CRASH_FREE + 3):
      self.run_two_steps()
      self.assertEqual(owner._starts(), [], 'a clean stop was counted as a death')
    self.assertEqual(owner.note_start(time.monotonic())[0], 0.0)

  def test_a_record_that_cannot_be_read_counts_nothing(self):
    gadget.STARTS.parent.mkdir(parents=True, exist_ok=True)
    for text in ('', 'garbage', '{"a": 1}', '["x", null]'):
      gadget.STARTS.write_text(text)
      self.assertEqual(owner.note_start(1000.0), (0.0, 0), text)
      gadget.STARTS.unlink()

  def test_the_wait_holds_nothing_and_says_why(self):
    now = time.monotonic()
    gadget.write_record(gadget.STARTS, [now - 30.0, now - 20.0, now - 10.0])   # three owners that died
    o = self.owner(presented=False)
    seen = []

    def sleep(seconds):
      seen.append(gadget.owner_status())
      o.stop = True   # manager stops it mid-wait
    with mock.patch.object(owner.time, 'sleep', side_effect=sleep), mock.patch.object(o, 'step') as step:
      o.run()
    step.assert_not_called()
    o.open_link.assert_not_called()
    o.port.update.assert_not_called()
    self.assertTrue(gadget.owner_alive(seen[0]))
    self.assertIn('keeps stopping (3 times in 10 min), waiting 10 s before starting it again', seen[0]['error'])
    # manager's stop is a clean one: it does not count against the next start
    self.assertEqual(len(owner._starts()), 3)

  def test_after_the_wait_it_starts_as_any_owner_does(self):
    now = time.monotonic()
    gadget.write_record(gadget.STARTS, [now - 30.0, now - 20.0, now - 10.0])
    o = self.owner()
    with mock.patch.object(owner, 'CRASH_BACKOFF', 0.05), mock.patch.object(owner, 'POLL', 0.01), \
         mock.patch.object(o, 'step', side_effect=lambda: setattr(o, 'stop', True)) as step:
      started = time.monotonic()
      o.run()
    step.assert_called_once()
    self.assertGreaterEqual(time.monotonic() - started, 0.05)
    self.assertIsNone(o.backing_off)

  def test_a_power_off_request_ends_the_wait(self):
    # hardwared waits 25 s for the owner; a wait of up to 5 min would leave
    # the Jetson on its own supply running
    now = time.monotonic()
    gadget.write_record(gadget.STARTS, [now - 30.0, now - 20.0, now - 10.0])
    gadget.SHUTDOWN_REQUEST.write_text(json.dumps({'reason': 'car battery'}))
    o = self.owner()
    with mock.patch.object(owner, 'CRASH_BACKOFF', 60.0), \
         mock.patch.object(o, 'step', side_effect=lambda: setattr(o, 'stop', True)) as step:
      started = time.monotonic()
      o.run()
    self.assertLess(time.monotonic() - started, 1.0)
    step.assert_called_once()

  def test_an_owner_that_dies_is_counted(self):
    o = self.owner()
    with mock.patch.object(o, 'step', side_effect=SystemExit('nothing catches this')), \
         mock.patch.object(o, 'forget_status', side_effect=SystemExit('killed in the teardown')), \
         self.assertRaises(SystemExit):
      o.run()
    self.assertEqual(len(owner._starts()), 1)

  def test_a_stop_asked_for_is_clean_however_long_its_teardown(self):
    # link Off with a run in a hello: stop_worker's grace is 10 s, and manager
    # SIGKILLs 5 s after its SIGINT, before any finally
    o = self.owner()
    o.born = time.monotonic()
    owner.note_start(o.born)
    seen = []

    def wait(timeout):
      seen.append(list(owner._starts()))
      if len(seen) < 3:
        raise subprocess.TimeoutExpired('run', timeout)
      return 0
    o.worker = mock.Mock(**{'wait.side_effect': wait})
    o.request_stop()
    o.published = 0.0
    with mock.patch.object(owner, 'POLL', 0.0):
      o.stop_worker()
    self.assertEqual(seen[1], [], 'still counted while the teardown waited: a SIGKILL there was a death')
    self.assertIs(gadget.owner_status()['stopping'], True)

  def test_a_stopping_owner_killed_in_its_teardown_leaves_no_alert(self):
    from jetlink.openpilot import status
    o = self.owner()
    o.request_stop()
    o.publish_status()
    record = gadget.owner_status()
    self.assertIs(record['stopping'], True)
    with mock.patch.object(gadget.time, 'monotonic', return_value=record['at'] + 60.0):
      self.assertEqual(status.owner_record(), (None, None))
    record['stopping'] = False
    gadget.write_record(gadget.STATUS, record)
    with mock.patch.object(gadget.time, 'monotonic', return_value=record['at'] + 60.0):
      self.assertEqual(status.owner_record(), (record, None))


class TestStartingAgain(OwnerTest):
  """An owner started after one that died finds its gadget and its records."""

  def test_its_records_are_cleared_and_its_gadget_used_as_it_is(self):
    gadget.note_link('cable', '192.168.60.3')
    gadget.DORMANT.write_text('999999')
    gadget.note_lender_error('address in use')
    o = self.owner()
    with mock.patch.object(gadget, 'setup_gadget') as setup:
      o.adopt()
      self.assertFalse(gadget.DORMANT.exists())
      self.assertIsNone(gadget.gadget_error())
      self.assertIsNone(gadget.link_peer())
      o.step()
    setup.assert_not_called()   # the configfs gadget is there, so nothing is rebuilt
    self.assertEqual(gadget.link_kind(), 'usb')

  def test_a_shutdown_request_waiting_for_it_is_kept(self):
    # hardwared's, not the dead owner's: the new owner still takes it
    gadget.SHUTDOWN_REQUEST.write_text(json.dumps({'reason': 'car battery'}))
    o = self.owner()
    o.adopt()
    o.step()
    o.spawn_worker.assert_called_once()
    self.assertIn('shut down', o.spawn_worker.call_args.args[0])


class TestOneOwnerAtATime(OwnerTest):
  """A manager SIGKILLed without its cleanup starts jetlinkd again while the
  orphaned owner still holds the gadget. The second one must leave it alone."""

  def test_a_second_owner_touches_nothing_and_exits(self):
    live = self.owner()
    self.assertTrue(live.take_lock())
    self.addCleanup(lambda: live.lock_fd is not None and os.close(live.lock_fd))
    self.assertFalse(os.get_inheritable(live.lock_fd), 'a provisioning run would inherit the lock')
    # the live owner's records
    gadget.note_link('cable', '192.168.60.3')
    gadget.DORMANT.write_text(str(os.getpid()))
    gadget.note_lender_error('address in use')
    second = self.owner()
    with mock.patch.object(second, 'step') as step, mock.patch.object(gadget, 'log') as log:
      second.run()
    step.assert_not_called()
    self.assertIn('another owner (pid %s) holds the gadget', log.error.call_args.args[0])
    self.assertEqual(log.error.call_args.args[1], str(os.getpid()))
    self.assertEqual(gadget.link_peer(), '192.168.60.3')
    self.assertTrue(gadget.DORMANT.exists())
    self.assertEqual(gadget.gadget_error(), 'the lender could not listen: address in use')
    self.assertFalse(gadget.STATUS.exists())
    self.assertEqual(owner._starts(), [])
    second.port.off.assert_not_called()
    second.lender.stop.assert_not_called()

  def test_the_lock_goes_with_the_owner(self):
    first = self.owner()
    self.stopped(first)
    self.assertIsNone(first.lock_fd)
    second = self.owner()
    second.stop = True
    with mock.patch.object(second, 'hold_the_gadget') as held:
      second.run()
    held.assert_called_once()

  def test_without_a_place_to_lock_it_runs_as_before(self):
    o = self.owner()
    o.stop = True
    with mock.patch.object(gadget, 'OWNER_LOCK', Path('/nonexistent-root/jetlink/owner.lock')), \
         mock.patch.object(o, 'hold_the_gadget') as held, mock.patch.object(gadget, 'log'):
      o.run()
    held.assert_called_once()


class TestABusyEp0(OwnerTest):
  """An owner started while a borrower from before it still has ep1 and ep2:
  modeld mid-drive, or the dead owner's own run. FunctionFS refuses ep0 until
  they are closed, and the borrower's link has to carry on meanwhile."""

  def setUp(self):
    super().setUp()
    self.configfs = self.tmp / 'configfs'
    self.configfs.mkdir()
    (self.configfs / 'UDC').write_text('a600000.dwc3\n')   # bound, by the owner that died
    for name, value in (('GADGET_PATH', self.configfs), ('FFS_MOUNT', self.tmp / 'ffs')):
      p = mock.patch.object(gadget, name, value)
      self.addCleanup(p.stop)
      p.start()

  def fresh(self):
    o = self.make()
    self.addCleanup(o.cable.close)
    o.lender = mock.Mock(lent=False, listening=True)
    return o

  def refused(self):
    """os.open as FunctionFS answers while somebody else has an endpoint file open."""
    real = os.open

    def refuse(path, *args, **kwargs):
      if os.path.basename(os.fsdecode(path)) == 'ep0':
        raise OSError(errno.EBUSY, os.strerror(errno.EBUSY), os.fsdecode(path))
      return real(path, *args, **kwargs)
    return mock.patch('jetlink.transport.ffs.os.open', side_effect=refuse)

  def test_the_borrowers_gadget_is_left_bound(self):
    o = self.fresh()
    with self.refused(), mock.patch.object(gadget, 'log') as log:
      self.assertFalse(o.open_link())
      self.assertFalse(o.open_link())
    self.assertEqual((self.configfs / 'UDC').read_text(), 'a600000.dwc3\n', 'the refused open unbound the gadget')
    self.assertIsNone(o.transport)
    log.warning.assert_called_once()   # said once, and no traceback
    log.exception.assert_not_called()
    self.assertAlmostEqual(o.next_attempt - time.monotonic(), owner.EP0_BUSY_RETRY, delta=0.25)

  def test_it_is_presented_once_the_borrower_lets_go(self):
    o = self.fresh()
    with self.refused():
      o.open_link()
    self.assertTrue(o.ep0_busy)
    with mock.patch('jetlink.transport.ffs.FfsTransport') as made:
      self.assertTrue(o.open_link())
    made.assert_called_once()
    self.assertFalse(o.ep0_busy)

  def test_onroad_it_is_asked_once_a_second_not_every_step(self):
    self.write('IsOffroad', b'0')
    o = self.fresh()
    busy = OSError(errno.EBUSY, 'Device or resource busy', str(self.tmp / 'ffs' / 'ep0'))
    with mock.patch('jetlink.transport.ffs.FfsTransport', side_effect=busy) as made:
      for _ in range(3):
        o.step()
      made.assert_called_once()
      o.next_attempt = 0.0
      o.step()
      self.assertEqual(made.call_count, 2)

  def test_a_busy_udc_is_not_a_busy_ep0(self):
    o = self.fresh()
    busy = OSError(errno.EBUSY, 'Device or resource busy', str(self.configfs / 'UDC'))
    with mock.patch('jetlink.transport.ffs.FfsTransport', side_effect=busy), mock.patch.object(gadget, 'log') as log:
      self.assertFalse(o.open_link())
    log.exception.assert_called_once()
    self.assertFalse(o.ep0_busy)
    self.assertAlmostEqual(o.next_attempt - time.monotonic(), owner.RECONNECT_BACKOFF, delta=0.25)


class TestSetup(OwnerTest):
  def test_the_owner_creates_the_gadget_when_there_is_none(self):
    # nothing sets it up at boot
    o = self.owner(presented=False)
    with mock.patch.object(gadget, 'link_configured', return_value=False), \
         mock.patch.object(gadget, 'setup_gadget', return_value=True) as setup:
      self.assertTrue(o.ensure_gadget(False))
      setup.assert_called_once()

  def test_a_failed_setup_is_not_retried_every_cycle(self):
    # off AGNOS too, where root.run is a False for everything
    o = self.owner(presented=False)
    with mock.patch.object(gadget, 'link_configured', return_value=False), \
         mock.patch.object(gadget, 'setup_gadget', return_value=False) as setup:
      for _ in range(3):
        self.assertFalse(o.ensure_gadget(False))
      setup.assert_called_once()


class TestTheWorker(OwnerTest):
  """The caller names the provisioning run; the owner knows no openpilot module."""

  def test_the_run_is_the_callers_argv_cwd_and_env(self):
    o = self.make(['python3', '-m', 'the.worker'], cwd='/data/openpilot', env={'PYTHONPATH': '/data/openpilot'})
    with mock.patch.object(owner.subprocess, 'Popen') as popen:
      o.spawn_worker('nothing has been checked since boot')
    (argv,), kwargs = popen.call_args
    self.assertEqual(argv, ['python3', '-m', 'the.worker'])
    self.assertEqual(kwargs['cwd'], '/data/openpilot')
    self.assertEqual(kwargs['env'], {**os.environ, 'PYTHONPATH': '/data/openpilot'})
    self.assertIs(o.worker, popen.return_value)

  def test_a_run_that_will_not_start_is_not_an_error(self):
    o = self.make()
    with mock.patch.object(owner.subprocess, 'Popen', side_effect=OSError('no such file')):
      o.spawn_worker('nothing has been checked since boot')
    self.assertIsNone(o.worker)



class TestTheOpenpilotEntry(OwnerTest):
  """jetlink.openpilot.owner.main: the owner as the fork's adapter starts it."""

  def config(self):
    from tests.openpilot.fakes import FakeOpenpilot
    op = FakeOpenpilot(self.tmp / 'op')
    return op.owner_config()

  def test_it_runs_an_owner_over_the_whole_config(self):
    config = self.config()
    with mock.patch.object(owner, 'Owner') as made, mock.patch.object(openpilot_owner.signal, 'signal') as handle, \
         mock.patch.object(gadget, 'set_logger') as set_logger:
      openpilot_owner.main(config)
    (worker,), kwargs = made.call_args
    # the run is jetlink's to name, over the adapter the fork names
    self.assertEqual(worker, [sys.executable, '-m', 'jetlink.openpilot.provision', '--adapter', 'tests.openpilot.fakes'])
    self.assertEqual((kwargs['cwd'], kwargs['env']), (str(config.cwd), dict(config.env)))
    self.assertEqual(kwargs['chestnut_ids'], config.chestnut_ids)
    made.return_value.run.assert_called_once()
    self.assertEqual(handle.call_args_list, [mock.call(openpilot_owner.signal.SIGTERM, made.return_value.request_stop),
                                             mock.call(openpilot_owner.signal.SIGINT, made.return_value.request_stop)])
    # its settings are read off the directory the config names
    s = kwargs['settings']
    self.assertEqual((s.mode(), s.offroad()), ('off', True))
    (config.params_dir / 'JetlinkLink').write_bytes(b'2')
    (config.params_dir / 'IsOffroad').write_bytes(b'0')
    self.assertEqual((s.mode(), s.offroad()), ('ios', False))
    # and it logs to the file the config names
    logger = set_logger.call_args.args[0]
    self.addCleanup(lambda: [logger.removeHandler(h) or h.close() for h in list(logger.handlers)])
    logger.warning('jetlink: a line for the file')
    self.assertIn('a line for the file', config.log_file.read_text())
    self.assertIsInstance(logger, logging.Logger)

  def test_the_port_takes_the_ids_the_owner_was_given(self):
    self.make(chestnut_ids={(1, 2)})
    owner.port.Port.assert_called_with({(1, 2)})

  def test_the_owner_knows_no_settings_or_ids_of_its_own(self):
    # the fork's adapter names both; a caller that forgets them fails at once
    # rather than running on a guess
    with self.assertRaises(TypeError):
      owner.Owner(())


class TestLending(OwnerTest):
  def test_lendable_only_once_the_endpoints_are_down(self):
    o = self.owner(lendable=False)
    self.assertFalse(o.lendable())
    o.transport.lendable = True
    self.assertTrue(o.lendable())
    o.transport = None
    self.assertFalse(o.lendable())

  def test_a_stuck_write_is_freed_by_the_owner(self):
    o = self.owner()
    o.transport.rebind.return_value = True
    self.assertTrue(o.bounce_gadget())
    o.transport.rebind.assert_called_once()

  def test_nothing_to_bounce_is_not_an_error(self):
    o = self.owner(presented=False)
    self.assertFalse(o.bounce_gadget())
    self.assertFalse(o.lendable())

  def test_a_bounce_that_raises_is_not_an_error(self):
    o = self.owner()
    o.transport.rebind.side_effect = OSError('no such device')
    self.assertFalse(o.bounce_gadget())

