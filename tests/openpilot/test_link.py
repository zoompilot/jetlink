"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Holding the link while the Jetson comes up, and making it ready for the pick.

The comma is the USB device: the link exists only while a process holds ep0
with the UDC bound, and every unbind is an unplug the far end has to recover
from. This is about not doing that and about modeld borrowing the endpoints
rather than the gadget. The one case that still needs an edge, the bounce in
wait_for_host, is jetlink.comma's and tested there.
"""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from jetlink.comma import gadget, lending
from jetlink.openpilot import link
from tests.openpilot.fakes import OpenpilotTest, RecordingLog


class FakeClock:
  """monotonic and sleep, so a 45 s wait costs no wall clock."""

  def __init__(self, now: float = 1000.0):
    self.now = now
    self.slept = 0.0

  def monotonic(self) -> float:
    return self.now

  def sleep(self, seconds: float) -> None:
    step = max(seconds, 0.01)
    self.now += step
    self.slept += step


class ClockedTest(OpenpilotTest):
  def setUp(self):
    super().setUp()
    self.clock = FakeClock()
    for module in (link, gadget):
      self.patch(module, 'time', self.clock)
    # USB unless a test says otherwise, whatever a previous owner's record holds
    self.patch(gadget, 'LINK', Path(tempfile.mkdtemp()) / 'link')

  def bus(self, udc: str, cc: bool = True):
    for name, value in (('udc_state', udc), ('port_has_host', cc)):
      self.patch(gadget, name, return_value=value)


class HoldingTheLink(ClockedTest):
  """A link the attempt could not use stays open for the next one.

  Closing it unbinds the UDC, and while a Jetson boots the join loop asks
  again every few seconds: that would be an unplug every cycle, and one of
  them can land on the enumeration.
  """

  def setUp(self):
    super().setUp()
    self.client = mock.Mock(dead=False)
    self.link = link.Link(self.op.log)
    self.link.client = self.client
    self.bus('powered', cc=False)

  def test_a_link_nobody_enumerated_is_kept(self):
    with self.assertRaises(TimeoutError):
      link.connect_patiently(self.link)
    assert self.link.client is self.client, 'unbound the gadget between attempts'
    assert self.client.close.call_count == 0
    assert self.op.log.has('gadget up, waiting for the jetson to enumerate')

  def test_endpoints_that_will_not_open_are_still_reported(self):
    # the owner lent them, but a provisioning run is still finishing an
    # exchange on them: there is no link to hold on to and the join loop
    # should hear why
    self.link.client = None
    with mock.patch.object(lending, 'borrow', return_value=mock.Mock(closed=False)), \
         mock.patch.object(link, 'connect', side_effect=OSError('ep0 busy')), \
         self.assertRaisesRegex(OSError, 'ep0 busy'):
      link.connect_patiently(self.link)
    assert self.op.log.has('link not ready (ep0 busy), retrying')

  def test_no_model_picked_yet_keeps_the_gadget_presented(self):
    # The panel says what is going on; a closed gadget would take the whole
    # link off the bus for the drive instead.
    with mock.patch.object(self.parts.models, 'selected_model', return_value=None):
      with self.assertRaises(RuntimeError):
        link.open_link(self.parts, self.link)
    assert self.link.client is self.client
    assert self.client.close.call_count == 0

  def test_closing_the_link_keeps_the_lease(self):
    # jetlinkd should hold the gadget for the whole drive, however many times
    # the join has to start over
    self.link.loan = mock.Mock(closed=False)
    self.link.close()
    assert self.link.client is None
    self.client.close.assert_called_once()
    assert self.link.loan.close.call_count == 0

  def test_a_close_that_raises_is_logged(self):
    self.client.close.side_effect = OSError('gone')
    self.link.close()
    assert self.link.client is None
    assert self.op.log.has('error closing the link', 'exception')


class BorrowingTheGadget(OpenpilotTest):
  """modeld does not bring the gadget up.

  jetlinkd holds ep0 and the bind for as long as the link is enabled, so the
  comma stays enumerated across the ignition edge; modeld asks for the endpoint
  files and gives them back by exiting. It never opens the gadget itself: only
  the owner holds ep0.
  """

  def setUp(self):
    super().setUp()
    self.link = link.Link(self.op.log)

  def test_the_lease_is_what_the_link_is_opened_over(self):
    loan = mock.Mock(closed=False)
    with mock.patch.object(lending, 'borrow', return_value=loan), \
         mock.patch.object(link, 'connect') as connect:
      self.link.open()
    assert connect.call_args.kwargs['loan'] is loan
    assert connect.call_args.kwargs['name'] == 'modeld'

  def test_the_lease_is_borrowed_once_and_reused_every_attempt(self):
    loan = mock.Mock(closed=False)
    with mock.patch.object(lending, 'borrow', return_value=loan) as borrow, \
         mock.patch.object(link, 'connect') as connect:
      self.link.open()
      self.link.client = None
      self.link.open()
    borrow.assert_called_once()
    assert all(c.kwargs['loan'] is loan for c in connect.call_args_list)

  def test_a_dead_client_is_replaced_not_reused(self):
    # a big model retired after a link loss closes its client; the next
    # attempt reused it and failed on EBADF, a whole retry after every loss
    dead = mock.Mock(dead=True)
    loan = mock.Mock(closed=False)
    self.link.client, self.link.loan = dead, loan
    with mock.patch.object(link, 'connect') as connect:
      assert self.link.open() is connect.return_value
    dead.close.assert_called_once()
    assert connect.call_args.kwargs['loan'] is loan

  def test_a_lease_that_ended_is_asked_for_again(self):
    with mock.patch.object(lending, 'borrow', return_value=mock.Mock(closed=True)) as borrow, \
         mock.patch.object(link, 'connect'):
      self.link.open()
      self.link.client = None
      self.link.open()
    assert borrow.call_count == 2

  def test_no_loan_is_no_join_and_no_gadget_of_our_own(self):
    # nobody lent the link: the small model drives and the join loop asks
    # again. Opening the endpoints here would be a second owner of ep0
    with mock.patch.object(lending, 'borrow', return_value=None), \
         mock.patch('jetlink.client.JetlinkClient') as client:
      with self.assertRaises(TimeoutError):
        self.link.open()
    assert client.method_calls == []
    assert self.link.client is None and self.link.loan is None

  def test_the_early_present_does_not_spend_its_whole_budget_asking(self):
    # PRESENT_TIMEOUT blocks modeld's main thread, and a borrow that outlasts
    # it leaves nothing to open the link with
    with mock.patch.object(lending, 'borrow', return_value=mock.Mock(closed=False)) as borrow, \
         mock.patch.object(link, 'connect') as connect:
      self.link.open(deadline=link.time.monotonic() + 1.5)
    assert borrow.call_args.kwargs['timeout'] <= 1.5
    assert connect.call_args.kwargs['wait'] <= 1.5, 'and nor does the wait for the phone'


class TestConnect(unittest.TestCase):
  """Which transport the client is opened over: whatever the owner lent, the
  endpoint files or on the cable the phone's next dial. Never the gadget
  itself. Both borrowers go through this."""

  def setUp(self):
    from jetlink.client import JetlinkClient
    self.client = mock.Mock(name='JetlinkClient')
    for name in ('open_socket', 'open_borrowed_ffs', 'open_ffs'):
      setattr(self.client, name, mock.patch.object(JetlinkClient, name).start())
    self.note_link = mock.patch.object(gadget, 'note_link').start()
    self.addCleanup(mock.patch.stopall)
    self.log = RecordingLog()

  def test_a_cable_loan_takes_the_phones_next_dial(self):
    sock = mock.Mock(name='sock')
    sock.getpeername.return_value = ('192.168.60.3', 50000)
    loan = mock.Mock(cable=True)
    loan.accept.return_value = sock
    link.connect(self.log, deadline=2.0, name='modeld', loan=loan)
    loan.accept.assert_called_once_with(lending.BORROW_TIMEOUT)
    self.client.open_socket.assert_called_once_with(sock, deadline=2.0, name='modeld')
    self.client.open_borrowed_ffs.assert_not_called()
    self.note_link.assert_called_once_with('cable', '192.168.60.3')
    assert self.log.has("the phone dialed in from 192.168.60.3")

  def test_the_wait_for_the_phone_is_the_callers(self):
    loan = mock.Mock(cable=True)
    loan.accept.return_value = mock.Mock(**{'getpeername.return_value': ('192.168.60.3', 1)})
    link.connect(self.log, loan=loan, wait=1.5)
    loan.accept.assert_called_once_with(1.5)

  def test_a_loan_of_the_endpoints_is_opened_over_them(self):
    # whatever the link record says: the owner decided once, when it lent, and
    # the record is for the panels
    loan = mock.Mock(cable=False, mount='/dev/ffs-jetlink', udc='udc0')
    with mock.patch.object(gadget, 'link_kind', return_value='cable') as kind:
      link.connect(self.log, name='modeld', loan=loan)
    self.client.open_borrowed_ffs.assert_called_once()
    assert self.client.open_borrowed_ffs.call_args.args[:2] == ('/dev/ffs-jetlink', 'udc0')
    assert self.client.open_borrowed_ffs.call_args.kwargs['bounce'] is loan.bounce
    self.client.open_socket.assert_not_called()
    self.client.open_ffs.assert_not_called()
    kind.assert_not_called()

  def test_the_deadline_is_a_frames_unless_it_is_given(self):
    from jetlink.client import FRAME_TIMEOUT
    link.connect(self.log, loan=mock.Mock(cable=False))
    assert self.client.open_borrowed_ffs.call_args.kwargs['deadline'] == FRAME_TIMEOUT


class PresentingEarly(OpenpilotTest):
  """The load takes the link from a helper thread and waits PRESENT_TIMEOUT
  for it at most; a helper that finishes later closes what it opened."""

  def setUp(self):
    super().setUp()
    self.patch(lending, 'borrow', mock.Mock(return_value=mock.Mock(closed=False)))
    self.patch(link, 'PRESENT_TIMEOUT', 0.05)
    self.link = link.Link(self.op.log)
    self.client = mock.Mock(dead=False)
    self.background = mock.Mock()

  def test_a_link_that_opens_in_time_is_kept(self):
    with mock.patch.object(link, 'connect', return_value=self.client):
      link.present_early(self.link, self.background)
    self.assertIs(self.link.client, self.client)
    self.client.close.assert_not_called()
    # off modeld's realtime core before the open makes a reader thread
    self.background.assert_called_once_with()

  def test_a_link_that_opens_late_is_closed(self):
    release = threading.Event()
    self.addCleanup(release.set)

    def slow(*args, **kwargs):
      release.wait(5)
      return self.client

    with mock.patch.object(link, 'connect', side_effect=slow):
      t0 = time.monotonic()
      link.present_early(self.link, self.background)
      self.assertLess(time.monotonic() - t0, 2.0, 'the load waited on a helper past its budget')
      release.set()
      for _ in range(200):
        if self.client.close.called:
          break
        time.sleep(0.01)
    self.client.close.assert_called_once()
    assert self.op.log.has('presenting the gadget took over')

  def test_a_link_that_never_opens_is_left_to_the_join(self):
    with mock.patch.object(link, 'connect', side_effect=OSError('busy')):
      link.present_early(self.link, self.background)
    self.assertIsNone(self.link.client)
    assert self.op.log.has('could not present the gadget early (busy), the join will')


class BuildingOnroad(OpenpilotTest):
  """The picked model is built with the small model driving.

  a provisioning run works offroad only, so without this a model picked in
  the driveway and driven off on would cost the whole drive.
  """

  ENTRY = {'name': 'CTM v2', 'ref': 'f' * 40, 'oid': 'a' * 64, 'size': 766 << 20}

  def setUp(self):
    super().setUp()
    self.client = mock.Mock()
    self.client.hello.return_value = {'device': 'orin', 'trt_version': '10.3', 'engine_state': 'ready', 'loaded': 'a' * 64}
    self.spec = mock.Mock(sha256=self.ENTRY['oid'])
    self.link = link.Link(self.op.log)
    self.patch(link, 'connect_patiently', return_value=self.client)
    for target, name, value in ((self.parts.models, 'selected_model', dict(self.ENTRY)),
                                (self.parts.models, 'shipped_model_path', None),
                                (self.parts.spec, 'ready_spec', None)):
      self.patch(target, name, return_value=value)
    self.ensure = self.patch(link, 'ensure', return_value=self.spec)

  def test_the_model_the_picker_names_is_what_gets_built(self):
    client, spec = link.open_link(self.parts, self.link)
    assert spec is self.spec and client is self.client
    assert self.ensure.call_args.args[2:4] == (self.ENTRY['oid'], self.ENTRY['size'])
    assert self.op.log.has('CTM v2 is not built yet, building it with the small model driving')
    assert self.op.log.has(f"orin trt 10.3, engine ready, loaded {'a' * 16}")

  def test_the_hello_carries_our_pick(self):
    self.patch(self.parts.models, 'hello_pick', return_value={'ref': 'f' * 40, 'age_s': 5.0})
    link.open_link(self.parts, self.link)
    self.client.hello.assert_called_once_with(timeout=10.0, pick={'ref': 'f' * 40, 'age_s': 5.0})

  def test_the_hello_carries_the_cameras_calibration_where_the_adapter_knows_it(self):
    self.patch(self.parts.models, 'hello_pick', return_value=None)
    calibration = {'wide_from_device_euler': [0.0, 0.0217, -0.0161], 'height': 1.374}
    self.parts.op.camera_calibration = lambda: calibration
    link.open_link(self.parts, self.link)
    self.client.hello.assert_called_once_with(timeout=10.0, pick=None, camera=calibration)

  def test_a_newer_pick_on_the_server_is_what_drives_and_is_ours_now(self):
    # picked on the Jetson's web page while parked: it drives at the next join
    theirs = {'name': 'Cinque Terre V3', 'ref': 'e' * 40, 'oid': 'e' * 64, 'size': 99}
    self.parts.models.selected_model.side_effect = [dict(self.ENTRY), theirs]
    adopt = self.patch(self.parts.models, 'adopt')
    link.open_link(self.parts, self.link)
    adopt.assert_called_once_with()
    self.assertEqual(self.ensure.call_args.args[2:4], ('e' * 64, 99))
    self.assertEqual(self.ensure.call_args.kwargs['ref'], 'e' * 40)

  def test_only_the_jetson_is_asked_to_download_the_pick(self):
    # the pick, when neither end has it, the Jetson fetches itself; a
    # stand-in is a model it has
    link.open_link(self.parts, self.link)
    self.assertTrue(self.ensure.call_args.kwargs['fetch'])
    self.patch(self.parts.spec, 'ready_spec', return_value=mock.Mock(sha256='b' * 64, nbytes=123))
    self.patch(self.parts.models, 'name_for', return_value='Cinque Terre V3')
    link.open_link(self.parts, self.link)
    self.assertFalse(self.ensure.call_args.kwargs['fetch'])

  def test_the_last_model_built_drives_while_the_pick_is_not_ready(self):
    # by design, not the small model. No upload and no
    # build here: either would cost the model that is steering
    standin = mock.Mock(sha256='b' * 64, nbytes=123)
    self.patch(self.parts.spec, 'ready_spec', return_value=standin)
    self.patch(self.parts.models, 'name_for', return_value='Cinque Terre V3')
    self.patch(self.parts.models, 'shipped_model_path', return_value='/data/model.onnx')
    link.open_link(self.parts, self.link)
    self.assertEqual(self.ensure.call_args.args[2:5], ('b' * 64, 123, None))
    assert self.op.log.has('CTM v2 is not ready yet, Cinque Terre V3 drives until it is')

  def test_a_record_of_the_pick_itself_is_no_stand_in(self):
    self.patch(self.parts.spec, 'ready_spec', return_value=mock.Mock(sha256=self.ENTRY['oid'], nbytes=1))
    link.open_link(self.parts, self.link)
    self.assertEqual(self.ensure.call_args.args[2:4], (self.ENTRY['oid'], self.ENTRY['size']))

  RECORD = mock.Mock(sha256='r' * 64, nbytes=7)

  def hello(self, loaded=None, cached=(), record=RECORD):
    """A server that lists what it has, and the comma's record of the last model built."""
    self.client.hello.return_value = {'device': 'iPhone', 'engine_state': 'none', 'loaded': loaded,
                                      'cached_models': list(cached)}
    self.parts.spec.ready_spec.return_value = record

  def test_the_model_the_server_has_loaded_drives_over_the_record(self):
    # the comma picked one model, the server has another loaded, and the
    # comma's record names a third from another server: the server must not
    # be made to switch to the record's
    size_for = self.patch(self.parts.models, 'size_for', return_value=99)
    self.patch(self.parts.models, 'name_for', return_value='Cinque Terre V3')
    self.hello(loaded='c' * 64, cached=['c' * 64, 'r' * 64])
    link.open_link(self.parts, self.link)
    self.assertEqual(self.ensure.call_args.args[2:5], ('c' * 64, 99, None))
    size_for.assert_called_once_with('c' * 64)
    assert self.op.log.has('CTM v2 is not ready yet, Cinque Terre V3 drives until it is')

  def test_the_pick_wins_whenever_the_server_has_it(self):
    for loaded, cached in ((self.ENTRY['oid'], ['r' * 64]), ('c' * 64, [self.ENTRY['oid']])):
      with self.subTest(loaded=loaded):
        self.hello(loaded=loaded, cached=cached)
        link.open_link(self.parts, self.link)
        self.assertEqual(self.ensure.call_args.args[2:4], (self.ENTRY['oid'], self.ENTRY['size']))

  def test_the_record_stands_in_only_where_the_server_has_it(self):
    self.hello(cached=['r' * 64])
    link.open_link(self.parts, self.link)
    self.assertEqual(self.ensure.call_args.args[2:5], ('r' * 64, 7, None))
    # a server without it would answer need_upload and cost the record
    self.hello(cached=['c' * 64])
    link.open_link(self.parts, self.link)
    self.assertEqual(self.ensure.call_args.args[2:4], (self.ENTRY['oid'], self.ENTRY['size']))

  def test_a_loaded_model_of_unknown_size_is_passed_over(self):
    # its size names it to the server; without one it would be need_upload
    self.patch(self.parts.models, 'size_for', return_value=None)
    self.hello(loaded='c' * 64, cached=['c' * 64], record=None)
    link.open_link(self.parts, self.link)
    self.assertEqual(self.ensure.call_args.args[2:4], (self.ENTRY['oid'], self.ENTRY['size']))

  def test_the_frame_deadline_is_set_before_the_link_is_handed_over(self):
    # ensure_engine waits minutes; the frame path must not inherit that
    client, _ = link.open_link(self.parts, self.link)
    assert client.deadline == link.INFERENCE_TIMEOUT

  def test_the_join_thread_can_be_stopped_through_the_build(self):
    stop = object()
    link.open_link(self.parts, self.link, should_stop=stop)
    assert self.ensure.call_args.kwargs['should_stop'] is stop

  def test_progress_reaches_the_panel(self):
    link.open_link(self.parts, self.link)
    assert self.ensure.call_args.kwargs['progress'] == self.parts.progress.report_with_eta

  def test_every_join_passes_the_hello_on_to_the_owner(self):
    # so the owner's idea of whether the far end sleeps is refreshed every
    # drive, and not only by a provisioning run that had work
    self.link.loan = mock.Mock(closed=False)
    link.open_link(self.parts, self.link)
    self.link.loan.note_server.assert_called_once_with(self.client.hello.return_value)

  def test_a_note_that_lost_the_lease_starts_the_attempt_over(self):
    # its bounce would go over a closed loan, and a stuck write could not be freed
    from jetlink.transport.base import LinkError
    self.link.loan = mock.Mock(closed=True, **{'note_server.return_value': False})
    self.link.client = self.client
    with self.assertRaises(LinkError):
      link.open_link(self.parts, self.link)
    self.client.close.assert_called_once()
    self.ensure.assert_not_called()

  def test_an_owner_too_old_for_the_note_is_no_reason_to_stop(self):
    self.link.loan = mock.Mock(closed=False, **{'note_server.return_value': False})
    client, spec = link.open_link(self.parts, self.link)
    assert spec is self.spec

  def test_without_a_lease_there_is_nobody_to_tell(self):
    self.link.loan = None
    link.open_link(self.parts, self.link)   # and nothing raised

  def test_bytes_neither_end_has_are_a_parked_job(self):
    # A Jetson that could not download the pick itself leaves it to the
    # comma, and a gigabyte on the comma is not something to start mid-drive.
    # The link is fine: closing it would cost a gadget bounce per retry for
    # the whole drive
    from jetlink.client import EngineMissing
    self.ensure.side_effect = EngineMissing('no engine')
    self.link.client = self.client
    with mock.patch.object(self.parts.spec, 'clear_ready') as cleared:
      with self.assertRaises(link.ModelMissing):
        link.open_link(self.parts, self.link)
    cleared.assert_called_once_with(self.ensure.call_args.args[2])
    self.client.close.assert_not_called()
    self.assertIs(self.link.client, self.client)


class FrameDeadline(unittest.TestCase):
  def test_a_silent_host_costs_one_short_frame(self):
    # a hung server or a phone's cable leaves no USB edge, so only the deadline
    # ends the frame. The fallback's small-model frame runs behind it, and both
    # have to fit inside modelV2's alive limit (10 frames at 20 Hz), or the
    # fallback also flashes commIssue
    import numpy as np

    from jetlink.client import JetlinkClient
    from jetlink.transport.base import LinkError
    from tests.test_protocol import _spec, make_pair
    self.assertLess(link.INFERENCE_TIMEOUT + 0.1, 10 / 20)
    a, b = make_pair()
    client = JetlinkClient(a, deadline=link.INFERENCE_TIMEOUT)
    client.spec = spec = _spec()
    try:
      t0 = time.monotonic()
      with self.assertRaises(LinkError):   # nothing is serving b
        client.infer(np.zeros(spec.warped_shape, np.uint8), np.zeros(spec.packed_nelem, np.float32))
      took = time.monotonic() - t0
    finally:
      client.close()
      b.close()
    self.assertGreaterEqual(took, link.INFERENCE_TIMEOUT * 0.9)
    self.assertLess(took, link.INFERENCE_TIMEOUT + 0.1)


if __name__ == '__main__':
  unittest.main()
