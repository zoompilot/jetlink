"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The rules the late join has to keep.

Both model states are fakes; the joining state is plumbing around them. What
is pinned: modeld gets a working model immediately, a swap never lands on an
engaged frame, a large model that dies mid-drive falls back without losing the
frame, and a small-model failure still belongs to modeld.
"""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from jetlink.openpilot import joining
from jetlink.openpilot.joining import REJOIN_DELAY_QUICK, STABLE_SECONDS, JoiningModelState
from tests.openpilot.fakes import RecordingLog, isolate


class FakeModel:
  def __init__(self, name, chestnut=False, client=None):
    self.name = name
    self.chestnut = chestnut
    self.client = client
    self.lat_delay = 0.0
    self.vision_input_names = ['img', 'big_img']
    self.calls = 0
    self.raises = None
    self.closed = False
    self.warmed = False
    # the large model's: why it should hand back after its last frame
    # (model_state's hold rules and the proof after a swap)
    self.behind = None

  def run(self, bufs, transforms, inputs, after_enqueue=None):
    self.calls += 1
    if self.raises is not None:
      raise self.raises
    if after_enqueue is not None:
      after_enqueue()
    return {'from': self.name}

  def warmup(self):
    self.warmed = True

  def close(self):
    self.closed = True


def wait_for(predicate, timeout=2.0) -> bool:
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    if predicate():
      return True
    time.sleep(0.005)
  return False


class JoiningBase(unittest.TestCase):
  """The joining state over fake models, and the helpers to drive it. No tests
  here, so each face below runs only its own."""

  def setUp(self):
    # a lost link reads the USB-C port's CC pin
    isolate(self, Path(tempfile.mkdtemp()))
    # the small model's warm-up frames at start have their own tests (WarmupTest);
    # everywhere else a join may swap on the first frame
    patcher = mock.patch.object(joining, 'SMALL_WARMUP_FRAMES', 0)
    patcher.start()
    self.addCleanup(patcher.stop)

    # The join reports what it waits on as progress; what it says is asserted here.
    self.progress = mock.Mock()
    self.log = RecordingLog()

    self.small = FakeModel('small')
    self.big = FakeModel('big', chestnut=True, client=object())
    self.joined = threading.Event()
    self.connect_calls = 0
    self.connect_error = None

  def _connect(self, should_stop=None):
    self.connect_calls += 1
    if self.connect_error is not None:
      raise self.connect_error
    self.joined.set()
    # A link the joining state may have to close on its own, when it is torn
    # down holding a join that never found a disengaged frame to land on.
    return (mock.MagicMock(name='client'), 'spec')

  def _build(self, client, spec):
    return self.big

  def _make(self, *args, **kwargs):
    return JoiningModelState(*args, progress=self.progress, log=self.log, **kwargs)

  def _state(self):
    s = self._make(self.small, self._connect, self._build)
    self.addCleanup(self._close, s)
    return s

  @staticmethod
  def _close(s):
    # the join thread is joined while setUp's patches are still on
    s.close()
    s._thread.join(5)

  def _wait_joined(self, s, timeout=5.0):
    # connect() returning is not publication: wait for the owner to hand off,
    # or for a frame to have taken the join up and built the large model
    deadline = time.monotonic() + timeout
    while s._joined is None and s._big is None and time.monotonic() < deadline:
      time.sleep(0.001)
    self.assertTrue(s._joined is not None or s._big is not None, 'never joined')

  def _run(self, s):
    return s.run({}, {}, {})

  def _driving(self, dead: bool):
    """A state with the large model driving, over a client that is dead or live."""
    s = self._state()
    self._wait_joined(s)
    s.in_control = False
    self.assertEqual(self._run(s), {'from': 'big'})
    self.big.client = mock.Mock(dead=dead)
    return s

  def _ready(self):
    """A state with a link ready and waiting for a window; engaged, so the
    small model drives on."""
    s = self._state()
    self._wait_joined(s)
    self.assertEqual(self._run(s), {'from': 'small'})
    return s

  def _said(self):
    """The panel's lines so far, in order."""
    return [c.args[2] for c in self.progress.report.call_args_list if c.args[0] == 'connect']

  def _wait_reported(self, s, msg, drops=0, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
      if any(c.args == ('connect', 0.0, msg) and c.kwargs.get('drops', 0) == drops
             for c in self.progress.report.call_args_list):
        return
      time.sleep(0.005)
    self.fail(f"never reported {msg!r}: {self.progress.report.call_args_list}")


class JoiningTest(JoiningBase):
  def test_only_the_join_thread_runs_and_it_leaves_modelds_realtime_core_first(self):
    # created after config_realtime_process(7, 54), it inherits SCHED_FIFO on
    # core 7 and drops it before anything else. Nothing polls for in_control:
    # the frame asks it
    events = []
    with mock.patch.object(joining, 'background_thread', lambda: events.append(threading.current_thread().name)):
      s = self._state()
      self.assertTrue(wait_for(lambda: events))
    self.assertEqual(events, [s._thread.name])

  def test_runs_the_small_model_immediately(self):
    s = self._state()
    # No waiting on a link: this is the whole point.
    self.assertEqual(self._run(s), {'from': 'small'})
    self.assertFalse(s.chestnut)
    self.assertIsNone(s.client)

  def test_does_not_swap_while_engaged(self):
    # stopped or moving: even stopped, longitudinal control can hold the brake
    # or request motion, so the window is modeld's in_control and nothing else
    s = self._state()
    self._wait_joined(s)
    s.in_control = True
    for _ in range(3):
      self.assertEqual(self._run(s), {'from': 'small'})
    self.assertFalse(s.chestnut)
    self.assertFalse(self.big.warmed)

  def test_asks_the_adapter_before_every_frame(self):
    answers = [True, True, False]
    s = self._make(self.small, self._connect, self._build, in_control=lambda: answers.pop(0))
    self.addCleanup(self._close, s)
    self._wait_joined(s)
    self.assertEqual([self._run(s) for _ in range(3)], [{'from': 'small'}, {'from': 'small'}, {'from': 'big'}])
    self.assertFalse(self.small.in_control, 'lands on the small model too')

  def test_late_boot_announces_availability_without_switching(self):
    booted = threading.Event()
    connect = self._connect

    def after_boot(should_stop=None):
      if not booted.wait(5):
        raise RuntimeError('test boot timeout')
      return connect()

    self._connect = after_boot
    s = self._state()
    self.addCleanup(booted.set)
    self.assertFalse(s.big_model_available)
    self.assertEqual(s.big_model_state, 'joining', 'nothing to swap in yet')
    self.assertEqual(self._run(s), {'from': 'small'})
    booted.set()
    self._wait_joined(s)
    self.assertTrue(s.big_model_available)
    self.assertEqual(s.big_model_state, 'ready')
    self.assertEqual(self._run(s), {'from': 'small'}, 'engaged: offered, not swapped')
    s.in_control = False
    self.assertEqual(self._run(s), {'from': 'big'})
    self.assertFalse(s.big_model_available)
    self.assertEqual(s.big_model_state, 'running')

  def test_a_model_neither_end_has_is_asked_for_again_slowly_and_said(self):
    # a hello and a gadget bounce every 7 s for a whole drive (2026-10-06):
    # only a provisioning run's download changes the answer
    from jetlink.openpilot.link import ModelMissing
    self.connect_error = ModelMissing('server has no engine for 1563b85f6bd00d9e (have 0 of the model)')
    self.connect_error.retry_after = 0.05
    with mock.patch.object(joining, 'REJOIN_DELAY', 30.0):
      s = self._state()
      deadline = time.monotonic() + 5
      while self.connect_calls < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    self.assertGreaterEqual(self.connect_calls, 2, 'waited REJOIN_DELAY, not MODEL_WAIT')
    self.progress.report.assert_any_call('connect', 0.0, 'big model not downloaded yet', drops=0)
    self.assertFalse(s.big_model_available)

  def test_close_with_pending_model_clears_availability(self):
    s = self._ready()
    self.assertTrue(s.big_model_available)
    s.close()
    self.assertFalse(s.big_model_available)

  def test_swaps_on_a_disengaged_frame(self):
    s = self._state()
    self._wait_joined(s)
    s.in_control = False
    self.assertEqual(self._run(s), {'from': 'big'})
    self.assertTrue(s.chestnut)
    self.assertIs(s.client, self.big.client)
    # No warmup at the swap: the warp was prepared in __init__ and a frame
    # over the link here was two dropped camera frames on the car.
    self.assertFalse(self.big.warmed)

  def test_large_model_failure_demotes_and_keeps_the_frame(self):
    s = self._state()
    self._wait_joined(s)
    s.in_control = False
    self._run(s)
    self.assertTrue(s.chestnut)

    self.big.raises = RuntimeError("link gone")
    # modeld still gets an output for this frame, from the small model.
    self.assertEqual(self._run(s), {'from': 'small'})
    self.assertFalse(s.chestnut)
    self.assertTrue(wait_for(lambda: self.big.closed))
    # And it tries again rather than staying small for the rest of the drive.
    self.assertTrue(s._rejoin_at > time.monotonic() or self.connect_calls > 1)

  def test_a_swap_whose_first_frame_fails_is_still_two_handovers(self):
    # swap and demote inside one run(): modelV2.big is false before and after,
    # and only the count tells modeld the frame's stall was a handover
    s = self._state()
    self._wait_joined(s)
    s.in_control = False
    self.big.raises = RuntimeError('first frame timed out')
    handovers = s.handovers
    self.assertFalse(s.chestnut)
    self.assertEqual(self._run(s), {'from': 'small'})
    self.assertFalse(s.chestnut)
    self.assertEqual(s.handovers, handovers + 2)

  def test_failed_first_inference_never_announces_ready(self):
    s = self._state()
    self._wait_joined(s)
    s.in_control = False
    self.big.raises = RuntimeError('first inference failed')
    self.assertEqual(self._run(s), {'from': 'small'})
    self.assertEqual(s.big_model_state, 'retrying')
    self.assertFalse(s.big_model_available)
    self.progress.clear.assert_not_called()

  def test_fallback_resets_history_without_waiting_for_teardown(self):
    entered, release = threading.Event(), threading.Event()
    reset = mock.Mock()

    def close():
      entered.set()
      release.wait(2)

    self.big.close = close
    s = self._make(self.small, self._connect, self._build, reset_small=reset)
    self.addCleanup(self._close, s)
    self.addCleanup(release.set)
    self._wait_joined(s)
    s.in_control = False
    self._run(s)
    self.big.raises = RuntimeError('failed')
    run = self.small.run

    def small_run(*args):
      reset.assert_called_once()
      return run(*args)

    self.small.run = small_run
    self.assertEqual(self._run(s), {'from': 'small'})
    self.assertTrue(entered.wait(1))
    self.assertFalse(release.is_set())

  def test_ready_is_announced_only_after_inference_returns(self):
    s = self._state()
    self._wait_joined(s)
    s.in_control = False
    run = self.big.run

    def inspect(*args):
      # Swapped in, but not announced: the first frame has not returned yet.
      self.assertTrue(s._loading)
      self.progress.clear.assert_not_called()
      return run(*args)

    self.big.run = inspect
    self.assertEqual(self._run(s), {'from': 'big'})
    self.assertFalse(s._loading)
    self.progress.clear.assert_called_once()

  def test_prepare_failure_uses_startup_fallback_not_a_slow_swap(self):
    order = []
    self.connect_calls = 0

    def prepare():
      order.append('prepare')
      raise RuntimeError("no warp today")

    with self.assertRaisesRegex(RuntimeError, 'no warp today'):
      self._make(self.small, self._connect, self._build, prepare)
    # Ran, and ran before anything else: modeld's main thread is blocked for
    # exactly as long as the constructor takes, so this is the only place the
    # GPU work can go without costing a frame.
    self.assertEqual(order, ['prepare'])
    self.assertEqual(self.connect_calls, 0)
    self.assertFalse(self.big.warmed)

  def test_loading_has_no_deadline(self):
    # no 60 s edge: selfdrived gates on it only while nothing publishes modelV2,
    # so a Jetson that takes a whole drive stays "getting ready"
    self.connect_error = RuntimeError("jetson still booting")
    s = self._state()
    self._run(s)
    with mock.patch.object(joining.time, 'monotonic',
                    return_value=time.monotonic() + 600.0):
      self._run(s)
    self.assertFalse(s.chestnut)
    self.assertEqual(s.big_model_state, 'joining')
    # A join that succeeds later still swaps.
    self.connect_error = None
    s._rejoin.set()
    self._wait_joined(s, 10.0)
    s.in_control = False
    for _ in range(20):
      if self._run(s) == {'from': 'big'}:
        break
      time.sleep(0.05)
    self.assertTrue(s.chestnut)

  def test_state_travels_in_the_message_not_in_params(self):
    # a chestnut's load is over once; this never is, so selfdrived's edge is
    # modelV2.big turning true and the UI reads acceleratorState
    s = self._ready()
    self.assertFalse(s.chestnut)
    # up and only a swap window away, which the icon draws steady rather than
    # pulsing "loading" for the rest of a drive with no stop in it
    self.assertEqual(s.big_model_state, 'ready')
    s.in_control = False
    self._run(s)
    self.assertTrue(s.chestnut)
    self.assertEqual(s.big_model_state, 'running')

    self.big.raises = RuntimeError("link gone")
    self._run(s)
    self.assertFalse(s.chestnut)
    self.assertEqual(s.big_model_state, 'retrying')
    # reported from the join thread, not the frame that lost the link
    self._wait_reported(s, 'reconnecting')

    s.close()
    self.assertEqual(s.big_model_state, 'unavailable')

  def test_a_lost_link_is_counted_and_repeated_drops_blame_the_cable(self):
    # every drop on the 2026-09-07 drives was the USB port letting go, and the
    # driver saw "Big Model Failed" six times with no hint of a cause
    s = self._state()
    self._wait_joined(s)
    s.in_control = False
    self._run(s)
    # held for a while: the rejoin is the quick one, so the test waits on it
    s._joined_at = time.monotonic() - (STABLE_SECONDS + 1)
    self.big.raises = RuntimeError("host dropped the gadget configuration (udc: not attached)")
    self._run(s)
    self._wait_reported(s, 'reconnecting')
    self.assertEqual(s._drops, 1)
    first = [c for c in self.progress.report.call_args_list if c.args[2] == 'reconnecting']
    self.assertEqual(first[-1].kwargs['drops'], 0)
    # a second drop in the drive names the cable, on every status from then on
    self.big.raises = None
    self._wait_joined(s)
    self._run(s)
    self.assertTrue(s.chestnut)
    s._joined_at = time.monotonic() - (STABLE_SECONDS + 1)
    self.big.raises = RuntimeError("gadget write failed: [Errno 19] No such device (udc: default)")
    self._run(s)
    self._wait_reported(s, 'reconnecting', drops=2)
    self.assertEqual(s._drops, 2)
    self._wait_reported(s, 'waiting for jetlink', drops=2)

  def test_a_lag_demote_says_why_over_the_live_link(self):
    # the phone's log carried no reason and none of the comma's numbers: a
    # session there was the big model's time plus the backoff, and nothing
    # said which. The leave carries what the comma measured
    s = self._driving(dead=False)
    self.big.trips = mock.Mock(summary=lambda: {'frames': 7, 'p50_ms': 41.5})
    s.frame_drop_ratio = joining.DROP_LIMIT * 2
    self.assertEqual(self._run(s), {'from': 'small'})
    self.assertTrue(wait_for(lambda: self.big.client.leave.called))
    self.big.client.leave.assert_called_once_with('behind', drops=0, lags=1, frames=7, p50_ms=41.5)

  def test_a_lag_demote_keeps_a_live_link_for_the_next_attempt(self):
    # the next attempt is a hello and a model request over the same
    # connection: no reconnect, and on a phone's cable no redial. The link is
    # closed only when it is dead
    s = self._driving(dead=False)
    s.frame_drop_ratio = joining.DROP_LIMIT * 2
    with mock.patch.object(joining, 'REJOIN_DELAY_QUICK', 0.05):
      self.assertEqual(self._run(s), {'from': 'small'})
      self.assertTrue(wait_for(lambda: self.connect_calls > 1))
    self.assertFalse(self.big.closed, 'a live link was closed')
    self.big.client.leave.assert_called_once()
    self.big.client.close.assert_not_called()

  def test_a_dead_link_is_closed_and_a_live_one_kept_whatever_went_wrong(self):
    # the client knows whether it is dead; the joining state only asks. A
    # failure on the comma's side (the warp, the parser) leaves the link up
    s = self._driving(dead=True)
    self.big.raises = RuntimeError('link gone')
    self._run(s)
    self.assertTrue(wait_for(lambda: self.big.closed))

    self.big = FakeModel('big', chestnut=True, client=object())
    s = self._driving(dead=False)
    self.big.raises = RuntimeError('a tensor was the wrong shape')
    with mock.patch.object(joining, 'REJOIN_DELAY_QUICK', 0.05):
      self._run(s)
      self.assertTrue(wait_for(lambda: self.big.client.leave.called))
    self.assertFalse(self.big.closed)
    self.big.client.leave.assert_called_once_with('lost', drops=1, lags=0)

  def test_closing_says_stopped_once_to_whatever_holds_the_link(self):
    # a link waiting for a window and a large model driving: each hears
    # modeld stop, once, however many times close() runs
    s = self._state()
    self._wait_joined(s)
    pending = s._joined[0]
    pending.dead = False
    s.close()
    s.close()
    pending.leave.assert_called_once_with('stopped', drops=0, lags=0)
    pending.close.assert_called()

    s = self._driving(dead=False)
    s.close()
    s.close()
    self.big.client.leave.assert_called_once_with('stopped', drops=0, lags=0)
    self.assertTrue(self.big.closed)

  def test_the_frame_that_loses_the_link_does_not_report_or_read_the_port(self):
    # the frame thread is SCHED_FIFO on modeld's core; params and sysfs are
    # for the join thread. The drop is only counted here
    s = self._state()
    self._wait_joined(s)
    s.in_control = False
    self._run(s)
    self.big.raises = RuntimeError("link gone")
    # the join thread is held asleep, so whatever reported did so on the frame
    with mock.patch.object(s._rejoin, 'set'), mock.patch.object(s, '_note_link_loss') as note:
      self.assertEqual(self._run(s), {'from': 'small'})
      self.assertIsNotNone(s._retired)
      self.assertEqual(s._drops, 1)
      note.assert_not_called()
      calls = [c for c in self.progress.report.call_args_list if c.args[2].startswith('lost')]
      self.assertEqual(calls, [])

  def test_build_failure_backs_off(self):
    self._build = mock.Mock(side_effect=RuntimeError("no warp"))
    s = self._make(self.small, self._connect, self._build)
    self.addCleanup(self._close, s)
    self._wait_joined(s)
    s.in_control = False
    self.assertEqual(self._run(s), {'from': 'small'})
    # Not straight back onto the link: the next attempt waits REJOIN_DELAY_QUICK.
    self.assertGreater(s._rejoin_at, time.monotonic() + 0.5)
    self.assertFalse(s.chestnut)
    self.assertEqual(s.big_model_state, 'retrying')

  def test_failures_back_off_and_a_stable_join_starts_over(self):
    # A demote is mostly one late frame: the first few are retried in a
    # second. A link that dies on its first frame every time used to cost a
    # swap, a demote and a chime every REJOIN_DELAY for the drive, so a streak
    # past QUICK_RETRIES doubles as before.
    s = self._state()
    s._joined_at = 0.0
    delays = []
    for _ in range(8):
      t = time.monotonic()
      s._back_off()
      delays.append(round(s._rejoin_at - t))
    self.assertEqual(delays, [1, 1, 1, 5, 10, 20, 40, 60])

    # A join that held is not that link, and must not inherit its delay: a
    # link that ran for minutes and then went is a USB drop, and the host
    # re-enumerates a rebound gadget in under a second.
    s._joined_at = time.monotonic() - (STABLE_SECONDS + 1)
    t = time.monotonic()
    s._back_off()
    self.assertEqual(round(s._rejoin_at - t), round(REJOIN_DELAY_QUICK))
    self.assertEqual(s._failures, 1)
    # and failures on its heels climb the same rungs
    for _ in range(joining.QUICK_RETRIES):
      s._back_off()
    self.assertEqual(round(s._rejoin_at - time.monotonic()), 5)

  def test_small_model_failure_is_modelds(self):
    s = self._state()
    self.small.raises = RuntimeError("vipc gone")
    with self.assertRaises(RuntimeError):
      self._run(s)

  def test_a_read_it_does_not_know_follows_the_model_that_drives(self):
    # modeld reads attributes off the model every frame; one a sync adds must
    # not be an AttributeError on the frame thread
    self.small.new_constant, self.big.new_constant = 'small', 'big'
    s = self._state()
    self.assertEqual(s.new_constant, 'small')
    self._wait_joined(s)
    s.in_control = False
    self._run(s)
    self.assertEqual(s.new_constant, 'big')
    self.big.raises = RuntimeError('link gone')
    self._run(s)
    self.assertEqual(s.new_constant, 'small')

  def test_a_name_neither_model_has_is_still_an_error(self):
    s = self._state()
    with self.assertRaises(AttributeError):
      _ = s.no_such_thing
    self.assertFalse(hasattr(s, 'no_such_thing'))

  def test_private_names_are_its_own(self):
    self.small._private = 1
    s = self._state()
    with self.assertRaises(AttributeError):
      _ = s._private

  def test_a_write_it_does_not_know_stays_on_it(self):
    # only reads are delegated: each write modeld makes has a setter that
    # reaches both models (lat_delay, PLANPLUS_CONTROL)
    s = self._state()
    s.something_new = 5
    self.assertFalse(hasattr(self.small, 'something_new'))

  def test_it_is_not_consulted_while_it_is_being_built(self):
    # a read before _active exists must not recurse
    s = JoiningModelState.__new__(JoiningModelState)
    with self.assertRaises(AttributeError):
      _ = s.anything

  def test_lat_delay_reaches_both_models(self):
    s = self._state()
    s.lat_delay = 0.25
    self.assertEqual(self.small.lat_delay, 0.25)
    self._wait_joined(s)
    s.in_control = False
    self._run(s)
    self.assertEqual(self.big.lat_delay, 0.25)


  def test_reports_joining_until_it_joins(self):
    # the UI reads this to tell "not up yet" from "failed"; modelV2.big is
    # false for the whole join
    s = self._state()
    self._run(s)
    self.assertIn(s.big_model_state, ('joining', 'ready'))

    self._wait_joined(s)
    s.in_control = False
    self._run(s)
    self.assertEqual(s.big_model_state, 'running')

    self.big.raises = RuntimeError("link gone")
    self._run(s)
    self.assertEqual(s.big_model_state, 'retrying')


class WaitingTest(JoiningBase):
  """Until the window opens the comma runs its own model and nothing else: the
  large model is built on the swap frame, and a ready link only hears a
  keepalive ping. Frames sent to the host while the small model drove (shadow
  frames) starved the driver monitoring model on a comma whose small model took
  most of the frame (2026-10-05)."""

  def test_nothing_reaches_the_large_model_until_the_swap(self):
    build = mock.Mock(wraps=self._build)
    s = self._make(self.small, self._connect, build)
    self.addCleanup(self._close, s)
    self._wait_joined(s)
    for _ in range(3):
      self.assertEqual(self._run(s), {'from': 'small'})
    build.assert_not_called()
    self.assertEqual(self.big.calls, 0)
    self.assertEqual(s._joined_at, 0.0, 'a link waiting is not a link held')
    s.in_control = False
    self.assertEqual(self._run(s), {'from': 'big'})
    build.assert_called_once()
    self.assertIsNone(s._joined)
    self.assertGreater(s._joined_at, 0.0)
    self.assertFalse(self.big.warmed)

  def test_the_panel_says_ready_once_the_link_is(self):
    s = self._ready()
    self._wait_reported(s, joining.READY)
    self.assertEqual(self._said()[-1], joining.READY)
    self.assertTrue(self.log.has('waiting for a window to swap', 'warning'))

  def test_a_ready_link_is_pinged_while_it_waits(self):
    with mock.patch.object(joining, 'KEEPALIVE_PERIOD', 0.01):
      s = self._state()
      self._wait_joined(s)
      client = s._joined[0]
      self.assertTrue(wait_for(lambda: client.ping.call_count >= 3))
    client.ping.assert_called_with(timeout=joining.PING_TIMEOUT)
    self.assertTrue(wait_for(lambda: s._joined is not None), 'put back after the ping')
    self.assertTrue(s.big_model_available)
    s.in_control = False
    self.assertTrue(wait_for(lambda: self._run(s) == {'from': 'big'}))

  def test_a_link_that_dies_while_it_waits_is_closed_and_retried(self):
    # a host that rebooted in a drive with no window would be found at the
    # swap: a build on a dead link, a demote and the backoff, on modeld's thread
    clients = []

    def connect(should_stop=None):
      self.connect_calls += 1
      client = mock.MagicMock(name='client')
      client.ping.side_effect = RuntimeError('no PONG') if self.connect_calls == 1 else None
      clients.append(client)
      return (client, 'spec')

    self._connect = connect
    with mock.patch.object(joining, 'KEEPALIVE_PERIOD', 0.01), mock.patch.object(joining, 'REJOIN_DELAY_QUICK', 0.05):
      s = self._state()
      self.assertTrue(wait_for(lambda: self.connect_calls >= 2))
    clients[0].close.assert_called()
    self.assertEqual(s._drops, 0, 'no frame ran on it: nothing about the cable')
    self.assertTrue(self.log.has('died before it could be used', 'warning'))
    self._wait_reported(s, 'reconnecting')
    self.assertTrue(wait_for(lambda: s.big_model_available))

  def test_a_frame_while_the_ping_has_the_link_stays_small(self):
    s = self._state()
    self._wait_joined(s)
    with s._lock:
      joined, s._joined = s._joined, None   # as _keep_alive takes it
    s.in_control = False
    self.assertEqual(self._run(s), {'from': 'small'})
    self.assertTrue(s.big_model_available, 'still ready: the ping puts it back')
    with s._lock:
      s._joined = joined
    self.assertEqual(self._run(s), {'from': 'big'})


class LagTest(JoiningBase):
  """A large model that answers, but late, is handed back as if it were lost:
  when it says it held too many frames (behind), or when modeld drops too
  many camera frames behind it."""

  def setUp(self):
    super().setUp()
    self.reset = mock.Mock()
    # modeld's filter of dropped camera frames, and its frames since a handover
    self.drops = 0.0
    self.run_count = 0
    self.small.new_constant, self.big.new_constant = 'small', 'big'
    self.s = self._make(self.small, self._connect, self._build, reset_small=self.reset)
    self.addCleanup(self._close, self.s)
    self._wait_joined(self.s)
    self.s.in_control = False
    self.swap()
    self.settle()

  def swap(self):
    self.assertEqual(self.frame(), {'from': 'big'})
    self.assertTrue(self.s.chestnut)

  def settle(self):
    # modeld forgives the dropped frames of the ten frames after a handover
    for _ in range(10):
      self.assert_big_drives()

  def rejoin(self):
    self.reset.reset_mock()
    self.s._rejoin_at = 0.0
    self.s._rejoin.set()
    self._wait_joined(self.s)
    self.swap()
    self.settle()

  def frame(self, skipped=0):
    """One modeld frame, after `skipped` camera frames modeld dropped. Before
    run() modeld writes its share of dropped frames onto the model, from the
    filter both modelds run (10 s at 20 Hz), held at zero for the ten frames
    after a handover. Kept here: jetlink cannot import it, and the fork's seam
    test runs the real one against DROP_LIMIT."""
    self.drops += 0.05 / (10. + 0.05) * (min(skipped, 10) - self.drops)
    if self.run_count < 10:
      self.drops = 0.
    self.run_count += 1
    self.s.frame_drop_ratio = self.drops / (1 + self.drops)
    handovers = self.s.handovers
    result = self._run(self.s)
    if self.s.handovers != handovers:
      self.run_count = 0
    return result

  def hand_back(self):
    self.behind()
    self.frame()

  def behind(self):
    """One frame after which the large model says it held too many
    (model_state.JetlinkModelState.behind), published as it came."""
    self.big.behind = 'held 5 frames in a row'
    try:
      return self.frame()
    finally:
      self.big.behind = None

  def assert_big_drives(self):
    self.assertEqual(self.frame(), {'from': 'big'})
    self.assertTrue(self.s.chestnut)

  def test_a_frame_behind_is_published_and_the_next_one_is_the_small_models(self):
    self.assert_big_drives()
    handovers = self.s.handovers
    self.assertEqual(self.behind(), {'from': 'big'})
    # the late frame's output is the large model's, and so is everything modeld
    # reads off the model for it, modelV2.big included. The handover count
    # moves now, which is what makes modeld forgive the stall
    self.assertTrue(self.s.chestnut)
    self.assertEqual(self.s.new_constant, 'big')
    self.assertEqual(self.s.handovers, handovers + 1)
    self.reset.assert_not_called()
    self.assertEqual(self.frame(), {'from': 'small'})
    self.assertEqual(self.s.handovers, handovers + 2)
    self.reset.assert_called_once()
    self.assertFalse(self.s.chestnut)
    self.assertEqual(self.s.new_constant, 'small')
    # backed off and reported like a loss, and the link let go; counted as lag,
    # which says nothing about the cable
    self.assertEqual((self.s._lags, self.s._drops), (1, 0))
    self.assertEqual(self.s.big_model_state, 'retrying')
    self.assertGreater(self.s._rejoin_at, joining.time.monotonic())
    self._wait_reported(self.s, 'fell behind, reconnecting')
    for _ in range(100):
      if self.big.closed:
        break
      time.sleep(0.01)
    self.assertTrue(self.big.closed)

  def test_a_model_that_says_it_held_too_many_is_handed_back_on_the_next_frame(self):
    handovers = self.s.handovers
    self.assertEqual(self.behind(), {'from': 'big'}, 'published as it came')
    self.assertEqual(self.s.handovers, handovers + 1)
    self.assertEqual(self.frame(), {'from': 'small'})
    self.assertEqual((self.s._lags, self.s._drops), (1, 0))
    self.assertTrue(any('held 5 frames' in line for line in self.log.lines('warning')))

  def test_after_a_hand_back_the_next_link_waits_for_the_window(self):
    self.hand_back()
    self.assertEqual(self.frame(), {'from': 'small'})
    calls = self.big.calls
    self.s._rejoin_at = 0.0
    self.s._rejoin.set()
    self._wait_joined(self.s)
    # engaged through the rejoin: the small model drives, and the large one
    # is built and swapped in only when the window opens
    self.s.in_control = True
    for _ in range(3):
      self.assertEqual(self.frame(), {'from': 'small'})
    self.assertEqual(self.big.calls, calls)
    self.assertEqual(self.s.big_model_state, 'ready')
    self.s.in_control = False
    self.swap()
    self.settle()

  def test_lag_never_blames_the_cable(self):
    for _ in range(joining.DROPS_TO_BLAME_CABLE):
      self.hand_back()
      self.rejoin()
    self._wait_reported(self.s, 'fell behind, reconnecting')
    self.assertFalse(any('cable' in c.args[2] for c in self.progress.report.call_args_list))

  # A host a little slower than the camera: no frame is held, but modeld
  # skips camera frames, and selfdrived soft-disables
  # past 1 % of them (modeldLagging)

  def assert_handed_back(self, handovers):
    # the small model's from this frame on, and the handover moved in this
    # frame's run(), which is what has modeld forgive the drops it counted
    self.assertEqual(self.s.handovers, handovers + 1)
    self.reset.assert_called_once()
    self.assertFalse(self.s.chestnut)
    self.assertEqual(self.s._lags, 1)
    self.assertEqual(self.s.big_model_state, 'retrying')
    self.assertTrue(self.log.has('of camera frames behind the large model'))

  def test_one_dropped_frame_is_forgiven(self):
    handovers = self.s.handovers
    self.assertEqual(self.frame(skipped=1), {'from': 'big'})
    for _ in range(5):
      self.assert_big_drives()
    self.assertEqual(self.s.handovers, handovers)
    self.reset.assert_not_called()

  def test_a_second_dropped_frame_soon_after_hands_back(self):
    self.frame(skipped=1)
    for _ in range(100):   # 5 s
      self.assert_big_drives()
    handovers = self.s.handovers
    self.assertEqual(self.frame(skipped=1), {'from': 'small'})
    self.assert_handed_back(handovers)
    self.assertEqual(self.frame(), {'from': 'small'})

  def test_one_frame_that_drops_two_hands_back(self):
    handovers = self.s.handovers
    self.assertEqual(self.frame(skipped=2), {'from': 'small'})
    self.assert_handed_back(handovers)

  def test_drops_often_enough_for_selfdrived_hand_back_and_rare_ones_never(self):
    # modeld's share passes selfdrived's 1 % at a steady drop every 136 frames
    # (6.8 s) or closer; the large model goes at anything closer than about
    # 11 s, and never further apart
    for spacing, hands_back in ((1, True), (5, True), (60, True), (136, True), (200, True), (241, False), (400, False)):
      with self.subTest(spacing=spacing):
        handed_back = any(self.frame(skipped=1 if i % spacing == 0 else 0) == {'from': 'small'} for i in range(2000))
        self.assertEqual(handed_back, hands_back)
        # the next spacing starts from a swap, with modeld's filter empty
        if not handed_back:
          self.hand_back()
        self.rejoin()

  def test_a_host_slower_than_the_camera_hands_back(self):
    # 55 ms a frame, never held: modeld falls 5 ms
    # further behind the camera each frame and skips one when it is a whole
    # frame behind, so every tenth: an iPhone that has warmed up
    behind, handed_back_at = 0, None
    for i in range(1, 41):
      skipped, behind = divmod(behind + 5, 50)
      if self.frame(skipped=skipped) == {'from': 'small'}:
        handed_back_at = i
        break
    # the second dropped frame, about a second in
    self.assertEqual(handed_back_at, 20)
    self.assertEqual(self.s._lags, 1)

  def test_the_small_models_dropped_frames_are_its_own(self):
    self.hand_back()
    for _ in range(20):
      self.assertEqual(self.frame(skipped=3), {'from': 'small'})
    self.assertEqual(self.s._lags, 1)

  def test_a_modeld_that_writes_no_share_leaves_the_timing_rules(self):
    # a fork older than the write: the share stays at zero
    for _ in range(20):
      self.assertEqual(self._run(self.s), {'from': 'big'})
    self.assertEqual(self.s.frame_drop_ratio, 0.)
    self.reset.assert_not_called()


class WarmupTest(JoiningBase):
  """The small model runs before the large one first drives, however soon that
  could be: its first run in a process is slow, and it must not be the frame
  of the first fallback."""

  def setUp(self):
    super().setUp()
    patcher = mock.patch.object(joining, 'SMALL_WARMUP_FRAMES', 3)
    patcher.start()
    self.addCleanup(patcher.stop)
    self.first_small_frame_at = None
    run = self.small.run

    def small_run(*args):
      if self.small.calls == 0:
        # the ~1.3 s first run, marked rather than waited out
        self.first_small_frame_at = self.big.calls
      return run(*args)
    self.small.run = small_run

  def test_the_small_model_drives_the_first_frames_with_the_large_one_ready(self):
    s = self._state()
    self._wait_joined(s)
    s.in_control = False
    for _ in range(joining.SMALL_WARMUP_FRAMES):
      self.assertEqual(self._run(s), {'from': 'small'})
    self.assertEqual(self._run(s), {'from': 'big'})

  def test_the_first_fallback_is_not_the_small_models_first_run(self):
    s = self._state()
    self._wait_joined(s)
    s.in_control = False
    while self._run(s) != {'from': 'big'}:
      pass
    self.assertEqual(self.first_small_frame_at, 0, "the large model drove before the small one had run")
    self.big.raises = RuntimeError('pulled')
    calls = self.small.calls
    self.assertEqual(self._run(s), {'from': 'small'})
    self.assertGreater(calls, 0)


class ReplugTest(JoiningBase):
  """The backoff after a loss is for a host that is still there. One that let
  go of the gadget and configured it again is a replug: try at once."""

  def setUp(self):
    super().setUp()
    self.attached = True
    patcher = mock.patch.object(joining.gadget, 'host_attached', lambda: self.attached)
    patcher.start()
    self.addCleanup(patcher.stop)
    patcher = mock.patch.object(joining, 'REPLUG_POLL', 0.01)
    patcher.start()
    self.addCleanup(patcher.stop)
    self.s = self._state()
    self._wait_joined(self.s)
    self.s.in_control = False
    self._run(self.s)
    self.assertTrue(self.s.chestnut)
    # past the quick retries: the shortcut matters once the backoff is long
    self.s._failures = joining.QUICK_RETRIES

  def lose_the_link(self):
    self.big.raises = RuntimeError('host dropped the gadget configuration (udc: not attached)')
    self._run(self.s)
    self.assertGreater(self.s._rejoin_at, time.monotonic() + 1.0, "no backoff to cut short")

  def connects_within(self, seconds, calls=2):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
      if self.connect_calls >= calls:
        return True
      time.sleep(0.005)
    return False

  def test_a_replug_ends_the_backoff(self):
    self.attached = False
    self.lose_the_link()
    self.assertFalse(self.connects_within(0.2))
    self.attached = True
    self.assertTrue(self.connects_within(1.0))
    self.assertTrue(self.log.has('the host configured the gadget again, retrying now'))

  def test_a_host_back_before_the_backoff_began_was_still_replugged(self):
    # the teardown between the loss and the backoff can take longer than a replug
    self.attached = False
    with mock.patch.object(self.s, '_close_retired', side_effect=lambda *why: setattr(self, 'attached', True)):
      self.lose_the_link()
      self.assertTrue(self.connects_within(1.0))

  def test_a_flapping_host_skips_the_backoff_once_per_streak(self):
    self.attached = False
    self.lose_the_link()
    self.assertFalse(self.connects_within(0.2))   # the join thread has seen the host go
    self.attached = True
    self.assertTrue(self.connects_within(1.0))
    # back, swapped in, and gone again at once: the same failure streak
    self._wait_joined(self.s)
    self.attached = False
    self.assertEqual(self._run(self.s), {'from': 'small'})
    self.assertGreater(self.s._rejoin_at, time.monotonic() + 5.0)
    self.assertFalse(self.connects_within(0.2, calls=3))
    self.attached = True
    self.assertFalse(self.connects_within(0.5, calls=3), 'a flapping host cycled the swap with no backoff')

  def test_a_join_that_held_starts_a_new_streak(self):
    self.attached = False
    self.lose_the_link()
    self.assertFalse(self.connects_within(0.2))
    self.attached = True
    self.assertTrue(self.connects_within(1.0))
    self._wait_joined(self.s)
    self.big.raises = None
    self._run(self.s)
    self.assertTrue(self.s.chestnut)
    self.s._joined_at = time.monotonic() - (STABLE_SECONDS + 1)
    self.attached = False
    self.big.raises = RuntimeError('pulled again, minutes later')
    self._run(self.s)
    self.assertFalse(self.connects_within(0.2, calls=3))
    self.attached = True
    self.assertTrue(self.connects_within(1.0, calls=3))

  def test_a_failure_with_the_host_present_waits_it_out(self):
    self.lose_the_link()
    self.assertFalse(self.connects_within(0.5))


class FakeV2Model(FakeModel):
  """A modeld_v2 ModelState: constants, smoothing and the action function are its own."""

  def __init__(self, name, chestnut=False, client=None, desire_key='desire', slots=('desire', 'lateral_control_params')):
    super().__init__(name, chestnut, client)
    self.constants = mock.Mock(name=f'{name}.constants', MODEL_FREQ=20, DESIRE_LEN=8)
    self.desire_key = desire_key
    self.numpy_inputs = dict.fromkeys(slots)
    self.LAT_SMOOTH_SECONDS = 0.1 if name == 'small' else 0.0
    self.LONG_SMOOTH_SECONDS = 0.2 if name == 'small' else 0.3
    self.PLANPLUS_CONTROL = 1.0
    self.seen_inputs = None

  def run(self, bufs, transforms, inputs, after_enqueue=None):
    self.seen_inputs = inputs
    return super().run(bufs, transforms, inputs, after_enqueue)

  def get_action_from_model(self, *args):
    return (self.name, args)


class ModeldV2FaceTest(JoiningBase):
  """What sunnypilot's modeld_tinygrad reads off the model: per-model, following the
  model that is driving, and the frame it built for the small bundle handed to
  whichever model takes it."""

  def setUp(self):
    super().setUp()
    self.small = FakeV2Model('small')
    self.big = FakeV2Model('big', chestnut=True, client=object(), desire_key='desire_pulse', slots=('desire', 'action_t', 'traffic_convention'))

  def _swap(self, s):
    self._wait_joined(s)
    s.in_control = False
    self._run(s)
    self.assertIs(s._active, self.big)

  def test_the_face_follows_the_model_that_drives(self):
    s = self._state()
    self.assertIs(s.constants, self.small.constants)
    self.assertEqual((s.LAT_SMOOTH_SECONDS, s.LONG_SMOOTH_SECONDS), (0.1, 0.2))
    self.assertEqual(s.get_action_from_model('out', 'prev'), ('small', ('out', 'prev')))
    self._swap(s)
    self.assertIs(s.constants, self.big.constants)
    self.assertEqual((s.LAT_SMOOTH_SECONDS, s.LONG_SMOOTH_SECONDS), (0.0, 0.3))
    self.assertEqual(s.get_action_from_model('out', 'prev'), ('big', ('out', 'prev')))

  def test_what_the_loop_writes_lands_on_both(self):
    s = self._state()
    s.PLANPLUS_CONTROL = 0.5
    self.assertEqual(self.small.PLANPLUS_CONTROL, 0.5)
    self._swap(s)
    self.assertEqual(self.big.PLANPLUS_CONTROL, 1.0)  # not yet written since the join
    s.PLANPLUS_CONTROL = 0.7
    self.assertEqual((self.small.PLANPLUS_CONTROL, self.big.PLANPLUS_CONTROL), (0.7, 0.7))

  def test_the_frame_is_built_for_the_small_bundle_and_handed_over_as_is(self):
    # the loop keys the desire input and probes the slots by the small bundle,
    # before and after the join; the large model takes the frame as built
    s = self._state()
    self.assertEqual(s.desire_key, 'desire')
    self.assertIs(s.numpy_inputs, self.small.numpy_inputs)
    self._wait_joined(s)
    s.in_control = False
    inputs = {'desire': object(), 'action_t': 1}
    self.assertEqual(s.run({}, {}, inputs), {'from': 'big'})
    self.assertIs(self.big.seen_inputs, inputs)
    self.assertEqual(s.desire_key, 'desire')
    self.assertIs(s.numpy_inputs, self.small.numpy_inputs)
    # and back on a demote, the same frame
    self.big.raises = RuntimeError('link died')
    self.assertEqual(s.run({}, {}, inputs), {'from': 'small'})
    self.assertIs(self.small.seen_inputs, inputs)


if __name__ == '__main__':
  unittest.main()
