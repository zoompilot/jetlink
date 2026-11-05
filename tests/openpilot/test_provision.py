"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

What a provisioning run does when the far end is attached but not serving.

That is the expensive state, not the one where no Jetson is plugged in: the
run has a host to talk to, and the owner starts another while the work is
unfinished, so anything a run repeats it repeats for as long as the car is
parked.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from jetlink import protocol as P
from jetlink.client import EngineMissing, JetlinkClient
from jetlink.comma import gadget, lending
from jetlink.openpilot import link, provision
from tests.openpilot import fakes
from tests.openpilot.fakes import OpenpilotTest

ROOT = Path(__file__).resolve().parents[2]


class FakeSpec:
  def __init__(self, sha256: str = 'deadbeef', nbytes: int = 1 << 20):
    self.sha256 = sha256
    self.nbytes = nbytes


class FakeSpecRecord:
  """The spec record in memory, counting its writes."""

  def __init__(self):
    self.spec = None
    self.ready = False
    self.stores = 0

  def load(self):
    return self.spec

  def store(self, spec) -> None:
    self.stores += 1
    self.spec, self.ready = spec, True

  def engine_ready_for(self, sha256) -> bool:
    return self.ready and self.spec is not None and self.spec.sha256 == sha256

  def clear_ready(self) -> None:
    self.ready = False


def serving_client(spec=None):
  """A client whose server already has the engine."""
  client = mock.Mock()
  client.hello.return_value = {'device': 'orin', 'trt_version': '10.3'}
  client.ensure_engine.return_value = spec or FakeSpec()
  return client


class TestProvisionCost(OpenpilotTest):
  """What provisioning is allowed to cost when nothing needs doing.

  The identity comes from the catalog's pointer, so a parked car asks the Jetson what it
  already has without reading, hashing or even having the ONNX.
  """

  ENTRY = {'name': 'Fake', 'ref': 'f' * 40, 'oid': 'deadbeef', 'size': 4096}

  def setUp(self):
    super().setUp()
    link._hashed.clear()
    self.model = self.tmp / 'big_driving_supercombo.onnx'
    self.model.write_bytes(b'x' * 4096)
    self.cache = FakeSpecRecord()
    self.hashed: list[str] = []
    self.patch(self.parts, 'spec', self.cache)
    self.progress = self.patch(self.parts, 'progress', mock.Mock())
    for name, value in (('shipped_model_path', self.model), ('selected_model', dict(self.ENTRY))):
      self.patch(self.parts.models, name, return_value=value)

    def sha256_file(path, *args, **kwargs):
      self.hashed.append(path)
      return 'deadbeef', 1 << 20

    self.patch(sys.modules['jetlink.spec'], 'sha256_file', side_effect=sha256_file)

  def run_with(self, client=None):
    d = provision.ProvisioningRun(self.parts)
    d.client = client or serving_client()
    return d

  def test_the_identity_comes_from_the_registry_not_the_file(self):
    d = self.run_with()
    assert d.provision() is True
    args = d.client.ensure_engine.call_args.args
    assert args[0] == self.ENTRY['oid'] and args[1] == self.ENTRY['size']
    assert self.op.log.has('provisioning Fake (0 MB, sha deadbeef)')
    assert self.op.log.has('engine ready for deadbeef')

  def test_a_model_the_comma_s_index_does_not_know_is_looked_up(self):
    d = self.run_with()
    with mock.patch.object(self.parts.models, 'selected_model', return_value={**self.ENTRY, 'oid': None, 'size': None}), \
         mock.patch.object(self.parts.models, 'resolve', return_value=('deadbeef', 4096)) as resolve:
      assert d.provision() is True
    resolve.assert_called_once_with('f' * 40, None)
    args = d.client.ensure_engine.call_args.args
    assert args[0] == 'deadbeef' and args[1] == 4096
    self.progress.report.assert_any_call('connect', 0.0, 'finding model', code='finding_model')

  def test_a_model_that_cannot_be_looked_up_is_a_failed_provision(self):
    # the ordinary failure path: logged, backed off, tried again next poll
    d = self.run_with()
    with mock.patch.object(self.parts.models, 'selected_model', return_value={**self.ENTRY, 'oid': None, 'size': None}), \
         mock.patch.object(self.parts.models, 'resolve', side_effect=OSError('offline')), \
         self.assertRaises(OSError):
      d.provision()
    d.client.ensure_engine.assert_not_called()

  def test_a_server_that_already_has_it_never_reads_the_file(self):
    # the steady state of a parked car: the 766 MB hash is not paid per retry
    d = self.run_with()
    for _ in range(3):
      assert d.provision() is True
    assert self.hashed == [], "hashed the model to ask a question the registry answers"

  def test_it_asks_even_with_no_model_on_disk(self):
    # The Jetson keeps its own copy of every ONNX and never prunes them, so a
    # comma that has deleted its own can still use an engine already built.
    d = self.run_with()
    with mock.patch.object(self.parts.models, 'shipped_model_path', return_value=None):
      assert d.provision() is True
    assert d.client.ensure_engine.call_args.kwargs['onnx_path'] is None

  def test_a_server_that_wants_the_bytes_gets_them_in_the_same_run(self):
    d = self.run_with()
    with mock.patch.object(self.parts.models, 'shipped_model_path', return_value=None), \
         mock.patch.object(d, 'fetch_model', return_value=self.model) as fetch, \
         mock.patch.object(link, 'ensure', side_effect=[EngineMissing('no engine'), FakeSpec()]) as ensure:
      # one run: the owner lends the link for all of it, and leaving after the
      # download put the upload and build five minutes out
      assert d.provision() is True
    fetch.assert_called_once()
    assert [c.args[4] for c in ensure.call_args_list] == [None, self.model]
    assert d.client.hello.call_count == 2, "a new session after the download"

  def test_the_server_is_asked_to_download_what_neither_end_has(self):
    # only the Jetson needs the model: it fetches it itself, and the comma
    # fetches it only once the server has said it cannot
    d = self.run_with()
    d.client.ensure_engine.side_effect = [EngineMissing('no internet'), EngineMissing('need upload'), FakeSpec()]
    with mock.patch.object(self.parts.models, 'shipped_model_path', return_value=None), \
         mock.patch.object(d, 'fetch_model', return_value=self.model) as fetch:
      assert d.provision() is True
    asked, again, uploaded = d.client.ensure_engine.call_args_list
    assert asked.kwargs['fetch'] is True and asked.kwargs['onnx_path'] is None
    assert asked.kwargs['ref'] == self.ENTRY['ref']
    # with the file here, the server is not asked to download it again
    assert again.kwargs['fetch'] is False and again.kwargs['onnx_path'] is None
    assert uploaded.kwargs['onnx_path'] == self.model
    fetch.assert_called_once()

  def test_a_server_that_downloads_it_needs_nothing_from_the_comma(self):
    d = self.run_with()
    with mock.patch.object(self.parts.models, 'shipped_model_path', return_value=None), \
         mock.patch.object(d, 'fetch_model') as fetch:
      assert d.provision() is True
    fetch.assert_not_called()
    assert d.client.ensure_engine.call_args.kwargs['fetch'] is True

  def test_the_hello_carries_our_pick_and_a_newer_one_on_the_server_is_provisioned(self):
    d = self.run_with()
    theirs = {**self.ENTRY, 'name': 'Picked on the Jetson', 'ref': 'e' * 40, 'oid': 'e' * 64}
    with mock.patch.object(self.parts.models, 'hello_pick', return_value={'ref': 'f' * 40, 'age_s': 9.0}), \
         mock.patch.object(self.parts.models, 'adopt') as adopt, \
         mock.patch.object(self.parts.models, 'selected_model', return_value=theirs):
      assert d.provision() is True
    d.client.hello.assert_called_once_with(timeout=10.0, pick={'ref': 'f' * 40, 'age_s': 9.0})
    adopt.assert_called_once_with()
    assert d.client.ensure_engine.call_args.args[0] == 'e' * 64
    assert self.op.log.has('provisioning Picked on the Jetson')

  def test_a_download_that_fails_leaves_the_engine_missing(self):
    d = self.run_with()
    d.client.ensure_engine.side_effect = EngineMissing('no engine')
    with mock.patch.object(self.parts.models, 'shipped_model_path', return_value=None), \
         mock.patch.object(d, 'fetch_model', return_value=None), \
         self.assertRaises(EngineMissing):
      d.provision()

  def _wants_the_bytes(self):
    d = self.run_with()
    d.client.ensure_engine.side_effect = EngineMissing('no engine')
    return d

  def test_a_file_that_is_not_the_registry_model_is_never_uploaded(self):
    # Trusting the pointer for the identity is right for asking and wrong for
    # answering: uploading under a sha the bytes do not have would leave the
    # Jetson with a plan whose name lies about its contents.
    d = self._wants_the_bytes()
    with mock.patch.object(self.parts.models, 'selected_model',
                           return_value={**self.ENTRY, 'oid': 'not-what-the-file-hashes-to'}), \
         self.assertRaises(EngineMissing):
      d.provision()
    assert all(c.kwargs['onnx_path'] is None for c in d.client.ensure_engine.call_args_list)
    assert self.op.log.has('hashes to deadbeef, the registry says not-what-the-f', 'error')

  def test_a_file_of_the_wrong_size_is_never_uploaded(self):
    d = self._wants_the_bytes()
    with mock.patch.object(self.parts.models, 'selected_model',
                           return_value={**self.ENTRY, 'size': 999999}), \
         self.assertRaises(EngineMissing):
      d.provision()
    assert all(c.kwargs['onnx_path'] is None for c in d.client.ensure_engine.call_args_list)
    assert self.hashed == [], "size is the cheap check and comes first"
    assert self.op.log.has('is 4096 bytes, the registry says 999999', 'error')

  def test_the_file_is_uploaded_once_it_is_proven_to_be_the_model(self):
    d = self._wants_the_bytes()
    d.client.ensure_engine.side_effect = [d.client.ensure_engine.side_effect, FakeSpec()]
    assert d.provision() is True
    calls = d.client.ensure_engine.call_args_list
    assert calls[0].kwargs['onnx_path'] is None, "asked without the file first"
    assert calls[1].kwargs['onnx_path'] == self.model
    assert self.hashed == [str(self.model)], "hashed once, on the path the bytes leave by"

  def test_a_file_already_hashed_is_not_hashed_again(self):
    # a join that fails and tries again would read a gigabyte every time
    for _ in range(2):
      d = self._wants_the_bytes()
      d.client.ensure_engine.side_effect = [EngineMissing('no engine'), FakeSpec()]
      assert d.provision() is True
    assert self.hashed == [str(self.model)]

  def test_a_ready_record_is_still_checked_with_the_server(self):
    # the Jetson's cache can be pruned or re-flashed under a record that says
    # ready. A run provisions once and then exits, so the check is once a run
    # and there is no second call to skip
    self.cache.store(FakeSpec())
    d = self.run_with()
    assert d.provision() is True
    assert d.client.ensure_engine.call_count == 1
    assert self.hashed == [], 'read the model to answer a question the server answers'
    assert self.cache.stores == 2 and self.cache.engine_ready_for('deadbeef')

  def test_the_shapes_come_from_the_server_not_the_file(self):
    d = self.run_with(serving_client(FakeSpec(sha256='deadbeef', nbytes=1 << 20)))
    assert d.provision() is True
    d.client.ensure_engine.assert_called_once()
    kwargs = d.client.ensure_engine.call_args.kwargs
    assert callable(kwargs['should_stop'])
    assert kwargs['build_timeout'] == link.BUILD_TIMEOUT
    assert kwargs['progress'] == self.parts.progress.report_with_eta
    assert self.cache.stores == 1 and self.cache.spec.sha256 == 'deadbeef'

  def test_stop_is_polled_through_the_long_wait(self):
    # a build started while parked can still be running when the driver pulls
    # away. The owner sends a stop at the onroad transition so modeld can take
    # the endpoints; the server's build thread carries on and modeld picks the
    # engine up over its own link
    d = self.run_with()
    d.provision()
    should_stop = d.client.ensure_engine.call_args.kwargs['should_stop']
    assert should_stop() is False
    d.request_stop()
    assert should_stop() is True

  def test_every_hello_is_passed_on_to_the_owner(self):
    # the owner never speaks the protocol; this is how it learns sleep_after
    client = serving_client()
    client.ensure_engine.side_effect = [EngineMissing('no bytes'), FakeSpec()]
    d = self.run_with(client)
    d.loan = mock.Mock()
    self.patch(self.parts.models, 'shipped_model_path', return_value=None)
    self.patch(d, 'fetch_model', return_value=self.model)
    assert d.provision() is True
    # the first hello, and the one after the download started a new session
    assert d.loan.note_server.call_args_list == [mock.call(client.hello.return_value)] * 2

  def test_no_pick_and_no_catalog_is_nothing_to_provision(self):
    d = self.run_with()
    with mock.patch.object(self.parts.models, 'selected_model', return_value=None):
      assert d.provision() is False
    assert self.cache.ready is False
    self.progress.clear.assert_called_once_with()
    # the hello came first, in case the server had a model picked on it
    d.client.hello.assert_called_once()
    d.client.ensure_engine.assert_not_called()


class TestFetching(OpenpilotTest):
  def test_a_download_reports_and_stops_with_the_run(self):
    d = provision.ProvisioningRun(self.parts)
    progress = self.patch(self.parts, 'progress', mock.Mock())
    with mock.patch.object(self.parts.models, 'fetch_shipped_model', return_value=Path('/x')) as fetch:
      self.assertEqual(d.fetch_model(), Path('/x'))
    fetch.call_args.kwargs['progress'](0.5)
    progress.report.assert_called_once_with('download', 0.5, 'downloading', code='downloading', args=None)
    self.assertFalse(fetch.call_args.kwargs['should_stop']())
    d.request_stop()
    self.assertTrue(fetch.call_args.kwargs['should_stop']())

  def test_a_failed_download_is_one_attempt_a_run(self):
    # retrying a gigabyte on a loop is worse than staying small
    d = provision.ProvisioningRun(self.parts)
    progress = self.patch(self.parts, 'progress', mock.Mock())
    with mock.patch.object(self.parts.models, 'fetch_shipped_model', side_effect=OSError('offline')) as fetch:
      self.assertIsNone(d.fetch_model())
      self.assertIsNone(d.fetch_model())
    fetch.assert_called_once()
    progress.report.assert_called_once_with('failed', 1.0, 'download failed, retrying later',
                                            code='download_failed')


class TestTheLoan(OpenpilotTest):
  """Only the owner that started this run holds ep0; the run borrows from it."""

  def test_no_loan_is_one_error_and_no_gadget_of_our_own(self):
    d = provision.ProvisioningRun(self.parts)
    with mock.patch.object(lending, 'borrow', return_value=None), \
         mock.patch('jetlink.client.JetlinkClient') as client:
      assert d.open_link() is False
    assert self.op.log.lines('error') == ["jetlink: the owner lent no link, nothing to provision over"]
    assert client.method_calls == []
    assert d.client is None

  def test_closing_the_link_says_the_run_is_done_while_it_can(self):
    d = provision.ProvisioningRun(self.parts)
    d.client = mock.Mock(dead=False)
    client = d.client
    d.close_link()
    client.leave.assert_called_once_with('provisioned')
    client.close.assert_called_once()
    d.client = mock.Mock(dead=True)
    client = d.client
    d.close_link()
    client.leave.assert_called_once_with('provisioned')   # and the client says nothing over a dead link
    client.close.assert_called_once()

  def test_a_loan_is_opened_over(self):
    d = provision.ProvisioningRun(self.parts)
    loan = mock.Mock(cable=False, mount='/dev/ffs-jetlink', udc='udc0')
    with mock.patch.object(lending, 'borrow', return_value=loan) as borrow, \
         mock.patch.object(link, 'connect') as connect:
      assert d.open_link() is True
    borrow.assert_called_once_with('provision')
    assert connect.call_args.args[1:] == (loan,)
    assert connect.call_args.kwargs == {'deadline': 5.0, 'name': 'provision'}
    assert d.client is connect.return_value

  def test_a_link_that_will_not_open_is_logged(self):
    d = provision.ProvisioningRun(self.parts)
    with mock.patch.object(lending, 'borrow', side_effect=OSError('no socket')):
      assert d.open_link() is False
    assert self.op.log.has('could not open the link', 'exception')


class TestTheRun(OpenpilotTest):
  """One round, then the process exits. What it leaves behind is what the owner
  cannot work out for itself."""

  def setUp(self):
    super().setUp()
    self.op.set_mode('usb')
    self.patch(self.parts, 'progress', mock.Mock())
    self.patch(gadget, 'SHUTDOWN_REQUEST', self.tmp / 'shutdown')

  def worker(self, work=True):
    d = provision.ProvisioningRun(self.parts)
    for name, value in (('has_work', work), ('open_link', True), ('provision', True)):
      self.patch(d, name, mock.Mock(return_value=value))
    # no host on the cable until a test puts one there
    for target, name, value in ((gadget, 'pending_shutdown', None), (gadget, 'wait_for_host', True),
                                (gadget, 'host_attached', False), (self.parts.warps, 'built', True)):
      self.patch(target, name, mock.Mock(return_value=value))
    return d

  def test_nothing_to_do_never_opens_the_link(self):
    d = self.worker(work=False)
    assert d.run() is True
    d.open_link.assert_not_called()

  def test_a_finished_round_says_so_and_lets_the_link_go(self):
    d = self.worker()
    d.client = mock.Mock()
    assert d.run() is True
    d.provision.assert_called_once()
    assert d.client is None, 'left the gadget open after the run'

  def test_a_round_that_fails_leaves_the_work_for_the_next_one(self):
    d = self.worker()
    d.provision.side_effect = RuntimeError('the jetson went away')
    assert d.run() is False
    self.parts.progress.report.assert_called_with('failed', 1.0, 'see the log', code='failed')

  def test_no_loan_is_left_for_the_next_run(self):
    d = self.worker()
    d.open_link.return_value = False
    assert d.run() is False
    d.provision.assert_not_called()

  def test_no_jetson_is_left_for_the_next_run(self):
    d = self.worker()
    gadget.wait_for_host.return_value = False
    assert d.run() is False
    d.provision.assert_not_called()

  def test_the_jetson_is_asked_before_any_download(self):
    # only the Jetson needs the model, and one that has it, or can download
    # it, needs nothing from the comma: a sleeping one wakes to the bind
    d = self.worker()
    order = []
    self.patch(d, 'needs_download', mock.Mock(return_value=True))
    self.patch(d, 'fetch_model', mock.Mock(side_effect=lambda: order.append('fetch') or Path('/m.onnx')))
    d.open_link.side_effect = lambda: order.append('borrow') or True
    d.provision.side_effect = lambda: order.append('provision') or True
    assert d.run() is True
    self.assertEqual(order, ['borrow', 'provision'])

  def test_with_no_jetson_the_comma_downloads_only_for_one_that_cannot(self):
    # a car whose Jetson is switched with the ignition has none while parked:
    # one that downloads for itself gets the model when it is on, and one
    # with no internet (or too old to say) gets it from the comma, fetched now
    for heard, fetched in (({'fetches': True}, False), ({'fetches': False}, True), ({}, True), (None, True)):
      with self.subTest(heard=heard):
        d = self.worker()
        self.patch(d, 'needs_download', mock.Mock(return_value=True))
        self.patch(d, 'fetch_model', mock.Mock(return_value=Path('/m.onnx')))
        gadget.wait_for_host.return_value = False
        if heard is None:
          gadget.SERVER.unlink(missing_ok=True)
        else:
          gadget.write_record(gadget.SERVER, heard)
        assert d.run() is False
        self.assertIs(d.fetch_model.called, fetched)
        d.provision.assert_not_called()
        if fetched:
          self.parts.progress.report.assert_called_with('waiting', 0.0, 'downloaded, builds when the jetson is on',
                                                        code='downloaded_waiting_host')
        else:
          self.parts.progress.report.assert_called_with('waiting', 0.0, 'the jetson downloads it when it is on',
                                                        code='waiting_for_host')

  def test_the_download_waits_until_the_link_is_let_go(self):
    # minutes on LTE: the loan goes back to the owner first
    d = self.worker()
    order = []
    self.patch(d, 'needs_download', mock.Mock(return_value=True))
    self.patch(d, 'fetch_model', mock.Mock(side_effect=lambda: order.append('fetch') or None))
    self.patch(d, 'close_link', mock.Mock(side_effect=lambda: order.append('close')))
    gadget.wait_for_host.return_value = False
    assert d.run() is False
    self.assertEqual(order[:2], ['close', 'fetch'])
    self.parts.progress.report.assert_not_called()

  def test_downloaded_with_no_jetson_says_it_builds_when_one_is_on(self):
    d = self.worker()
    self.patch(d, 'needs_download', mock.Mock(return_value=False))
    self.patch(d, 'fetch_model', mock.Mock(return_value=Path('/m.onnx')))
    gadget.wait_for_host.return_value = False
    assert d.run() is False
    d.fetch_model.assert_not_called()
    self.parts.progress.report.assert_called_with('waiting', 0.0, 'downloaded, builds when the jetson is on',
                                                  code='downloaded_waiting_host')

  def test_a_jetson_awake_on_the_bus_is_asked_for_its_pick(self):
    # nothing to build here, but a model picked on the server since is the
    # comma's now: a hello says, and needs no wake
    d = self.worker(work=False)
    gadget.host_attached.return_value = True
    assert d.run() is True
    d.provision.assert_called_once_with()

  def test_a_phone_on_the_cable_is_not_waited_for_with_nothing_to_do(self):
    # the cable is there only once the app dials in
    self.op.set_mode('ios')
    d = self.worker(work=False)
    gadget.host_attached.return_value = True
    assert d.run() is True
    d.open_link.assert_not_called()

  def test_only_a_pick_neither_built_nor_here_is_downloaded(self):
    d = provision.ProvisioningRun(self.parts)
    entry = {'name': 'ResAction', 'ref': 'r' * 40, 'oid': 'a' * 64, 'size': 10}
    self.patch(self.parts.models, 'selected_model', return_value=entry)
    for built, here, want in ((False, None, True), (True, None, False), (False, Path('/m.onnx'), False)):
      with self.subTest(built=built, here=here):
        with mock.patch.object(self.parts.spec, 'engine_ready_for', return_value=built), \
             mock.patch.object(self.parts.models, 'shipped_model_path', return_value=here):
          self.assertIs(d.needs_download(), want)

  def test_nothing_is_left_behind_in_dev_shm(self):
    # what the owner needs goes over the loan and in the exit status
    d = self.worker(work=False)
    with mock.patch('pathlib.Path.write_text', side_effect=AssertionError('a record written')):
      assert d.run() is True

  def test_a_shutdown_request_is_the_whole_round(self):
    d = self.worker()
    gadget.pending_shutdown.return_value = 'car battery'
    with mock.patch.object(d, 'shutdown_jetson') as shutdown:
      assert d.run() is True
    shutdown.assert_called_once_with('car battery')
    d.provision.assert_not_called()

  def test_no_warp_for_this_camera_is_nothing_to_provision_for(self):
    # the engine could never run; waking the Jetson to build it changes nothing
    d = self.worker()
    self.parts.warps.built.return_value = False
    assert d.run() is True
    d.open_link.assert_not_called()

  def test_the_link_off_is_not_a_round(self):
    d = self.worker()
    self.op.set_mode('off')
    assert d.run() is True
    d.open_link.assert_not_called()

  def test_a_chestnut_is_not_the_runs_to_judge(self):
    # the owner started this run for the setting; the chestnut veto is manager's
    d = self.worker()
    self.op.chestnut = True
    assert d.run() is True
    d.provision.assert_called_once()


class ReplayDroppingServer:
  """The server's session rule, scripted (JetlinkServer's Session.handle): a
  hello starts the session over at its seq, and any other message at or below
  the last seq seen is a replay and dropped without an answer. It answers a
  hello and a shutdown request, and records the rest."""

  def __init__(self, transport, last_seq: int):
    self.t = transport
    self.last = last_seq
    self.dropped: list[int] = []
    self.shutdowns: list[dict] = []
    self.done = threading.Event()
    threading.Thread(target=self.serve, daemon=True).start()

  def serve(self) -> None:
    try:
      while not self.done.is_set():
        msg = self.t.recv(timeout=10.0)
        if msg.msg_type == P.Msg.HELLO_REQ:
          self.last = msg.seq
          self.t.send_json(P.Msg.HELLO_RESP, msg.seq, {'protocol': P.VERSION, 'device': 'orin', 'sleep_after': 0.0})
        elif msg.seq <= self.last:
          self.dropped.append(msg.msg_type)
        else:
          self.last = msg.seq
          if msg.msg_type == P.Msg.SHUTDOWN_REQ:
            self.shutdowns.append(json.loads(bytes(msg.payload)))
            self.t.send_json(P.Msg.SHUTDOWN_RESP, msg.seq, {'ok': True})
            self.done.set()
    except Exception:
      self.done.set()


class TestShuttingTheJetsonDown(OpenpilotTest):
  def setUp(self):
    super().setUp()
    self.request = self.tmp / 'shutdown'
    self.patch(gadget, 'SHUTDOWN_REQUEST', self.request)
    self.request.write_text(json.dumps({'reason': 'car battery'}))
    self.d = provision.ProvisioningRun(self.parts)
    self.d.client = mock.Mock()
    self.wait = self.patch(gadget, 'wait_for_host', return_value=True)

  def test_the_jetson_is_asked_and_the_request_taken(self):
    self.d.shutdown_jetson('car battery')
    self.d.client.shutdown.assert_called_once_with('car battery', timeout=5.0)
    self.assertFalse(self.request.exists(), 'hardwared is waiting on the file')

  def test_a_session_that_outlived_the_last_borrower_still_hears_it(self):
    # the gadget stayed bound since the last borrower (an always-on Jetson, a
    # build just stopped, inside the dormant hold), so the server's session
    # did too, at the last borrower's seq. This client starts at seq 1
    from jetlink.transport.tcp import TcpTransport
    srv = TcpTransport.listen('127.0.0.1', 0)
    ours = TcpTransport.connect('127.0.0.1', srv.getsockname()[1])
    theirs = TcpTransport(srv.accept()[0])
    srv.close()
    server = ReplayDroppingServer(theirs, last_seq=40)
    self.addCleanup(ours.close)
    self.addCleanup(theirs.close)
    self.d.client = JetlinkClient(ours, name='provision')
    self.d.loan = mock.Mock()
    self.d.shutdown_jetson('car battery')
    server.done.wait(5.0)
    self.assertEqual(server.shutdowns, [{'reason': 'car battery'}])
    self.assertEqual(server.dropped, [])
    # and, like every hello, it reaches the owner
    self.d.loan.note_server.assert_called_once_with({'protocol': P.VERSION, 'device': 'orin', 'sleep_after': 0.0})
    self.assertTrue(self.op.log.has("jetson answered the shutdown request: {'ok': True}"))
    self.assertFalse(self.request.exists())

  def test_the_request_is_taken_whatever_happens(self):
    self.wait.return_value = False
    self.d.shutdown_jetson('car battery')
    self.d.client.shutdown.assert_not_called()
    self.assertFalse(self.request.exists())
    self.assertTrue(self.op.log.has('could not shut the jetson down', 'exception'))

  def test_a_bounce_goes_over_the_lease(self):
    self.d.client.rebind.return_value = True
    self.assertTrue(self.d.bounce())
    self.d.client.rebind.side_effect = OSError('gone')
    self.assertFalse(self.d.bounce())
    self.d.client = None
    self.assertFalse(self.d.bounce())


class TestTheEntryPoint(unittest.TestCase):
  """python -m jetlink.openpilot.provision --adapter MODULE, as the owner starts it."""

  def test_it_runs_one_round_over_the_adapter_it_is_named(self):
    root = Path(tempfile.mkdtemp())
    # the link off: a round that ends at once, without touching the gadget
    env = {**os.environ, 'JETLINK_FAKE_ROOT': str(root), 'PYTHONPATH': str(ROOT)}
    run = subprocess.run([sys.executable, '-m', 'jetlink.openpilot.provision', '--adapter', 'tests.openpilot.fakes'],
                         cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    self.assertEqual(run.returncode, 0, run.stderr)

  def test_a_round_with_work_and_no_owner_leaves_it_for_the_next_run(self):
    # the whole run in its own process: bound to the adapter, the link on,
    # a warp built, no spec yet, and no owner to lend the link. The fake
    # adapter points the comma layer's files under its root there
    root = Path(tempfile.mkdtemp())
    (root / 'params' / 'd').mkdir(parents=True)
    (root / 'params' / 'd' / 'JetlinkLink').write_bytes(b'1')
    warp = fakes.FakeOpenpilot(root).warp_path(*fakes.TICI)
    warp.parent.mkdir(parents=True)
    warp.write_bytes(b'built')
    env = {**os.environ, 'JETLINK_FAKE_ROOT': str(root), 'JETLINK_FAKE_ISOLATE': '1', 'PYTHONPATH': str(ROOT)}
    run = subprocess.run([sys.executable, '-m', 'jetlink.openpilot.provision', '--adapter', 'tests.openpilot.fakes'],
                         cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    # unfinished is the exit status, and the owner tries again after its backoff
    self.assertEqual(run.returncode, 1, run.stderr)
    self.assertFalse((root / 'dev' / 'state').exists())

  def test_a_borrower_asks_the_socket_as_it_is_when_it_asks(self):
    # a default bound when lending was imported would ask the real owner
    here = Path(tempfile.mkdtemp()) / 'lend.sock'
    with mock.patch.object(lending, 'SOCKET', here), \
         mock.patch.object(lending.socket.socket, 'connect', side_effect=OSError('nobody')) as connect:
      self.assertIsNone(lending.borrow('provision', timeout=0.1))
      self.assertEqual(lending.Lender(lambda: True, lambda: True).path, here)
    self.assertEqual(connect.call_args.args[-1], str(here))

  def test_the_adapter_is_required(self):
    with self.assertRaises(SystemExit):
      provision.main([])

  def main(self, finished: bool):
    """main() over a round that finishes or not: its exit status, the round, the signals."""
    # binding points jetlink.comma's log at the adapter's; put it back after
    with mock.patch.object(provision.ProvisioningRun, 'run', return_value=finished) as run, \
         mock.patch.object(provision.signal, 'signal') as signal, \
         mock.patch.object(gadget, 'log', gadget.log), mock.patch.object(gadget.root, 'log', gadget.root.log), \
         mock.patch.dict(os.environ, {'JETLINK_FAKE_ROOT': tempfile.mkdtemp()}), \
         self.assertRaises(SystemExit) as exited:
      provision.main(['--adapter', 'tests.openpilot.fakes'])
    return exited.exception.code, run, signal

  def test_stops_are_signals(self):
    code, run, signal = self.main(finished=True)
    run.assert_called_once_with()
    self.assertEqual({c.args[0] for c in signal.call_args_list}, {provision.signal.SIGTERM, provision.signal.SIGINT})

  def test_the_exit_status_says_whether_work_is_left(self):
    self.assertEqual(self.main(finished=True)[0], 0)
    self.assertEqual(self.main(finished=False)[0], 1)


if __name__ == '__main__':
  unittest.main()
