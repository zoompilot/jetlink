"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

An add-on's link never opens before the gadget is configured, and leaves the
caller's own model to run on any doubt.
"""
from pathlib import Path
from types import SimpleNamespace
import os
import struct

import numpy as np
import pytest

from jetlink.addons import link as link_mod
from jetlink.addons import wire as W
from jetlink.addons.link import AddonLink, HostModel, JetsonNet, blob_to_u8

SHA = "ab12cd34" + "0" * 56
TAG = W.tag(SHA)


def function(tmp_path: Path):
  for name in ("ep0", "ep1", "ep2"):
    (tmp_path / name).write_bytes(b"")
  return SimpleNamespace(name="vasm", ep0=tmp_path / "ep0", ep_out=tmp_path / "ep1", ep_in=tmp_path / "ep2")


class Clock:
  def __init__(self):
    self.t = 100.0

  def __call__(self):
    return self.t


@pytest.fixture
def no_threads(monkeypatch):
  monkeypatch.setattr(link_mod.threading, "Thread", lambda *a, **k: SimpleNamespace(start=lambda: None))


def test_wire_round_trips():
  image = bytes(range(12))
  seq, tag, shape, encoding, payload = W.unpack_request(W.pack_request(7, TAG, image, (3, 2, 2)))
  assert (seq, tag, shape, encoding, bytes(payload)) == (7, TAG, (3, 2, 2), W.ENC_U8, image)
  floats = np.arange(12, dtype=np.float32).tobytes()
  assert W.unpack_request(W.pack_request(8, TAG, floats, (3, 2, 2), W.ENC_F32))[3] == W.ENC_F32
  out = np.arange(4, dtype=np.float32)
  seq, tag, status, us, n, payload = W.unpack_answer(W.pack_answer(7, TAG, W.STATUS_OK, 55, out.tobytes()))
  assert (seq, tag, status, us, n) == (7, TAG, W.STATUS_OK, 55, 4)
  assert np.array_equal(np.frombuffer(payload, np.float32), out)
  seq, version, status, _, n, payload = W.unpack_answer(W.pack_hello([TAG, 5], W.STATUS_YIELD))
  assert (seq, version, status, W.hello_tags(payload, n)) == (W.HELLO_SEQ, W.VERSION, W.STATUS_YIELD, frozenset({TAG, 5}))
  good = W.pack_request(1, TAG, image, (3, 2, 2))
  for bad in (b"", good[:-1], good + b"\0", b"XXXX" + good[4:], good[:21] + b"\x09" + good[22:]):
    with pytest.raises(ValueError):
      W.unpack_request(bad)
  with pytest.raises(ValueError):
    W.unpack_answer(W.pack_answer(1, TAG, W.STATUS_OK, 0, b"\0" * 4)[:-1])
  with pytest.raises(ValueError):
    W.pack_request(1, TAG, image, (3, 2, 3))


def test_whole_packet_messages_carry_a_pad_byte():
  image = bytes(W.PACKET * 2 - W.REQ.size)
  message = W.pack_request(1, TAG, image, (1, 1, len(image)))
  assert len(message) == W.PACKET * 2 + 1
  assert bytes(W.unpack_request(message)[4]) == image
  floats = bytes(W.PACKET - W.RESP.size)
  answer = W.pack_answer(1, TAG, W.STATUS_OK, 0, floats)
  assert len(answer) % W.PACKET == 1 and bytes(W.unpack_answer(answer)[5]) == floats
  assert len(W.pack_request(1, TAG, bytes(12), (3, 2, 2))) == W.REQ.size + 12


def test_never_opens_before_the_gadget_is_configured(tmp_path, monkeypatch, no_threads):
  opened = []
  monkeypatch.setattr(link_mod.os, "open", lambda *a: opened.append(a) or 99)
  clock = Clock()
  link = AddonLink(function(tmp_path), host_ready=lambda: False, clock=clock)
  assert link.infer(TAG, np.zeros((3, 2, 2), np.uint8), 3) is None
  clock.t += 10
  link.infer(TAG, np.zeros((3, 2, 2), np.uint8), 3)
  assert opened == [] and link._rx is None


def test_absent_interface_is_never_opened(tmp_path, monkeypatch, no_threads):
  opened = []
  monkeypatch.setattr(link_mod.os, "open", lambda *a: opened.append(a) or 99)
  fn = SimpleNamespace(ep0=tmp_path / "missing", ep_out=tmp_path / "ep1", ep_in=tmp_path / "ep2")
  link = AddonLink(fn, host_ready=lambda: True, clock=Clock())
  assert not link_mod.functions.present(fn)
  link.infer(TAG, np.zeros((3, 2, 2), np.uint8), 3)
  assert opened == []


def test_a_failed_endpoint_closes_both(tmp_path, monkeypatch, no_threads):
  clock = Clock()
  link = AddonLink(function(tmp_path), host_ready=lambda: True, clock=clock)
  link._ensure_open()
  fds = (link._rx.fd, link._tx.fd)
  assert min(fds) >= 0 and link.stats["reopens"] == 1
  link.handle(W.pack_hello([TAG]))
  assert link.live(TAG)
  link._close()
  assert (link._rx, link._tx, link.tags) == (None, None, frozenset())
  for fd in fds:
    with pytest.raises(OSError):
      os.fstat(fd)
  link._ensure_open()
  assert link._rx is None   # waits REOPEN_S
  clock.t += link_mod.REOPEN_S
  link._ensure_open()
  assert link._rx is not None
  link._close()


def test_an_endpoint_in_use_closes_when_its_call_returns(tmp_path, no_threads):
  link = AddonLink(function(tmp_path), host_ready=lambda: True, clock=Clock())
  link._ensure_open()
  rx = link._rx
  fd = rx.acquire()   # a read in flight
  link._close()
  os.fstat(fd)        # still open: its number cannot be reused under the read
  assert rx.acquire() == -1
  rx.release()
  with pytest.raises(OSError):
    os.fstat(fd)


def test_a_late_error_never_closes_the_pair_opened_after_it(tmp_path, no_threads):
  clock = Clock()
  link = AddonLink(function(tmp_path), host_ready=lambda: True, clock=clock)
  link._ensure_open()
  old = link._tx
  link._close()
  clock.t += link_mod.REOPEN_S
  link._ensure_open()
  current = link._rx
  link._close(old)
  assert link._rx is current
  link._close()


def test_reads_go_in_pieces_and_a_message_ends_at_a_short_one(tmp_path, monkeypatch):
  link = AddonLink(function(tmp_path), host_ready=lambda: True, clock=Clock())
  link._ensure_open = lambda: None
  link._rx = link_mod._Endpoint(1000)
  big = np.arange(10_000, dtype=np.float32)   # 40 KB: three pieces
  messages = [W.pack_hello([TAG]), W.pack_answer(9, TAG, W.STATUS_OK, 1, big.tobytes())]
  stream = [m[i:i + link_mod.READ_CHUNK] for m in messages for i in range(0, len(m), link_mod.READ_CHUNK)]
  sizes = []

  def readv(fd, bufs):
    sizes.append(len(bufs[0]))
    if not stream:
      raise SystemExit
    piece = stream.pop(0)
    bufs[0][:len(piece)] = piece
    return len(piece)
  monkeypatch.setattr(link_mod.os, "readv", readv)
  monkeypatch.setattr(link_mod, "_mask_signals", lambda: None)
  slot = [link_mod.threading.Event(), None]
  link._waiting[9] = slot
  with pytest.raises(SystemExit):
    link._read_forever()
  assert set(sizes) == {link_mod.READ_CHUNK}
  assert link.tags == {TAG} and slot[0].is_set() and np.array_equal(slot[1][2], big)


def test_over_a_cable_network_the_phone_dials_and_answers_are_framed_by_header(monkeypatch):
  import socket
  import threading
  from jetlink.transport import tcp
  monkeypatch.setattr(tcp, "CABLE_ADDRESS", "127.0.0.1")
  probe = socket.socket()
  probe.bind(("127.0.0.1", 0))
  port = probe.getsockname()[1]
  probe.close()
  link = AddonLink(SimpleNamespace(name="vasm", port=port), cable_network=lambda: True)
  assert link.poll(TAG) is False   # listening now, nothing dialed yet

  def phone():
    conn = socket.create_connection(("127.0.0.1", port), timeout=2)
    hello = W.pack_hello([TAG])
    conn.sendall(hello[:5])
    conn.sendall(hello[5:])
    request = conn.recv(1 << 16)
    seq = W.unpack_request(request)[0]
    answer = W.pack_answer(seq, TAG, W.STATUS_OK, 7, np.asarray((0.25, 0.5, 0.25), np.float32).tobytes())
    conn.sendall(answer + W.pack_hello([TAG]))   # an answer and a hello in one segment
    conn.recv(1)
    conn.close()
  threading.Thread(target=phone, daemon=True).start()
  link._next_open = 0
  for _ in range(200):
    link._next_open = 0
    if link.poll(TAG):
      break
    threading.Event().wait(0.01)
  assert link._stream and link.ready(TAG)
  monkeypatch.setattr(link_mod, "REPLY_TIMEOUT_S", 2.0)
  assert np.allclose(link.infer(TAG, np.zeros((3, 2, 2), np.uint8), 3), (0.25, 0.5, 0.25))
  link._close()
  assert link._conn is None and link._rx is None


def test_a_stream_answer_is_its_header_and_floats_and_any_pad():
  assert W.answer_length(W.pack_hello([TAG])) == W.RESP.size + 4
  whole = W.pack_answer(1, TAG, W.STATUS_OK, 0, bytes(W.PACKET - W.RESP.size))
  assert W.answer_length(whole) == len(whole) == W.PACKET + 1
  with pytest.raises(ValueError):
    W.answer_length(b"XXXX" + whole[4:])


def answering(link, status=W.STATUS_OK, values=(0.1, 0.8, 0.1)):
  def put(msg):
    seq, tag, _, _, _ = W.unpack_request(msg)
    link.handle(W.pack_answer(seq, tag, status, 900, np.asarray(values, np.float32).tobytes()))
  link._requests = SimpleNamespace(put_nowait=put)


def test_answers_only_for_a_fresh_hello_listing_the_model(tmp_path, no_threads):
  clock = Clock()
  link = AddonLink(function(tmp_path), host_ready=lambda: False, clock=clock)
  answering(link)
  image = np.zeros((3, 2, 2), np.uint8)
  assert link.infer(TAG, image, 3) is None
  link.handle(W.pack_hello([TAG + 1]))
  assert link.infer(TAG, image, 3) is None
  link.handle(W.pack_hello([TAG]))
  assert np.allclose(link.infer(TAG, image, 3), (0.1, 0.8, 0.1))
  assert link.infer(TAG, image, 4) is None   # wrong size
  clock.t += link_mod.HELLO_FRESH_S + 1
  assert link.infer(TAG, image, 3) is None


def test_a_miss_rests_the_link(tmp_path, no_threads, monkeypatch):
  clock = Clock()
  link = AddonLink(function(tmp_path), host_ready=lambda: False, clock=clock)
  link.handle(W.pack_hello([TAG]))
  monkeypatch.setattr(link_mod, "REPLY_TIMEOUT_S", 0.001)
  link._requests = SimpleNamespace(put_nowait=lambda msg: None)   # never answered
  image = np.zeros((3, 2, 2), np.uint8)
  assert link.infer(TAG, image, 3) is None and link.stats["timeouts"] == 1
  link.handle(W.pack_hello([TAG]))   # a late answer or hello keeps it live, but not ready
  assert link.live(TAG) and not link.ready(TAG)
  clock.t += link_mod.MISS_HOLD_S
  link.handle(W.pack_hello([TAG]))
  assert link.ready(TAG)
  answering(link, values=(1.0, 2.0))   # wrong size
  assert link.infer(TAG, image, 3) is None and not link.ready(TAG)


def test_a_yield_answer_is_for_that_request_alone(tmp_path, no_threads):
  link = AddonLink(function(tmp_path), host_ready=lambda: False, clock=Clock())
  link.handle(W.pack_hello([TAG]))
  answering(link, W.STATUS_YIELD, ())
  image = np.zeros((3, 2, 2), np.uint8)
  assert link.infer(TAG, image, 3) is None and not link.yield_seen and link.ready(TAG)
  assert link.stats["yields"] == 1 and link.stats["bad_answers"] == 0
  answering(link)
  assert link.infer(TAG, image, 3) is not None


def test_a_yield_hello_holds_until_a_plain_hello(tmp_path, no_threads):
  link = AddonLink(function(tmp_path), host_ready=lambda: False, clock=Clock())
  link.handle(W.pack_hello([TAG], W.STATUS_YIELD))
  answering(link)
  image = np.zeros((3, 2, 2), np.uint8)
  assert link.infer(TAG, image, 3) is None and link.yield_seen
  link.handle(W.pack_hello([TAG]))
  assert link.infer(TAG, image, 3) is not None


def test_blob_round_trip_is_exact():
  u8 = np.random.default_rng(0).integers(0, 256, (3, 8, 8), dtype=np.uint8)
  assert np.array_equal(blob_to_u8((u8.astype(np.float32) / 255.0)[None]), u8)


class CpuNet:
  def __init__(self):
    self.calls = []

  def setInput(self, blob):
    self.blob = blob

  def forward(self, names=None):
    self.calls.append(names)
    return ("cpu", names)


class FakeLink:
  def __init__(self, flat):
    self.flat, self.sent, self.images = flat, [], []

  def infer(self, tag, image, size):
    self.sent.append((tag, image.shape, size))
    self.images.append(image)
    return self.flat

  def poll(self, tag):
    return True

  def want(self, tag):
    pass


def test_jetson_net_splits_outputs_by_name_and_falls_back():
  outputs = (("output0", (1, 2)), ("stop/output0", (1, 3)))
  flat = np.arange(5, dtype=np.float32)   # name order: output0, stop/output0
  cpu = CpuNet()
  net = JetsonNet(cpu, FakeLink(flat), SHA, outputs, scaled_u8=True)
  net.setInput(np.zeros((1, 3, 4, 4), np.float32))
  speed, stops = net.forward(["output0", "stop/output0"])
  assert speed.tolist() == [[0, 1]] and stops.tolist() == [[2, 3, 4]]
  (only,) = net.forward(["stop/output0"])
  assert only.shape == (1, 3)
  assert net.forward().tolist() == [[0, 1]]
  assert net.link.sent[0] == (TAG, (3, 4, 4), 5)
  assert net.forward(["inner/layer"]) == ("cpu", ["inner/layer"])
  net.link.flat = None
  assert net.forward(["output0"]) == ("cpu", ["output0"])


def test_forward_with_no_names_is_the_first_output_given_not_the_first_by_name():
  outputs = (("scores", (1, 2)), ("boxes", (1, 3)))   # the wire: boxes, then scores
  net = JetsonNet(CpuNet(), FakeLink(np.arange(5, dtype=np.float32)), SHA, outputs)
  net.setInput(np.zeros((1, 3, 4, 4), np.float32))
  assert net.forward().tolist() == [[3, 4]]
  assert net.forward(["boxes"])[0].tolist() == [[0, 1, 2]]


def test_unscaled_blobs_go_as_f32_by_default():
  link = FakeLink(np.arange(2, dtype=np.float32))
  net = JetsonNet(CpuNet(), link, SHA, (("output0", (1, 2)),))
  net.setInput(np.full((1, 3, 4, 4), -0.5, np.float32))
  assert net.forward().tolist() == [[0, 1]]
  assert link.images[0].dtype == np.float32


def test_any_runtime_asks_the_host_first_and_runs_its_own_model_otherwise():
  outputs = (("scores", (1, 2)), ("boxes", (1, 3)))   # the wire: boxes, then scores
  link = FakeLink(np.arange(5, dtype=np.float32))
  ran = []
  model = HostModel(link, SHA, outputs, local=lambda blob: ran.append(blob.shape) or ("own", "model"))
  blob = np.zeros((1, 3, 4, 4), np.float32)
  scores, boxes = model(blob)
  assert scores.tolist() == [[3, 4]] and boxes.tolist() == [[0, 1, 2]] and not ran
  link.flat = None
  assert model(blob) == ("own", "model") and ran == [(1, 3, 4, 4)]
  model.enabled = lambda: False
  link.flat = np.arange(5, dtype=np.float32)
  assert model(blob) == ("own", "model") and ran == [(1, 3, 4, 4)] * 2


def test_reports_go_to_the_callback(tmp_path, no_threads):
  heard = []
  clock = Clock()
  link = AddonLink(function(tmp_path), host_ready=lambda: False, report=heard.append, clock=clock)
  clock.t += link_mod.REPORT_S
  link.infer(TAG, np.zeros((3, 2, 2), np.uint8), 3)
  assert heard and heard[0]["link"]["sent"] == 0 and "connected" in heard[0]


def test_a_report_callback_that_raises_never_reaches_the_reader(tmp_path, no_threads):
  def broken(report):
    raise RuntimeError("the fork's panel")
  clock = Clock()
  link = AddonLink(function(tmp_path), host_ready=lambda: False, report=broken, clock=clock)
  clock.t += link_mod.REPORT_S
  link.handle(W.pack_hello([TAG]))   # the reader's call: it must come back
  assert link.tags == frozenset({TAG})


def test_the_reader_lives_through_an_error_in_a_message(tmp_path, monkeypatch):
  link = AddonLink(function(tmp_path), host_ready=lambda: True, clock=Clock())
  link._ensure_open = lambda: None
  link._rx = link_mod._Endpoint(1000)
  calls = []

  def read_forever():
    calls.append(1)
    if len(calls) == 1:
      raise RuntimeError("deque mutated during iteration")
    raise SystemExit   # ends the test's loop
  monkeypatch.setattr(link, "_read_forever", read_forever)
  monkeypatch.setattr(link, "_close", lambda *a: calls.append("closed"))
  monkeypatch.setattr(link_mod, "_mask_signals", lambda: None)
  with pytest.raises(SystemExit):
    link._forever(link._read_forever, "reader")
  assert calls == [1, "closed", 1]


def test_round_trips_can_be_taken_while_added(tmp_path):
  import threading
  trips = link_mod.RoundTrips()
  stop = threading.Event()

  def add():
    while not stop.is_set():
      trips.add(1000, 500)
  adder = threading.Thread(target=add)
  adder.start()
  try:
    for _ in range(2000):
      trips.take()
  finally:
    stop.set()
    adder.join()


def test_two_threads_open_a_shared_link_once(tmp_path, monkeypatch):
  import threading
  link = AddonLink(function(tmp_path), host_ready=lambda: True, clock=Clock())
  link._threads_started = True   # no reader on these plain files
  entered, release, opens = threading.Event(), threading.Event(), []
  real = link._open_endpoints

  def slow_open():
    opens.append(1)
    entered.set()
    release.wait(5)
    return real()
  monkeypatch.setattr(link, "_open_endpoints", slow_open)
  first = threading.Thread(target=link._ensure_open)
  first.start()
  entered.wait(5)
  link._ensure_open()   # the other thread: does not open a second pair
  release.set()
  first.join()
  assert len(opens) == 1 and link._rx is not None
  link._close()


def test_a_failure_the_server_says_to_retry_is_left_for_the_next_run():
  from jetlink.client import AddonFailed, JetlinkClient
  from jetlink.transport.base import LinkError
  client = JetlinkClient.__new__(JetlinkClient)

  def provide(state):
    def run(*args, **kwargs):
      client._engine_state = state
      raise LinkError("engine failed")
    return run
  client._provide = provide({"state": "failed", "detail": "out of memory", "retry": True})
  with pytest.raises(LinkError) as raised:
    client.ensure_addon("a" * 64, 1, "model.onnx", (0xFF, 1, 2))
  assert not isinstance(raised.value, AddonFailed)
  client._provide = provide({"state": "failed", "detail": "outputs are not float"})
  with pytest.raises(AddonFailed):
    client.ensure_addon("a" * 64, 1, "model.onnx", (0xFF, 1, 2))


def test_hello_tags_unpack_many():
  tags = list(range(10))
  payload = struct.pack("<10I", *tags)
  assert W.hello_tags(payload, 10) == frozenset(tags)


def test_the_report_names_models_the_host_lacks(tmp_path, no_threads):
  heard = []
  clock = Clock()
  link = AddonLink(function(tmp_path), host_ready=lambda: False, report=heard.append, clock=clock)
  JetsonNet(CpuNet(), link, SHA, (("output0", (1, 3)),))
  assert link.report()["missing"] == []   # no host, nothing to compare
  link.handle(W.pack_hello([TAG + 1]))
  assert link.report()["missing"] == [TAG]
  clock.t += link_mod.REPORT_S
  link.handle(W.pack_hello([TAG]))   # a hello reports, with no caller
  assert heard and heard[-1]["missing"] == [] and heard[-1]["connected"]


def test_one_link_per_interface_and_the_plain_net_on_failure(tmp_path, monkeypatch):
  monkeypatch.setattr(link_mod, "_shared", {})
  fn = function(tmp_path)
  a = link_mod.with_host(CpuNet(), fn, SHA, (("output0", (1, 3)),))
  b = link_mod.with_host(CpuNet(), fn, SHA, (("output0", (1, 3)),))
  assert isinstance(a, JetsonNet) and a.link is b.link
  cpu = CpuNet()
  assert link_mod.with_host(cpu, fn, "not hex", (("output0", (1, 3)),)) is cpu


def test_no_image_is_built_while_the_host_is_not_ready(tmp_path, monkeypatch, no_threads):
  built = []
  monkeypatch.setattr(link_mod, "blob_to_u8", lambda blob: built.append(blob))
  link = AddonLink(function(tmp_path), host_ready=lambda: True, clock=Clock())
  net = JetsonNet(CpuNet(), link, SHA, (("output0", (1, 3)),), scaled_u8=True)
  net.setInput(np.zeros((1, 3, 4, 4), np.float32))
  assert net.forward() == ("cpu", None)
  assert built == [] and link._rx is not None   # the link still opens, to hear a hello
  link._close()


def test_a_switched_off_net_never_asks_and_switches_back_on(tmp_path, no_threads):
  on = [False]
  link = FakeLink(np.arange(3, dtype=np.float32))
  net = JetsonNet(CpuNet(), link, SHA, (("output0", (1, 3)),), enabled=lambda: on[0])
  net.setInput(np.zeros((1, 3, 4, 4), np.float32))
  assert net.forward() == ("cpu", None) and link.sent == []
  on[0] = True
  assert net.forward().tolist() == [[0, 1, 2]] and len(link.sent) == 1


def test_a_broken_switch_is_off():
  def broken():
    raise OSError("no params")
  net = JetsonNet(CpuNet(), FakeLink(None), SHA, (("output0", (1, 3)),), enabled=broken)
  assert not net.on()


def test_attributes_set_on_the_wrapper_land_on_the_net():
  cpu = CpuNet()
  net = JetsonNet(cpu, FakeLink(None), SHA, (("output0", (1, 3)),))
  net.custom = 7
  assert cpu.custom == 7 and net.custom == 7
  net.blob = None
  assert not hasattr(cpu, "blob")
