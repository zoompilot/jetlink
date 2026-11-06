"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The comma's client over a live socket pair: the leave, and what a close does
to a connection another process still holds a copy of.
"""
import json
import os
import socket
import threading
import time
import unittest
from unittest import mock

import numpy as np

from jetlink import protocol as P
from jetlink.client import JetlinkClient
from jetlink.transport.base import LinkError
from jetlink.transport.tcp import TcpTransport
from tests.test_protocol import _spec, make_pair, reply


class SocketPairTest(unittest.TestCase):
  def setUp(self):
    self.ours, self.peer = make_pair()
    self.addCleanup(self.ours.close)
    self.addCleanup(self.peer.close)
    self.client = JetlinkClient(self.ours)

  def pong_once(self) -> threading.Thread:
    """The far end answers one PING."""
    def serve():
      msg = self.peer.recv(timeout=2.0)
      assert msg.msg_type == P.Msg.PING
      self.peer.send(P.Msg.PONG, msg.seq)
    t = threading.Thread(target=serve, daemon=True)
    t.start()
    return t


class TheLeave(SocketPairTest):
  def test_a_dead_link_carries_no_leave(self):
    self.client.dead = True
    self.client.leave('lost')
    with self.assertRaises(LinkError):
      self.peer.recv(timeout=0.2)

  def test_it_says_why_with_what_was_measured_and_wants_no_answer(self):
    self.client.leave('behind', frames=3, p50_ms=41.2)
    msg = self.peer.recv(timeout=1.0)
    self.assertEqual(msg.msg_type, P.Msg.LEAVE)
    self.assertEqual(json.loads(bytes(msg.payload)), {'reason': 'behind', 'frames': 3, 'p50_ms': 41.2})
    self.assertFalse(self.client.dead)

  def test_an_older_servers_error_to_it_is_not_this_links_failure(self):
    # a v0.8.0 server answers unknown_message; the next exchange, a hello or a
    # ping, reads the socket next and must not take that as its own failure
    self.client.leave('stopped')
    msg = self.peer.recv(timeout=1.0)
    self.peer.send_json(P.Msg.ERROR, msg.seq, {'error': 'unknown_message', 'detail': 'type 19'})
    t = self.pong_once()
    self.client.ping(timeout=2.0)
    t.join(2.0)
    self.assertFalse(self.client.dead)

  def test_an_error_to_anything_else_still_is(self):
    self.peer.send_json(P.Msg.ERROR, 999, {'error': 'no_hello', 'detail': 'say hello again'})
    with self.assertRaises(LinkError):
      self.client.ping(timeout=1.0)


class ClosingTheSocket(SocketPairTest):
  def test_the_peer_hears_a_close_though_another_process_holds_a_copy(self):
    # the comma's owner keeps a copy of the phone's dial for the drive; a
    # plain close here left the phone connected to nobody until the owner let
    # its copy go, at the next join attempt, seconds to a minute later
    copy = socket.socket(fileno=os.dup(self.ours.sock.fileno()))
    self.addCleanup(copy.close)
    self.ours.close()
    with self.assertRaises(LinkError) as closed:
      self.peer.recv(timeout=1.0)
    self.assertIn('peer closed', str(closed.exception))


class FramesInFlight(SocketPairTest):
  """Frames the comma stops waiting for (a held frame), and how their answers
  are read later."""

  def setUp(self):
    super().setUp()
    self.client.spec = _spec()
    self.client.deadline = 0.5

  def send(self, frame_id: int) -> int:
    spec = self.client.spec
    return self.client.infer_begin(bytes(spec.warped_nbytes), bytes(spec.packed_nbytes), frame_id=frame_id)

  def answer(self, status=P.Status.OK) -> None:
    """The far end takes one request and answers it with its frame id in
    every output."""
    msg = self.peer.recv(timeout=2.0)
    self.assertEqual(msg.msg_type, P.Msg.INFER_REQ)
    frame_id = P.unpack_infer_req(msg.payload)[0]
    payload = reply(np.full(self.client.spec.reply_nelem, frame_id, np.float32), frame_id=frame_id)
    if status != P.Status.OK:
      payload = P.pack_infer_resp(frame_id, status, 0, 0, 0) + payload[P.INFER_RESP_SIZE:]
    self.peer.send(P.Msg.INFER_RESP, msg.seq, (payload,))

  def answering(self, n: int) -> threading.Thread:
    """The far end answering `n` requests as they come: a 393 KB frame does
    not fit a Linux loopback socket's buffers twice over, so a peer that
    only reads afterwards blocks the second send."""
    t = threading.Thread(target=lambda: [self.answer() for _ in range(n)], daemon=True)
    t.start()
    return t

  def test_a_drain_takes_what_has_arrived_and_waits_for_nothing(self):
    t0 = time.monotonic()
    self.assertEqual(self.client.drain(), 0)
    self.assertLess(time.monotonic() - t0, 0.05, 'a drain with nothing to read waited')
    t = self.answering(2)
    self.send(1)
    self.send(2)
    t.join(2.0)
    for _ in range(50):   # the replies cross a socket; give them a moment
      if self.client.drain():
        break
      time.sleep(0.005)
    self.assertEqual(self.client.waiting_for(), 0.0)
    self.assertEqual(self.client.last_output[0], 2.0, 'the newest reply is what a held frame publishes')
    self.assertFalse(self.client.dead)

  def test_a_frame_given_up_on_is_read_quietly_by_the_next(self):
    seq1 = self.send(1)
    request = self.peer.recv(timeout=2.0)   # not answered yet
    self.assertIsNone(self.client.infer_end(seq1, hold=0.01))
    self.assertFalse(self.client.dead, 'a hold that passed is not a failure')
    self.assertGreater(self.client.waiting_for(), 0.0)
    # the late answer lands, then the next frame goes out and is answered
    frame_id = P.unpack_infer_req(request.payload)[0]
    self.peer.send(P.Msg.INFER_RESP, request.seq, (reply(np.ones(self.client.spec.reply_nelem), frame_id=frame_id),))
    seq2 = self.send(2)
    self.answer()
    with self.assertNoLogs('jetlink.client', level='WARNING'):
      out = self.client.infer_end(seq2)
    self.assertEqual(out[0], 2.0)
    self.assertEqual(self.client.waiting_for(), 0.0)

  def test_a_quiet_host_shows_in_how_long_the_oldest_frame_has_waited(self):
    self.send(1)
    time.sleep(0.02)
    self.assertGreater(self.client.waiting_for(), 0.015)
    self.assertEqual(self.client.drain(), 0)

  def test_a_host_that_answers_nothing_for_the_deadline_fails_the_link_before_the_next_frame(self):
    # held frames keep going out without waiting, so a quiet host shows
    # here, not in a wait
    self.client.deadline = 0.05
    self.send(1)
    time.sleep(0.06)
    with self.assertRaises(LinkError) as quiet:
      self.send(2)
    self.assertIn('no answer to 1 frames', str(quiet.exception))
    self.assertTrue(self.client.dead)

  def test_a_frame_the_server_failed_fails_the_link_when_drained(self):
    self.send(1)
    self.answer(status=P.Status.INFER_FAILED)
    time.sleep(0.02)
    with self.assertRaises(LinkError):
      self.client.drain()
    self.assertTrue(self.client.dead)

  def test_a_frame_the_server_found_not_finite_is_raised_and_the_link_kept(self):
    # the server's session goes on, keeping nothing of that frame
    # (Session.infer), so the next frame is answered over the same link
    for i, status in enumerate((P.Status.OK, P.Status.NOT_FINITE, P.Status.OK), 1):
      seq = self.send(i)
      self.answer(status)
      if status == P.Status.OK:
        self.assertEqual(self.client.infer_end(seq)[0], i)
        continue
      with self.assertRaisesRegex(P.NotFinite, 'frame 2'):
        self.client.infer_end(seq)
      self.assertFalse(self.client.dead)
      self.assertEqual(self.client.waiting_for(), 0.0)
      self.assertEqual(self.client.last_output[0], 1.0, 'a held frame publishes the last good one')

  def test_a_hello_forgets_the_frames_of_the_session_before(self):
    self.send(1)

    def serve():
      msg = self.peer.recv(timeout=2.0)   # the frame
      msg = self.peer.recv(timeout=2.0)
      assert msg.msg_type == P.Msg.HELLO_REQ
      self.peer.send_json(P.Msg.HELLO_RESP, msg.seq, {'device': 'test'})
    t = threading.Thread(target=serve, daemon=True)
    t.start()
    self.client.hello(timeout=2.0)
    t.join(2.0)
    self.assertEqual(self.client.waiting_for(), 0.0)
    self.assertIsNone(self.client.last_output)

  def test_a_hello_says_what_the_comma_knows_of_its_cameras(self):
    said = []

    def serve():
      for _ in range(2):
        msg = self.peer.recv(timeout=2.0)
        said.append(json.loads(bytes(msg.payload))['client'])
        self.peer.send_json(P.Msg.HELLO_RESP, msg.seq, {'device': 'test'})
    t = threading.Thread(target=serve, daemon=True)
    t.start()
    self.client.hello(timeout=2.0, camera={'wide_from_device_euler': [0.0, 0.02, -0.016]})
    self.client.hello(timeout=2.0)
    t.join(2.0)
    self.assertEqual(said[0]['camera'], {'wide_from_device_euler': [0.0, 0.02, -0.016]})
    self.assertNotIn('camera', said[1], 'a comma that knows nothing says nothing')


if __name__ == '__main__':
  unittest.main()


class FramesAsDatagrams(FramesInFlight):
  """Over the cable a phone's server offers a UDP port in its hello, and
  frames the comma may lose go there as datagrams; everything else, and any
  frame it cannot lose, stays on the stream. Loopback stands in for the
  cable."""

  def setUp(self):
    super().setUp()
    cable = mock.patch.object(TcpTransport, 'on_the_cable', return_value=True)
    cable.start()
    self.addCleanup(cable.stop)
    self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.udp.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
    self.udp.bind(('127.0.0.1', 0))
    self.udp.settimeout(2.0)
    self.addCleanup(self.udp.close)

  def greet(self, **offer) -> None:
    """A hello the far end answers with `offer` in it."""
    def serve():
      msg = self.peer.recv(timeout=2.0)
      self.peer.send_json(P.Msg.HELLO_RESP, msg.seq, {'device': 'test', **offer})
    t = threading.Thread(target=serve, daemon=True)
    t.start()
    self.client.hello(timeout=2.0)
    t.join(2.0)

  def offer(self) -> dict:
    return {'frame_port': self.udp.getsockname()[1], 'frame_token': 77}

  def lose(self, frame_id: int) -> int:
    spec = self.client.spec
    return self.client.infer_begin(bytes(spec.warped_nbytes), bytes(spec.packed_nbytes), frame_id=frame_id,
                                   skip_if_busy=True)

  def take_frame(self) -> tuple[int, bytes]:
    """One frame off the UDP socket, put back together: (seq, its stream bytes)."""
    got, total, seq = {}, None, None
    while total is None or sum(len(b) for b in got.values()) < total:
      datagram = self.udp.recv(1 << 16)
      token, seq, offset, total = P.unpack_datagram_header(datagram)
      self.assertEqual(token, 77)
      got[offset] = datagram[P.DATAGRAM_HEADER_SIZE:]
    return seq, b''.join(got[o] for o in sorted(got))

  def test_an_offer_sends_a_frame_that_may_be_lost_as_the_streams_bytes(self):
    self.greet(**self.offer())
    self.assertTrue(self.client.t.datagrams)
    seq = self.lose(5)
    got_seq, data = self.take_frame()
    self.assertEqual(got_seq, seq)
    spec = self.client.spec
    stream = b''.join(bytes(b) for b in self.client.t._frame(
      P.Msg.INFER_REQ, seq, (P.pack_infer_req(5, 0), bytes(spec.warped_nbytes), bytes(spec.packed_nbytes)), 0))
    self.assertEqual(data, stream)
    self.assertGreater(len(P.datagram_pieces(len(stream))), 1)
    with self.assertRaises(LinkError):
      self.peer.recv(timeout=0.1)   # nothing on the stream

  def test_a_frame_that_cannot_be_lost_stays_on_the_stream(self):
    self.greet(**self.offer())
    self.send(1)
    self.assertEqual(self.peer.recv(timeout=2.0).msg_type, P.Msg.INFER_REQ)

  def test_no_offer_no_cable_or_turned_off_keeps_frames_on_the_stream(self):
    self.greet()
    self.assertFalse(self.client.t.datagrams)
    self.client.allow_datagrams = False
    self.greet(**self.offer())
    self.assertFalse(self.client.t.datagrams)
    self.client.allow_datagrams = True
    with mock.patch.object(TcpTransport, 'on_the_cable', return_value=False):
      self.greet(**self.offer())
    self.assertFalse(self.client.t.datagrams, 'off the cable')
    self.lose(1)
    self.assertEqual(self.peer.recv(timeout=2.0).msg_type, P.Msg.INFER_REQ)

  def test_a_lost_frame_is_taken_out_of_flight_by_the_next_reply(self):
    self.greet(**self.offer())
    first = self.lose(1)
    self.take_frame()   # the phone never got it whole
    self.assertIsNone(self.client.infer_end(first, hold=0.01))
    second = self.lose(2)
    seq, _ = self.take_frame()
    self.peer.send(P.Msg.INFER_RESP, seq, (reply(np.full(self.client.spec.reply_nelem, 2.0), frame_id=2),))
    self.assertEqual(self.client.infer_end(second)[0], 2.0)
    self.assertEqual((self.client.frames_lost, self.client.waiting_for()), (1, 0.0))
    self.assertTrue(self.client.t.datagrams, 'a lost frame is a held one, nothing more')

  def test_a_send_the_kernel_refuses_fails_the_link(self):
    # the phone's port is gone with its server, and the stream with it
    port = self.udp.getsockname()[1]
    self.udp.close()
    self.greet(frame_port=port, frame_token=77)
    with self.assertRaises(LinkError):
      for frame_id in range(1, 4):
        self.lose(frame_id)
    self.assertTrue(self.client.dead)
