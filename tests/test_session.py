"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

End-to-end: a real client talking to a real Session over a real socket.

Only the TensorRT engine is faked. Everything else - framing, the infer
request/response encoding, the history queues, the hidden-state feedback, the
piggybacked telemetry - is the code that will run in the car. This is the test
that covers the paths the car would otherwise be the first to execute.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from tests.fake_trt import install_stubs

install_stubs()

from jetlink import protocol as P                          # noqa: E402
from jetlink.client import JetlinkClient                   # noqa: E402
from jetlink.queues import PolicyQueues                    # noqa: E402
from jetlink.server.builder import EngineCache             # noqa: E402
from jetlink.server.session import ModelSlot, Session      # noqa: E402
from jetlink.spec import ModelSpec                         # noqa: E402
from jetlink.transport.base import LinkError               # noqa: E402
from jetlink.transport.tcp import TcpTransport             # noqa: E402

BIG = {
  'img': (1, 12, 128, 256), 'big_img': (1, 12, 128, 256),
  'desire_pulse': (1, 33, 8), 'traffic_convention': (1, 2),
  'action_t': (1, 2), 'features_buffer': (1, 32, 32, 512),
}


def make_spec() -> ModelSpec:
  return ModelSpec(sha256='b' * 64, nbytes=1234, frame_skip=4, input_shapes=BIG,
                   output_shapes={'outputs': (1, 18452)},
                   output_slices={'hidden_state': slice(2066, 18450),
                                  'plan': slice(917, 1907)},
                   checkpoint=None)


class FakeEngine:
  """Returns something deterministic that depends on the inputs, so a wiring
  mistake between the queues and the engine cannot pass unnoticed."""

  def __init__(self, spec: ModelSpec):
    self.spec = spec
    self.inputs = {k: None for k in spec.input_shapes}
    self._host = {k: np.zeros(v, np.float16) for k, v in spec.input_shapes.items()}
    self._out = np.zeros(spec.output_nelem, np.float16)
    self.last_gpu_us = 1234
    self.calls = 0
    self.nonfinite = False

  def host_input(self, name):
    return self._host[name]

  def run(self):
    self.calls += 1
    # Fold a couple of inputs into the output so the test can verify the queues
    # actually fed the engine. Sample rather than reduce: a float16 sum over
    # 393216 elements overflows to inf, which the server correctly rejects.
    self._out[:] = self._host['img'][0, 0, 0, 0]
    self._out[0] = self._host['features_buffer'][0, 0, 0, 0]
    self._out[1] = np.float16(self.calls)
    self._out[2] = self._host['img'][0, 6, 0, 0]
    if self.nonfinite:
      self._out[5] = np.float16('nan')
    return {'outputs': self._out}


def ready_session(spec, transport, engine=None):
  session = Session(transport, EngineCache('/tmp/jetlink-test-cache'))
  engine = engine or FakeEngine(spec)
  slot = ModelSlot(spec)
  slot.state = 'ready'
  slot.engine = engine
  slot.queues = PolicyQueues(spec)
  slot.host_inputs = {n: engine.host_input(n) for n in spec.input_shapes}
  session.slot = slot
  return session, engine


@pytest.fixture
def link():
  spec = make_spec()
  srv = TcpTransport.listen('127.0.0.1', 0)
  port = srv.getsockname()[1]
  client_t = TcpTransport.connect('127.0.0.1', port)
  server_t, _ = TcpTransport.accept(srv)
  srv.close()

  session, engine = ready_session(spec, server_t)
  thread = threading.Thread(target=session.serve_forever, daemon=True)
  thread.start()

  client = JetlinkClient(client_t, deadline=10.0)
  client.spec = spec
  yield client, session, engine, spec
  client.close()
  server_t.close()


def test_infer_round_trip(link):
  client, session, engine, spec = link
  rng = np.random.default_rng(0)
  warped = rng.integers(0, 256, spec.warped_shape, dtype=np.uint8)
  packed = np.zeros(spec.packed_nelem, np.float32)

  out = client.infer(warped, packed, frame_id=7)
  assert out.shape == (spec.output_nelem,)
  assert out.dtype == np.float32
  assert engine.calls == 1
  assert out[1] == 1  # the engine saw exactly one call
  gpu_us, queue_us, total_us = client.last_timings
  assert gpu_us == 1234 and total_us >= 0


def test_hidden_state_feeds_back_into_the_queues(link):
  """prev_feat goes out, comes back as features_buffer on the next frame."""
  client, session, engine, spec = link
  warped = np.zeros(spec.warped_shape, np.uint8)
  packed = np.zeros(spec.packed_nelem, np.float32)
  hidden = spec.output_slices['hidden_state']

  client.infer(warped, packed, frame_id=1)
  packed[-(hidden.stop - hidden.start):] = 3.0   # a distinctive prev_feat
  client.infer(warped, packed, frame_id=2)

  # feat_q is sampled at logical [0, 4, ... 124] of 128, so a value pushed now
  # takes several frames to appear. Drive it until it does.
  seen = False
  for i in range(3, 200):
    out = client.infer(warped, packed, frame_id=i)
    if float(out[0]) == 3.0:
      seen = True
      break
  assert seen, "prev_feat never reached the engine's features_buffer"


def test_queues_reset_flag_clears_history(link):
  client, session, engine, spec = link
  warped = np.full(spec.warped_shape, 9, np.uint8)
  packed = np.zeros(spec.packed_nelem, np.float32)
  for i in range(6):
    client.infer(warped, packed, frame_id=i)
  assert session.slot.queues.img_q.buf.any()
  client.infer(np.zeros(spec.warped_shape, np.uint8), packed, frame_id=99, reset=True)
  assert not session.slot.queues.img_q.buf.any()


def test_nonfinite_output_is_reported_not_returned(link):
  """openpilot treats a non-finite big-model output as fatal; so must we."""
  client, session, engine, spec = link
  engine.nonfinite = True
  with pytest.raises(LinkError, match='NOT_FINITE'):
    client.infer(np.zeros(spec.warped_shape, np.uint8),
                 np.zeros(spec.packed_nelem, np.float32))


def test_telemetry_piggybacks_without_an_extra_round_trip(link):
  client, session, engine, spec = link
  warped = np.zeros(spec.warped_shape, np.uint8)
  packed = np.zeros(spec.packed_nelem, np.float32)

  client.infer(warped, packed, want_state=False)
  assert client.last_state is None
  client.infer(warped, packed, want_state=True)
  assert client.last_state is not None
  assert 'tempC' in client.last_state and 'pcieLtssm' in client.last_state


def test_not_ready_is_reported_rather_than_crashing():
  spec = make_spec()
  srv = TcpTransport.listen('127.0.0.1', 0)
  port = srv.getsockname()[1]
  client_t = TcpTransport.connect('127.0.0.1', port)
  server_t, _ = TcpTransport.accept(srv)
  srv.close()
  session = Session(server_t, EngineCache('/tmp/jetlink-test-cache'))  # no slot
  threading.Thread(target=session.serve_forever, daemon=True).start()

  client = JetlinkClient(client_t, deadline=10.0)
  client.spec = spec
  try:
    with pytest.raises(LinkError, match='NOT_READY'):
      client.infer(np.zeros(spec.warped_shape, np.uint8),
                   np.zeros(spec.packed_nelem, np.float32))
  finally:
    client.close()
    server_t.close()


def test_ping_and_state_requests(link):
  client, session, engine, spec = link
  assert client.ping(timeout=5) < 5.0
  state = client.state(timeout=5)
  assert state['engine_state'] == 'ready'
  assert 'frames_served' in state
  hello = client.hello(timeout=5)
  assert hello['protocol'] == P.VERSION


def test_wrong_sized_request_is_rejected(link):
  """A client on a different model must be told, not silently fed garbage.

  The server reads `packed` at an offset derived from its own spec, so a
  mismatched request would have the scalars and the 16384-float hidden state
  read out of the middle of the image - and the result would look finite.
  """
  client, session, engine, spec = link
  seq = client._next_seq()
  client.t.send(P.Msg.INFER_REQ, seq, (P.pack_infer_req(1, 0), b'\x00' * 1000))
  msg = client._expect(P.Msg.INFER_RESP, seq, 5.0)
  _, status, _, _, _ = P.unpack_infer_resp(msg.payload)
  assert status == P.Status.BAD_SHAPE
  assert engine.calls == 0, "the engine must not have run on a malformed request"
