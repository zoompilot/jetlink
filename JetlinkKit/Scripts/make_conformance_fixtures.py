#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

What the comma's Python does, written down for the Swift server to match.
Every file comes from the code the comma runs, on fixed inputs:

  wire      JetlinkKit/Tests/JetlinkServerTests/Fixtures/conformance/wire*
            headers and INFER bodies from protocol.py, and the byte streams
            StreamTransport frames for TCP, for a USB host and for the gadget
            (FfsTransport's 16 KB bursts)
  staging   .../conformance/staging*: the tensors PolicyQueues.step feeds for
            the tiny queued graph at frame_skip 1, 2 and 4, with the hidden
            state each frame's output feeds back, a hello, a non-finite frame,
            a reset, and desires with NaNs and signed zeros included
  layout    .../conformance/layout.json: where a reply leaves hidden_state out,
            for slices with open ends and ends counted from the back, and
            whether a queued graph's queues can feed it back
  registry  tests/fixtures/conformance/registry.json: LFS pointers, model
            identities, catalog parsing and merging. Its `cache` block (one
            cache directory's catalog and inventory payloads) was written by
            the Python server's registry and is frozen: this copies it through.

  .venv/bin/python JetlinkKit/Scripts/make_conformance_fixtures.py [--root DIR]

from the root of this checkout. --root writes the same tree somewhere else;
tests/test_conformance.py does that and compares byte for byte. The server's
own goldens beside these (stats.json, the golden frames, the control events)
are the Swift's, written once and never regenerated (docs/conformance.md).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# This checkout's jetlink and tests, ahead of any jetlink the environment has
# installed: a venv's editable install can point at another checkout.
sys.path.insert(0, str(ROOT))

SERVER = Path('JetlinkKit/Tests/JetlinkServerTests/Fixtures/conformance')
REGISTRY = Path('tests/fixtures/conformance/registry.json')

SHA_A = 'a086d5249fc308bb73993d1e64630c669d4c7df5bde85f42ad61902543648525'
REF_A = 'f877d7a0ccc3cce943c76e285214c020cd65c899'


def dump(value) -> str:
  return json.dumps(value, indent=2, sort_keys=True) + '\n'


# -- wire ---------------------------------------------------------------------

def payload(seq: int, n: int) -> bytes:
  """The bytes of a binary payload; the Swift tests make the same ones."""
  return bytes((seq * 31 + i * 7) % 251 for i in range(n))


# (type, seq, flags, parts) with parts a list of lengths, or a JSON text. The
# lengths put header plus payload on each side of the 512 and 1024 byte packets
# and the 16 KB burst, where the PADDED byte and the gadget's padding change.
WIRE_MESSAGES = [
  ('HELLO_REQ', 1, 0, '{"client":{"name":"modeld","nonce":7}}'),
  ('HELLO_RESP', 1, 0, '{"protocol":3,"engine_state":"none","sleep_after":0.0}'),
  ('PING', 2, 0, []),
  ('PONG', 2, 0, []),
  ('ENGINE_REQ', 3, 0, '{"sha256":"' + SHA_A + '","nbytes":765953504,"frame_skip":4}'),
  ('INFER_REQ', 4, 3, [8, 800, 184]),       # 1024 on the wire: padded
  ('INFER_RESP', 4, 0, [20, 971]),          # 1023
  ('UPLOAD_CHUNK', 5, 0, [8, 985]),         # 1025
  ('UPLOAD_CHUNK', 6, 0, [8, 2008]),        # 2048: padded
  ('INFER_RESP', 12, 0, [20, 460]),         # 512, a high-speed packet: padded
  ('UPLOAD_CHUNK', 13, 0, [8, 1496]),       # 1536: padded
  ('INFER_RESP', 14, 0, [20, 461]),         # 513
  ('STATE_RESP', 7, 0, [16352]),            # 16384: padded, and a whole burst
  ('ERROR', 8, 0, [16353]),                 # 16385
  ('INFER_RESP', 9, 0, [20, 2066 * 4, 2 * 4, 64]),  # the big models' outputs either side of hidden_state, telemetry
  ('INFER_RESP', 11, 0, [20, 73808]),       # the whole vector, on WANT_HIDDEN
  ('SHUTDOWN_REQ', 10, 0, []),
  ('PROGRESS', 0, 0, '{"stage":"build","frac":0.5,"msg":"half"}'),
]

def _memory(tx_align: int = 0):
  """A transport over bytes in memory that pads what it sends to `tx_align`
  (0: the PADDED byte) and reads what it is given whole, as the comma does."""
  from jetlink.transport.base import LinkError, StreamTransport

  class Memory(StreamTransport):
    def __init__(self, incoming: bytes = b''):
      super().__init__()
      self.sent = bytearray()
      self.incoming = memoryview(incoming)
      self.pos = 0

    def _write(self, bufs) -> int:
      n = 0
      for b in bufs:
        self.sent += b
        n += memoryview(b).nbytes
      return n

    def _read_into(self, dest, timeout) -> int:
      left = self.incoming.nbytes - self.pos
      if left <= 0:
        raise LinkError('end of the fixture stream')
      n = min(dest.nbytes, left)
      dest[:n] = self.incoming[self.pos:self.pos + n]
      self.pos += n
      return n

    def close(self) -> None:
      pass

  Memory.tx_align = tx_align
  return Memory


class _UsbHostReader:
  """How the USB host reads the gadget's stream: each message padded to the
  gadget's 16 KB burst, so every one ends on a whole packet. No Python reads
  as a USB host any more; the Swift server is held to this."""

  def __init__(self, stream: bytes):
    self.stream, self.pos = memoryview(stream), 0

  def recv(self, timeout=None):
    from jetlink import protocol as P
    from jetlink.transport.base import Message
    start = self.pos
    _, _, msg_type, seq, flags, length, _ = P.unpack_header(self.stream[start:start + P.HEADER_SIZE])
    end = start + P.HEADER_SIZE + length
    self.pos = end + -(end - start) % P.GADGET_TX_ALIGN
    assert self.pos <= len(self.stream) and self.pos % P.USB_MAX_PACKET == 0
    return Message(msg_type, seq, flags, self.stream[start + P.HEADER_SIZE:end])


def _message_bytes(spec) -> tuple[list[bytes], bytes]:
  _, seq, _, parts = spec
  if isinstance(parts, str):
    return [parts.encode()], parts.encode()
  body = payload(seq, sum(parts))
  out, at = [], 0
  for n in parts:
    out.append(body[at:at + n])
    at += n
  return out, body


def wire(root: Path) -> None:
  from jetlink import protocol as P
  from jetlink.transport.ffs import FfsTransport
  from jetlink.transport.tcp import TcpTransport

  host, gadget, tcp = _memory(0), _memory(FfsTransport.tx_align), _memory(TcpTransport.tx_align)
  out = root / SERVER
  out.mkdir(parents=True, exist_ok=True)

  headers = []
  for i, msg in enumerate(P.Msg):
    fields = {'msg_type': int(msg), 'seq': 1000 * i + 7, 'flags': [0, 1, 2, 128, 131][i % 5],
              'length': [0, 1, 1023, 1 << 20, 16 << 20][i % 5], 'reserved': [0, 99, 2 ** 63][i % 3]}
    raw = P.pack_header(fields['msg_type'], fields['seq'], fields['length'], fields['flags'], fields['reserved'])
    headers.append({'name': msg.name, **fields, 'hex': raw.hex()})
  infer_req = [{'frame_id': f, 'flags': fl, 'hex': P.pack_infer_req(f, fl).hex()}
               for f, fl in ((0, 0), (1, 1), (4242, 3), (2 ** 32 - 1, 2))]
  infer_resp = [{'frame_id': f, 'status': int(s), 'gpu_us': g, 'queue_us': q, 'total_us': t,
                 'hex': P.pack_infer_resp(f, s, g, q, t).hex()}
                for f, s, g, q, t in ((42, P.Status.NOT_FINITE, 28000, 130, 29000), (0, P.Status.NOT_READY, 0, 0, 0),
                                      (2 ** 32 - 1, P.Status.OK, 2 ** 32 - 1, 1, 2 ** 31))]

  # usb_host: what a USB host sends and the gadget reads; usb_gadget: the
  # other way, in whole bursts; tcp: both ways over a socket
  senders = {'tcp': (tcp, tcp), 'usb_host': (host, gadget), 'usb_gadget': (gadget, _UsbHostReader)}
  streams = {}
  for name, (sender_cls, receiver_cls) in senders.items():
    sender = sender_cls()
    for spec in WIRE_MESSAGES:
      parts, _ = _message_bytes(spec)
      sender.send(P.Msg[spec[0]], spec[1], parts, spec[2])
    stream = bytes(sender.sent)
    (out / f'wire.{name}.bin').write_bytes(stream)

    # read it back the way the other end does: the fixture must be a stream
    # the receiving side takes in whole
    receiver = receiver_cls(stream)
    offsets = []
    for spec in WIRE_MESSAGES:
      m = receiver.recv(timeout=None)
      _, body = _message_bytes(spec)
      assert (m.msg_type, m.seq, bytes(m.payload)) == (P.Msg[spec[0]], spec[1], body), (name, spec[0])
      offsets.append(receiver.pos)
    assert receiver.pos == len(stream), name
    streams[name] = {'file': f'wire.{name}.bin', 'bytes': len(stream), 'ends': offsets}

  messages = [{'type': P.Msg[t].value, 'name': t, 'seq': s, 'flags': f,
               'parts': p if isinstance(p, list) else None, 'json': p if isinstance(p, str) else None}
              for t, s, f, p in WIRE_MESSAGES]
  (out / 'wire.json').write_text(dump({
    'payload': 'byte i of a binary payload is (seq * 31 + i * 7) % 251',
    'headers': headers, 'infer_req': infer_req, 'infer_resp': infer_resp,
    'messages': messages, 'streams': streams, 'datagrams': _datagrams(out),
  }))


# A frame the comma sends as datagrams: three pieces, the last message padded
DATAGRAM_FRAME = ('INFER_REQ', 21, 1, [8, 140000, 184])
DATAGRAM_TOKEN = 0xC0FFEE11


def _datagrams(out: Path) -> dict:
  """Datagram headers, how messages are cut, and the datagrams
  TcpTransport.send_datagrams sends for DATAGRAM_FRAME, in wire.datagrams.bin
  one after another."""
  from jetlink import protocol as P
  from jetlink.transport.tcp import TcpTransport, datagrams

  headers = [{'token': t, 'seq': q, 'offset': o, 'total': n, 'hex': P.pack_datagram_header(t, q, o, n).hex()}
             for t, q, o, n in ((1, 1, 0, 32), (DATAGRAM_TOKEN, 2 ** 32 - 1, 65000, 398376), (2 ** 32 - 1, 7, 7, 16 << 20))]
  pieces = [{'total': n, 'pieces': [list(p) for p in P.datagram_pieces(n)]} for n in (32, 65000, 65001, 130001, 398376)]

  parts, _ = _message_bytes(DATAGRAM_FRAME)
  stream = _memory(TcpTransport.tx_align)()
  bufs = stream._frame(P.Msg[DATAGRAM_FRAME[0]], DATAGRAM_FRAME[1], parts, DATAGRAM_FRAME[2])
  sent = [b''.join(bytes(b) for b in d) for d in datagrams(bufs, DATAGRAM_TOKEN, DATAGRAM_FRAME[1])]
  (out / 'wire.datagrams.bin').write_bytes(b''.join(sent))
  name, seq, flags, lengths = DATAGRAM_FRAME
  return {'headers': headers, 'pieces': pieces,
          'frame': {'name': name, 'type': P.Msg[name].value, 'seq': seq, 'flags': flags, 'parts': lengths,
                    'token': DATAGRAM_TOKEN, 'file': 'wire.datagrams.bin', 'sizes': [len(d) for d in sent]}}


# -- staging ------------------------------------------------------------------

STAGING_FRAMES = 12
STAGING_RESET_BEFORE = 8
# a new client: nothing fed back on its first frame, the queues kept
STAGING_HELLO_BEFORE = 5
# an output with a NaN in it: its hidden state is not fed back
STAGING_NOT_FINITE = 3
STAGING_INPUTS = ('img', 'big_img', 'features_buffer', 'desire_pulse', 'traffic_convention', 'action_t')
# Desires whose max over a frame_skip group numpy decides by its own rules: a
# NaN wins and keeps its payload (the first of two), and of two equal values
# the first stays (0 then -0 gives 0, -0 then 0 gives -0). One column each, as
# float32 bits over four frames; frames 0 to 3 and again 8 to 11, after the
# reset, carry them instead of random desires.
DESIRE_EDGES = np.array([
  [0x3F800000, 0x7FC00000, 0x40000000, 0x40400000],   # 1, NaN, 2, 3
  [0x3F800000, 0x7FC02000, 0x7FC04000, 0x40C00000],   # 1, two NaN payloads, 6
  [0x00000000, 0x80000000, 0xBF800000, 0xC0000000],   # 0, -0, -1, -2
  [0x80000000, 0x00000000, 0xBF800000, 0xC0000000],   # -0, 0, -1, -2
  [0x3F800000, 0x40400000, 0x40000000, 0xBF800000],   # 1, 3, 2, -1
  [0xFFC00000, 0x3F800000, 0x40000000, 0x40400000],   # a negative NaN first
  [0xFF800000, 0xC0A00000, 0x7FC00000, 0x40E00000],   # -inf, -5, NaN, 7
  [0x40000000, 0x7F800000, 0x40400000, 0x40800000],   # 2, inf, 3, 4
], np.uint32).view(np.float32).T


def _desire_edges(f: int) -> np.ndarray | None:
  return DESIRE_EDGES[f % 4] if f % 8 < 4 else None


def staging(root: Path) -> None:
  from jetlink.queues import PolicyQueues
  from jetlink.spec import DRIVING_OUTPUT, spec_from_onnx
  from tests import tiny_model

  out = root / SERVER
  out.mkdir(parents=True, exist_ok=True)
  cases = []
  with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / 'tiny_queued.onnx'
    tiny_model.write(path, shapes=True)
    for skip in (1, 2, 4):
      spec = spec_from_onnx(str(path), frame_skip=skip)
      rng = np.random.default_rng(20260927 + skip)
      queues = PolicyQueues(spec)
      frames, staged = bytearray(), bytearray()
      for f in range(STAGING_FRAMES):
        warped = rng.integers(0, 256, spec.warped_shape, dtype=np.uint8)
        packed = (rng.standard_normal(spec.packed_nelem) * 2.0).astype(np.float32)
        edges = _desire_edges(f)
        if edges is not None:
          packed[spec.packed_layout['desire'][0]] = edges
        # what the engine returned for this frame, which the next feeds back
        output = (rng.standard_normal(spec.output_nelem) * 2.0).astype(np.float32)
        if f == STAGING_NOT_FINITE:
          output[spec.hidden_range[0]] = np.nan
        if f == STAGING_RESET_BEFORE:
          queues.reset()
        if f == STAGING_HELLO_BEFORE:
          queues.new_client()
        feed = queues.step(warped, packed)
        if np.all(np.isfinite(output)):
          queues.after_run({DRIVING_OUTPUT: output})
        frames += warped.tobytes() + packed.tobytes() + output.tobytes()
        for name in STAGING_INPUTS:
          assert feed[name].dtype == np.float16, name
          staged += feed[name].tobytes()
      stem = f'staging.fs{skip}'
      (out / f'{stem}.spec.json').write_text(dump(spec.to_dict()))
      (out / f'{stem}.frames.bin').write_bytes(bytes(frames))
      (out / f'{stem}.staged.bin').write_bytes(bytes(staged))
      cases.append({'frame_skip': skip, 'spec': f'{stem}.spec.json', 'frames': f'{stem}.frames.bin',
                    'staged': f'{stem}.staged.bin',
                    'inputs': [{'name': n, 'shape': list(spec.input_shapes[n])} for n in STAGING_INPUTS]})
  (out / 'staging.json').write_text(dump({
    'frames': STAGING_FRAMES, 'reset_before': STAGING_RESET_BEFORE, 'hello_before': STAGING_HELLO_BEFORE,
    'desire_edges': 'frames 0 to 3 and 8 to 11 carry desires with NaNs, signed zeros and infinities',
    'dtype': 'float16',
    'frame_layout': 'warped uint8, packed float32, then the driving output float32 the frame returned, per frame',
    'feedback': 'the output\'s hidden_state is fed back after the frame when every value is finite',
    'staged_layout': 'each input in `inputs` order, float16, per frame',
    'cases': cases,
  }))


# -- layout -------------------------------------------------------------------

# A queued graph as small as the tiny one: 64 output floats, and a hidden state
# of 32 (features_buffer's 4 x 8) for the queues to feed back.
LAYOUT_SPEC = {
  'sha256': 'cd' * 32, 'nbytes': 4096, 'frame_skip': 4, 'checkpoint': None,
  'input_shapes': {'img': [1, 12, 8, 16], 'big_img': [1, 12, 8, 16], 'desire_pulse': [1, 33, 8],
                   'traffic_convention': [1, 2], 'action_t': [1, 2], 'features_buffer': [1, 32, 4, 8]},
  'output_shapes': {'outputs': [1, 64]},
}
# hidden_state as output_slices carries it, [start, stop], each an int or None;
# None for a model with no such slice
LAYOUT_SLICES = [
  [32, 64], [-32, None], [32, None], [-32, 64], [None, 32], [30, -2],
  [40, 100], [-100, 10], [50, 40], [64, None], [None, None], None,
]


def layout(root: Path) -> None:
  from jetlink.queues import PolicyQueues
  from jetlink.spec import ModelSpec

  out = root / SERVER
  out.mkdir(parents=True, exist_ok=True)
  cases = []
  for bounds in LAYOUT_SLICES:
    slices = {'plan': [0, 16], **({'hidden_state': bounds} if bounds is not None else {})}
    spec = ModelSpec.from_dict({**LAYOUT_SPEC, 'output_slices': slices})
    try:
      PolicyQueues(spec)
      feeds_back = True
    except ValueError:
      feeds_back = False
    hidden = spec.hidden_range
    cases.append({'hidden_state': bounds, 'hidden_range': list(hidden) if hidden else None,
                  'reply_nelem': spec.reply_nelem, 'infer_resp_nbytes': spec.infer_resp_nbytes, 'feeds_back': feeds_back})
  (out / 'layout.json').write_text(dump({
    'spec': LAYOUT_SPEC,
    'slices': 'each case adds output_slices {"plan": [0, 16], "hidden_state": its bounds}, or no hidden_state for null',
    'cases': cases,
  }))


# -- registry -----------------------------------------------------------------

INDEX_FILE = 'model_index_v1.json'
SHA_B = 'b' * 64
REF_B, REF_C = 'b' * 40, 'c' * 40


def _index_cases() -> list[tuple[str, object]]:
  ok = {'ref': REF_A, 'oid': SHA_A, 'size': 765953504}

  def models(*more):
    return {'schema': 1, 'models': [ok, *more]}

  return [
    ('fields', {'schema': 1, 'models': [
      {'ref': REF_B, 'oid': SHA_B, 'size': 20, 'name': 'Beta (October 29, 2026)', 'folder': '2026 Deep RL Models', 'short_name': 'B',
       'build_time': '2026-10-29T00:00:00Z', 'index': 20, 'is_20hz': True},
      {**ok, 'pr': 39235}]}),
    ('missing_fields', models({'ref': REF_B, 'oid': SHA_B, 'size': 2, 'name': 7, 'folder': '', 'short_name': None, 'build_time': 1})),
    ('catalog_fields', models({'ref': REF_B, 'oid': SHA_B, 'size': 2, 'is_20hz': False, 'generation': '13', 'environment': 'release',
                               'runner': 'tinygrad'},
                              {'ref': REF_C, 'oid': SHA_B, 'size': 2, 'is_20hz': 'yes', 'generation': 13, 'environment': '', 'runner': None})),
    ('default_listed', {**models({'ref': REF_B, 'oid': SHA_B, 'size': 2}), 'default': REF_B}),
    ('default_not_listed', {**models(), 'default': REF_B, 'refs': {REF_B: {'oid': SHA_B, 'size': 2}}}),
    ('default_rubbish', {**models(), 'default': 7}),
    ('pr_values', models(*({'ref': c * 40, 'oid': SHA_B, 'size': 2, 'pr': pr} for c, pr in zip('bcde', (True, 0, '5', -1), strict=True)))),
    ('no_ref', models({'oid': SHA_B, 'size': 1})),
    ('bad_ref', models({'ref': REF_A.upper(), 'oid': SHA_B, 'size': 1})),
    ('bad_oid', models({'ref': REF_B, 'oid': SHA_B[:63], 'size': 1})),
    ('size_zero', models({'ref': REF_B, 'oid': SHA_B, 'size': 0})),
    ('size_text', models({'ref': REF_B, 'oid': SHA_B, 'size': '1'})),
    ('size_bool', models({'ref': REF_B, 'oid': SHA_B, 'size': True})),
    ('size_fraction', models({'ref': REF_B, 'oid': SHA_B, 'size': 1.5})),
    ('size_big', models({'ref': REF_B, 'oid': SHA_B, 'size': 1 << 40})),
    ('duplicate', models({**ok, 'oid': SHA_B, 'name': 'the second one'})),
    ('rubbish_models', models(1, 'two', None, [])),
    ('refs_and_onnx', {**models(), 'refs': {REF_B: {'oid': SHA_B, 'size': 2, 'method': 'weights'}, REF_C: {'oid': 'x', 'size': 2},
                                           'short': {'oid': SHA_B, 'size': 2}, 'd' * 40: 'junk'},
                       'onnx': {'c' * 64: {'size': 3, 'export': 'e/1/big_driving_supercombo.onnx'}, 'd' * 64: {'size': -1},
                                'e' * 64: 'junk', 'nothex': {'size': 4}}}),
    ('a_listed_ref_wins_over_refs', {**models(), 'refs': {REF_A: {'oid': SHA_B, 'size': 9}}}),
    ('refs_not_an_object', {**models(), 'refs': [REF_B], 'onnx': 'x'}),
    ('schema_2', {**models(), 'schema': 2}),
    ('schema_text', {**models(), 'schema': '1'}),
    ('no_schema', {'models': [ok]}),
    ('nothing_listed', {'schema': 1, 'models': [], 'refs': {REF_B: {'oid': SHA_B, 'size': 2}}}),
    ('models_not_a_list', {'schema': 1, 'models': {'a': ok}}),
    ('not_an_object', ['models']),
  ]


def _parsed(index) -> dict:
  return {'default': index.default, 'models': [vars(m) for m in index.models], 'refs': {r: vars(p) for r, p in sorted(index.refs.items())},
          'sizes': dict(sorted(index.sizes.items()))}


def registry(root: Path) -> None:
  from jetlink.registry.index import IndexInvalid, is_ref, is_sha256, parse_index
  from jetlink.registry.lfs import LFS_ENDPOINTS

  # frozen: the server's own cache layout, which the Swift side writes
  frozen = json.loads((ROOT / REGISTRY).read_text())['cache']
  out = root / REGISTRY
  out.parent.mkdir(parents=True, exist_ok=True)

  def parse(data):
    try:
      return _parsed(parse_index(data))
    except IndexInvalid:
      return None

  identities = [{'value': v, 'is_ref': is_ref(v), 'is_sha256': is_sha256(v)}
                for v in (REF_A, SHA_A, REF_A.upper(), SHA_A[:63], SHA_A + '0', '', 'g' * 40, 'g' * 64)]
  published = json.loads((ROOT / 'tests/fixtures' / INDEX_FILE).read_text())
  # null for an index refused whole: the caller keeps the last one that parsed
  parses = [{'name': 'published', 'index_file': INDEX_FILE, 'expected': parse(published)}]
  parses += [{'name': name, 'index': data, 'expected': parse(data)} for name, data in _index_cases()]

  out.write_text(dump({'identities': identities, 'parses': parses, 'endpoints': list(LFS_ENDPOINTS), 'cache': frozen}))


# -----------------------------------------------------------------------------

PARTS = {'wire': wire, 'staging': staging, 'layout': layout, 'registry': registry}


def generate(root: Path = ROOT, parts=tuple(PARTS)) -> None:
  for name in parts:
    PARTS[name](root)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="The conformance fixtures: what the comma's Python does, for the Swift to match.")
  parser.add_argument('--root', type=Path, default=ROOT, help='write the tree under this directory (default: this checkout)')
  parser.add_argument('parts', nargs='*', metavar='PART', help=f"some of {', '.join(PARTS)} (default: all)")
  args = parser.parse_args()
  unknown = set(args.parts) - set(PARTS)
  if unknown:
    parser.error(f"no such fixtures: {', '.join(sorted(unknown))}")
  generate(args.root, args.parts or tuple(PARTS))
