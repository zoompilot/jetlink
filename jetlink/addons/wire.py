"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The add-on wire: one request, one answer, each a single USB transfer on the
add-on's own interface. docs/addons.md describes it; JetlinkKit's AddonWire is
the host's copy. Standard library only.

  request  JAQ1: <4sIIHHHBBI magic, seq, tag, c, h, w, encoding, version, nbytes>
           + the image, [c, h, w]: u8 for ENC_U8 (the server divides by 255),
           f32 for ENC_F32
  answer   JAR1: <4sIIiII magic, seq, tag, status, server_us, nfloats>
           + f32 [nfloats]: every graph output, flattened, in name order
  hello    JAR1 with seq HELLO_SEQ, tag VERSION, status STATUS_HELLO or
           STATUS_YIELD, and nfloats u32 model tags in place of the floats

A model's tag is the first four bytes of its ONNX's sha256, big-endian, so a
host answers only for the exact file the comma checked. A message whose length
is a multiple of 512 carries one zero byte more: a bulk transfer of whole
packets has no end the other side can see without a zero-length packet, and
neither end sends one.

From Peter Nguyen's V-ASM and sign-vision wires (trung791997/openpilot).
"""
from __future__ import annotations

import struct

VERSION = 1
REQ = struct.Struct('<4sIIHHHBBI')
RESP = struct.Struct('<4sIIiII')
REQ_MAGIC, RESP_MAGIC = b'JAQ1', b'JAR1'
ENC_U8, ENC_F32 = 0, 1
ITEM = {ENC_U8: 1, ENC_F32: 4}
STATUS_OK, STATUS_BAD_REQUEST, STATUS_INFER_FAILED, STATUS_UNKNOWN_MODEL = 0, 1, 2, 3
STATUS_HELLO, STATUS_YIELD = 100, 101
HELLO_SEQ = 0xFFFFFFFF
PACKET = 512
MAX_REQUEST = 4 << 20
MAX_ANSWER = 4 << 20


def interface_triple(iface_class) -> str:
  """An interface class as the server names it: (0xFF, 0x56, 0x41) is 'ff/56/41'."""
  return '/'.join(f'{b:02x}' for b in iface_class)


def tag(sha256: str) -> int:
  return int(sha256[:8], 16)


def padded(message: bytes) -> bytes:
  return message + b'\0' if len(message) % PACKET == 0 else message


def _trim(buf, size: int):
  """The message without its pad byte; ValueError for any other length."""
  if len(buf) == size or (len(buf) == size + 1 and size % PACKET == 0):
    return memoryview(buf)[:size]
  raise ValueError(f'{len(buf)} bytes for a {size}-byte message')


def pack_request(seq: int, model_tag: int, image: bytes, shape: tuple[int, int, int], encoding: int = ENC_U8) -> bytes:
  c, h, w = shape
  if encoding not in ITEM or len(image) != c * h * w * ITEM[encoding]:
    raise ValueError(f'{len(image)} bytes is not a {c}x{h}x{w} image in encoding {encoding}')
  return padded(REQ.pack(REQ_MAGIC, seq, model_tag, c, h, w, encoding, VERSION, len(image)) + image)


def unpack_request(buf) -> tuple[int, int, tuple[int, int, int], int, memoryview]:
  """seq, tag, (c, h, w), encoding and the image. ValueError for anything malformed."""
  if len(buf) < REQ.size:
    raise ValueError(f'short request, {len(buf)} bytes')
  magic, seq, model_tag, c, h, w, encoding, version, nbytes = REQ.unpack_from(buf)
  if magic != REQ_MAGIC or version != VERSION or encoding not in ITEM or nbytes != c * h * w * ITEM[encoding]:
    raise ValueError(f'bad request header {magic!r} v{version}, {c}x{h}x{w} encoding {encoding}, {nbytes} bytes')
  return seq, model_tag, (c, h, w), encoding, _trim(buf, REQ.size + nbytes)[REQ.size:]


def pack_answer(seq: int, model_tag: int, status: int, server_us: int = 0, floats: bytes = b'') -> bytes:
  if len(floats) % 4:
    raise ValueError('floats must be whole f32s')
  return padded(RESP.pack(RESP_MAGIC, seq, model_tag, status, server_us, len(floats) // 4) + floats)


def pack_hello(tags, status: int = STATUS_HELLO) -> bytes:
  tags = sorted(tags)
  return padded(RESP.pack(RESP_MAGIC, HELLO_SEQ, VERSION, status, 0, len(tags)) + struct.pack(f'<{len(tags)}I', *tags))


def unpack_answer(buf) -> tuple[int, int, int, int, int, memoryview]:
  """seq, tag, status, server_us, count and the payload. ValueError for anything malformed."""
  if len(buf) < RESP.size:
    raise ValueError(f'short answer, {len(buf)} bytes')
  magic, seq, model_tag, status, server_us, count = RESP.unpack_from(buf)
  if magic != RESP_MAGIC:
    raise ValueError(f'bad answer magic {magic!r}')
  return seq, model_tag, status, server_us, count, _trim(buf, RESP.size + 4 * count)[RESP.size:]


def answer_length(buf) -> int:
  """A whole answer's length on a byte stream, its pad byte included, from its
  header. ValueError for a bad one."""
  magic, _, _, _, _, count = RESP.unpack_from(buf)
  size = RESP.size + 4 * count
  if magic != RESP_MAGIC or size > MAX_ANSWER:
    raise ValueError(f'bad answer header {magic!r}, {count} floats')
  return size + 1 if size % PACKET == 0 else size


def hello_tags(payload, count: int) -> frozenset[int]:
  return frozenset(struct.unpack_from(f'<{count}I', payload))
