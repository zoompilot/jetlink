"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Wire protocol.

Transport-agnostic: every message is a fixed 32-byte header followed by an
opaque payload. The header is padded to 32 bytes so that float32 payloads land
8-byte aligned, which lets both ends build numpy views over the receive buffer
without copying.

The hot path (INFER) is deliberately not JSON or capnp. It is a fixed struct
plus two raw arrays whose sizes both ends agree on at handshake time, so a
request costs one vectored write and a response costs one read into a
preallocated buffer.
"""
from __future__ import annotations

import struct
from enum import IntEnum

MAGIC = 0x4B4E4C4A  # b'JLNK'
VERSION = 1

# magic, version, msg_type, seq, flags, length, reserved, 4 pad
HEADER_FMT = '<IHHIIIQ4x'
HEADER_SIZE = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 32

_header = struct.Struct(HEADER_FMT)


class Msg(IntEnum):
  HELLO_REQ = 1        # {} -> server describes itself
  HELLO_RESP = 2       # json: server caps, loaded engine, shapes
  ENGINE_REQ = 3       # json: {sha256, size, frame_skip} -> does the server have this engine?
  ENGINE_RESP = 4      # json: {state: ready|need_upload|building, ...}
  UPLOAD_CHUNK = 5     # u64 offset + bytes
  UPLOAD_DONE = 6      # json: {sha256}
  PROGRESS = 7         # json: {stage, frac, msg} - unsolicited, server -> client
  INFER_REQ = 8        # InferHeader + warped(u8) + packed(f32)
  INFER_RESP = 9       # InferRespHeader + outputs(f32)
  # 10, 11 were RESET_REQ/RESP; queues are cleared with Flag.RESET_QUEUES
  STATE_REQ = 12       # telemetry
  STATE_RESP = 13      # json
  ERROR = 14           # json: {error, detail}
  PING = 15
  PONG = 16


class Flag(IntEnum):
  RESET_QUEUES = 1 << 0   # on INFER_REQ: warm-start, clear history before this frame
  WANT_STATE = 1 << 1     # on INFER_REQ: append telemetry json to the response.
                          # Piggybacked because at 20 Hz there is no gap in which
                          # to run a separate request/response without racing a
                          # frame, and health data must not cost a frame.


# INFER_REQ: frame_id, flags. Sizes of the two arrays come from the handshake.
INFER_REQ_FMT = '<II'
INFER_REQ_SIZE = struct.calcsize(INFER_REQ_FMT)
_infer_req = struct.Struct(INFER_REQ_FMT)

# INFER_RESP: frame_id, status, server-side timings in microseconds
INFER_RESP_FMT = '<IIIII'
INFER_RESP_SIZE = struct.calcsize(INFER_RESP_FMT)
_infer_resp = struct.Struct(INFER_RESP_FMT)


class ProtocolError(RuntimeError):
  pass


def pack_header(msg_type: int, seq: int, length: int, flags: int = 0, reserved: int = 0) -> bytes:
  return _header.pack(MAGIC, VERSION, int(msg_type), seq, flags, length, reserved)


def unpack_header(buf) -> tuple[int, int, int, int, int, int, int]:
  magic, version, msg_type, seq, flags, length, reserved = _header.unpack_from(buf)
  if magic != MAGIC:
    raise ProtocolError(f"bad magic 0x{magic:08x} (link desynced or not a jetlink peer)")
  if version != VERSION:
    raise ProtocolError(f"peer speaks protocol v{version}, we speak v{VERSION}")
  return magic, version, msg_type, seq, flags, length, reserved


def pack_infer_req(frame_id: int, flags: int = 0) -> bytes:
  return _infer_req.pack(frame_id, flags)


def unpack_infer_req(buf, offset: int = 0) -> tuple[int, int]:
  return _infer_req.unpack_from(buf, offset)


def pack_infer_resp(frame_id: int, status: int, gpu_us: int, queue_us: int, total_us: int) -> bytes:
  return _infer_resp.pack(frame_id, status, gpu_us, queue_us, total_us)


def unpack_infer_resp(buf, offset: int = 0) -> tuple[int, int, int, int, int]:
  return _infer_resp.unpack_from(buf, offset)


class Status(IntEnum):
  OK = 0
  NOT_READY = 1       # no engine loaded
  BAD_SHAPE = 2
  INFER_FAILED = 3
  NOT_FINITE = 4      # model produced NaN/Inf; caller must fall back
