// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.
//
// The wire protocol, byte for byte what jetlink/protocol.py speaks.
//
// Every message is a 32-byte little-endian header and an opaque payload:
// magic, version, msg_type, seq, flags, length, reserved, 4 pad
// ('<IHHIIIQ4x'). INFER is a fixed struct plus two raw arrays sized at the
// handshake.

import Foundation

public enum Wire {
  public static let magic: UInt32 = 0x4B4E_4C4A  // b'JLNK'
  public static let version: UInt16 = 2
  public static let headerSize = 32
  // A message whose header plus payload is an exact multiple of this carries
  // one pad byte and Flag.padded; see protocol.PACKET_MULTIPLE.
  public static let packetMultiple = 1024
  // INFER_REQ: frame_id, flags ('<II')
  public static let inferReqSize = 8
  // INFER_RESP: frame_id, status, gpu_us, queue_us, total_us ('<IIIII')
  public static let inferRespSize = 20
  // Stops a corrupt length field making the receive buffer allocate gigabytes.
  public static let maxMessage = 16 << 20
}

public enum Msg: UInt16, Sendable {
  case helloReq = 1
  case helloResp = 2
  case engineReq = 3
  case engineResp = 4
  case uploadChunk = 5
  case uploadDone = 6
  case progress = 7
  case inferReq = 8
  case inferResp = 9
  case stateReq = 12
  case stateResp = 13
  case error = 14
  case ping = 15
  case pong = 16
  case shutdownReq = 17
  case shutdownResp = 18
}

public struct Flag: OptionSet, Sendable {
  public let rawValue: UInt32
  public init(rawValue: UInt32) { self.rawValue = rawValue }
  public static let resetQueues = Flag(rawValue: 1 << 0)
  public static let wantState = Flag(rawValue: 1 << 1)
  public static let padded = Flag(rawValue: 1 << 7)
}

public enum InferStatus: UInt32, Sendable {
  case ok = 0
  case notReady = 1
  case badShape = 2
  case inferFailed = 3
  case notFinite = 4
}

public struct ProtocolError: Error, CustomStringConvertible {
  public let description: String
  public init(_ description: String) { self.description = description }
}

public struct Header: Equatable, Sendable {
  public var msgType: UInt16
  public var seq: UInt32
  public var flags: UInt32
  public var length: UInt32
  public var reserved: UInt64 = 0

  public init(msgType: UInt16, seq: UInt32, flags: UInt32, length: UInt32, reserved: UInt64 = 0) {
    self.msgType = msgType
    self.seq = seq
    self.flags = flags
    self.length = length
    self.reserved = reserved
  }

  public func encode(into p: UnsafeMutableRawPointer) {
    p.storeBytes(of: Wire.magic.littleEndian, toByteOffset: 0, as: UInt32.self)
    p.storeBytes(of: Wire.version.littleEndian, toByteOffset: 4, as: UInt16.self)
    p.storeBytes(of: msgType.littleEndian, toByteOffset: 6, as: UInt16.self)
    p.storeBytes(of: seq.littleEndian, toByteOffset: 8, as: UInt32.self)
    p.storeBytes(of: flags.littleEndian, toByteOffset: 12, as: UInt32.self)
    p.storeBytes(of: length.littleEndian, toByteOffset: 16, as: UInt32.self)
    p.storeBytes(of: reserved.littleEndian, toByteOffset: 20, as: UInt64.self)
    p.storeBytes(of: UInt32(0), toByteOffset: 28, as: UInt32.self)
  }

  public var bytes: [UInt8] {
    var out = [UInt8](repeating: 0, count: Wire.headerSize)
    out.withUnsafeMutableBytes { encode(into: $0.baseAddress!) }
    return out
  }

  /// Parses and validates a header, as protocol.unpack_header does.
  public static func decode(_ p: UnsafeRawPointer) throws -> Header {
    let magic = UInt32(littleEndian: p.loadUnaligned(fromByteOffset: 0, as: UInt32.self))
    let version = UInt16(littleEndian: p.loadUnaligned(fromByteOffset: 4, as: UInt16.self))
    if magic != Wire.magic {
      throw ProtocolError(String(format: "bad magic 0x%08x (link desynced or not a jetlink peer)", magic))
    }
    if version != Wire.version {
      throw ProtocolError("peer speaks protocol v\(version), we speak v\(Wire.version)")
    }
    return Header(
      msgType: UInt16(littleEndian: p.loadUnaligned(fromByteOffset: 6, as: UInt16.self)),
      seq: UInt32(littleEndian: p.loadUnaligned(fromByteOffset: 8, as: UInt32.self)),
      flags: UInt32(littleEndian: p.loadUnaligned(fromByteOffset: 12, as: UInt32.self)),
      length: UInt32(littleEndian: p.loadUnaligned(fromByteOffset: 16, as: UInt32.self)),
      reserved: UInt64(littleEndian: p.loadUnaligned(fromByteOffset: 20, as: UInt64.self)))
  }
}

public enum InferWire {
  public static func encodeResp(
    into p: UnsafeMutableRawPointer, frameID: UInt32, status: InferStatus, gpuUs: UInt32, queueUs: UInt32,
    totalUs: UInt32
  ) {
    p.storeBytes(of: frameID.littleEndian, toByteOffset: 0, as: UInt32.self)
    p.storeBytes(of: status.rawValue.littleEndian, toByteOffset: 4, as: UInt32.self)
    p.storeBytes(of: gpuUs.littleEndian, toByteOffset: 8, as: UInt32.self)
    p.storeBytes(of: queueUs.littleEndian, toByteOffset: 12, as: UInt32.self)
    p.storeBytes(of: totalUs.littleEndian, toByteOffset: 16, as: UInt32.self)
  }

  public static func resp(frameID: UInt32, status: InferStatus, gpuUs: UInt32 = 0, queueUs: UInt32 = 0, totalUs: UInt32 = 0)
    -> [UInt8]
  {
    var out = [UInt8](repeating: 0, count: Wire.inferRespSize)
    out.withUnsafeMutableBytes {
      encodeResp(into: $0.baseAddress!, frameID: frameID, status: status, gpuUs: gpuUs, queueUs: queueUs, totalUs: totalUs)
    }
    return out
  }

  /// (frame_id, flags) from the front of an INFER_REQ payload.
  public static func decodeReq(_ p: UnsafeRawPointer) -> (frameID: UInt32, flags: UInt32) {
    (
      UInt32(littleEndian: p.loadUnaligned(fromByteOffset: 0, as: UInt32.self)),
      UInt32(littleEndian: p.loadUnaligned(fromByteOffset: 4, as: UInt32.self))
    )
  }
}

/// Clamps a microsecond count to the UInt32 the wire carries.
@inline(__always)
func wireMicros(_ us: Int) -> UInt32 {
  us <= 0 ? 0 : (us >= Int(UInt32.max) ? UInt32.max : UInt32(us))
}
