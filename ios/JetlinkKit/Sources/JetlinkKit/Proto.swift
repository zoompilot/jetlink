// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.
//
// Just enough protobuf to rewrite an ONNX file without holding it twice.
//
// The file is mapped, and a message is decoded only when something reads or
// changes it. A field nobody touched is written back as the bytes it was,
// straight from the mapping, so 700 MB of weights pass through as a copy.
// A field that is new can be produced lazily at write time, which is how the
// transposed Gemm weights avoid sitting in memory all at once.

import Foundation

public struct ProtoDecodeError: Error, CustomStringConvertible {
  public let description: String
  init(_ d: String) { description = d }
}

/// A file mapped read-only for the lifetime of the object.
public final class MappedFile: @unchecked Sendable {
  public let base: UnsafePointer<UInt8>
  public let count: Int
  private let mapping: UnsafeMutableRawPointer?

  public init(path: String) throws {
    let fd = open(path, O_RDONLY)
    if fd < 0 { throw ProtoDecodeError("cannot open \(path): \(String(cString: strerror(errno)))") }
    defer { close(fd) }
    var st = stat()
    if fstat(fd, &st) != 0 { throw ProtoDecodeError("cannot stat \(path)") }
    count = Int(st.st_size)
    if count == 0 {
      mapping = nil
      base = UnsafePointer(UnsafeMutablePointer<UInt8>.allocate(capacity: 1))
      return
    }
    guard let p = mmap(nil, count, PROT_READ, MAP_PRIVATE, fd, 0), p != MAP_FAILED else {
      throw ProtoDecodeError("cannot map \(path): \(String(cString: strerror(errno)))")
    }
    // Read once front to back: the rewrite walks the whole file in order.
    madvise(p, count, MADV_SEQUENTIAL)
    mapping = p
    base = UnsafePointer(p.assumingMemoryBound(to: UInt8.self))
  }

  deinit {
    if let mapping {
      munmap(mapping, count)
    } else {
      UnsafeMutablePointer(mutating: base).deallocate()
    }
  }

  func string(_ r: Range<Int>) -> String {
    String(decoding: UnsafeBufferPointer(start: base + r.lowerBound, count: r.count), as: UTF8.self)
  }

  func bytes(_ r: Range<Int>) -> [UInt8] {
    Array(UnsafeBufferPointer(start: base + r.lowerBound, count: r.count))
  }
}

// MARK: - reading

enum WireType: Int {
  case varint = 0, fixed64 = 1, lengthDelimited = 2, fixed32 = 5
}

struct ProtoReader {
  let base: UnsafePointer<UInt8>
  var pos: Int
  let end: Int

  init(_ file: MappedFile, _ range: Range<Int>) {
    base = file.base
    pos = range.lowerBound
    end = range.upperBound
  }

  var atEnd: Bool { pos >= end }

  mutating func varint() throws -> UInt64 {
    var result: UInt64 = 0
    var shift: UInt64 = 0
    while true {
      guard pos < end else { throw ProtoDecodeError("truncated varint") }
      let b = base[pos]
      pos += 1
      if shift == 63 && b > 1 { throw ProtoDecodeError("varint overflows 64 bits") }
      result |= UInt64(b & 0x7F) << shift
      if b & 0x80 == 0 { return result }
      shift += 7
      if shift > 63 { throw ProtoDecodeError("varint too long") }
    }
  }

  /// The next field's number, wire type, and where its value lies. Groups,
  /// which ONNX never uses, are an error rather than a guess.
  mutating func next() throws -> (number: Int, wire: WireType, value: Range<Int>)? {
    if atEnd { return nil }
    let tag = try varint()
    let number = Int(tag >> 3)
    guard number > 0, let wire = WireType(rawValue: Int(tag & 7)) else {
      throw ProtoDecodeError("unsupported wire type \(tag & 7) or field 0")
    }
    let start = pos
    switch wire {
    case .varint: _ = try varint()
    case .fixed64: pos += 8
    case .fixed32: pos += 4
    case .lengthDelimited:
      let n = try varint()
      guard n <= UInt64(end - pos) else { throw ProtoDecodeError("a field runs past its message") }
      let valueStart = pos
      pos += Int(n)
      return (number, wire, valueStart..<pos)
    }
    guard pos <= end else { throw ProtoDecodeError("a field runs past its message") }
    return (number, wire, start..<pos)
  }
}

// MARK: - the message model

/// Bytes on their way out: copied from the file, encoded here, or produced
/// by a closure when the writer reaches them.
public enum Chunk {
  case source(Range<Int>)
  case bytes([UInt8])
  case lazy(Int, (inout ProtoWriter) throws -> Void)

  var count: Int {
    switch self {
    case .source(let r): r.count
    case .bytes(let b): b.count
    case .lazy(let n, _): n
    }
  }
}

public indirect enum Payload {
  case varint(UInt64)
  case fixed64(UInt64)
  case fixed32(UInt32)
  /// A length-delimited value nobody has decoded: string, bytes, packed
  /// scalars, or a message kept as it was.
  case raw(Chunk)
  case message(ProtoMessage)
}

public struct ProtoField {
  public var number: Int
  public var payload: Payload

  init(_ number: Int, _ payload: Payload) {
    self.number = number
    self.payload = payload
  }

  static func string(_ number: Int, _ s: String) -> ProtoField { ProtoField(number, .raw(.bytes(Array(s.utf8)))) }
  static func int(_ number: Int, _ v: Int64) -> ProtoField { ProtoField(number, .varint(UInt64(bitPattern: v))) }
}

public struct ProtoMessage {
  public var fields: [ProtoField] = []

  init(fields: [ProtoField] = []) { self.fields = fields }

  /// Decodes one level; nested messages stay raw until asked for.
  static func decode(_ file: MappedFile, _ range: Range<Int>) throws -> ProtoMessage {
    var r = ProtoReader(file, range)
    var out = ProtoMessage()
    while let (number, wire, value) = try r.next() {
      switch wire {
      case .varint:
        var v = ProtoReader(base: file.base, pos: value.lowerBound, end: value.upperBound)
        out.fields.append(ProtoField(number, .varint(try v.varint())))
      case .fixed64:
        out.fields.append(ProtoField(number, .fixed64(loadLE(file, value.lowerBound, UInt64.self))))
      case .fixed32:
        out.fields.append(ProtoField(number, .fixed32(loadLE(file, value.lowerBound, UInt32.self))))
      case .lengthDelimited:
        out.fields.append(ProtoField(number, .raw(.source(value))))
      }
    }
    return out
  }

  // MARK: encoding

  /// Encoded size, fields in field-number order as protobuf itself writes a
  /// message it has parsed.
  var size: Int {
    var n = 0
    for f in fields { n += ProtoMessage.fieldSize(f) }
    return n
  }

  static func fieldSize(_ f: ProtoField) -> Int {
    let tag = varintSize(UInt64(f.number) << 3)
    switch f.payload {
    case .varint(let v): return tag + varintSize(v)
    case .fixed64: return tag + 8
    case .fixed32: return tag + 4
    case .raw(let c): return tag + varintSize(UInt64(c.count)) + c.count
    case .message(let m):
      let s = m.size
      return tag + varintSize(UInt64(s)) + s
    }
  }

  func write(to w: inout ProtoWriter) throws {
    for f in sortedFields {
      try ProtoMessage.writeField(f, to: &w)
    }
  }

  var sortedFields: [ProtoField] {
    // Stable: repeated fields keep their order.
    fields.enumerated().sorted { a, b in
      a.element.number != b.element.number ? a.element.number < b.element.number : a.offset < b.offset
    }.map(\.element)
  }

  static func writeField(_ f: ProtoField, to w: inout ProtoWriter) throws {
    switch f.payload {
    case .varint(let v):
      try w.varint(UInt64(f.number) << 3 | 0)
      try w.varint(v)
    case .fixed64(var v):
      try w.varint(UInt64(f.number) << 3 | 1)
      v = v.littleEndian
      try withUnsafeBytes(of: &v) { try w.write($0) }
    case .fixed32(var v):
      try w.varint(UInt64(f.number) << 3 | 5)
      v = v.littleEndian
      try withUnsafeBytes(of: &v) { try w.write($0) }
    case .raw(let c):
      try w.varint(UInt64(f.number) << 3 | 2)
      try w.varint(UInt64(c.count))
      try w.chunk(c)
    case .message(let m):
      try w.varint(UInt64(f.number) << 3 | 2)
      try w.varint(UInt64(m.size))
      try m.write(to: &w)
    }
  }

  /// The whole message as bytes. For small messages only.
  func encoded(file: MappedFile?) throws -> [UInt8] {
    var w = ProtoWriter(memory: file)
    try write(to: &w)
    return w.memory
  }

  // MARK: field access

  func all(_ number: Int) -> [ProtoField] { fields.filter { $0.number == number } }
  func first(_ number: Int) -> ProtoField? { fields.first { $0.number == number } }
  func has(_ number: Int) -> Bool { fields.contains { $0.number == number } }

  mutating func remove(_ number: Int) { fields.removeAll { $0.number == number } }

  /// Sets a singular field. Position is irrelevant: fields are written in
  /// field-number order.
  mutating func set(_ field: ProtoField) {
    remove(field.number)
    fields.append(field)
  }

  /// Replaces every occurrence of a repeated field, keeping their new order.
  mutating func replaceAll(_ number: Int, with new: [ProtoField]) {
    remove(number)
    fields.append(contentsOf: new)
  }
}

func varintSize(_ v: UInt64) -> Int {
  var n = 1
  var v = v >> 7
  while v != 0 {
    n += 1
    v >>= 7
  }
  return n
}

func loadLE<T: FixedWidthInteger>(_ file: MappedFile, _ offset: Int, _: T.Type) -> T {
  T(littleEndian: UnsafeRawPointer(file.base + offset).loadUnaligned(as: T.self))
}

// MARK: - decoding helpers for scalar payloads

extension ProtoField {
  func string(_ file: MappedFile) -> String? {
    switch payload {
    case .raw(.source(let r)): return file.string(r)
    case .raw(.bytes(let b)): return String(decoding: b, as: UTF8.self)
    default: return nil
    }
  }

  var int: Int64? {
    if case .varint(let v) = payload { return Int64(bitPattern: v) }
    return nil
  }

  /// A nested message, decoding it if it is still raw.
  func message(_ file: MappedFile) throws -> ProtoMessage? {
    switch payload {
    case .message(let m): return m
    case .raw(.source(let r)): return try ProtoMessage.decode(file, r)
    default: return nil
    }
  }

  /// Every int64 in a repeated field that may be packed or not.
  static func int64s(_ fields: [ProtoField], _ file: MappedFile) throws -> [Int64] {
    var out: [Int64] = []
    for f in fields {
      switch f.payload {
      case .varint(let v): out.append(Int64(bitPattern: v))
      case .raw(.source(let r)):
        var reader = ProtoReader(base: file.base, pos: r.lowerBound, end: r.upperBound)
        while !reader.atEnd { out.append(Int64(bitPattern: try reader.varint())) }
      default: throw ProtoDecodeError("unexpected encoding for a repeated integer")
      }
    }
    return out
  }
}

extension ProtoReader {
  init(base: UnsafePointer<UInt8>, pos: Int, end: Int) {
    self.base = base
    self.pos = pos
    self.end = end
  }
}

// MARK: - writing

/// Buffered output to a file descriptor, or to memory for small messages.
public struct ProtoWriter {
  private let fd: Int32
  private var buffer: [UInt8]
  private(set) var memory: [UInt8] = []
  private let toMemory: Bool
  let file: MappedFile?
  private(set) var written = 0
  static let bufferSize = 1 << 20

  init(fd: Int32, file: MappedFile?) {
    self.fd = fd
    self.file = file
    buffer = []
    buffer.reserveCapacity(ProtoWriter.bufferSize)
    toMemory = false
  }

  init(memory file: MappedFile?) {
    fd = -1
    self.file = file
    buffer = []
    toMemory = true
  }

  mutating func varint(_ v: UInt64) throws {
    var v = v
    var tmp = [UInt8]()
    tmp.reserveCapacity(10)
    while v >= 0x80 {
      tmp.append(UInt8(v & 0x7F) | 0x80)
      v >>= 7
    }
    tmp.append(UInt8(v))
    try tmp.withUnsafeBytes { try write($0) }
  }

  mutating func write(_ bytes: UnsafeRawBufferPointer) throws {
    written += bytes.count
    if toMemory {
      memory.append(contentsOf: bytes)
      return
    }
    if buffer.count + bytes.count > ProtoWriter.bufferSize {
      try flush()
      if bytes.count >= ProtoWriter.bufferSize {
        try ProtoWriter.writeAll(fd, bytes)
        return
      }
    }
    buffer.append(contentsOf: bytes)
  }

  mutating func chunk(_ c: Chunk) throws {
    switch c {
    case .source(let r):
      guard let file else { throw ProtoDecodeError("a source chunk with no file") }
      try write(UnsafeRawBufferPointer(start: file.base + r.lowerBound, count: r.count))
    case .bytes(let b):
      try b.withUnsafeBytes { try write($0) }
    case .lazy(let n, let produce):
      let before = written
      try produce(&self)
      if written - before != n {
        throw ProtoDecodeError("a lazily written field produced \(written - before) bytes, promised \(n)")
      }
    }
  }

  mutating func flush() throws {
    if toMemory || buffer.isEmpty { return }
    try buffer.withUnsafeBytes { try ProtoWriter.writeAll(fd, $0) }
    buffer.removeAll(keepingCapacity: true)
  }

  private static func writeAll(_ fd: Int32, _ bytes: UnsafeRawBufferPointer) throws {
    var off = 0
    while off < bytes.count {
      let n = Darwin.write(fd, bytes.baseAddress! + off, bytes.count - off)
      if n < 0 {
        if errno == EINTR { continue }
        throw ProtoDecodeError("write failed: \(String(cString: strerror(errno)))")
      }
      off += n
    }
  }
}
