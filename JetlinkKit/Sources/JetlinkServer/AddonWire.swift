// The add-on server: a fork's own models beside the driving model, each on its
// own interface of the comma's gadget. docs/addons.md. From Peter Nguyen's
// V-ASM and sign-vision servers (trung791997/openpilot).
import Foundation
import JetlinkKit

/// The add-on wire, jetlink/addons/wire.py's format: one request, one answer,
/// little-endian.
///
///   request  "JAQ1" seq:u32 tag:u32 c:u16 h:u16 w:u16 encoding:u8 version:u8 nbytes:u32, then the image
///   answer   "JAR1" seq:u32 tag:u32 status:i32 server_us:u32 count:u32, then count f32
///   hello    an answer with seq `helloSeq`, tag `version`, status hello or yield, and count u32 tags
///
/// A message whose length is a multiple of 512 carries one zero byte more, so
/// it always ends in a short packet.
public enum AddonWire {
  public static let version: UInt8 = 1
  public static let headerBytes = 24
  public static let packet = 512
  /// Every request and answer is under these, header included, as the comma's
  /// jetlink/addons/wire.py MAX_REQUEST and MAX_ANSWER: a model whose input or
  /// outputs cannot fit is refused when it is built, not on every frame.
  public static let maxRequest = 4 << 20
  public static let maxAnswer = 4 << 20
  public static let helloSeq: UInt32 = 0xFFFF_FFFF
  static let requestMagic = Array("JAQ1".utf8)
  static let answerMagic = Array("JAR1".utf8)

  public enum Encoding: UInt8, Sendable {
    /// u8, which the server divides by 255: a blobFromImage(1/255) image's own bytes
    case u8 = 0
    case f32 = 1
    /// u8 the model is given as the same values, whatever its input type
    case bytes = 2

    public var itemBytes: Int { self == .f32 ? 4 : 1 }
  }

  public enum Status: Int32, Sendable {
    case ok = 0
    case badRequest = 1
    case inferFailed = 2
    case unknownModel = 3
    case hello = 100
    case yield = 101
  }

  public struct Request: Equatable, Sendable {
    public let seq: UInt32
    public let tag: UInt32
    public let shape: [Int]
    public let encoding: Encoding
    public let nbytes: Int

    /// The bytes that follow the header, the pad byte included.
    public var bodyBytes: Int { padded(headerBytes + nbytes) - headerBytes }
  }

  public enum WireError: Error, CustomStringConvertible {
    case malformed(String)

    public var description: String {
      switch self {
      case .malformed(let why): "bad add-on request: \(why)"
      }
    }
  }

  /// The first four bytes of a sha256, as the tag the comma sends.
  public static func tag(sha256: String) -> UInt32? {
    UInt32(sha256.prefix(8), radix: 16)
  }

  public static func padded(_ count: Int) -> Int {
    count % packet == 0 ? count + 1 : count
  }

  public static func request(header bytes: UnsafeRawBufferPointer) throws -> Request {
    guard bytes.count >= headerBytes else { throw WireError.malformed("\(bytes.count) header bytes") }
    guard Array(bytes[0..<4]) == requestMagic else { throw WireError.malformed("magic \(Array(bytes[0..<4]))") }
    let channels = Int(load(UInt16.self, bytes, 12))
    let height = Int(load(UInt16.self, bytes, 14))
    let width = Int(load(UInt16.self, bytes, 16))
    guard let encoding = Encoding(rawValue: bytes[18]) else { throw WireError.malformed("encoding \(bytes[18])") }
    guard bytes[19] == version else { throw WireError.malformed("version \(bytes[19])") }
    let nbytes = Int(load(UInt32.self, bytes, 20))
    guard nbytes == channels * height * width * encoding.itemBytes, headerBytes + nbytes < maxRequest else {
      throw WireError.malformed("\(nbytes) bytes for \(channels)x\(height)x\(width) in \(encoding)")
    }
    return Request(
      seq: load(UInt32.self, bytes, 4), tag: load(UInt32.self, bytes, 8), shape: [channels, height, width], encoding: encoding, nbytes: nbytes)
  }

  public static func answer(seq: UInt32, tag: UInt32, status: Status, serverUs: UInt32 = 0, floats: [Float] = []) -> [UInt8] {
    var out = head(seq: seq, tag: tag, status: status.rawValue, serverUs: serverUs, count: UInt32(floats.count))
    floats.withUnsafeBytes { raw in
      for value in raw.bindMemory(to: UInt32.self) { append(value, to: &out) }
    }
    return pad(out)
  }

  public static func hello(tags: [UInt32], yielding: Bool) -> [UInt8] {
    var out = head(seq: helloSeq, tag: UInt32(version), status: (yielding ? Status.yield : .hello).rawValue, serverUs: 0, count: UInt32(tags.count))
    for tag in tags.sorted() { append(tag, to: &out) }
    return pad(out)
  }

  private static func head(seq: UInt32, tag: UInt32, status: Int32, serverUs: UInt32, count: UInt32) -> [UInt8] {
    var out = answerMagic
    out.reserveCapacity(headerBytes + Int(count) * 4 + 1)
    for value in [seq, tag, UInt32(bitPattern: status), serverUs, count] { append(value, to: &out) }
    return out
  }

  private static func pad(_ out: [UInt8]) -> [UInt8] {
    out.count % packet == 0 ? out + [0] : out
  }

  private static func append(_ value: UInt32, to out: inout [UInt8]) {
    withUnsafeBytes(of: value.littleEndian) { out.append(contentsOf: $0) }
  }

  private static func load<T: FixedWidthInteger>(_: T.Type, _ bytes: UnsafeRawBufferPointer, _ offset: Int) -> T {
    T(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: T.self))
  }
}

extension AddonWire {
  /// "ff/56/41" as bytes, or nil for anything else, Jetlink's own class included.
  public static func interfaceClass(_ triple: String) -> [UInt8]? {
    let parts = triple.split(separator: "/", omittingEmptySubsequences: false)
    let bytes = parts.compactMap { $0.count == 2 ? UInt8($0, radix: 16) : nil }
    guard parts.count == 3, bytes.count == 3, bytes != Pinned.usbVendorClass else { return nil }
    return bytes
  }

  /// The class as "ff/56/41".
  public static func triple(_ bytes: [UInt8]) -> String {
    bytes.map { String(format: "%02x", $0) }.joined(separator: "/")
  }
}
