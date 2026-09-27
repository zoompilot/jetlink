// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.
//
// JSON as the Python side reads it.
//
// The peer does arithmetic on what it decodes (a frame_skip of 4.0 would
// make numpy shapes out of floats), so integers and doubles stay distinct
// here, both ways. Objects keep their keys sorted on the way out so every
// payload is deterministic.

import Foundation

public enum JSON: Equatable, Sendable {
  case null
  case bool(Bool)
  case int(Int)
  case double(Double)
  case string(String)
  case array([JSON])
  case object([String: JSON])

  // MARK: reading

  public static func parse(_ data: Data) throws -> JSON {
    if data.isEmpty { throw ProtocolError("empty JSON") }
    let any = try JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
    return try JSON(any: any)
  }

  public static func parse(_ bytes: UnsafeRawBufferPointer) throws -> JSON {
    try parse(Data(bytes))
  }

  init(any: Any) throws {
    switch any {
    case is NSNull:
      self = .null
    case let n as NSNumber:
      if CFGetTypeID(n) == CFBooleanGetTypeID() {
        self = .bool(n.boolValue)
      } else if CFNumberIsFloatType(n) {
        self = .double(n.doubleValue)
      } else {
        self = .int(n.intValue)
      }
    case let s as String:
      self = .string(s)
    case let a as [Any]:
      self = .array(try a.map(JSON.init(any:)))
    case let d as [String: Any]:
      self = .object(try d.mapValues(JSON.init(any:)))
    default:
      throw ProtocolError("unsupported JSON value \(type(of: any))")
    }
  }

  public subscript(key: String) -> JSON? {
    if case .object(let d) = self { return d[key] }
    return nil
  }

  public var string: String? {
    if case .string(let s) = self { return s }
    return nil
  }

  /// An integer, accepting a double with no fractional part as Python's int() would.
  public var int: Int? {
    switch self {
    case .int(let i): return i
    case .double(let d) where d.rounded() == d && abs(d) < 1e15: return Int(d)
    case .bool(let b): return b ? 1 : 0
    default: return nil
    }
  }

  public var double: Double? {
    switch self {
    case .int(let i): return Double(i)
    case .double(let d): return d
    default: return nil
    }
  }

  public var bool: Bool? {
    if case .bool(let b) = self { return b }
    return nil
  }

  public var array: [JSON]? {
    if case .array(let a) = self { return a }
    return nil
  }

  public var object: [String: JSON]? {
    if case .object(let d) = self { return d }
    return nil
  }

  public var isNull: Bool { self == .null }

  // MARK: writing

  public var data: Data { Data(encoded.utf8) }

  public var encoded: String {
    var out = ""
    write(to: &out)
    return out
  }

  private func write(to out: inout String) {
    switch self {
    case .null: out += "null"
    case .bool(let b): out += b ? "true" : "false"
    case .int(let i): out += String(i)
    case .double(let d):
      if d.isFinite {
        // Python's repr of a float round-trips; so does Swift's description.
        // A whole number keeps its ".0" so it reads back as a float.
        out += d.rounded() == d && abs(d) < 1e16 ? String(format: "%.1f", d) : "\(d)"
      } else {
        // json.dumps writes these; json.loads reads them back.
        out += d.isNaN ? "NaN" : (d > 0 ? "Infinity" : "-Infinity")
      }
    case .string(let s): JSON.quote(s, into: &out)
    case .array(let a):
      out += "["
      for (i, v) in a.enumerated() {
        if i > 0 { out += ", " }
        v.write(to: &out)
      }
      out += "]"
    case .object(let d):
      out += "{"
      for (i, k) in d.keys.sorted().enumerated() {
        if i > 0 { out += ", " }
        JSON.quote(k, into: &out)
        out += ": "
        d[k]!.write(to: &out)
      }
      out += "}"
    }
  }

  private static func quote(_ s: String, into out: inout String) {
    out += "\""
    for u in s.unicodeScalars {
      switch u {
      case "\"": out += "\\\""
      case "\\": out += "\\\\"
      case "\n": out += "\\n"
      case "\r": out += "\\r"
      case "\t": out += "\\t"
      case _ where u.value < 0x20:
        out += String(format: "\\u%04x", u.value)
      default:
        out.unicodeScalars.append(u)
      }
    }
    out += "\""
  }
}

extension JSON: ExpressibleByStringLiteral, ExpressibleByIntegerLiteral, ExpressibleByBooleanLiteral,
  ExpressibleByFloatLiteral, ExpressibleByNilLiteral, ExpressibleByArrayLiteral, ExpressibleByDictionaryLiteral
{
  public init(stringLiteral value: String) { self = .string(value) }
  public init(integerLiteral value: Int) { self = .int(value) }
  public init(booleanLiteral value: Bool) { self = .bool(value) }
  public init(floatLiteral value: Double) { self = .double(value) }
  public init(nilLiteral: ()) { self = .null }
  public init(arrayLiteral elements: JSON...) { self = .array(elements) }
  public init(dictionaryLiteral elements: (String, JSON)...) {
    self = .object(Dictionary(elements, uniquingKeysWith: { _, b in b }))
  }
}

extension JSON {
  public static func str(_ s: String?) -> JSON { s.map(JSON.string) ?? .null }
}

/// Python's round(x, n), for the progress fractions and stats the peer displays.
func pyRound(_ x: Double, _ digits: Int) -> Double {
  let scale = pow(10.0, Double(digits))
  return (x * scale).rounded(.toNearestOrEven) / scale
}
