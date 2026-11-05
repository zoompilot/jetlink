import Foundation

/// A JSON value that behaves like the `json` module's: an object keeps its keys
/// in the order they were read or first set, as a Python dict does, and an
/// integer stays an integer.
///
/// The registry's state files and the model index go through this rather
/// than `JSONSerialization` because the Python registry depends on dict order
/// in places: the index's `refs` are walked in the order it lists them, and
/// the first ref with a given oid names an unlisted model. An unordered
/// dictionary would name a different one, and a size of 1.0 or true would
/// read as an integer.
enum JSON: Sendable, Equatable {
  case null
  case bool(Bool)
  case int(Int64)
  case double(Double)
  case string(String)
  case array([JSON])
  case object(JSONObject)

  // MARK: access

  var object: JSONObject? {
    if case .object(let value) = self { return value }
    return nil
  }

  var array: [JSON]? {
    if case .array(let value) = self { return value }
    return nil
  }

  var string: String? {
    if case .string(let value) = self { return value }
    return nil
  }

  /// `d.get(key)` on an object; nil for a missing key or for anything that is
  /// not an object.
  subscript(key: String) -> JSON? {
    object?[key]
  }

  // MARK: Python's coercions

  /// Python's truth value.
  var truthy: Bool {
    switch self {
    case .null: return false
    case .bool(let value): return value
    case .int(let value): return value != 0
    case .double(let value): return value != 0
    case .string(let value): return !value.isEmpty
    case .array(let value): return !value.isEmpty
    case .object(let value): return !value.isEmpty
    }
  }

  /// `int(x)`, or nil where Python raises TypeError or ValueError.
  var pythonInt: Int64? {
    switch self {
    case .bool(let value): return value ? 1 : 0
    case .int(let value): return value
    case .double(let value):
      guard value.isFinite, let truncated = Int64(exactly: value.rounded(.towardZero)) else { return nil }
      return truncated
    case .string(let value): return JSON.parsePythonInt(value)
    case .null, .array, .object: return nil
    }
  }

  /// `isinstance(x, (int, float))` then `float(x)`. A bool is an int in Python.
  var pythonNumber: Double? {
    switch self {
    case .bool(let value): return value ? 1 : 0
    case .int(let value): return Double(value)
    case .double(let value): return value
    default: return nil
    }
  }

  /// `isinstance(x, int)`, which a bool also is.
  var pythonIntValue: Int64? {
    switch self {
    case .bool(let value): return value ? 1 : 0
    case .int(let value): return value
    default: return nil
    }
  }

  /// `str(x)`.
  var pythonString: String {
    switch self {
    case .null: return "None"
    case .bool(let value): return value ? "True" : "False"
    case .int(let value): return String(value)
    case .double(let value): return JSON.pythonRepr(value)
    case .string(let value): return value
    case .array, .object: return serialized()
    }
  }

  /// `int(text)` for a str: surrounding whitespace, one sign, and single
  /// underscores between digits are all allowed.
  static func parsePythonInt(_ text: String) -> Int64? {
    var body = Substring(text.trimmingCharacters(in: .whitespacesAndNewlines))
    var negative = false
    if let first = body.first, first == "+" || first == "-" {
      negative = first == "-"
      body = body.dropFirst()
    }
    guard let first = body.first, first.isASCII, first.isNumber, body.last != "_" else { return nil }
    var digits = ""
    var previousUnderscore = false
    for character in body {
      if character == "_" {
        if previousUnderscore { return nil }
        previousUnderscore = true
        continue
      }
      guard character.isASCII, character.isNumber else { return nil }
      previousUnderscore = false
      digits.append(character)
    }
    guard let magnitude = Int64(digits) else { return nil }
    return negative ? -magnitude : magnitude
  }

  static func pythonRepr(_ value: Double) -> String {
    if value.isNaN { return "nan" }
    if value.isInfinite { return value < 0 ? "-inf" : "inf" }
    return value.description
  }
}

/// A dict: keys in first-insertion order, and setting an existing key keeps
/// its place. Equality ignores order, as Python's does.
struct JSONObject: Sendable, Equatable, Sequence {
  private(set) var keys: [String] = []
  private var storage: [String: JSON] = [:]

  init() {}

  init(_ pairs: [(String, JSON)]) {
    for (key, value) in pairs {
      self[key] = value
    }
  }

  var isEmpty: Bool { keys.isEmpty }
  var count: Int { keys.count }

  subscript(key: String) -> JSON? {
    get { storage[key] }
    set {
      if let newValue {
        if storage.updateValue(newValue, forKey: key) == nil {
          keys.append(key)
        }
      } else if storage.removeValue(forKey: key) != nil {
        keys.removeAll { $0 == key }
      }
    }
  }

  func contains(_ key: String) -> Bool {
    storage[key] != nil
  }

  func makeIterator() -> AnyIterator<(key: String, value: JSON)> {
    var index = 0
    return AnyIterator {
      guard index < keys.count else { return nil }
      defer { index += 1 }
      let key = keys[index]
      return (key, storage[key] ?? .null)
    }
  }

  static func == (lhs: JSONObject, rhs: JSONObject) -> Bool {
    lhs.storage == rhs.storage
  }
}

// MARK: - parsing

struct JSONParseError: Error, CustomStringConvertible {
  let description: String
}

extension JSON {
  /// `json.loads(data.decode())`: strict UTF-8, and nothing but whitespace
  /// after the value.
  static func parse(_ data: Data) throws -> JSON {
    var parser = Parser(bytes: [UInt8](data))
    parser.skipWhitespace()
    let value = try parser.value(depth: 0)
    parser.skipWhitespace()
    guard parser.index == parser.bytes.count else {
      throw JSONParseError(description: "Extra data at byte \(parser.index)")
    }
    return value
  }

  private struct Parser {
    let bytes: [UInt8]
    var index = 0

    // Python's default recursion limit turns deeper documents into an error
    // too; this keeps a hostile one off the stack.
    static let maxDepth = 512

    mutating func skipWhitespace() {
      while index < bytes.count, [0x20, 0x09, 0x0A, 0x0D].contains(bytes[index]) {
        index += 1
      }
    }

    func fail(_ what: String) -> JSONParseError {
      JSONParseError(description: "\(what) at byte \(index)")
    }

    mutating func value(depth: Int) throws -> JSON {
      guard depth < Parser.maxDepth else { throw fail("Nesting too deep") }
      guard index < bytes.count else { throw fail("Expecting value") }
      switch bytes[index] {
      case UInt8(ascii: "{"): return try object(depth: depth)
      case UInt8(ascii: "["): return try array(depth: depth)
      case UInt8(ascii: "\""): return .string(try string())
      case UInt8(ascii: "t"): return try literal("true", .bool(true))
      case UInt8(ascii: "f"): return try literal("false", .bool(false))
      case UInt8(ascii: "n"): return try literal("null", .null)
      case UInt8(ascii: "N"): return try literal("NaN", .double(.nan))
      case UInt8(ascii: "I"): return try literal("Infinity", .double(.infinity))
      case UInt8(ascii: "-"):
        if index + 1 < bytes.count, bytes[index + 1] == UInt8(ascii: "I") {
          index += 1
          return try literal("Infinity", .double(-.infinity))
        }
        return try number()
      case UInt8(ascii: "0")...UInt8(ascii: "9"): return try number()
      default: throw fail("Expecting value")
      }
    }

    mutating func literal(_ word: String, _ result: JSON) throws -> JSON {
      let expected = Array(word.utf8)
      guard index + expected.count <= bytes.count, Array(bytes[index..<index + expected.count]) == expected else {
        throw fail("Expecting value")
      }
      index += expected.count
      return result
    }

    mutating func number() throws -> JSON {
      let start = index
      var isInteger = true
      if bytes[index] == UInt8(ascii: "-") { index += 1 }
      guard index < bytes.count, isDigit(bytes[index]) else { throw fail("Expecting value") }
      if bytes[index] == UInt8(ascii: "0") {
        index += 1
      } else {
        while index < bytes.count, isDigit(bytes[index]) { index += 1 }
      }
      if index + 1 < bytes.count, bytes[index] == UInt8(ascii: "."), isDigit(bytes[index + 1]) {
        isInteger = false
        index += 1
        while index < bytes.count, isDigit(bytes[index]) { index += 1 }
      }
      if index < bytes.count, bytes[index] == UInt8(ascii: "e") || bytes[index] == UInt8(ascii: "E") {
        var probe = index + 1
        if probe < bytes.count, bytes[probe] == UInt8(ascii: "+") || bytes[probe] == UInt8(ascii: "-") { probe += 1 }
        if probe < bytes.count, isDigit(bytes[probe]) {
          isInteger = false
          index = probe
          while index < bytes.count, isDigit(bytes[index]) { index += 1 }
        }
      }
      let text = String(decoding: bytes[start..<index], as: UTF8.self)
      if isInteger, let value = Int64(text) {
        return .int(value)
      }
      // An integer too large for 64 bits is a Python int; a double is the
      // nearest this side can hold, and no field the registry reads is one.
      guard let value = Double(text) else { throw fail("Bad number") }
      return .double(value)
    }

    func isDigit(_ byte: UInt8) -> Bool {
      byte >= UInt8(ascii: "0") && byte <= UInt8(ascii: "9")
    }

    mutating func string() throws -> String {
      index += 1  // the opening quote
      var out = String.UnicodeScalarView()
      var runStart = index
      func flush(_ end: Int) throws {
        guard runStart < end else { return }
        guard let run = String(validating: bytes[runStart..<end], as: UTF8.self) else {
          throw JSONParseError(description: "Invalid UTF-8 at byte \(runStart)")
        }
        out.append(contentsOf: run.unicodeScalars)
      }
      while true {
        guard index < bytes.count else { throw fail("Unterminated string") }
        let byte = bytes[index]
        if byte == UInt8(ascii: "\"") {
          try flush(index)
          index += 1
          return String(out)
        }
        if byte < 0x20 {
          throw fail("Invalid control character")
        }
        if byte != UInt8(ascii: "\\") {
          index += 1
          continue
        }
        try flush(index)
        index += 1
        guard index < bytes.count else { throw fail("Unterminated string") }
        let escape = bytes[index]
        index += 1
        switch escape {
        case UInt8(ascii: "\""): out.append("\"")
        case UInt8(ascii: "\\"): out.append("\\")
        case UInt8(ascii: "/"): out.append("/")
        case UInt8(ascii: "b"): out.append("\u{08}")
        case UInt8(ascii: "f"): out.append("\u{0C}")
        case UInt8(ascii: "n"): out.append("\n")
        case UInt8(ascii: "r"): out.append("\r")
        case UInt8(ascii: "t"): out.append("\t")
        case UInt8(ascii: "u"):
          let unit = try hex4()
          if (0xD800..<0xDC00).contains(unit), index + 1 < bytes.count,
            bytes[index] == UInt8(ascii: "\\"), bytes[index + 1] == UInt8(ascii: "u")
          {
            let save = index
            index += 2
            let low = try hex4()
            if (0xDC00..<0xE000).contains(low) {
              let scalar = 0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00)
              out.append(Unicode.Scalar(scalar) ?? "\u{FFFD}")
              break
            }
            index = save
          }
          // A lone surrogate is legal in a Python str and not in a Swift one.
          out.append(Unicode.Scalar(unit) ?? "\u{FFFD}")
        default:
          throw fail("Invalid \\escape")
        }
        runStart = index
      }
    }

    mutating func hex4() throws -> UInt32 {
      guard index + 4 <= bytes.count else { throw fail("Invalid \\uXXXX escape") }
      var value: UInt32 = 0
      for byte in bytes[index..<index + 4] {
        guard let digit = hexDigit(byte) else { throw fail("Invalid \\uXXXX escape") }
        value = value << 4 | digit
      }
      index += 4
      return value
    }

    func hexDigit(_ byte: UInt8) -> UInt32? {
      switch byte {
      case UInt8(ascii: "0")...UInt8(ascii: "9"): return UInt32(byte - UInt8(ascii: "0"))
      case UInt8(ascii: "a")...UInt8(ascii: "f"): return UInt32(byte - UInt8(ascii: "a") + 10)
      case UInt8(ascii: "A")...UInt8(ascii: "F"): return UInt32(byte - UInt8(ascii: "A") + 10)
      default: return nil
      }
    }

    mutating func array(depth: Int) throws -> JSON {
      index += 1
      var items: [JSON] = []
      skipWhitespace()
      if index < bytes.count, bytes[index] == UInt8(ascii: "]") {
        index += 1
        return .array(items)
      }
      while true {
        skipWhitespace()
        items.append(try value(depth: depth + 1))
        skipWhitespace()
        guard index < bytes.count else { throw fail("Unterminated array") }
        if bytes[index] == UInt8(ascii: ",") {
          index += 1
        } else if bytes[index] == UInt8(ascii: "]") {
          index += 1
          return .array(items)
        } else {
          throw fail("Expecting ',' delimiter")
        }
      }
    }

    mutating func object(depth: Int) throws -> JSON {
      index += 1
      var out = JSONObject()
      skipWhitespace()
      if index < bytes.count, bytes[index] == UInt8(ascii: "}") {
        index += 1
        return .object(out)
      }
      while true {
        skipWhitespace()
        guard index < bytes.count, bytes[index] == UInt8(ascii: "\"") else {
          throw fail("Expecting property name enclosed in double quotes")
        }
        let key = try string()
        skipWhitespace()
        guard index < bytes.count, bytes[index] == UInt8(ascii: ":") else { throw fail("Expecting ':' delimiter") }
        index += 1
        skipWhitespace()
        // A repeated key takes the last value and keeps the first place, as dict() does.
        out[key] = try value(depth: depth + 1)
        skipWhitespace()
        guard index < bytes.count else { throw fail("Unterminated object") }
        if bytes[index] == UInt8(ascii: ",") {
          index += 1
        } else if bytes[index] == UInt8(ascii: "}") {
          index += 1
          return .object(out)
        } else {
          throw fail("Expecting ',' delimiter")
        }
      }
    }
  }
}

// MARK: - writing

extension JSON {
  /// `json.dumps(value)`: ", " and ": " separators and ASCII only, so the
  /// files read the same as the ones the Python registry writes.
  func serialized() -> String {
    var out = ""
    write(to: &out)
    return out
  }

  func data() -> Data {
    Data(serialized().utf8)
  }

  private func write(to out: inout String) {
    switch self {
    case .null: out += "null"
    case .bool(let value): out += value ? "true" : "false"
    case .int(let value): out += String(value)
    case .double(let value):
      if value.isNaN {
        out += "NaN"
      } else if value.isInfinite {
        out += value < 0 ? "-Infinity" : "Infinity"
      } else {
        out += value.description
      }
    case .string(let value): JSON.writeString(value, to: &out)
    case .array(let items):
      out += "["
      for (i, item) in items.enumerated() {
        if i > 0 { out += ", " }
        item.write(to: &out)
      }
      out += "]"
    case .object(let object):
      out += "{"
      for (i, pair) in object.enumerated() {
        if i > 0 { out += ", " }
        JSON.writeString(pair.key, to: &out)
        out += ": "
        pair.value.write(to: &out)
      }
      out += "}"
    }
  }

  private static func writeString(_ value: String, to out: inout String) {
    out += "\""
    for unit in value.utf16 {
      switch unit {
      case 0x22: out += "\\\""
      case 0x5C: out += "\\\\"
      case 0x0A: out += "\\n"
      case 0x0D: out += "\\r"
      case 0x09: out += "\\t"
      case 0x08: out += "\\b"
      case 0x0C: out += "\\f"
      case 0x20..<0x7F: out.unicodeScalars.append(Unicode.Scalar(UInt8(unit)))
      default:
        let hex = String(unit, radix: 16)
        out += "\\u" + String(repeating: "0", count: 4 - hex.count) + hex
      }
    }
    out += "\""
  }
}

// MARK: - literals, for building values in code

extension JSON: ExpressibleByStringLiteral, ExpressibleByIntegerLiteral, ExpressibleByFloatLiteral,
  ExpressibleByBooleanLiteral, ExpressibleByArrayLiteral, ExpressibleByDictionaryLiteral, ExpressibleByNilLiteral
{
  init(stringLiteral value: String) { self = .string(value) }
  init(integerLiteral value: Int64) { self = .int(value) }
  init(floatLiteral value: Double) { self = .double(value) }
  init(booleanLiteral value: Bool) { self = .bool(value) }
  init(arrayLiteral elements: JSON...) { self = .array(elements) }
  init(dictionaryLiteral elements: (String, JSON)...) { self = .object(JSONObject(elements)) }
  init(nilLiteral: ()) { self = .null }
}

extension JSONObject: ExpressibleByDictionaryLiteral {
  init(dictionaryLiteral elements: (String, JSON)...) { self.init(elements) }
}

extension JSON {
  /// `repr()` of a str, for the messages that quote a value back.
  static func pythonRepr(_ value: String) -> String {
    let quote: Character = value.contains("'") && !value.contains("\"") ? "\"" : "'"
    var out = String(quote)
    for scalar in value.unicodeScalars {
      switch scalar {
      case "\\": out += "\\\\"
      case "\n": out += "\\n"
      case "\r": out += "\\r"
      case "\t": out += "\\t"
      default:
        if Character(scalar) == quote {
          out += "\\" + String(quote)
        } else if scalar.value < 0x20 || scalar.value == 0x7F {
          let hex = String(scalar.value, radix: 16)
          out += "\\x" + String(repeating: "0", count: 2 - hex.count) + hex
        } else {
          out.unicodeScalars.append(scalar)
        }
      }
    }
    out.append(quote)
    return out
  }
}
