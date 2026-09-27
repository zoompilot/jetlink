// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.
//
// openpilot stores the output slice map in the model's metadata_props as a
// base64 pickle of {name: slice(start, stop)}. This reads exactly that and
// nothing else: the opcodes pickle protocols 2 to 5 use for a dict of str to
// slice of int or None. The only callable it will construct is
// builtins.slice, so a hostile file cannot make it do anything but fail.

import Foundation

enum PickleValue {
  case none
  case int(Int)
  case string(String)
  case slice(Int?, Int?, Int?)
  case tuple([PickleValue])
  case dict([(String, PickleValue)])
  case sliceClass
  case mark
}

public struct PickleError: Error, CustomStringConvertible {
  public let description: String
  init(_ d: String) { description = d }
}

/// The {name: slice} map a pickle holds, in the pickle's order.
func unpickleSliceMap(_ bytes: [UInt8]) throws -> [(String, (Int?, Int?))] {
  var r = PickleReader(bytes)
  guard case .dict(let items) = try r.run() else { throw PickleError("output_slices is not a dict") }
  return try items.map { name, value in
    guard case .slice(let start, let stop, let step) = value else {
      throw PickleError("output_slices[\(name)] is not a slice")
    }
    if let step, step != 1 { throw PickleError("output_slices[\(name)] has a step of \(step)") }
    return (name, (start, stop))
  }
}

private struct PickleReader {
  let b: [UInt8]
  var i = 0
  var stack: [PickleValue] = []
  var memo: [Int: PickleValue] = [:]

  init(_ bytes: [UInt8]) { b = bytes }

  mutating func byte() throws -> UInt8 {
    guard i < b.count else { throw PickleError("pickle ends early") }
    defer { i += 1 }
    return b[i]
  }

  mutating func take(_ n: Int) throws -> ArraySlice<UInt8> {
    guard n >= 0, i + n <= b.count else { throw PickleError("pickle ends early") }
    defer { i += n }
    return b[i..<i + n]
  }

  mutating func uint(_ n: Int) throws -> UInt64 {
    var v: UInt64 = 0
    for (k, x) in try take(n).enumerated() { v |= UInt64(x) << (8 * UInt64(k)) }
    return v
  }

  mutating func line() throws -> String {
    guard let end = b[i...].firstIndex(of: 0x0A) else { throw PickleError("pickle ends early") }
    defer { i = end + 1 }
    return String(decoding: b[i..<end], as: UTF8.self)
  }

  mutating func pop() throws -> PickleValue {
    guard let v = stack.popLast() else { throw PickleError("pickle stack underflow") }
    return v
  }

  mutating func popToMark() throws -> [PickleValue] {
    guard let m = stack.lastIndex(where: { if case .mark = $0 { return true } else { return false } }) else {
      throw PickleError("pickle has no mark")
    }
    let items = Array(stack[(m + 1)...])
    stack.removeSubrange(m...)
    return items
  }

  mutating func string(_ n: Int) throws -> PickleValue {
    guard let s = String(bytes: try take(n), encoding: .utf8) else { throw PickleError("pickle string is not UTF-8") }
    return .string(s)
  }

  static func sliceIndex(_ v: PickleValue) throws -> Int? {
    switch v {
    case .none: return nil
    case .int(let i): return i
    default: throw PickleError("a slice bound is not an int or None")
    }
  }

  mutating func setItems(_ items: [PickleValue]) throws {
    guard items.count % 2 == 0, case .dict(var d) = try pop() else { throw PickleError("SETITEMS without a dict") }
    for k in stride(from: 0, to: items.count, by: 2) {
      guard case .string(let key) = items[k] else { throw PickleError("a dict key is not a string") }
      d.append((key, items[k + 1]))
    }
    stack.append(.dict(d))
  }

  mutating func run() throws -> PickleValue {
    while true {
      let op = try byte()
      switch op {
      case 0x80: _ = try byte()  // PROTO
      case 0x95: _ = try take(8)  // FRAME: a length hint
      case 0x7D: stack.append(.dict([]))  // EMPTY_DICT '}'
      case 0x28: stack.append(.mark)  // MARK '('
      case 0x4E: stack.append(.none)  // NONE 'N'
      case 0x29: stack.append(.tuple([]))  // EMPTY_TUPLE ')'
      case 0x4B: stack.append(.int(Int(try byte())))  // BININT1 'K'
      case 0x4D: stack.append(.int(Int(try uint(2))))  // BININT2 'M'
      case 0x4A: stack.append(.int(Int(Int32(bitPattern: UInt32(try uint(4))))))  // BININT 'J'
      case 0x8A:  // LONG1: little-endian two's complement
        let n = Int(try byte())
        guard n <= 8 else { throw PickleError("integer too large") }
        if n == 0 {
          stack.append(.int(0))
        } else {
          let raw = try uint(n)
          let shift = UInt64(64 - 8 * n)
          stack.append(.int(Int(Int64(bitPattern: raw << shift) >> Int64(shift))))
        }
      case 0x8C: stack.append(try string(Int(try byte())))  // SHORT_BINUNICODE
      case 0x58: stack.append(try string(Int(try uint(4))))  // BINUNICODE 'X'
      case 0x8D:  // BINUNICODE8
        let n = try uint(8)
        guard n < UInt64(b.count) else { throw PickleError("pickle ends early") }
        stack.append(try string(Int(n)))
      case 0x94: memo[memo.count] = stack.last  // MEMOIZE
      case 0x71: memo[Int(try byte())] = stack.last  // BINPUT 'q'
      case 0x72: memo[Int(try uint(4))] = stack.last  // LONG_BINPUT 'r'
      case 0x68, 0x6A:  // BINGET 'h', LONG_BINGET 'j'
        let k = op == 0x68 ? Int(try byte()) : Int(try uint(4))
        guard let v = memo[k] else { throw PickleError("pickle refers to memo \(k), which is empty") }
        stack.append(v)
      case 0x93:  // STACK_GLOBAL
        let name = try pop()
        let module = try pop()
        guard case .string("builtins") = module, case .string("slice") = name else {
          throw PickleError("output_slices refers to a class other than builtins.slice")
        }
        stack.append(.sliceClass)
      case 0x63:  // GLOBAL 'c'
        let module = try line()
        let name = try line()
        guard module == "builtins" || module == "__builtin__", name == "slice" else {
          throw PickleError("output_slices refers to \(module).\(name)")
        }
        stack.append(.sliceClass)
      case 0x85, 0x86, 0x87:  // TUPLE1, TUPLE2, TUPLE3
        let n = Int(op - 0x84)
        guard stack.count >= n else { throw PickleError("pickle stack underflow") }
        let items = Array(stack.suffix(n))
        stack.removeLast(n)
        stack.append(.tuple(items))
      case 0x74: stack.append(.tuple(try popToMark()))  // TUPLE 't'
      case 0x52:  // REDUCE 'R'
        guard case .tuple(let args) = try pop(), case .sliceClass = try pop() else {
          throw PickleError("REDUCE of something other than slice(...)")
        }
        guard (1...3).contains(args.count) else { throw PickleError("slice takes 1 to 3 arguments") }
        let a = try args.map(PickleReader.sliceIndex)
        // slice(stop) has only a stop, as in Python.
        stack.append(a.count == 1 ? .slice(nil, a[0], nil) : .slice(a[0], a[1], a.count == 3 ? a[2] : nil))
      case 0x75: try setItems(try popToMark())  // SETITEMS 'u'
      case 0x73:  // SETITEM 's'
        let v = try pop()
        let k = try pop()
        try setItems([k, v])
      case 0x2E: return try pop()  // STOP '.'
      default:
        throw PickleError(String(format: "unsupported pickle opcode 0x%02x in output_slices", op))
      }
    }
  }
}
