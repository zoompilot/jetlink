// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.

import CryptoKit
import Foundation

/// The file's SHA-256 as lowercase hex, and its size: a model's identity.
public func sha256File(_ path: String, progress: ((Int) -> Void)? = nil, shouldStop: (() -> Bool)? = nil) throws
  -> (String, Int)
{
  guard let h = FileHandle(forReadingAtPath: path) else { throw OnnxError("cannot open \(path)") }
  defer { try? h.close() }
  var hasher = SHA256()
  var n = 0
  while true {
    if shouldStop?() == true { throw CancellationError() }
    let chunk = try autoreleasepool { try h.read(upToCount: 4 << 20) ?? Data() }
    if chunk.isEmpty { break }
    hasher.update(data: chunk)
    n += chunk.count
    progress?(n)
  }
  return (hex(hasher.finalize()), n)
}

func hex<D: Sequence>(_ digest: D) -> String where D.Element == UInt8 {
  digest.map { String(format: "%02x", $0) }.joined()
}

/// Exactly `count` lowercase hex digits and nothing else, as re.fullmatch.
private func isLowerHex(_ s: String, count: Int) -> Bool {
  s.utf8.count == count && s.utf8.allSatisfy { (0x30...0x39).contains($0) || (0x61...0x66).contains($0) }
}

/// A model identity: a SHA-256 as 64 lowercase hex digits, which is also the LFS oid.
public func isSHA256(_ s: String) -> Bool { isLowerHex(s, count: 64) }

/// A comma openpilot commit: 40 lowercase hex digits.
public func isRef(_ s: String) -> Bool { isLowerHex(s, count: 40) }

/// A NUL-terminated C string held in an array, as Swift text.
func cText(_ buf: [CChar]) -> String {
  String(decoding: buf.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
}
