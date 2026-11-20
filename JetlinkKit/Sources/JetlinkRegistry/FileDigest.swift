import Foundation

#if canImport(CryptoKit)
  import CryptoKit
#else
  import Crypto
#endif

extension Files {
  /// The file's SHA-256, lowercase hex, read a megabyte at a time.
  package static func sha256(of url: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    var hasher = SHA256()
    while let chunk = try handle.read(upToCount: 1 << 20), !chunk.isEmpty {
      hasher.update(data: chunk)
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
  }
}
