import Foundation
import JetlinkTestSupport
import Synchronization
import Testing

@testable import JetlinkRegistry

#if canImport(CryptoKit)
  import CryptoKit
#else
  import Crypto
#endif
#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

// MARK: - fixtures

extension RegistryFixture {
  static func json(_ name: String) -> JSON {
    (try? JSON.parse(data(name))) ?? .null
  }
}

func sha256Hex(_ data: Data) -> String {
  SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

func makeSpec(_ sha256: String) -> JSON {
  [
    "sha256": .string(sha256), "nbytes": 1234, "frame_skip": 4, "checkpoint": "b9facbcc",
    "input_shapes": ["img": [1, 12, 128, 256]],
    "output_shapes": ["outputs": [1, 18452]],
    "output_slices": ["hidden_state": [2066, 18450]],
  ]
}

// MARK: - a scratch cache

extension TemporaryDirectory {
  var layout: CacheLayout { CacheLayout(root: url) }
}

// MARK: - the network

// MockNet and LocalServer are JetlinkTestSupport's, which the command's tests
// share.

#if canImport(Darwin) || canImport(Glibc)
  extension LocalServer {
    /// The SHA-256 of what is served, without holding it.
    var sha256: String {
      var hasher = SHA256()
      var left = total
      while left > 0 {
        let n = Int(min(Int64(pattern.count), left))
        hasher.update(data: pattern.prefix(n))
        left -= Int64(n)
      }
      return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
  }
#endif

/// Collects progress calls from whatever thread makes them.
final class ProgressLog: Sendable {
  private let values = Mutex<[Double]>([])

  var callback: @Sendable (Double) -> Void {
    { [self] value in values.withLock { $0.append(value) } }
  }

  var all: [Double] { values.withLock { $0 } }
}

/// A shouldStop that answers from a list, then with its last answer.
final class StopScript: Sendable {
  private let answers: Mutex<[Bool]>

  init(_ answers: [Bool]) {
    self.answers = Mutex(answers)
  }

  var callback: @Sendable () -> Bool {
    { [self] in
      answers.withLock { list in
        list.count > 1 ? list.removeFirst() : (list.first ?? false)
      }
    }
  }
}
