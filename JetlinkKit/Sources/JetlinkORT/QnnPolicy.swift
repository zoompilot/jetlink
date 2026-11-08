import CoreFoundation
import Foundation
import JetlinkServer

/// Optional QNN HTP behavior. Defaults preserve the existing provider and caches.
/// Shared I/O changes host allocation; strict mode changes fallback admission;
/// disabling activation fusion is a separate numerical-correctness workaround.
public struct QnnPolicy: Sendable, Equatable {
  public let sharedIO: Bool
  public let strict: Bool
  public let disableAdvancedActivationFusion: Bool
  public let libraryDirectory: String?
  public let runtimeIdentity: String?

  public init(
    sharedIO: Bool = false, strict: Bool = false, disableAdvancedActivationFusion: Bool = false,
    libraryDirectory: String? = nil, runtimeIdentity: String? = nil
  ) {
    self.sharedIO = sharedIO
    self.strict = strict
    self.disableAdvancedActivationFusion = disableAdvancedActivationFusion
    self.libraryDirectory = libraryDirectory
    self.runtimeIdentity = runtimeIdentity
  }

  /// App configuration accepts JSON Booleans, not strings or numeric lookalikes.
  public init(configuration: [String: Any]) throws {
    func flag(_ key: String) throws -> Bool {
      guard let value = configuration[key] else { return false }
      guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else {
        throw HostError.invalid("\(key) must be a Boolean")
      }
      return number.boolValue
    }
    self.init(
      sharedIO: try flag("qnn_shared_io"), strict: try flag("qnn_strict"),
      disableAdvancedActivationFusion: try flag("qnn_no_activation_fusion"),
      libraryDirectory: configuration["native_library_dir"] as? String,
      runtimeIdentity: configuration["qnn_runtime_identity"] as? String)
    try validate()
  }

  var sessionConfig: [String: String] { strict ? ["session.disable_cpu_ep_fallback": "1"] : [:] }

  var cacheSuffix: String {
    guard sharedIO || strict || disableAdvancedActivationFusion else { return "" }
    let sdk = disableAdvancedActivationFusion ? "-sdk\(runtimeIdentity ?? "missing")" : ""
    return "-qnn-v1-shared\(sharedIO ? 1 : 0)-strict\(strict ? 1 : 0)-nofusion\(disableAdvancedActivationFusion ? 1 : 0)\(sdk)"
  }

  func validate() throws {
    guard disableAdvancedActivationFusion else { return }
    guard let libraryDirectory, libraryDirectory.hasPrefix("/"), !libraryDirectory.contains("\0"),
      let runtimeIdentity, runtimeIdentity.count == 64,
      runtimeIdentity.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
      throw HostError.invalid("the no-fusion adapter requires a matching SDK directory and runtime identity")
    }
  }

  var htpBackend: [String: String] {
    guard disableAdvancedActivationFusion else { return ["backend_type": "htp"] }
    let directory = libraryDirectory.map { URL(fileURLWithPath: $0, isDirectory: true) }
    let path = directory?.appendingPathComponent("libJetlinkQnnHtpNoFusion.so").path ?? "libJetlinkQnnHtpNoFusion.so"
    return ["backend_path": path]
  }
}
