import Foundation
import os

/// Where the interpreter that runs the server comes from.
struct PythonRuntime: Sendable, Equatable {
  enum Source: Sendable, Equatable {
    case bundled
    case environment(String)
    case settings(String)
  }

  let executable: URL
  let source: Source

  private static let log = Logger(subsystem: "io.zoompilot.jetlink", category: "python")

  /// Environment first, then the settings override, then the copy inside the bundle.
  @MainActor static func locate(settings: AppSettings) -> Result<PythonRuntime, PythonRuntimeError> {
    if let fromEnvironment = ProcessInfo.processInfo.environment["JETLINK_PYTHON"], !fromEnvironment.isEmpty {
      guard isRunnable(fromEnvironment) else { return .failure(.overrideMissing(fromEnvironment)) }
      return .success(PythonRuntime(executable: URL(filePath: fromEnvironment), source: .environment(fromEnvironment)))
    }
    if let override = settings.pythonOverride, !override.isEmpty {
      guard isRunnable(override) else { return .failure(.overrideMissing(override)) }
      return .success(PythonRuntime(executable: URL(filePath: override), source: .settings(override)))
    }
    guard let resources = Bundle.main.resourceURL else { return .failure(.notBundled) }
    let bundled = resources.appending(path: "python/bin/python3")
    guard isRunnable(bundled.path(percentEncoded: false)) else { return .failure(.notBundled) }
    return .success(PythonRuntime(executable: bundled, source: .bundled))
  }

  /// The embedded runtime's manifest, flattened for the About and Settings
  /// displays: the top level keys stay, and every entry of "packages" is lifted
  /// to the top level, so "onnxruntime" and "tinygrad" are keys of their own.
  /// Nil when there is no bundled runtime.
  static func manifest() -> [String: String]? {
    guard let resources = Bundle.main.resourceURL else { return nil }
    let url = resources.appending(path: "python/MANIFEST.json")
    guard let data = try? Data(contentsOf: url) else { return nil }
    guard let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return nil }
    var out: [String: String] = [:]
    for (key, value) in object where key != "packages" {
      out[key] = describe(value)
    }
    if let packages = object["packages"] as? [String: Any] {
      for (name, version) in packages {
        out[name] = describe(version)
      }
    }
    return out
  }

  private static func describe(_ value: Any) -> String {
    if let text = value as? String { return text }
    if let number = value as? NSNumber { return number.stringValue }
    return String(describing: value)
  }

  private static func isRunnable(_ path: String) -> Bool {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory), !isDirectory.boolValue else { return false }
    return FileManager.default.isExecutableFile(atPath: path)
  }
}

enum PythonRuntimeError: Error, LocalizedError, Equatable {
  case notBundled
  case overrideMissing(String)

  var errorDescription: String? {
    switch self {
    case .notBundled:
      return "This build has no bundled Python runtime. Run `make python` in macos/, or set JETLINK_PYTHON to a Python 3.14 interpreter with the jetlink package installed."
    case .overrideMissing(let path):
      return "The Python interpreter at \(path) does not exist or is not executable."
    }
  }
}
