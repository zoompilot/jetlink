import Foundation
import Observation

@MainActor
@Observable
final class AppSettings {
  enum Key {
    static let backend = "backend"
    static let transport = "transport"
    static let tcpPort = "tcpPort"
    static let cacheDirectory = "cacheDirectory"
    static let startServerOnLaunch = "startServerOnLaunch"
    static let keepAwakeWhileServing = "keepAwakeWhileServing"
    static let logLevel = "logLevel"
    static let pythonOverride = "pythonOverride"
  }

  @ObservationIgnored private let defaults: UserDefaults

  var backend: BackendChoice {
    didSet { defaults.set(backend.rawValue, forKey: Key.backend) }
  }

  var transport: TransportChoice {
    didSet { defaults.set(transport.rawValue, forKey: Key.transport) }
  }

  var tcpPort: Int {
    didSet { defaults.set(tcpPort, forKey: Key.tcpPort) }
  }

  var cacheDirectory: URL {
    didSet { defaults.set(cacheDirectory.path(percentEncoded: false), forKey: Key.cacheDirectory) }
  }

  var startServerOnLaunch: Bool {
    didSet { defaults.set(startServerOnLaunch, forKey: Key.startServerOnLaunch) }
  }

  var keepAwakeWhileServing: Bool {
    didSet { defaults.set(keepAwakeWhileServing, forKey: Key.keepAwakeWhileServing) }
  }

  var logLevel: String {
    didSet { defaults.set(logLevel, forKey: Key.logLevel) }
  }

  var pythonOverride: String? {
    didSet {
      if let pythonOverride, !pythonOverride.isEmpty {
        defaults.set(pythonOverride, forKey: Key.pythonOverride)
      } else {
        defaults.removeObject(forKey: Key.pythonOverride)
      }
    }
  }

  init(defaults: UserDefaults = .standard) {
    self.defaults = defaults
    backend = BackendChoice(rawValue: defaults.string(forKey: Key.backend) ?? "") ?? .auto
    transport = TransportChoice(rawValue: defaults.string(forKey: Key.transport) ?? "") ?? .usb
    let storedPort = defaults.integer(forKey: Key.tcpPort)
    tcpPort = storedPort > 0 ? storedPort : AppSettings.defaultTCPPort
    if let stored = defaults.string(forKey: Key.cacheDirectory), !stored.isEmpty {
      cacheDirectory = URL(filePath: stored)
    } else {
      cacheDirectory = AppSettings.defaultCacheDirectory
    }
    startServerOnLaunch = defaults.object(forKey: Key.startServerOnLaunch) as? Bool ?? true
    keepAwakeWhileServing = defaults.object(forKey: Key.keepAwakeWhileServing) as? Bool ?? true
    logLevel = defaults.string(forKey: Key.logLevel) ?? "INFO"
    pythonOverride = defaults.string(forKey: Key.pythonOverride)
  }

  nonisolated static let defaultTCPPort = 5599

  nonisolated static var applicationSupportDirectory: URL {
    let base =
      FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
      ?? URL(filePath: NSHomeDirectory()).appending(path: "Library/Application Support")
    return base.appending(path: "Jetlink")
  }

  nonisolated static var defaultCacheDirectory: URL {
    applicationSupportDirectory.appending(path: "cache")
  }

  /// The control socket lives in the per user temporary directory, because an
  /// AF_UNIX path on macOS may not exceed 104 bytes.
  nonisolated static var controlSocketURL: URL {
    URL(filePath: NSTemporaryDirectory()).appending(path: "jetlink-control.sock")
  }

  nonisolated static var logFileURL: URL {
    let base =
      FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask).first
      ?? URL(filePath: NSHomeDirectory()).appending(path: "Library")
    return base.appending(path: "Logs/Jetlink/server.log")
  }

  nonisolated static let previewSuiteName = "io.zoompilot.jetlink.preview"

  /// A settings object with the shipped defaults, for previews and tests. It
  /// never touches the real preferences.
  static func preview() -> AppSettings {
    let defaults = UserDefaults(suiteName: previewSuiteName) ?? .standard
    defaults.removePersistentDomain(forName: previewSuiteName)
    return AppSettings(defaults: defaults)
  }
}
