import Foundation
import JetlinkORT
import Observation

/// Where the server runs the model. The Swift server has the two the Python
/// server's CoreML backend had; tinygrad went with the Python server.
enum BackendChoice: String, CaseIterable, Codable, Sendable {
  case auto, coreml

  /// What the Swift server is asked to be.
  var profile: OrtProfile {
    switch self {
    case .auto: .ane
    case .coreml: .coreml
    }
  }

  var title: String {
    switch self {
    case .auto: "CoreML with Neural Engine"
    case .coreml: "CoreML (GPU)"
    }
  }

  var shortTitle: String {
    switch self {
    case .auto: "Neural Engine"
    case .coreml: "CoreML GPU"
    }
  }
}

/// How the comma reaches the server: its USB gadget, or TCP for a bench client.
enum TransportChoice: String, CaseIterable, Codable, Sendable {
  case usb, tcp
}

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
    static let keepAwakeOnBattery = "keepAwakeOnBattery"
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

  var keepAwakeOnBattery: Bool {
    didSet { defaults.set(keepAwakeOnBattery, forKey: Key.keepAwakeOnBattery) }
  }

  init(defaults: UserDefaults = .standard) {
    self.defaults = defaults
    // A stored "tinygrad" (the removed Python backend) or "ane" (an older name
    // for the split) reads as Automatic.
    backend = BackendChoice(rawValue: defaults.string(forKey: Key.backend) ?? "") ?? .auto
    transport = TransportChoice(rawValue: defaults.string(forKey: Key.transport) ?? "") ?? .usb
    let storedPort = defaults.integer(forKey: Key.tcpPort)
    tcpPort = storedPort > 0 ? storedPort : AppSettings.defaultTCPPort
    if let stored = defaults.string(forKey: Key.cacheDirectory), !stored.isEmpty {
      cacheDirectory = URL(filePath: stored)
    } else {
      cacheDirectory = AppSettings.defaultCacheDirectory
    }
    // bool(forKey:) also reads "YES" and "NO" from launch arguments
    // (-startServerOnLaunch NO), which an `as? Bool` cast ignores.
    startServerOnLaunch = defaults.object(forKey: Key.startServerOnLaunch) == nil ? true : defaults.bool(forKey: Key.startServerOnLaunch)
    keepAwakeWhileServing = defaults.object(forKey: Key.keepAwakeWhileServing) == nil ? true : defaults.bool(forKey: Key.keepAwakeWhileServing)
    keepAwakeOnBattery = defaults.object(forKey: Key.keepAwakeOnBattery) == nil ? false : defaults.bool(forKey: Key.keepAwakeOnBattery)
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
