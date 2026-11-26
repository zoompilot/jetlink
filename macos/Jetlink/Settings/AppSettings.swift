import Foundation
import JetlinkORT
import JetlinkServer
import Observation

/// Where the server runs the model, in the picker's order.
enum BackendChoice: String, CaseIterable, Codable, Sendable {
  case auto
  case aneBoth = "ane-both"
  case ane
  case coreml

  /// What Automatic is on `chip`, as `HostChip.name()` names it. The Neural
  /// Engine alone draws the least power and heats the Mac least, so it is the
  /// default everywhere but the M1 Pro, Max and Ultra, whose Neural Engine is
  /// slower with the whole model than with the GPU's help (M1 Pro, Light
  /// Speed 2: 32 ms a frame against 28).
  nonisolated static func automatic(chip: String) -> OrtProfile {
    ["Apple M1 Pro", "Apple M1 Max", "Apple M1 Ultra"].contains(chip) ? .ane : .aneBoth
  }

  /// What the Swift server is asked to be.
  var profile: OrtProfile {
    switch self {
    case .auto: BackendChoice.automatic(chip: HostChip.name())
    case .aneBoth: .aneBoth
    case .ane: .ane
    case .coreml: .coreml
    }
  }

  /// A processor as the Settings picker and the status lines name it.
  nonisolated static func title(_ profile: OrtProfile) -> String {
    switch profile {
    case .ane: "Neural Engine + GPU"
    case .aneBoth: "Neural Engine Only"
    case .coreml: "GPU Only"
    default: profile.rawValue
    }
  }

  /// Automatic reads as what it runs on here.
  var title: String { BackendChoice.title(profile) }
}

/// How the comma reaches the server: its USB gadget, or TCP for a bench client.
enum TransportChoice: String, Sendable {
  case usb, tcp
}

@MainActor
@Observable
final class AppSettings {
  enum Key {
    static let backend = "backend"
    static let developerTransport = "developerTransport"
    static let tcpPort = "tcpPort"
    static let cacheDirectory = "cacheDirectory"
    static let startServerOnLaunch = "startServerOnLaunch"
    static let keepAwakeWhileServing = "keepAwakeWhileServing"
    static let keepAwakeLidClosed = "keepAwakeLidClosed"
    static let keepGPUAwake = "keepGPUAwake"
    static let keepCPUWarm = "keepCPUWarm"
  }

  @ObservationIgnored private let defaults: UserDefaults

  var backend: BackendChoice {
    didSet { defaults.set(backend.rawValue, forKey: Key.backend) }
  }

  /// A small GPU job between frames so the GPU does not clock down in the
  /// gaps. On unless turned off: on an M1 Pro the split ran 36.5 ms a frame
  /// with it against 43.3 without.
  var keepGPUAwake: Bool {
    didSet { defaults.set(keepGPUAwake, forKey: Key.keepGPUAwake) }
  }

  /// A CPU core kept busy between frames, so the next frame is not waiting on
  /// a core that went to sleep. It shows as one whole core in Activity Monitor.
  var keepCPUWarm: Bool {
    didSet { defaults.set(keepCPUWarm, forKey: Key.keepCPUWarm) }
  }

  /// Read once at launch, never written: only `defaults write` or a launch
  /// argument (-developerTransport tcp -tcpPort 5599) turns TCP on.
  let transport: TransportChoice

  let tcpPort: Int

  var cacheDirectory: URL {
    didSet { defaults.set(cacheDirectory.path(percentEncoded: false), forKey: Key.cacheDirectory) }
  }

  var startServerOnLaunch: Bool {
    didSet { defaults.set(startServerOnLaunch, forKey: Key.startServerOnLaunch) }
  }

  var keepAwakeWhileServing: Bool {
    didSet { defaults.set(keepAwakeWhileServing, forKey: Key.keepAwakeWhileServing) }
  }

  var keepAwakeLidClosed: Bool {
    didSet { defaults.set(keepAwakeLidClosed, forKey: Key.keepAwakeLidClosed) }
  }

  init(defaults: UserDefaults = .standard) {
    self.defaults = defaults
    // A stored "tinygrad" (the removed Python backend) reads as Automatic.
    backend = BackendChoice(rawValue: defaults.string(forKey: Key.backend) ?? "") ?? .auto
    // Not "transport": that is what the removed Settings picker stored, and a
    // stored "tcp" there must not keep the app off USB.
    transport = TransportChoice(rawValue: defaults.string(forKey: Key.developerTransport) ?? "") ?? .usb
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
    keepAwakeLidClosed = defaults.bool(forKey: Key.keepAwakeLidClosed)
    keepGPUAwake = defaults.object(forKey: Key.keepGPUAwake) == nil ? true : defaults.bool(forKey: Key.keepGPUAwake)
    keepCPUWarm = defaults.object(forKey: Key.keepCPUWarm) == nil ? true : defaults.bool(forKey: Key.keepCPUWarm)
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
