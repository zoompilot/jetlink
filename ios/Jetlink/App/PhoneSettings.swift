import Foundation
import JetlinkORT
import Observation

/// The few things worth changing on a phone, kept in UserDefaults.
@MainActor
@Observable
final class PhoneSettings {
  private let defaults: UserDefaults

  /// The TCP port the server listens on while `listening`, where a comma with
  /// Jetlink set to Wi-Fi, and bench tools such as `bench_link.py --host`,
  /// reach the phone over Wi-Fi. Over the cable the phone dials the comma instead.
  var port: UInt16 {
    didSet { defaults.set(Int(port), forKey: Keys.port) }
  }

  /// Developer settings shown. Seven taps on the version turn it on or off.
  var developer: Bool {
    didSet { defaults.set(developer, forKey: Keys.developer) }
  }

  /// Connect over Wi-Fi, a developer setting: the server listens on `port`
  /// for a comma with Jetlink set to Wi-Fi, or a bench tool. Off by default,
  /// since it listens on every network the phone is on.
  var overWiFi: Bool {
    didSet { defaults.set(overWiFi, forKey: Keys.overWiFi) }
  }

  /// Does the server listen on `port`? Only with Connect over Wi-Fi on and the
  /// developer settings shown; otherwise the phone only dials the comma over
  /// the cable, which is all driving needs.
  var listening: Bool { developer && overWiFi }

  /// What runs the model. Automatic until one is chosen by hand.
  var device: Processor {
    didSet { defaults.set(device.rawValue, forKey: Keys.device) }
  }

  /// What Compare Processors found best for each model, by its sha256, as
  /// the profile's name. Automatic runs a model on its entry, and on
  /// `Processor.fallback` until it has one.
  var automaticPicks: [String: String] {
    didSet { defaults.set(automaticPicks, forKey: Keys.automaticPicks) }
  }

  /// The profile the server runs `sha256` on.
  func profile(for sha256: String?) -> OrtProfile {
    if let profile = device.profile { return profile }
    return sha256.flatMap { automaticPicks[$0] }.flatMap(OrtProfile.init(rawValue:)) ?? Processor.fallback
  }

  /// A small GPU job between frames so the GPU does not clock down in the gaps.
  /// Off unless chosen: on an iPad Air (M1) with Light Speed 2 over 3,600
  /// frames at 20 Hz it cost more than it saved, 159 frames over 46 ms and a
  /// p99 of 49.4 ms with it on against none and 41.9 ms with it off.
  var keepGPUAwake: Bool {
    didSet { defaults.set(keepGPUAwake, forKey: Keys.keepGPUAwake) }
  }

  /// A CPU core kept busy between frames while the Neural Engine runs the
  /// model, so the next frame is not waiting on a core that went to sleep.
  var keepCPUWarm: Bool {
    didSet { defaults.set(keepCPUWarm, forKey: Keys.keepCPUWarm) }
  }

  /// The screen stays on while Jetlink is open: iOS suspends an app whose
  /// phone locks, and a suspended app serves nothing.
  var keepScreenOn: Bool {
    didSet { defaults.set(keepScreenOn, forKey: Keys.keepScreenOn) }
  }

  init(defaults: UserDefaults = .standard) {
    self.defaults = defaults
    let port = defaults.integer(forKey: Keys.port)
    self.port = port > 0 && port < 65_536 ? UInt16(port) : 5599
    self.developer = defaults.bool(forKey: Keys.developer)
    self.overWiFi = defaults.bool(forKey: Keys.overWiFi)
    // A device stored before Automatic existed was chosen by hand, and stays.
    self.device = defaults.string(forKey: Keys.device).flatMap(Processor.init(rawValue:)) ?? .automatic
    self.automaticPicks = defaults.dictionary(forKey: Keys.automaticPicks) as? [String: String] ?? [:]
    self.keepGPUAwake = defaults.object(forKey: Keys.keepGPUAwake) as? Bool ?? false
    self.keepCPUWarm = defaults.object(forKey: Keys.keepCPUWarm) as? Bool ?? true
    self.keepScreenOn = defaults.object(forKey: Keys.keepScreenOn) as? Bool ?? true
  }

  /// Where models and prepared engines live. Application Support, not Caches:
  /// iOS empties Caches when space runs low, and a missing engine costs the
  /// comma its big model on the next drive.
  static var cacheDirectory: URL {
    URL.applicationSupportDirectory.appending(path: "Jetlink/cache", directoryHint: .isDirectory)
  }

  private enum Keys {
    static let port = "port"
    static let developer = "developer"
    static let overWiFi = "overWiFi"
    static let device = "device"
    static let automaticPicks = "automaticPicks"
    static let keepGPUAwake = "keepGPUAwake"
    static let keepCPUWarm = "keepCPUWarm"
    static let keepScreenOn = "keepScreenOn"
  }
}

/// The Processor setting: Automatic, or one of the server's profiles by its
/// own name, which is also what is stored.
enum Processor: String, Hashable {
  case automatic = "auto"
  case aneBoth = "ane-both"
  case ane
  case coreml
  case cpu

  /// What Automatic runs a model on unless a comparison has picked another
  /// for it: the Neural Engine alone, which draws the least power and heats
  /// the device least (iPad Air M1, Light Speed 2: 25.1 ms median against
  /// 32.2 with the GPU). Its policy runs in lower precision, and only a
  /// comparison checks a model's outputs on it.
  #if targetEnvironment(simulator)
    // No Neural Engine there: the split it has always run, on the CPU.
    static let fallback = OrtProfile.ane
  #else
    static let fallback = OrtProfile.aneBoth
  #endif

  /// The profile chosen by hand, or nil for Automatic.
  var profile: OrtProfile? {
    self == .automatic ? nil : OrtProfile(rawValue: rawValue)
  }

  var title: String {
    profile?.title ?? "Automatic"
  }
}

extension OrtProfile {
  /// How the Settings picker and the screens name each device.
  var title: String {
    switch self {
    case .ane: "Neural Engine + GPU"
    case .aneBoth: "Neural Engine Only"
    case .coreml: "GPU Only"
    case .cpu: "CPU"
    // the Mac's and Android's, never offered here
    case .aneWhole, .htp, .htpWhole, .gpu: rawValue
    }
  }
}
