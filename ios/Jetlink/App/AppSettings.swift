// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.

import Foundation
import JetlinkKit
import Observation

/// The user's choices, kept in UserDefaults.
@MainActor
@Observable
final class AppSettings {
  @ObservationIgnored private let defaults: UserDefaults

  var compute: ComputePreference { didSet { defaults.set(compute.rawValue, forKey: Keys.compute) } }
  var port: Int { didSet { defaults.set(port, forKey: Keys.port) } }
  var keepScreenAwake: Bool { didSet { defaults.set(keepScreenAwake, forKey: Keys.keepScreenAwake) } }
  var startOnLaunch: Bool { didSet { defaults.set(startOnLaunch, forKey: Keys.startOnLaunch) } }
  var gpuKeepAlive: Bool { didSet { defaults.set(gpuKeepAlive, forKey: Keys.gpuKeepAlive) } }
  var cpuKeepWarm: Bool { didSet { defaults.set(cpuKeepWarm, forKey: Keys.cpuKeepWarm) } }

  enum Keys {
    static let compute = "compute"
    static let port = "port"
    static let keepScreenAwake = "keepScreenAwake"
    static let startOnLaunch = "startOnLaunch"
    static let gpuKeepAlive = "gpuKeepAlive"
    static let cpuKeepWarm = "cpuKeepWarm"
  }

  init(defaults: UserDefaults = .standard) {
    self.defaults = defaults
    defaults.register(defaults: [
      Keys.compute: ComputePreference.auto.rawValue,
      Keys.port: Int(JetlinkKit.defaultPort),
      Keys.keepScreenAwake: true,
      Keys.startOnLaunch: true,
      Keys.gpuKeepAlive: true,
      Keys.cpuKeepWarm: true,
    ])
    compute = ComputePreference(rawValue: defaults.string(forKey: Keys.compute) ?? "") ?? .auto
    let p = defaults.integer(forKey: Keys.port)
    port = (1024...65535).contains(p) ? p : Int(JetlinkKit.defaultPort)
    keepScreenAwake = defaults.bool(forKey: Keys.keepScreenAwake)
    startOnLaunch = defaults.bool(forKey: Keys.startOnLaunch)
    gpuKeepAlive = defaults.bool(forKey: Keys.gpuKeepAlive)
    cpuKeepWarm = defaults.bool(forKey: Keys.cpuKeepWarm)
  }
}
