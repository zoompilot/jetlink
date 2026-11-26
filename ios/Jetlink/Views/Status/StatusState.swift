import Foundation
import JetlinkKit
import JetlinkUI
import SwiftUI

/// Everything the Status tab draws, as plain values, so the views are a
/// function of it and previews need no server.
struct StatusState: Equatable {
  var runState: ServerRunState = .serving
  var link: LinkEvent = .waiting
  var engine: EngineEvent = .none
  /// The engine's model by name, when the catalog knows it.
  var modelName: String?
  /// The last ten seconds of frames: steadier than one second for a number
  /// read at a glance, and still quick to show a change.
  var recent: StatsEvent?
  /// One summary a second, for the history chart.
  var history: [StatsSample] = []
  /// The phone's address on the comma's cable network, while the cable is in.
  var cableAddress: String?
  /// What the connected comma's link is carried over: USB 3, USB 2 or TCP.
  var linkMedium: LinkMedium?
  var health = DeviceHealth()
  /// "iPhone" or "iPad", over the device's own tiles.
  var deviceName = "iPhone"
  /// The model a comma nobody changed asks for, for the empty state.
  var defaultModel: ModelRow?
  var hasPreparedModel = false
  /// The model list could not be fetched, and there is none cached.
  var catalogUnavailable = false
  /// iOS refuses Jetlink the local network, so the comma cannot connect.
  var localNetworkDenied = false
  /// The scene is not in front while serving: iOS will suspend the app.
  var needsForeground = false
  /// Too little free space to keep the model in use on the Neural Engine.
  var storageWarning: String?
  /// Frames far over the budget: the Neural Engine is not running the model.
  var slowWarning: String?

  /// The comma's frames, while the main card shows them.
  var servedStats: StatsEvent? {
    if case .budget(let stats) = hero { stats } else { nil }
  }

  /// What the main card shows.
  enum Hero: Equatable {
    /// The comma is connected to a loaded model and sending it frames.
    case budget(StatsEvent)
    /// The comma's drive holds the link, with no frames in the last ten
    /// seconds: it keeps its own model while engaged and switches only once
    /// nothing is, which is also where a lost link reconnects to mid-drive.
    case smallModel
    /// Connected to a loaded model with no drive on the link: the call the
    /// comma holds while parked, or its model fetch.
    case idle
    case progress(EngineEvent)
    case waiting
    case noModel
    case failed(String)
  }

  var hero: Hero {
    if case .failed(let reason) = runState {
      return .failed(reason)
    }
    switch engine.state {
    case .building, .loading:
      return .progress(engine)
    case .failed:
      return .failed(engine.detail.isEmpty ? "The model could not be prepared." : engine.detail)
    case .ready:
      guard link.state == .connected else { return .waiting }
      return recent.map { .budget($0) } ?? (link.isDrive ? .smallModel : .idle)
    case .none:
      return hasPreparedModel ? .waiting : .noModel
    }
  }

  /// One word or two for where things stand, and what it means.
  struct Summary: Equatable {
    let title: String
    let symbol: String
    let tone: StatusBadge.Tone
  }

  var summary: Summary {
    switch runState {
    case .failed: return Summary(title: "Stopped", symbol: "exclamationmark.octagon.fill", tone: .bad)
    case .starting: return Summary(title: "Starting", symbol: "hourglass", tone: .info)
    case .stopping, .stopped: return Summary(title: "Stopped", symbol: "stop.circle.fill", tone: .neutral)
    case .serving: break
    }
    switch engine.state {
    case .building: return Summary(title: "Preparing Model", symbol: "gearshape.2.fill", tone: .info)
    case .loading: return Summary(title: "Loading Model", symbol: "arrow.down.circle.fill", tone: .info)
    case .failed: return Summary(title: "Model Failed", symbol: "exclamationmark.triangle.fill", tone: .bad)
    case .ready, .none: break
    }
    if localNetworkDenied && link.state != .connected {
      return Summary(title: "Local Network Off", symbol: "wifi.exclamationmark", tone: .bad)
    }
    switch link.state {
    case .connected:
      // orange over USB 2: the title is the one line both orientations show
      return engine.state == .ready
        ? Summary(title: "Connected", symbol: "car.fill", tone: linkMedium?.isSlow == true ? .warning : .good)
        : Summary(title: "No Model", symbol: "shippingbox.fill", tone: .warning)
    case .waiting:
      return Summary(title: "Waiting for Comma", symbol: "cable.connector", tone: .neutral)
    case .disconnected:
      return Summary(title: "Disconnected", symbol: "cable.connector.slash", tone: .warning)
    }
  }

  /// The summary with the link it is over: "Connected over USB 3".
  var headline: String {
    if summary.title == "Connected", let linkMedium {
      return "Connected over \(linkMedium.phoneTitle)"
    }
    return summary.title
  }

  /// The line under the title: the summary and the model.
  var subtitle: String {
    [headline, modelName].compactMap { $0 }.joined(separator: " · ")
  }
}

extension LinkMedium {
  /// The link as the phone names it: anything but the cable reaches the
  /// phone over Wi-Fi.
  var phoneTitle: String {
    self == .tcp ? "Wi-Fi" : title
  }
}
