import AppKit
import SwiftUI

/// The menu behind the menu bar icon: what is happening, and the two things
/// worth doing without opening the window.
struct MenuBarView: View {
  @Environment(ServerStore.self) private var server
  @Environment(ModelStore.self) private var models
  @Environment(\.openWindow) private var openWindow

  var body: some View {
    Button(statusLine) { }
      .disabled(true)
    Button(modelLine) { }
      .disabled(true)
    Divider()
    switch server.runState {
    case .stopped, .failed:
      Button("Start server") { server.start() }
    case .serving:
      Button("Stop server") { server.stop() }
    case .starting, .stopping:
      Button("Start server") { }
        .disabled(true)
    }
    Button("Open Jetlink") {
      openWindow(id: "main")
      NSApp.activate(ignoringOtherApps: true)
    }
    Divider()
    Button("Quit Jetlink") { NSApp.terminate(nil) }
      .keyboardShortcut("q")
  }

  /// "Serving, waiting for comma", plus the frame rate once frames are flowing.
  private var statusLine: String {
    let (text, _) = StatusBadge.summary(runState: server.runState, link: server.link, engine: server.engine)
    if server.link.state == .connected, let stats = server.stats, stats.fps > 0 {
      return "Comma connected, \(stats.fps.formatted(.number.precision(.fractionLength(1)))) frames per second"
    }
    return text
  }

  private var modelLine: String {
    let engine = server.engine
    switch engine.state {
    case .none:
      return "No model"
    case .building, .loading:
      let name = engine.state == .building ? "Preparing" : "Loading"
      return "\(name): \(Int((engine.frac * 100).rounded()))%"
    case .ready:
      return "Loaded: \(loadedName)"
    case .failed:
      return "The model failed"
    }
  }

  private var loadedName: String {
    guard let sha = server.engine.sha256 else { return "unknown model" }
    return models.rows.first { $0.sha256 == sha }?.name ?? String(sha.prefix(16))
  }
}
