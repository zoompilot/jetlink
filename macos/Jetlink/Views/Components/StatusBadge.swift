import SwiftUI

/// A capsule with a coloured dot and a short piece of text, used wherever a
/// state has to be readable at a glance.
struct StatusBadge: View {
  enum Tone: Equatable, Sendable {
    case neutral, info, good, warning, bad

    var color: Color {
      switch self {
      case .neutral: .secondary
      case .info: .blue
      case .good: .green
      case .warning: .orange
      case .bad: .red
      }
    }
  }

  let text: String
  let tone: Tone
  var showsProgress: Bool = false

  init(text: String, tone: Tone, showsProgress: Bool = false) {
    self.text = text
    self.tone = tone
    self.showsProgress = showsProgress
  }

  var body: some View {
    HStack(spacing: 6) {
      Circle()
        .fill(tone.color)
        .frame(width: 6, height: 6)
      Text(text)
      if showsProgress {
        ProgressView()
          .controlSize(.small)
          .progressViewStyle(.circular)
      }
    }
    .padding(.horizontal, 8)
    .padding(.vertical, 3)
    .background(Capsule().fill(tone.color.opacity(0.12)))
    .accessibilityElement(children: .combine)
  }

  /// The one line that describes the whole app: what the server is doing, and
  /// then what the comma and the engine are doing when it is serving.
  static func summary(runState: ServerRunState, link: LinkEvent, engine: EngineEvent) -> (String, Tone) {
    switch runState {
    case .stopped:
      return ("Stopped", .neutral)
    case .starting:
      return ("Starting…", .info)
    case .stopping:
      return ("Stopping…", .neutral)
    case .failed:
      return ("Failed", .bad)
    case .serving:
      break
    }

    if link.state == .disconnected {
      return ("Comma disconnected", .warning)
    }
    let connected = link.state == .connected
    switch engine.state {
    case .building, .loading:
      return (connected ? "Comma connected, preparing" : "Preparing a model", .info)
    case .failed:
      return (connected ? "Comma connected, model failed" : "Model failed", .bad)
    case .none, .ready:
      return connected ? ("Comma connected", .good) : ("Waiting for comma", .neutral)
    }
  }
}

#Preview {
  VStack(alignment: .leading, spacing: 8) {
    StatusBadge(text: "Stopped", tone: .neutral)
    StatusBadge(text: "Starting…", tone: .info, showsProgress: true)
    StatusBadge(text: "Serving", tone: .good)
    StatusBadge(text: "Disconnected", tone: .warning)
    StatusBadge(text: "Failed", tone: .bad)
  }
  .padding()
}
