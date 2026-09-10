import SwiftUI

/// A coloured dot and a short piece of text in a capsule, used in form rows
/// wherever a state has to be readable at a glance. The toolbar's summary is
/// `ToolbarActivityView`, which shares `summary` with the menu bar.
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
      progress
    }
    .padding(.horizontal, 8)
    .padding(.vertical, 3)
    .modifier(PillBackground(tone: tone))
    .accessibilityElement(children: .combine)
  }

  @ViewBuilder
  private var progress: some View {
    if showsProgress {
      ProgressView()
        .controlSize(.small)
        .progressViewStyle(.circular)
    }
  }

  /// Liquid Glass takes the tint where the system has it, a plain fill before that.
  private struct PillBackground: ViewModifier {
    let tone: Tone

    @ViewBuilder
    func body(content: Content) -> some View {
      if #available(macOS 26, *) {
        content.glassEffect(.regular.tint(tone.color.opacity(0.35)), in: .capsule)
      } else {
        content.background(Capsule().fill(tone.color.opacity(0.12)))
      }
    }
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

    switch engine.state {
    case .building:
      return ("Preparing model", .info)
    case .loading:
      return ("Loading model", .info)
    case .failed:
      return ("Model failed", .bad)
    case .none, .ready:
      break
    }

    switch link.state {
    case .connected:
      return ("Comma connected", .good)
    case .waiting:
      return ("Waiting for comma", .neutral)
    case .disconnected:
      return ("Comma disconnected", .warning)
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
