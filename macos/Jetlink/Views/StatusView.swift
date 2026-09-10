import AppKit
import SwiftUI

/// What the server, the comma and the engine are doing right now.
struct StatusView: View {
  @Environment(ServerStore.self) private var server
  @Environment(ModelStore.self) private var models
  @Environment(AppSettings.self) private var settings
  @Environment(LogBuffer.self) private var logs
  @Binding var selection: SidebarItem?
  @State private var confirmingUnload = false

  var body: some View {
    Form {
      serverSection
      commaSection
      engineSection
      Section {
        HStack {
          Button("Reveal cache in Finder") {
            NSWorkspace.shared.activateFileViewerSelecting([cacheURL])
          }
          if server.engine.state == .ready {
            Button("Unload model") { confirmingUnload = true }
          }
        }
      }
    }
    .formStyle(.grouped)
    .confirmationDialog("Unload the model?", isPresented: $confirmingUnload) {
      Button("Unload model") { models.unload() }
      Button("Cancel", role: .cancel) {}
    } message: {
      Text("The comma will fall back to its small model until a model is loaded again.")
    }
  }

  // MARK: Server

  @ViewBuilder
  private var serverSection: some View {
    Section("Server") {
      LabeledContent("State") {
        StatusBadge(text: serverStateText, tone: serverStateTone, showsProgress: isBusy)
      }
      LabeledContent("Backend") {
        VStack(alignment: .trailing, spacing: 2) {
          Text(StatusView.backendDescription(backend: server.info?.backend, device: server.info?.device))
          if let info = server.info {
            Text(StatusView.runtimeLine(backend: info.backend, version: info.runtimeVersion, device: info.device))
              .font(.callout)
              .foregroundStyle(.secondary)
          }
        }
      }
      LabeledContent("Uptime") {
        if let startedAt = server.startedAt, server.runState == .serving {
          TimelineView(.periodic(from: .now, by: 30)) { context in
            Text(StatusView.uptimeText(from: startedAt, to: context.date))
          }
        } else {
          Text("Not running")
            .foregroundStyle(.secondary)
        }
      }
      if case let .failed(reason) = server.runState {
        VStack(alignment: .leading, spacing: 8) {
          Text("The server could not start.")
          Text(failureDetail(reason))
            .font(.system(size: 12, design: .monospaced))
            .foregroundStyle(.red)
            .textSelection(.enabled)
            .fixedSize(horizontal: false, vertical: true)
          if !lastLogLines.isEmpty {
            VStack(alignment: .leading, spacing: 1) {
              ForEach(Array(lastLogLines.enumerated()), id: \.offset) { _, line in
                Text(line)
                  .font(.system(size: 12, design: .monospaced))
                  .textSelection(.enabled)
                  .frame(maxWidth: .infinity, alignment: .leading)
              }
            }
          }
          Button("Show logs") { selection = .logs }
        }
      }
    }
  }

  private var serverStateText: String {
    switch server.runState {
    case .stopped: "Stopped"
    case .starting: "Starting…"
    case .serving: "Serving"
    case .stopping: "Stopping…"
    case .failed: "Failed"
    }
  }

  private var serverStateTone: StatusBadge.Tone {
    switch server.runState {
    case .stopped: .neutral
    case .starting: .info
    case .serving: .good
    case .stopping: .neutral
    case .failed: .bad
    }
  }

  private var isBusy: Bool {
    server.runState == .starting || server.runState == .stopping
  }

  // MARK: Comma

  @ViewBuilder
  private var commaSection: some View {
    Section {
      LabeledContent("Link") {
        VStack(alignment: .trailing, spacing: 2) {
          StatusBadge(text: linkText, tone: linkTone)
          if showsLinkDetail {
            Text(server.link.detail)
              .font(.callout)
              .foregroundStyle(.secondary)
          }
        }
      }
      if server.link.state == .connected, let stats = server.stats {
        LabeledContent("Frames", value: stats.frames.formatted(.number.grouping(.automatic)))
        LabeledContent("Rate", value: "\(stats.fps.formatted(.number.precision(.fractionLength(1)))) per second")
        LabeledContent("Frame time", value: StatusView.frameTimeText(stats))
        LabeledContent("GPU time", value: "\(stats.gpuMs.mean.formatted(.number.precision(.fractionLength(1)))) ms mean")
        LabeledContent("Slow frames") {
          Text(stats.slow.formatted())
            .foregroundStyle(stats.slow > 0 ? .red : .primary)
        }
        .help("Frames over 60 ms in the last second")
      }
    } header: {
      Text("Comma")
    } footer: {
      Text("The comma connects when it is plugged into a USB-A port with an A-to-C data cable. The small model keeps driving whenever the link is down.")
        .font(.callout)
        .foregroundStyle(.secondary)
    }
  }

  private var linkText: String {
    switch server.link.state {
    case .waiting:
      return "Waiting for comma"
    case .connected:
      let peer = server.link.peer ?? ""
      if peer.isEmpty || peer == "usb" {
        return "Connected over USB"
      }
      return "Connected over TCP from \(peer)"
    case .disconnected:
      return "Disconnected"
    }
  }

  /// The disconnect reason, and the address a TCP server is listening on. The
  /// USB "waiting for a jetlink gadget" line only repeats the badge.
  private var showsLinkDetail: Bool {
    guard !server.link.detail.isEmpty else { return false }
    switch server.link.state {
    case .disconnected: return true
    case .waiting: return server.info?.transport == "tcp"
    case .connected: return false
    }
  }

  private var linkTone: StatusBadge.Tone {
    switch server.link.state {
    case .waiting: .neutral
    case .connected: .good
    case .disconnected: .warning
    }
  }

  // MARK: Engine

  @ViewBuilder
  private var engineSection: some View {
    Section("Engine") {
      if showsEmptyState {
        VStack(alignment: .leading, spacing: 6) {
          Label("No model prepared", systemImage: "shippingbox")
            .font(.headline)
          Text(
            "Download and prepare the model your comma uses in Models. Keep Jetlink running afterwards; the model stays loaded and the comma connects to it immediately."
          )
          .font(.callout)
          .foregroundStyle(.secondary)
          .fixedSize(horizontal: false, vertical: true)
          Button("Open Models") { selection = .models }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
      } else {
        LabeledContent("Model") {
          VStack(alignment: .trailing, spacing: 2) {
            Text(loadedModelName)
            if let sha = server.engine.sha256 {
              Text(String(sha.prefix(16)))
                .font(.system(.callout, design: .monospaced))
                .foregroundStyle(.secondary)
            }
          }
        }
        LabeledContent("State") {
          StatusBadge(text: engineStateText, tone: engineStateTone)
        }
        if server.engine.state == .building || server.engine.state == .loading {
          ProgressRow(stage: server.engine.stage, frac: server.engine.frac, msg: server.engine.msg)
        }
        if server.engine.state == .failed, !server.engine.detail.isEmpty {
          Text(server.engine.detail)
            .foregroundStyle(.red)
            .textSelection(.enabled)
            .fixedSize(horizontal: false, vertical: true)
        }
      }
    }
  }

  private var showsEmptyState: Bool {
    server.engine.state == .none && (models.inventory?.artifacts.isEmpty ?? true)
  }

  private var loadedModelName: String {
    guard let sha = server.engine.sha256 else { return "None" }
    return models.rows.first { $0.sha256 == sha }?.displayName ?? "Unknown model"
  }

  private var engineStateText: String {
    switch server.engine.state {
    case .none: "None"
    case .building: "Preparing"
    case .loading: "Loading"
    case .ready: "Ready"
    case .failed: "Failed"
    }
  }

  private var engineStateTone: StatusBadge.Tone {
    switch server.engine.state {
    case .none: .neutral
    case .building, .loading: .info
    case .ready: .good
    case .failed: .bad
    }
  }

  // MARK: Failures

  /// What went wrong, in the server's own words, unless the app can see the
  /// reason itself: a build with no interpreter in it cannot start anything.
  private func failureDetail(_ reason: String) -> String {
    let overridden = settings.pythonOverride != nil || ProcessInfo.processInfo.environment["JETLINK_PYTHON"] != nil
    if EmbeddedPython.manifest() == nil, !overridden {
      return StatusView.missingPythonMessage
    }
    let failure = server.lastFailure ?? reason
    return failure.isEmpty ? reason : failure
  }

  static let missingPythonMessage = """
    This build has no bundled Python runtime. Run `make python` in macos/, or set JETLINK_PYTHON to a Python 3.14 \
    interpreter with the jetlink package installed.
    """

  private var lastLogLines: [String] {
    Array(logs.lines.suffix(20))
  }

  // MARK: Helpers

  private var cacheURL: URL {
    if let cache = server.info?.cache, !cache.isEmpty {
      return URL(fileURLWithPath: cache)
    }
    return settings.cacheDirectory
  }

  /// "tinygrad 0.14.0, Apple M1 Pro": what is actually running, under the
  /// backend's plain name. The device loses the backend prefix it repeats.
  static func runtimeLine(backend: String, version: String, device: String) -> String {
    let runtime =
      switch backend {
      case "ort": "onnxruntime"
      case "trt": "TensorRT"
      default: backend
      }
    let version = version.split(separator: "+", maxSplits: 1).first.map(String.init) ?? version
    var hardware = device
    if let dash = device.firstIndex(of: "-") {
      hardware = String(device[device.index(after: dash)...])
    }
    hardware = hardware.replacingOccurrences(of: "_", with: " ")
    let head = version.isEmpty ? runtime : "\(runtime) \(version)"
    return hardware.isEmpty ? head : "\(head), \(hardware)"
  }

  /// "CoreML on the GPU", "CoreML with the Neural Engine", "tinygrad on Metal",
  /// or the raw backend name when the server reports something else.
  static func backendDescription(backend: String?, device: String?) -> String {
    let device = device ?? ""
    if device.hasPrefix("ane") {
      return "CoreML with the Neural Engine"
    }
    switch backend {
    case "ort":
      return device.hasPrefix("coreml") || device.isEmpty ? "CoreML on the GPU" : "onnxruntime on \(device)"
    case "ane":
      return "CoreML with the Neural Engine"
    case "tinygrad":
      return "tinygrad on Metal"
    case let other?:
      return other
    case nil:
      return "Unknown"
    }
  }

  /// "31.2 ms mean, 38.0 ms p99, 41.5 ms max".
  static func frameTimeText(_ stats: StatsEvent) -> String {
    let one = FloatingPointFormatStyle<Double>.number.precision(.fractionLength(1))
    return "\(stats.totalMs.mean.formatted(one)) ms mean, \(stats.totalMs.p99.formatted(one)) ms p99, \(stats.totalMs.max.formatted(one)) ms max"
  }

  static func uptimeText(from start: Date, to now: Date) -> String {
    let seconds = max(0, Int(now.timeIntervalSince(start)))
    return Duration.seconds(seconds).formatted(.units(allowed: [.hours, .minutes], width: .wide))
  }
}

#Preview("Connected") {
  @Previewable @State var selection: SidebarItem? = .status
  StatusView(selection: $selection)
    .environment(
      ServerStore.preview(
        runState: .serving, info: PreviewData.serverInfo, link: PreviewData.linkConnected,
        engine: PreviewData.engineReady, stats: PreviewData.stats)
    )
    .environment(ModelStore.preview(catalog: PreviewData.catalog, inventory: PreviewData.inventory, engine: PreviewData.engineReady))
    .environment(AppSettings.preview())
    .environment(LogBuffer.preview(lines: PreviewData.logLines))
    .frame(width: 640, height: 620)
}

#Preview("Building") {
  @Previewable @State var selection: SidebarItem? = .status
  StatusView(selection: $selection)
    .environment(
      ServerStore.preview(
        runState: .serving, info: PreviewData.serverInfo, link: PreviewData.linkWaiting,
        engine: PreviewData.engineBuilding)
    )
    .environment(ModelStore.preview(catalog: PreviewData.catalog, inventory: PreviewData.inventory, engine: PreviewData.engineBuilding))
    .environment(AppSettings.preview())
    .environment(LogBuffer.preview(lines: PreviewData.logLines))
    .frame(width: 640, height: 620)
}

#Preview("Nothing prepared") {
  @Previewable @State var selection: SidebarItem? = .status
  StatusView(selection: $selection)
    .environment(ServerStore.preview(runState: .stopped, info: nil, link: PreviewData.linkWaiting, engine: PreviewData.engineNone))
    .environment(ModelStore.preview(catalog: nil, inventory: PreviewData.emptyInventory, engine: PreviewData.engineNone))
    .environment(AppSettings.preview())
    .environment(LogBuffer.preview(lines: PreviewData.logLines))
    .frame(width: 640, height: 620)
}
