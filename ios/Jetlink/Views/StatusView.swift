// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.

import JetlinkKit
import SwiftUI

/// What the server, the comma and the engine are doing right now.
struct StatusView: View {
  @Environment(AppModel.self) private var model
  @Environment(AppSettings.self) private var settings
  @State private var confirmingUnload = false

  var body: some View {
    Form {
      #if DEBUG
        Section {
          Label(
            "This is a Debug build. Frame times are only meaningful from a Release build: in Xcode, Product > Scheme > Edit Scheme > Run > Build Configuration > Release.",
            systemImage: "hammer.fill"
          )
          .foregroundStyle(.orange)
        }
      #endif
      if !model.inForeground {
        Section {
          Label("Keep Jetlink on screen while driving. iOS suspends background apps.", systemImage: "exclamationmark.triangle.fill")
            .foregroundStyle(.orange)
        }
      }
      serverSection
      commaSection
      engineSection
      if model.stats != nil { performanceSection }
      phoneSection
    }
    .navigationTitle("Jetlink")
    .confirmationDialog("Unload the model?", isPresented: $confirmingUnload) {
      Button("Unload model", role: .destructive) { model.unload() }
      Button("Cancel", role: .cancel) {}
    } message: {
      Text("The comma drives on its small model until one is loaded again.")
    }
  }

  // MARK: server

  private var serverSection: some View {
    Section {
      LabeledContent("Server") {
        StateText(
          text: model.serverRunning ? "Listening on port \(settings.port)" : (model.serverError == nil ? "Stopped" : "Could not start"),
          tone: model.serverRunning ? .good : (model.serverError == nil ? .neutral : .bad))
      }
      if let error = model.serverError {
        Text(error).font(.footnote).foregroundStyle(.red)
      }
      let wired = model.addresses.filter(\.isWired)
      if wired.isEmpty {
        LabeledContent("Wired link") {
          Text("None").foregroundStyle(.secondary)
        }
      }
      ForEach(model.addresses) { a in
        LabeledContent(a.kind) {
          Text("\(a.address):\(String(settings.port))")
            .monospacedDigit()
            .textSelection(.enabled)
            .foregroundStyle(a.isWired ? .primary : .secondary)
        }
      }
      Button(model.serverRunning ? "Stop server" : "Start server") {
        model.serverRunning ? model.stopServer() : model.startServer()
      }
    } header: {
      Text("Server")
    } footer: {
      Text(
        "The comma connects to a wired address. Set its JetlinkEndpoint to that address, or give the phone the fixed address the comma expects in Settings > Ethernet."
      )
    }
  }

  // MARK: comma

  private var commaSection: some View {
    Section("Comma") {
      switch model.link {
      case .stopped:
        LabeledContent("Link") { StateText(text: "Server stopped", tone: .neutral) }
      case .waiting:
        LabeledContent("Link") { StateText(text: "Waiting for the comma", tone: .neutral, busy: true) }
      case .connected(let peer):
        LabeledContent("Link") { StateText(text: "Connected", tone: .good) }
        LabeledContent("Comma address") { Text(peer).monospacedDigit() }
      case .disconnected(let detail):
        LabeledContent("Link") { StateText(text: "Disconnected", tone: .warning) }
        if !detail.isEmpty { Text(detail).font(.footnote).foregroundStyle(.secondary) }
      }
    }
  }

  // MARK: engine

  private var engineSection: some View {
    Section("Model") {
      let e = model.engine
      LabeledContent("State") { StateText(text: engineStateText(e), tone: engineTone(e), busy: e.isBusy) }
      if let sha = e.sha256 {
        LabeledContent("Model") {
          VStack(alignment: .trailing) {
            Text(model.name(for: sha) ?? "Unnamed model")
            Text(sha.prefix(16)).font(.caption).monospaced().foregroundStyle(.secondary)
          }
        }
      }
      if let units = e.units {
        LabeledContent("Running on", value: units.label)
      }
      if e.isBusy {
        VStack(alignment: .leading, spacing: 6) {
          ProgressView(value: min(max(e.frac, 0), 1))
          Text(e.msg.isEmpty ? (e.stage ?? "") : e.msg).font(.footnote).foregroundStyle(.secondary)
        }
      }
      if e.state == "failed", !e.detail.isEmpty {
        Text(e.detail).font(.footnote).foregroundStyle(.red)
      }
      NavigationLink("Benchmark") { BenchmarkView() }
      if e.state == "ready" {
        Button("Unload model", role: .destructive) { confirmingUnload = true }
      }
    }
  }

  private func engineStateText(_ e: EngineStatus) -> String {
    switch e.state {
    case "ready": "Ready"
    case "building": "Preparing"
    case "loading": "Loading"
    case "failed": "Failed"
    default: "No model loaded"
    }
  }

  private func engineTone(_ e: EngineStatus) -> StateText.Tone {
    switch e.state {
    case "ready": .good
    case "failed": .bad
    default: .neutral
    }
  }

  // MARK: performance

  private var performanceSection: some View {
    Section {
      if let s = model.stats {
        LabeledContent("Frames per second", value: String(format: "%.1f", s.fps))
        LabeledContent("Frame time, mean", value: String(format: "%.1f ms", s.meanMs))
        LabeledContent("Frame time, p99") {
          Text(String(format: "%.1f ms", s.p99Ms)).foregroundStyle(s.p99Ms > 50 ? .red : .primary)
        }
        LabeledContent("Slowest", value: String(format: "%.1f ms", s.maxMs))
        LabeledContent("Accelerator, mean", value: String(format: "%.1f ms", s.gpuMeanMs))
        if s.slow > 0 { LabeledContent("Over 60 ms", value: "\(s.slow)") }
      }
    } header: {
      Text("Last 5 seconds")
    } footer: {
      Text("Server-side time for each frame. The comma has 50 ms for the whole frame, including the link.")
    }
  }

  // MARK: phone

  private var phoneSection: some View {
    Section {
      LabeledContent("Temperature") {
        StateText(
          text: model.thermal.label,
          tone: model.thermal == .nominal || model.thermal == .fair ? .good : (model.thermal == .serious ? .warning : .bad))
      }
      LabeledContent("Battery") {
        Text(batteryText)
      }
      if model.memoryAvailable > 0 {
        LabeledContent("Memory the app may still use") {
          Text(ByteCountFormatter.string(fromByteCount: Int64(model.memoryAvailable), countStyle: .memory))
            .foregroundStyle(model.memoryAvailable < 1 << 30 ? .orange : .primary)
        }
      }
    } header: {
      Text("Phone")
    } footer: {
      Text("A hot phone runs its chip slower. Keep it out of direct sun and on a charger, ideally MagSafe or a hub with power pass-through.")
    }
  }

  private var batteryText: String {
    guard model.batteryLevel >= 0 else { return "Unknown" }
    let pct = Int((model.batteryLevel * 100).rounded())
    switch model.batteryState {
    case .charging: return "\(pct)%, charging"
    case .full: return "Full"
    case .unplugged: return "\(pct)%, not charging"
    default: return "\(pct)%"
    }
  }
}

/// A status word with a colored dot, and a spinner while something is under way.
struct StateText: View {
  enum Tone { case good, warning, bad, neutral }
  let text: String
  let tone: Tone
  var busy = false

  var body: some View {
    HStack(spacing: 6) {
      if busy {
        ProgressView().controlSize(.small)
      } else {
        Circle().fill(color).frame(width: 8, height: 8)
      }
      Text(text)
    }
  }

  private var color: Color {
    switch tone {
    case .good: .green
    case .warning: .orange
    case .bad: .red
    case .neutral: .secondary
    }
  }
}
