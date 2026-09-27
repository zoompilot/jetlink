// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.

import JetlinkKit
import SwiftUI
import UIKit

/// Is this phone fast enough, and does it stay fast enough? Measured on the
/// phone, then over the real link from the comma, before any drive.
struct BenchmarkView: View {
  @Environment(AppModel.self) private var model
  @Environment(AppSettings.self) private var settings

  var body: some View {
    Form {
      phoneSection
      if let r = model.benchReport { resultSection(r) }
      commaSection
      accuracySection
    }
    .navigationTitle("Benchmark")
  }

  // MARK: on the phone

  private var canRun: Bool {
    model.engine.state == "ready" && !model.linkConnected && !model.benchRunning
  }

  private var phoneSection: some View {
    Section {
      if model.engine.state != "ready" {
        Text("Load a model first: Models tab, then Prepare or Load.").foregroundStyle(.secondary)
      } else if let units = model.engine.units {
        LabeledContent("Model", value: model.name(for: model.engine.sha256) ?? String(model.engine.sha256?.prefix(16) ?? ""))
        LabeledContent("Running on", value: units.label)
      }
      if model.linkConnected {
        Text("A comma is connected. Disconnect it to run this; its live numbers are on the Status tab.")
          .font(.footnote).foregroundStyle(.orange)
      }
      if let p = model.benchProgress {
        VStack(alignment: .leading, spacing: 6) {
          ProgressView(value: min(p.elapsed / max(p.total, 1), 1))
          Text("\(Int(p.elapsed)) of \(Int(p.total)) s, \(p.frames) frames" + (p.lastWindow.map { String(format: ", last 10 s p99 %.1f ms", $0.frame.p99) } ?? ""))
            .font(.footnote).monospacedDigit().foregroundStyle(.secondary)
        }
        Button("Stop", role: .destructive) { model.cancelBenchmark() }
      } else {
        Button("Run for 1 minute") { model.startBenchmark(seconds: 60) }.disabled(!canRun)
        Button("Run for 10 minutes, to see it heat up") { model.startBenchmark(seconds: 600) }.disabled(!canRun)
      }
      if let e = model.benchError { Text(e).font(.footnote).foregroundStyle(.red) }
    } header: {
      Text("On this phone")
    } footer: {
      Text(
        "Runs the loaded model 20 times a second, as the comma will, with made-up camera frames. It times the phone's share of each frame: the history queues, the model, and reading the answer back. The cable is not included; the test from the comma below adds it. Leave the phone as it will be in the car: charging, in its case or mount."
      )
    }
  }

  private func resultSection(_ r: BenchmarkReport) -> some View {
    Section {
      LabeledContent("Verdict") { Text(verdict(r)).foregroundStyle(verdictColor(r)).multilineTextAlignment(.trailing) }
      LabeledContent("Frame, mean", value: String(format: "%.1f ms", r.frame.mean))
      LabeledContent("Frame, p99") {
        Text(String(format: "%.1f ms", r.frame.p99)).foregroundStyle(r.frame.p99 <= 35 ? .green : (r.frame.p99 <= 50 ? .orange : .red))
      }
      LabeledContent("Slowest", value: String(format: "%.1f ms", r.frame.max))
      LabeledContent("Model alone, mean", value: String(format: "%.1f ms", r.accelerator.mean))
      LabeledContent("Over 35 ms / over 50 ms", value: "\(r.over35) / \(r.over50) of \(r.frames)")
      LabeledContent("Temperature", value: "\(thermalLabel(r.thermalAtStart)) → \(thermalLabel(r.thermalAtEnd))")
      if r.windows.count > 1 {
        DisclosureGroup("Every 10 seconds") {
          ForEach(r.windows, id: \.startSecond) { w in
            HStack {
              Text("\(w.startSecond) s").frame(width: 50, alignment: .leading)
              Text(String(format: "mean %.1f  p99 %.1f", w.frame.mean, w.frame.p99))
              Spacer()
              Text(thermalLabel(w.thermal)).foregroundStyle(.secondary)
            }
            .font(.caption.monospacedDigit())
          }
        }
      }
      ShareLink(item: r.text) { Label("Share the results", systemImage: "square.and.arrow.up") }
    } header: {
      Text(r.cancelled ? "Result (stopped early)" : "Result")
    } footer: {
      Text(
        "The comma allows 50 ms for a whole frame, cable included. 35 ms or less here leaves room for the cable; the test from the comma measures it. Compare the first and last 10-second windows of a long run to see the phone slow as it warms."
      )
    }
  }

  private func verdict(_ r: BenchmarkReport) -> String {
    if r.frame.p99 <= 35 && r.over50 == 0 { return "Fast enough, with room for the link" }
    if r.frame.p99 <= 50 { return "Tight: little room for the link" }
    return "Too slow for 20 frames a second"
  }

  private func verdictColor(_ r: BenchmarkReport) -> Color {
    r.frame.p99 <= 35 && r.over50 == 0 ? .green : (r.frame.p99 <= 50 ? .orange : .red)
  }

  // MARK: from the comma

  private var commaCommand: String? {
    guard let sha = model.engine.sha256, let n = model.loadedModelBytes, let ip = model.serverAddress else { return nil }
    return """
      cd /data/openpilot/jetlink_repo && PYTHONPATH=/data/openpilot /usr/local/venv/bin/python3 \
      scripts/bench_link.py --host \(ip) --port \(settings.port) --sha256 \(sha) --nbytes \(n) --rate 20 --n 1200
      """
  }

  private var commaSection: some View {
    Section {
      if let cmd = commaCommand {
        CommandRow(command: cmd)
      } else {
        Text("Load a model and connect the link to get the command.").foregroundStyle(.secondary)
      }
    } header: {
      Text("Over the link, from the comma")
    } footer: {
      Text(
        "Parked, with the link up. Run it on the comma over SSH; zoompilot already carries this repository as jetlink_repo. Turn Accelerator Link off first, so the comma's own client is not holding the phone's server. It sends 1,200 real-sized frames at 20 a second and reports the round trip the car will see; \"frames over the 50 ms budget\" should be 0."
      )
    }
  }

  // MARK: accuracy

  private var parityCommand: String? {
    guard let sha = model.engine.sha256, let n = model.loadedModelBytes,
      let ip = model.addresses.first(where: { $0.kind == "Wi-Fi" })?.address ?? model.serverAddress
    else { return nil }
    return """
      python3 scripts/verify_parity.py capture --host \(ip) --port \(settings.port) --sha256 \(sha) --nbytes \(n) --dir parity-iphone \
      && python3 scripts/verify_parity.py reference --onnx "$HOME/Library/Application Support/Jetlink/cache/models/\(sha.prefix(16)).onnx" --dir parity-iphone \
      && python3 scripts/verify_parity.py compare --dir parity-iphone
      """
  }

  private var accuracySection: some View {
    Section {
      if let cmd = parityCommand {
        CommandRow(command: cmd)
      } else {
        Text("Load a model to get the command.").foregroundStyle(.secondary)
      }
    } header: {
      Text("Accuracy, from your Mac")
    } footer: {
      Text(
        "Checks that this phone's \(model.engine.units?.label ?? "accelerator") computes what onnxruntime does on a computer, the same gate the Mac server passes. Run it in the jetlink checkout on a Mac on the same Wi-Fi. It reads the model where the Mac app keeps it; if your copy of the .onnx is elsewhere, change the path after --onnx. Wi-Fi is slow, but this test does not care. It should end with OK."
      )
    }
  }
}

/// A shell command with a Copy button.
struct CommandRow: View {
  let command: String
  @State private var copied = false

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Text(command).font(.caption.monospaced()).textSelection(.enabled)
      Button(copied ? "Copied" : "Copy", systemImage: copied ? "checkmark" : "doc.on.doc") {
        UIPasteboard.general.string = command
        copied = true
      }
      .buttonStyle(.borderless)
    }
  }
}
