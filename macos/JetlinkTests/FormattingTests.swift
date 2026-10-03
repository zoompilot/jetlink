import Foundation
import JetlinkUI
import Testing

@testable import Jetlink

/// The text only the Mac's own views produce. What both apps show is tested in
/// JetlinkKit's FormattingTests.
@MainActor
@Suite("Formatting")
struct FormattingTests {
  @Test("The runtime line names onnxruntime, its version and the hardware")
  func runtimeLine() {
    #expect(StatusView.runtimeLine(version: "1.29.0", device: "coreml-Apple_M1_Pro") == "onnxruntime 1.29.0, Apple M1 Pro")
    #expect(StatusView.runtimeLine(version: "1.29.0+abc", device: "") == "onnxruntime 1.29.0")
  }

  @Test("Backends are named the way the Settings picker names them")
  func backendTitles() {
    #expect(BackendChoice.auto.title == "CoreML with Neural Engine")
    #expect(BackendChoice.coreml.title == "CoreML (GPU)")
    #expect(BackendChoice.auto.shortTitle == "Neural Engine")
  }

  @Test("An uptime under a minute says so instead of showing zero")
  func uptime() {
    let start = Date(timeIntervalSince1970: 1_757_440_000)
    #expect(StatusView.uptimeText(from: start, to: start) == "Less than a minute")
    #expect(StatusView.uptimeText(from: start, to: start.addingTimeInterval(59)) == "Less than a minute")
    #expect(StatusView.uptimeText(from: start, to: start.addingTimeInterval(60)) == "1 minute")
    #expect(StatusView.uptimeText(from: start, to: start.addingTimeInterval(150)) == "2 minutes")
    #expect(StatusView.uptimeText(from: start, to: start.addingTimeInterval(3900)) == "1 hour, 5 minutes")
  }

  @Test("Log lines are coloured by their level")
  func logTone() {
    #expect(LogTone.color(for: PreviewData.logLines[5]) == .red)
    #expect(LogTone.color(for: PreviewData.logLines[3]) == .orange)
    #expect(LogTone.color(for: PreviewData.logLines[0]) == .primary)
  }
}
