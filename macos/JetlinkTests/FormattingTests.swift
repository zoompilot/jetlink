import Foundation
import Testing

@testable import Jetlink

@MainActor
@Suite("Formatting")
struct FormattingTests {
  // MARK: StatusBadge.summary

  private func summary(_ runState: ServerRunState, _ link: LinkEvent, _ engine: EngineEvent) -> (String, StatusBadge.Tone) {
    StatusBadge.summary(runState: runState, link: link, engine: engine)
  }

  /// Every summary the app can produce, one row per state it can be in.
  @Test("Every summary says one thing, and never what the badge already shows")
  func summaryTable() {
    let links: [(LinkState, LinkEvent)] = [
      (.waiting, PreviewData.linkWaiting),
      (.connected, PreviewData.linkConnected),
      (.disconnected, PreviewData.linkDisconnected),
    ]
    let engines: [(EngineState, EngineEvent)] = [
      (.none, PreviewData.engineNone),
      (.ready, PreviewData.engineReady),
      (.building, PreviewData.engineBuilding),
      (.loading, PreviewData.engineLoading),
      (.failed, PreviewData.engineFailed),
    ]

    // The server's own state is the whole story until it is serving.
    for (_, link) in links {
      for (_, engine) in engines {
        #expect(summary(.stopped, link, engine) == ("Stopped", .neutral))
        #expect(summary(.starting, link, engine) == ("Starting…", .info))
        #expect(summary(.stopping, link, engine) == ("Stopping…", .neutral))
        #expect(summary(.failed("no backend came up"), link, engine) == ("Failed", .bad))
      }
    }

    // While serving, a job in flight wins, then the link.
    for (linkState, link) in links {
      for (engineState, engine) in engines {
        let expected: (String, StatusBadge.Tone) =
          switch engineState {
          case .building: ("Preparing model", .info)
          case .loading: ("Loading model", .info)
          case .failed: ("Model failed", .bad)
          case .none, .ready:
            switch linkState {
            case .connected: ("Comma connected", .good)
            case .waiting: ("Waiting for comma", .neutral)
            case .disconnected: ("Comma disconnected", .warning)
            }
          }
        #expect(summary(.serving, link, engine) == expected)
      }
    }
  }

  // MARK: ByteCount

  @Test("A rate is a size with a per second suffix")
  func rate() {
    let rate = ByteCount.rate(41_000_000)
    #expect(rate.hasSuffix("/s"))
    #expect(rate.contains("41"))
    #expect(rate.contains("MB"))
    #expect(ByteCount.rate(0).hasSuffix("/s"))
    #expect(ByteCount.rate(-1) == ByteCount.rate(0))
    #expect(ByteCount.rate(.nan) == ByteCount.rate(0))
  }

  @Test("Sizes are decimal units with one decimal at most")
  func sizes() {
    #expect(ByteCount.string(765_953_504) == "766 MB")
    #expect(ByteCount.string(777_200_000) == "777.2 MB")
    #expect(ByteCount.string(23_200_000_000) == "23.2 GB")
    #expect(ByteCount.string(1_800_000_000) == "1.8 GB")
    #expect(ByteCount.string(5_900_000_000) == "5.9 GB")
    #expect(ByteCount.string(512) == "512 bytes")
    #expect(ByteCount.string(0) == "0 bytes")
    #expect(ByteCount.string(999_999) == "1 MB")
  }

  @Test("A model name loses its trailing build date, and nothing else")
  func displayName() {
    #expect(PreviewData.loadedRow.displayName == "BMRLNAP Model v4")
    #expect(PreviewData.notDownloadedRow.displayName == "Cinque Terre Model V2")
    #expect(PreviewData.localRow.displayName == "big_driving_supercombo.onnx")
    var parenthesised = PreviewData.localRow
    parenthesised.name = "Some Model (experimental)"
    #expect(parenthesised.displayName == "Some Model (experimental)")
    var midName = PreviewData.localRow
    midName.name = "Model (August 30, 2026) rev 2"
    #expect(midName.displayName == "Model (August 30, 2026) rev 2")
  }

  @Test("The runtime line names the runtime, its version and the hardware")
  func runtimeLine() {
    #expect(StatusView.runtimeLine(backend: "tinygrad", version: "0.14.0+1241484386bc", device: "METAL-Apple_M1_Pro") == "tinygrad 0.14.0, Apple M1 Pro")
    #expect(StatusView.runtimeLine(backend: "ort", version: "1.29.0", device: "coreml-Apple_M1_Pro") == "onnxruntime 1.29.0, Apple M1 Pro")
  }

  // MARK: Build times

  @Test("Catalog build times parse with and without fractional seconds")
  func buildTimeParsing() throws {
    let plain = try #require(BuildTime.date("2026-08-30T09:41:12Z"))
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime]
    #expect(formatter.string(from: plain) == "2026-08-30T09:41:12Z")

    let fractional = try #require(BuildTime.date("2026-09-08T11:04:00.123Z"))
    #expect(abs(fractional.timeIntervalSince1970 - 0.123 - fractional.timeIntervalSince1970.rounded(.down)) < 0.001)

    #expect(BuildTime.date(nil) == nil)
    #expect(BuildTime.date("") == nil)
    #expect(BuildTime.date("September 8, 2026") == nil)
  }

  @Test("An unknown build time shows nothing at all")
  func buildTimeText() {
    #expect(BuildTime.text(nil).isEmpty)
    #expect(BuildTime.text("not a date").isEmpty)
    #expect(!BuildTime.text("2026-08-30T09:41:12Z").isEmpty)
  }

  // MARK: ProgressRow

  @Test("Every stage the server sends has plain English for it")
  func stageNames() {
    #expect(ProgressRow.stageName("upload") == "Receiving model")
    #expect(ProgressRow.stageName("patch") == "Preparing the model")
    #expect(ProgressRow.stageName("parse") == "Reading the model")
    #expect(ProgressRow.stageName("build") == "Building")
    #expect(ProgressRow.stageName("save") == "Saving")
    #expect(ProgressRow.stageName("load") == "Loading")
    #expect(ProgressRow.stageName("failed") == "Failed")
    #expect(ProgressRow.stageName(nil) == "Working")
    #expect(ProgressRow.stageName("something new") == "Working")
  }

  // MARK: Status view text

  @Test("Backends are named the way the Settings picker names them")
  func backendDescription() {
    #expect(StatusView.backendDescription(backend: "ort", device: "coreml-Apple_M1_Pro") == "CoreML on the GPU")
    #expect(StatusView.backendDescription(backend: "ort", device: "ane-Apple_M1_Pro") == "CoreML with the Neural Engine")
    #expect(StatusView.backendDescription(backend: "tinygrad", device: "METAL") == "tinygrad on Metal")
    #expect(StatusView.backendDescription(backend: "trt", device: "cuda") == "trt")
    #expect(StatusView.backendDescription(backend: nil, device: nil) == "Unknown")
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

  @Test("Frame times read as mean, p99 and max in milliseconds")
  func frameTime() {
    let text = StatusView.frameTimeText(PreviewData.stats)
    #expect(text.contains("ms mean"))
    #expect(text.contains("ms p99"))
    #expect(text.contains("ms max"))
  }

  @Test("Prepared backends are listed once each, in plain names")
  func preparedFor() {
    #expect(ModelsView.preparedForText(PreviewData.loadedRow) == "CoreML")
    #expect(ModelsView.preparedForText(PreviewData.notDownloadedRow).isEmpty)
    #expect(ModelsView.backendName("trt") == "TensorRT")
    #expect(ModelsView.backendName("tinygrad") == "tinygrad")
  }

  @Test("Log lines are coloured by their level")
  func logTone() {
    #expect(LogsView.tone(for: PreviewData.logLines[5]) == .red)
    #expect(LogsView.tone(for: PreviewData.logLines[3]) == .orange)
    #expect(LogsView.tone(for: PreviewData.logLines[0]) == .primary)
  }
}
