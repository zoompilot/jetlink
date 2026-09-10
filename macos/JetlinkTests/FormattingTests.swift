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

  @Test("The server's own state wins over the link and the engine")
  func summaryWhenNotServing() {
    let link = PreviewData.linkConnected
    let engine = PreviewData.engineReady
    #expect(summary(.stopped, link, engine) == ("Stopped", .neutral))
    #expect(summary(.starting, link, engine) == ("Starting…", .info))
    #expect(summary(.stopping, link, engine) == ("Stopping…", .neutral))
    #expect(summary(.failed("the server could not start"), link, engine) == ("Failed", .bad))
  }

  @Test("Serving reads as the link state when the engine is idle or ready")
  func summaryWhileServing() {
    for engine in [PreviewData.engineNone, PreviewData.engineReady] {
      #expect(summary(.serving, PreviewData.linkWaiting, engine) == ("Serving, waiting for comma", .neutral))
      #expect(summary(.serving, PreviewData.linkConnected, engine) == ("Comma connected", .good))
      #expect(summary(.serving, PreviewData.linkDisconnected, engine) == ("Serving, the comma disconnected", .warning))
    }
  }

  @Test("A build or a load is worth saying, connected or not")
  func summaryWhilePreparing() {
    for engine in [PreviewData.engineBuilding, PreviewData.engineLoading] {
      #expect(summary(.serving, PreviewData.linkWaiting, engine) == ("Serving, preparing a model", .info))
      #expect(summary(.serving, PreviewData.linkConnected, engine) == ("Comma connected, preparing a model", .info))
    }
  }

  @Test("A failed engine reads as failed while the server keeps serving")
  func summaryWhenTheEngineFailed() {
    #expect(summary(.serving, PreviewData.linkWaiting, PreviewData.engineFailed) == ("Serving, the model failed", .bad))
    #expect(summary(.serving, PreviewData.linkConnected, PreviewData.engineFailed) == ("Comma connected, the model failed", .bad))
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

  @Test("Sizes are the Finder's decimal units")
  func sizes() {
    #expect(ByteCount.string(765_953_504).contains("MB"))
    #expect(ByteCount.string(5_900_000_000).contains("GB"))
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
