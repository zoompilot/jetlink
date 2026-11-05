#if canImport(SwiftUI)
  import Foundation
  import Testing

  @testable import JetlinkKit
  @testable import JetlinkUI

  /// The text both apps show: the status summary, sizes, model rows and the
  /// frame budget. The Mac's own views are tested in its JetlinkTests.
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
      #expect(ProgressRow.stageName("upload") == "Receiving")
      #expect(ProgressRow.stageName("patch") == "Preparing")
      #expect(ProgressRow.stageName("parse") == "Reading")
      #expect(ProgressRow.stageName("convert") == "Converting")
      #expect(ProgressRow.stageName("compile") == "Compiling")
      #expect(ProgressRow.stageName("build") == "Building")
      #expect(ProgressRow.stageName("save") == "Saving")
      #expect(ProgressRow.stageName("load") == "Loading")
      #expect(ProgressRow.stageName("failed") == "Failed")
      #expect(ProgressRow.stageName(nil) == "Working")
      #expect(ProgressRow.stageName("something new") == "Working")
    }

    // MARK: Frame budget

    @Test("The headline is the room left at p99, or how far over it is")
    func headroom() {
      #expect(FrameBudgetView.headroomText(p99: 38.4) == "11.6 ms headroom")
      #expect(FrameBudgetView.headroomText(p99: 50) == "0.0 ms headroom")
      #expect(FrameBudgetView.headroomText(p99: 53.2) == "3.2 ms over")
      #expect(FrameBudgetView.Room(headroomMs: 11.6) == .plenty)
      #expect(FrameBudgetView.Room(headroomMs: 10) == .plenty)
      #expect(FrameBudgetView.Room(headroomMs: 9.9) == .tight)
      #expect(FrameBudgetView.Room(headroomMs: 0) == .tight)
      #expect(FrameBudgetView.Room(headroomMs: -0.1) == .over)
    }

    @Test("The bar always shows the whole budget, and stretches for frames past it")
    func budgetBarScale() {
      #expect(abs(FrameStageBar.domainMax(PreviewData.stats) - 55) < 1e-9)
      #expect(FrameStageBar.ticks(55) == [0, 10, 20, 30, 40, 50])
      let slow = StatsEvent(
        frames: 1, fps: 20, servedMs: StatsEvent.Total(mean: 60, p99: 70, max: 80),
        stagesMs: StatsEvent.Stages(queue: 1, gpu: 55, other: 3, send: 1), slow: 5, windowS: 1)
      #expect(FrameStageBar.domainMax(slow) == 70 * 1.08)
      #expect(FrameStageBar.ticks(FrameStageBar.domainMax(slow)).last == 70)
    }

    @Test("The chart's time axis counts back from now")
    func chartAxis() {
      #expect(FrameTimeChart.axisLabel(0) == "now")
      #expect(FrameTimeChart.axisLabel(-30) == "30 s")
      #expect(FrameTimeChart.axisLabel(-120) == "2 min")
      #expect(FrameTimeChart.agoText(0.2) == "Just now")
      #expect(FrameTimeChart.agoText(12) == "12 s ago")
    }

    // MARK: Model rows

    @Test("Use Model takes a model from wherever it is to in use")
    func useAction() {
      var notDownloaded = PreviewData.loadedRow
      notDownloaded.status = .notDownloaded
      #expect(ModelStore.canUse(notDownloaded))
      #expect(ModelFormatting.useHelp(notDownloaded, device: "this Mac") == "Downloads 766 MB, prepares it for this Mac and starts using it")
      #expect(!ModelStore.canUse(PreviewData.loadedRow))
      var prepared = PreviewData.loadedRow
      prepared.status = .prepared
      #expect(ModelStore.canUse(prepared))
      #expect(ModelFormatting.useHelp(prepared, device: "this Mac").hasPrefix("Starts using it"))
      var downloading = PreviewData.loadedRow
      downloading.status = .downloading(frac: 0.4, rateBps: 1)
      #expect(!ModelStore.canUse(downloading))
      // A row with no checksum (a server from before the model index): nothing to ask for.
      #expect(PreviewData.notDownloadedRow.sha256 == nil)
      #expect(!ModelStore.canUse(PreviewData.notDownloadedRow))
    }

    @Test("A row's second line is its date, size and what is on disk")
    func detailLine() {
      #expect(ModelFormatting.detailLine(PreviewData.loadedRow) == "\(BuildTime.text(PreviewData.loadedRow.buildTime)) · 766 MB · Prepared for CoreML")
      #expect(ModelFormatting.detailLine(PreviewData.localRow) == "766 MB · Downloaded")
      var notDownloaded = PreviewData.loadedRow
      notDownloaded.status = .notDownloaded
      notDownloaded.preparedFor = []
      #expect(!ModelFormatting.detailLine(notDownloaded).contains("Prepared"))
      #expect(
        ModelFormatting.diskSummary(models: 2_300_000_000, engines: 6_900_000_000, free: 13_600_000_000)
          == "Downloads 2.3 GB · Prepared engines 6.9 GB · 13.6 GB available")
      #expect(ModelFormatting.downloadCaption(frac: 0.421, rateBps: 0) == "Downloading 42%")
      #expect(ModelFormatting.downloadCaption(frac: 0.421, rateBps: 41_000_000).hasPrefix("Downloading 42%, 41"))
    }

    @Test("Prepared backends are listed once each, in plain names")
    func preparedFor() {
      #expect(ModelFormatting.preparedForText(PreviewData.loadedRow) == "CoreML")
      #expect(ModelFormatting.preparedForText(PreviewData.notDownloadedRow).isEmpty)
      #expect(ModelFormatting.backendName("trt") == "TensorRT")
      #expect(ModelFormatting.backendName("tinygrad") == "tinygrad")
    }
  }
#endif
