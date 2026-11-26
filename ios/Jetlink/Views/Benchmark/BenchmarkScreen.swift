import JetlinkKit
import JetlinkUI
import SwiftUI

/// Is this iPhone or iPad fast enough, and does it stay fast enough? The
/// loaded model at the comma's pace on the device alone.
struct BenchmarkScreen: View {
  @Environment(AppModel.self) private var app
  @Environment(\.horizontalSizeClass) private var horizontalSizeClass
  @State private var refusal: String?
  @State private var starting = false
  /// What Compare Processors would add on disk, asked before it starts.
  @State private var comparePrompt: ComparePrompt?

  struct ComparePrompt: Identifiable {
    let id = UUID()
    let sha256: String
    let modelName: String
    /// Processors the model is not prepared for yet.
    let missing: Int
    let need: Int64
    let free: Int64
    var fits: Bool { free >= need }
  }
  /// `-benchmark 60` on the command line runs one once a model is loaded,
  /// for screenshots from the simulator.
  @State private var scripted: Double? = UserDefaults.standard.double(forKey: "benchmark") > 0 ? UserDefaults.standard.double(forKey: "benchmark") : nil

  var body: some View {
    NavigationStack {
      ScrollView {
        Group {
          // An iPad has room for the run and its verdict beside the
          // numbers behind it; a narrower screen has them one under the other.
          if horizontalSizeClass == .regular, let report, report.frames > 0 {
            HStack(alignment: .top, spacing: StatusContent.spacing) {
              VStack(spacing: StatusContent.spacing) {
                runCard
                comparisonCard
                VerdictCard(report: report)
              }
              VStack(spacing: StatusContent.spacing) { results(report) }
            }
          } else {
            VStack(spacing: StatusContent.spacing) {
              runCard
              comparisonCard
              if let report {
                VerdictCard(report: report)
                if report.frames > 0 { results(report) }
              }
            }
            .frame(maxWidth: readableContentWidth)
            .frame(maxWidth: .infinity)
          }
        }
        .padding(.bottom, 24)
      }
      .contentMargins(.horizontal, StatusContent.margin, for: .scrollContent)
      .background(Color.groupedBackground)
      .navigationTitle("Benchmark")
      .onChange(of: sha256, initial: true) {
        if let seconds = scripted, blocker == nil {
          scripted = nil
          start(seconds: seconds)
        }
      }
      .alert(
        comparePrompt.map { $0.fits ? "Compare Processors?" : "Not Enough Space to Compare" } ?? "",
        isPresented: Binding(get: { comparePrompt != nil }, set: { if !$0 { comparePrompt = nil } }), presenting: comparePrompt
      ) { prompt in
        if prompt.fits {
          Button("Compare") { startComparison(sha256: prompt.sha256) }
          Button("Cancel", role: .cancel) {}
        } else {
          Button("OK", role: .cancel) {}
        }
      } message: { prompt in
        let more =
          "It prepares \(prompt.modelName) for \(prompt.missing) more processor\(prompt.missing == 1 ? "" : "s"), "
          + "and the Neural Engine compiles it once more for each: about \(StorageNeed.gigabytes(prompt.need)) free "
          + "with room left for the system. \(StorageNeed.gigabytes(prompt.free)) is free."
        Text(prompt.fits ? more : more + " Free some space first.")
      }
      .toolbar {
        if let text = app.comparison.outcome?.text ?? report?.text {
          ToolbarItem(placement: .topBarTrailing) {
            ShareLink(item: text, preview: SharePreview("Jetlink Benchmark"))
          }
        }
      }
    }
  }

  /// The numbers behind the verdict: the totals, then the run ten seconds at a time.
  @ViewBuilder
  private func results(_ report: BenchmarkReport) -> some View {
    totals(report)
    if report.windows.count > 1 {
      WindowsCard(windows: report.windows)
    }
  }

  // MARK: state

  private var event: BenchmarkEvent? { app.server.state.benchmark }
  private var running: Bool { event.map { !$0.isFinished } ?? false }
  private var report: BenchmarkReport? { event?.report }
  private var sha256: String? { app.server.engine.state == .ready ? app.server.engine.sha256 : nil }
  private var connected: Bool { app.server.link.state == .connected }

  /// Why a run cannot start now, in a few words; nil when it can.
  private var blocker: String? {
    BenchmarkBlocker.reason(serving: app.server.runState == .serving, modelLoaded: sha256 != nil, commaConnected: connected)
  }

  // MARK: the run

  private var runCard: some View {
    SummaryCard(title: "Benchmark", systemImage: "stopwatch.fill", tint: .blue, trailing: app.server.profile.title) {
      VStack(alignment: .leading, spacing: 14) {
        Text(sha256.flatMap(app.modelName) ?? (sha256 == nil ? "No Model" : "Model"))
          .font(.title2.weight(.bold))
        if case .running(let title, let step, let index) = app.comparison.phase {
          comparing(title: title, step: step, index: index)
        } else if let event, running {
          progress(event)
        } else {
          buttons
          compareButton
        }
        if let text = refusal ?? (event?.state == "failed" ? "The benchmark failed. See Logs for details." : nil) {
          Text(text)
            .font(.footnote)
            .foregroundStyle(.red)
        } else if !running, let blocker {
          Text(blocker)
            .font(.footnote)
            .foregroundStyle(.orange)
        }
        Text("Run it with the \(ThisDevice.name) charging and in its mount.")
          .font(.footnote)
          .foregroundStyle(.secondary)
      }
    }
  }

  private var buttons: some View {
    HStack(spacing: 12) {
      Button {
        start(seconds: 60)
      } label: {
        Label("1 Minute", systemImage: "play.fill")
          .frame(maxWidth: .infinity)
      }
      .buttonStyle(.glassProminent)
      Button {
        start(seconds: 600)
      } label: {
        Label("10 Minutes", systemImage: "flame.fill")
          .frame(maxWidth: .infinity)
      }
      .buttonStyle(.glass)
    }
    .controlSize(.large)
    .disabled(blocker != nil || starting)
  }

  /// Every processor in turn, and Automatic set to the best of them.
  private var compareButton: some View {
    VStack(alignment: .leading, spacing: 8) {
      Button {
        guard let sha256 else { return }
        refusal = nil
        askToCompare(sha256: sha256)
      } label: {
        Label("Compare Processors", systemImage: "arrow.triangle.branch")
          .frame(maxWidth: .infinity)
      }
      .buttonStyle(.glass)
      .controlSize(.large)
      .disabled(blocker != nil || starting)
      Text(
        "Runs this model on each processor for 20 seconds and sets Automatic to the best one. It prepares the model for each first, which takes minutes, and asks first how much space that needs."
      )
      .font(.footnote)
      .foregroundStyle(.secondary)
    }
  }

  private func comparing(title: String?, step: String, index: Int) -> some View {
    VStack(alignment: .leading, spacing: 12) {
      Text(title.map { "\($0): \(step)" } ?? step)
        .font(.headline)
      Text("Processor \(index + 1) of \(ProcessorComparison.candidates.count)")
        .font(.subheadline)
        .foregroundStyle(.secondary)
      if let event, running {
        figures(event)
      } else {
        ProgressView()
      }
      cancelButton { app.comparison.cancel() }
    }
  }

  /// What the last comparison found, or why it found nothing.
  @ViewBuilder
  private var comparisonCard: some View {
    if let outcome = app.comparison.outcome {
      ComparisonCard(outcome: outcome, automatic: app.settings.device == .automatic)
    } else if case .failed(let text) = app.comparison.phase {
      SummaryCard(title: "Comparison", systemImage: "arrow.triangle.branch", tint: .orange) {
        Text(text)
          .font(.subheadline)
      }
    }
  }

  private func progress(_ event: BenchmarkEvent) -> some View {
    VStack(alignment: .leading, spacing: 12) {
      figures(event)
      cancelButton(cancel)
    }
  }

  /// How far a running benchmark is, and its numbers so far.
  @ViewBuilder
  private func figures(_ event: BenchmarkEvent) -> some View {
    ProgressView(value: min(event.elapsed / max(event.total, 1), 1))
      .tint(.blue)
    HStack {
      Text("\(BenchmarkClock.text(event.elapsed)) of \(BenchmarkClock.text(event.total))")
      Spacer()
      Text("\(event.frames.formatted()) frames")
    }
    .font(.subheadline.monospacedDigit())
    .foregroundStyle(.secondary)
    HStack(spacing: 0) {
      Figure("P50", ms: event.frame?.p50)
      Divider().frame(height: 32)
      Figure("P99", ms: event.frame?.p99)
    }
  }

  private func cancelButton(_ action: @escaping () -> Void) -> some View {
    Button(role: .destructive, action: action) {
      Label("Cancel", systemImage: "stop.fill")
        .frame(maxWidth: .infinity)
    }
    .buttonStyle(.glass)
    .controlSize(.large)
  }

  private func start(seconds: Double) {
    refusal = nil
    starting = true
    Task {
      defer { starting = false }
      do {
        let reply = try await app.server.send(.benchmark(seconds: seconds))
        if !reply.ok { refuse(reply.error) }
      } catch {
        refuse(error.localizedDescription)
      }
    }
  }

  /// A short line on the card, and the server's reason in Logs.
  /// Compare Processors prepares and compiles the model once for each
  /// processor it is not prepared for: say what that takes first, and refuse
  /// it when the disk has no room, as a build would.
  private func askToCompare(sha256: String) {
    let prepared = Set(
      (app.models.inventory?.artifacts ?? []).filter { $0.sha256 == sha256 }.compactMap { artifact in
        ProcessorComparison.candidates.filter { artifact.device.hasPrefix("\($0.rawValue)-") }.max { $0.rawValue.count < $1.rawValue.count }
      })
    let missing = ProcessorComparison.candidates.filter { !prepared.contains($0) }
    guard !missing.isEmpty, let bytes = app.models.row(for: sha256)?.bytes, let free = app.models.inventory?.disk.freeBytes else {
      startComparison(sha256: sha256)
      return
    }
    let need = StorageNeed.coreML(modelBytes: bytes).toPrepare(profiles: missing.count, onNeuralEngine: missing.filter { $0 != .coreml }.count)
    comparePrompt = ComparePrompt(sha256: sha256, modelName: app.modelName(sha256) ?? "this model", missing: missing.count, need: need, free: free)
  }

  private func startComparison(sha256: String) {
    app.comparison.start(
      sha256: sha256, modelName: app.modelName(sha256) ?? "Model",
      chip: (app.server.state.server?.device).map(SettingsScreen.chip) ?? ThisDevice.name)
  }

  private func refuse(_ reason: String?) {
    refusal = "Couldn't start the benchmark. See Logs for details."
    app.server.note(.warning, "the benchmark could not start: \(reason ?? "no reason given")")
  }

  private func cancel() {
    Task { _ = try? await app.server.send(.cancelBenchmark) }
  }

  // MARK: totals

  private func totals(_ report: BenchmarkReport) -> some View {
    MetricGrid {
      GridRow {
        MetricTile(
          title: "Frames", systemImage: "film.stack", tint: .teal, value: report.frames.formatted(),
          note: "\(BenchmarkClock.text(report.seconds)) at 20 Hz")
        MetricTile(
          title: "Over Budget", systemImage: "tortoise.fill", tint: .pink, value: report.over50.formatted(),
          note: report.over35 > 0 ? "\(report.over35.formatted()) over 35 ms" : "None over 35 ms", noteTone: report.over50 > 0 ? .red : nil)
      }
      GridRow {
        MetricTile(
          title: "Temperature", systemImage: ThermalLevel(label: report.thermalAtEnd).symbol, tint: .orange,
          value: ThermalLevel(label: report.thermalAtEnd).title,
          note: report.thermalAtStart == report.thermalAtEnd ? "Throughout" : "From \(ThermalLevel(label: report.thermalAtStart).title.lowercased())",
          noteTone: ThermalLevel(label: report.thermalAtEnd).tone)
        MetricTile(
          title: "Model", systemImage: "cpu", tint: .purple, value: report.accelerator.mean.formatted(.number.precision(.fractionLength(1))), unit: "ms",
          note: "Mean, model alone")
      }
    }
  }
}

/// Fast enough, tight, or too slow, and the numbers that say so.
struct VerdictCard: View {
  let report: BenchmarkReport

  var body: some View {
    let verdict = BenchmarkVerdict(report)
    SummaryCard(title: "Verdict", systemImage: verdict.symbol, tint: verdict.tone, trailing: report.cancelled ? "Stopped Early" : nil) {
      BenchmarkVerdictSummary(report: report)
    }
  }
}

/// Each processor's run side by side, and the one Automatic now uses.
struct ComparisonCard: View {
  let outcome: ProcessorComparison.Outcome
  /// Whether the Processor setting is Automatic, so the pick is in use.
  let automatic: Bool

  var body: some View {
    let picked = outcome.trials.first { $0.id == outcome.decision.pick }
    SummaryCard(title: "Comparison", systemImage: "arrow.triangle.branch", tint: .teal, trailing: picked?.title) {
      VStack(alignment: .leading, spacing: 12) {
        ForEach(outcome.trials, id: \.id) { trial in
          let isPick = trial.id == outcome.decision.pick
          let report = trial.report.flatMap { $0.frames > 0 ? $0 : nil }
          VStack(alignment: .leading, spacing: 2) {
            HStack {
              Label(trial.title, systemImage: isPick ? "checkmark.circle.fill" : "circle")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(isPick ? .green : .primary)
              Spacer()
              if let report {
                Text("P50 \(FrameBudgetView.ms(report.frame.p50))")
                  .font(.subheadline.monospacedDigit())
              }
            }
            if let report {
              Text(
                "P99 \(FrameBudgetView.ms(report.frame.p99)), \(report.over50.formatted()) over budget, \(ThermalLevel(label: report.thermalAtEnd).title.lowercased()) at the end"
              )
              .font(.footnote.monospacedDigit())
              .foregroundStyle(.secondary)
            }
            if let why = outcome.decision.passedOver[trial.id] {
              Text(why)
                .font(.footnote)
                .foregroundStyle(.orange)
            }
          }
        }
        Divider()
        Text(outcome.decision.reason)
          .font(.subheadline)
        if let picked {
          Text(
            automatic
              ? "Automatic now runs this model on \(picked.title)."
              : "Set Processor to Automatic in Settings to run this model on \(picked.title)."
          )
          .font(.footnote)
          .foregroundStyle(.secondary)
        }
      }
    }
  }
}

/// The run ten seconds at a time, with the phone's temperature as each closed.
struct WindowsCard: View {
  let windows: [BenchmarkWindow]

  var body: some View {
    SummaryCard(title: "Over Time", systemImage: "chart.bar.fill", tint: .indigo, trailing: "10 s each") {
      BenchmarkWindowRows(windows: windows)
    }
  }
}
