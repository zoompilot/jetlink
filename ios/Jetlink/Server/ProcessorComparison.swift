import Foundation
import JetlinkKit
import JetlinkORT
import JetlinkServer
import Observation

/// Compare Processors: the loaded model run on each processor in turn, and
/// Automatic set to the best of them for that model (`ProcessorPick`).
///
/// Each candidate takes a server of its own, so the server is restarted on
/// it, the model prepared there and a benchmark run, with the outputs of the
/// two that are compared captured on fixed frames first. The runs are short
/// and each starts no warmer than the first did, so the heat one leaves does
/// not count against the next.
@MainActor
@Observable
final class ProcessorComparison {
  enum Phase: Equatable {
    case idle
    /// Which candidate, nil before the first, and what it is doing.
    case running(title: String?, step: String, index: Int)
    case failed(String)
  }

  /// One finished comparison, for the screen and the share sheet.
  struct Outcome: Equatable {
    let trials: [ProcessorPick.Trial]
    let decision: ProcessorPick.Decision
    let text: String
  }

  private(set) var phase: Phase = .idle
  private(set) var outcome: Outcome?

  var running: Bool {
    if case .running = phase { true } else { false }
  }

  /// In order of preference: the Neural Engine draws the least power.
  static let candidates: [OrtProfile] = [.aneBoth, .ane, .coreml]
  /// The one whose outputs the others are held to: it has passed the parity
  /// check against the CPU wherever that was run.
  static let reference = OrtProfile.ane
  /// The one that runs part of the model in lower precision than the
  /// reference, and is taken only when its outputs match it.
  static let checked = OrtProfile.aneBoth
  /// Each benchmark's length: 1,200 frames, enough for a P99, and short
  /// enough that the device does not heat through it.
  static let seconds = 60.0
  /// The longest a run waits for the device to cool to where the first run
  /// started. Past it the run goes ahead, and its report says how warm it was.
  static let cooldownTimeout: TimeInterval = 120
  /// A first preparation is minutes; one that takes this long is stuck.
  static let prepareTimeout: TimeInterval = 30 * 60

  private let server: PhoneServer
  private let settings: PhoneSettings
  @ObservationIgnored private var task: Task<Void, Never>?

  init(server: PhoneServer, settings: PhoneSettings) {
    self.server = server
    self.settings = settings
  }

  func start(sha256: String, modelName: String, chip: String) {
    guard !running else { return }
    outcome = nil
    phase = .running(title: nil, step: "Starting", index: 0)
    task = Task { await run(sha256: sha256, modelName: modelName, chip: chip) }
  }

  func cancel() {
    task?.cancel()
    Task { _ = try? await server.send(.cancelBenchmark) }
  }

  private struct Failure: Error {
    let text: String
  }

  private func run(sha256: String, modelName: String, chip: String) async {
    // least preferred first: what heat is left over counts against the one preferred
    let order = Array(ProcessorComparison.candidates.reversed())
    let reference = ProcessorComparison.reference
    let checked = ProcessorComparison.checked
    var reports: [OrtProfile: BenchmarkReport] = [:]
    var captures: [OrtProfile: ProbeCapture] = [:]
    var failures: [OrtProfile: String] = [:]
    let startedAt = ProcessInfo.processInfo.thermalState.rawValue
    defer {
      // back to what the setting says, with the pick this run made
      server.restart(on: nil)
      task = nil
    }
    for (index, profile) in order.enumerated() {
      func step(_ text: String) {
        phase = .running(title: profile.title, step: text, index: index)
      }
      // A comma's frames come first, and a partial comparison picks nothing.
      if server.link.state == .connected {
        server.note(.warning, "the processor comparison stopped: a comma connected")
        phase = .failed("A comma connected, so the comparison stopped.")
        return
      }
      do {
        step("Preparing the model")
        server.restart(on: profile)
        try await prepare(sha256)
        if profile == reference || profile == checked {
          step("Checking its outputs")
          captures[profile] = try await server.probe()
        }
        if ProcessInfo.processInfo.thermalState.rawValue > startedAt {
          step("Cooling down")
          try await cool(to: startedAt)
        }
        step("Measuring")
        reports[profile] = try await benchmark()
        try Task.checkCancellation()
      } catch {
        if error is CancellationError || Task.isCancelled {
          server.note(.info, "the processor comparison was cancelled")
          phase = .idle
          return
        }
        failures[profile] = (error as? Failure)?.text ?? "It could not run this model."
        server.note(.warning, "comparison: \(profile.rawValue) failed: \(error)")
      }
    }

    // Without the reference there is nothing to hold the others to.
    guard reports[reference] != nil else {
      phase = .failed("The comparison did not finish. See Logs for details.")
      return
    }
    var parity: OutputParity.Result?
    if let ours = captures[checked], let theirs = captures[reference] {
      parity = OutputParity.compare(candidate: ours.outputs, reference: theirs.outputs, slices: theirs.slices)
    }
    let trials = ProcessorComparison.candidates.map { profile in
      ProcessorPick.Trial(
        id: profile.rawValue, title: profile.title, report: reports[profile], outputsMatch: profile == checked ? parity?.passed : nil,
        needsOutputCheck: profile == checked, failure: failures[profile])
    }
    let decision = ProcessorPick.decide(trials)
    settings.automaticPicks[sha256] = decision.pick
    let text = ProcessorPick.text(
      trials, decision: decision, parity: parity, reference: reference.title, chip: chip, model: modelName, sha256: sha256)
    server.note(.info, text)
    outcome = Outcome(trials: trials, decision: decision, text: text)
    phase = .idle
  }

  /// Asks for the model and waits until the engine has it.
  private func prepare(_ sha256: String) async throws {
    let reply = try await server.send(.prepare(sha256: sha256, frameSkip: ModelStore.defaultFrameSkip))
    guard reply.ok else { throw Failure(text: "The model could not be prepared.") }
    let deadline = Date().addingTimeInterval(ProcessorComparison.prepareTimeout)
    while Date() < deadline {
      try await Task.sleep(for: .milliseconds(300))
      let engine = server.engine
      guard engine.sha256 == sha256 || engine.sha256 == nil else { throw Failure(text: "Another model was loaded.") }
      switch engine.state {
      case .ready where engine.sha256 == sha256: return
      case .failed: throw Failure(text: "The model could not be prepared.")
      default: break
      }
      if case .failed = server.runState { throw Failure(text: "The server stopped.") }
    }
    throw Failure(text: "Preparing took too long.")
  }

  /// Waits for the device to be no warmer than `level`, a thermal state.
  private func cool(to level: Int) async throws {
    let deadline = Date().addingTimeInterval(ProcessorComparison.cooldownTimeout)
    while ProcessInfo.processInfo.thermalState.rawValue > level, Date() < deadline {
      try await Task.sleep(for: .seconds(1))
    }
  }

  /// Runs one benchmark and waits for its report.
  private func benchmark() async throws -> BenchmarkReport {
    let before = server.state.benchmark
    let reply = try await server.send(.benchmark(seconds: ProcessorComparison.seconds))
    guard reply.ok else { throw Failure(text: "The benchmark could not start.") }
    while true {
      try await Task.sleep(for: .milliseconds(300))
      guard let event = server.state.benchmark, event != before, event.isFinished else { continue }
      if event.state == "cancelled" { throw CancellationError() }
      guard event.state == "done", let report = event.report else { throw Failure(text: "The benchmark failed.") }
      return report
    }
  }
}
