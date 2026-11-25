import Foundation
import JetlinkKit
import JetlinkORT
import JetlinkServer
import Observation

/// Compare Processors: the loaded model run on each processor in turn, and
/// Automatic set to the best of them for that model (`ProcessorPick`).
///
/// Each candidate takes a server of its own, so the server is restarted on
/// it, the model prepared there, its outputs captured on fixed frames and a
/// benchmark run. The least preferred goes first: the device warms as the
/// runs go, so the one preferred has to hold up warm, as on a drive.
@MainActor
@Observable
final class ProcessorComparison {
  enum Phase: Equatable {
    case idle
    /// Which candidate of how many, and what it is doing.
    case running(title: String, step: String, index: Int, count: Int)
    case done
    case failed(String)
  }

  /// One finished comparison, for the screen and the share sheet.
  struct Outcome: Equatable {
    let sha256: String
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
  /// The ones that run part of the model in lower precision than the
  /// reference, and are taken only when their outputs match it.
  static let checked: Set<OrtProfile> = [.aneBoth]
  /// Each benchmark's length. A minute misleads: on an iPad Air (M1) one
  /// 60 s run read 30.0 ms median where three minutes read 33.5.
  static let seconds = 180.0
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
    phase = .running(title: "", step: "Starting", index: 0, count: ProcessorComparison.candidates.count)
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
    let order = Array(ProcessorComparison.candidates.reversed())
    var reports: [OrtProfile: BenchmarkReport] = [:]
    var captures: [OrtProfile: ProbeCapture] = [:]
    var failures: [OrtProfile: String] = [:]
    defer {
      // back to what the setting says, with the pick this run made
      server.profileOverride = nil
      server.restart()
      task = nil
    }
    for (index, profile) in order.enumerated() {
      func step(_ text: String) {
        phase = .running(title: profile.title, step: text, index: index, count: order.count)
      }
      // A comma's frames come first, and a partial comparison picks nothing.
      if server.link.state == .connected {
        server.note(.warning, "the processor comparison stopped: a comma connected")
        phase = .failed("A comma connected, so the comparison stopped.")
        return
      }
      do {
        step("Preparing the model")
        server.profileOverride = profile
        server.restart()
        try await prepare(sha256)
        step("Checking its outputs")
        captures[profile] = try await server.probe()
        step("Measuring")
        reports[profile] = try await benchmark()
        try Task.checkCancellation()
      } catch is CancellationError {
        server.note(.info, "the processor comparison was cancelled")
        phase = .idle
        return
      } catch {
        if Task.isCancelled {
          phase = .idle
          return
        }
        let text = (error as? Failure)?.text ?? "It could not run this model."
        failures[profile] = text
        server.note(.warning, "comparison: \(profile.rawValue) failed: \(error)")
      }
    }

    // Without the reference there is nothing to hold the others to.
    guard reports[ProcessorComparison.reference] != nil else {
      phase = .failed("The comparison did not finish. See Logs for details.")
      return
    }
    var parity: OutputParity.Result?
    var matches: [OrtProfile: Bool] = [:]
    if let reference = captures[ProcessorComparison.reference] {
      for profile in ProcessorComparison.checked {
        guard let capture = captures[profile] else { continue }
        let result = OutputParity.compare(candidate: capture.outputs, reference: reference.outputs, slices: reference.slices)
        matches[profile] = result.passed
        parity = result
        server.note(result.passed ? .info : .warning, "comparison: \(profile.rawValue) outputs \(result.passed ? "match" : "differ")\n\(result.text)")
      }
    }
    let trials = ProcessorComparison.candidates.map { profile in
      ProcessorPick.Trial(
        id: profile.rawValue, title: profile.title, report: reports[profile], outputsMatch: matches[profile],
        needsOutputCheck: ProcessorComparison.checked.contains(profile), failure: failures[profile])
    }
    let decision = ProcessorPick.decide(trials)
    settings.automaticPicks[sha256] = decision.pick
    let text = ProcessorPick.text(trials, decision: decision, parity: parity, chip: chip, model: modelName, sha256: sha256)
    server.note(.info, text)
    outcome = Outcome(sha256: sha256, trials: trials, decision: decision, text: text)
    phase = decision.pick == nil ? .failed(decision.reason) : .done
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
