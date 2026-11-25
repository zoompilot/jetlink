import Foundation

/// Which processor a device should run a model on, from a benchmark of each.
///
/// The candidates come in order of preference: the Neural Engine alone, the
/// Neural Engine with the GPU, the GPU alone. The Neural Engine draws less
/// power and heats the device less (iPad Air M1, 3 minutes each: 12.8 W alone
/// against 14.0 W with the GPU), so the first in that order that leaves about
/// as much room in the frame budget as the fastest is taken, not the fastest
/// itself.
public enum ProcessorPick {
  /// A processor may be this much slower at P99 than the fastest and still
  /// count as leaving the same room: 2 ms of a 50 ms budget.
  public static let similarP99Ms = 2.0

  /// One processor's run.
  public struct Trial: Sendable, Equatable {
    /// The processor as the server names it: "ane-both", "ane", "coreml".
    public let id: String
    public let title: String
    /// Nil when it could not be prepared or run.
    public let report: BenchmarkReport?
    /// Whether its outputs matched the reference processor's on the same
    /// frames. Nil where that was not asked of it, or could not be checked.
    public let outputsMatch: Bool?
    /// It runs part of the model in lower precision, so it is taken only
    /// when `outputsMatch` is true.
    public let needsOutputCheck: Bool
    /// Why there is no report, for the screen.
    public let failure: String?

    public init(id: String, title: String, report: BenchmarkReport?, outputsMatch: Bool? = nil, needsOutputCheck: Bool = false, failure: String? = nil) {
      self.id = id
      self.title = title
      self.report = report
      self.outputsMatch = outputsMatch
      self.needsOutputCheck = needsOutputCheck
      self.failure = failure
    }
  }

  public struct Decision: Sendable, Equatable {
    /// The trial taken, or nil when none finished: the default stays.
    public let pick: String?
    /// Why, in a sentence for the screen and the shared report.
    public let reason: String
    /// Why each trial passed over was, by id.
    public let passedOver: [String: String]
  }

  /// `trials` in order of preference.
  public static func decide(_ trials: [Trial]) -> Decision {
    var passedOver: [String: String] = [:]
    var ran: [(trial: Trial, report: BenchmarkReport)] = []
    for trial in trials {
      guard let report = trial.report else {
        passedOver[trial.id] = trial.failure ?? "It did not run."
        continue
      }
      if report.cancelled {
        passedOver[trial.id] = "Stopped early."
      } else if report.frames == 0 {
        passedOver[trial.id] = "Too slow to finish a frame."
      } else if trial.needsOutputCheck && trial.outputsMatch != true {
        passedOver[trial.id] = trial.outputsMatch == nil ? "Its outputs could not be checked." : "Its outputs differ for this model."
      } else {
        ran.append((trial, report))
      }
    }
    guard let fewestOver = ran.map(\.report.over50).min() else {
      return Decision(pick: nil, reason: "No processor finished the run.", passedOver: passedOver)
    }
    let fastest = ran.filter { $0.report.over50 == fewestOver }.map(\.report.frame.p99).min()!
    let someCool = ran.contains { !hot($0.report.thermalAtEnd) }
    for (trial, report) in ran {
      let slower = report.frame.p99 - fastest
      if report.over50 > fewestOver {
        passedOver[trial.id] = "More frames over the 50 ms budget."
      } else if hot(report.thermalAtEnd) && someCool {
        passedOver[trial.id] = "It ended the run hot."
      } else if slower > similarP99Ms {
        passedOver[trial.id] = "\(String(format: "%.1f", slower)) ms slower at P99."
      } else {
        let reason =
          slower <= 0
          ? "\(trial.title) is the fastest here."
          : "\(trial.title) leaves about the same room as the fastest, and runs cooler."
        return Decision(pick: trial.id, reason: reason, passedOver: passedOver)
      }
    }
    return Decision(pick: nil, reason: "None was both fast enough and cool enough.", passedOver: passedOver)
  }

  private static func hot(_ thermal: String) -> Bool { thermal == "serious" || thermal == "critical" }

  /// The runs and the outcome as text, to send to the developers.
  public static func text(_ trials: [Trial], decision: Decision, parity: OutputParity.Result?, reference: String, chip: String, model: String, sha256: String) -> String {
    var lines = ["Jetlink processor comparison", "\(chip), \(model) (\(sha256.prefix(16)))"]
    for trial in trials {
      if let r = trial.report {
        lines.append(
          "\(trial.title): p50 \(r.frame.p50) p99 \(r.frame.p99) max \(r.frame.max) ms, \(r.over35) over 35 ms, \(r.over50) over 50 ms of \(r.frames), "
            + "\(r.thermalAtStart) to \(r.thermalAtEnd)"
            + (trial.needsOutputCheck ? ", outputs \(trial.outputsMatch.map { $0 ? "match" : "differ" } ?? "not checked")" : ""))
      } else {
        lines.append("\(trial.title): \(trial.failure ?? "did not run")")
      }
    }
    let title = trials.first { $0.id == decision.pick }?.title
    lines.append("Pick: \(title ?? "none"). \(decision.reason)")
    if let parity {
      lines.append("Outputs against \(reference):")
      lines.append(parity.text)
    }
    return lines.joined(separator: "\n")
  }
}
