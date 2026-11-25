import Foundation
import JetlinkKit
import Testing

/// The on-device output check and the pick it feeds.
struct ProcessorComparisonTests {
  static let slices: [(name: String, range: Range<Int>)] = [
    ("lead", 0..<144), ("lead_prob", 144..<147), ("road_transform", 147..<159), ("meta", 159..<214), ("plan", 214..<1204),
  ]

  static func reference(_ frame: Int) -> [Float] {
    (0..<1204).map { Float(sin(0.37 * Double($0) + 0.9 * Double(frame)) * Double(1 + $0 % 7)) }
  }

  static func candidate(_ frame: Int, noise: Double) -> [Float] {
    (0..<1204).map { Float(sin(0.37 * Double($0) + 0.9 * Double(frame)) * Double(1 + $0 % 7)) + Float(noise * cos(1.3 * Double($0) + Double(frame))) }
  }

  /// The slices scripts/verify_parity.py's report_slices fails on the same
  /// eight frames, run once on 2026-11-25: the port must agree with it.
  @Test(
    "The output check fails what verify_parity.py fails",
    arguments: [
      (0.0, [String]()), (0.02, []), (0.08, []), (0.1, ["road_transform"]), (0.15, ["road_transform"]),
      (0.2, ["lead", "plan", "road_transform"]), (0.3, ["lead", "meta", "plan", "road_transform"]),
      (0.8, ["lead", "lead_prob", "meta", "plan", "road_transform"]),
    ])
  func parity(noise: Double, fails: [String]) {
    let result = OutputParity.compare(
      candidate: (0..<8).map { Self.candidate($0, noise: noise) }, reference: (0..<8).map(Self.reference), slices: Self.slices)
    #expect(result.failed == fails, "\(result.text)")
    #expect(result.passed == fails.isEmpty)
  }

  @Test("A negated head too small to correlate fails by its error, and bad input never passes")
  func parityEdges() {
    let reference = (0..<8).map(Self.reference)
    var negated = reference
    for f in negated.indices {
      for i in 144..<147 { negated[f][i] = -negated[f][i] }
    }
    #expect(OutputParity.compare(candidate: negated, reference: reference, slices: Self.slices).failed == ["lead_prob"])
    var broken = reference
    broken[3][500] = .nan
    #expect(OutputParity.compare(candidate: broken, reference: reference, slices: Self.slices).failed == ["plan"])
    #expect(!OutputParity.compare(candidate: Array(reference.dropLast()), reference: reference, slices: Self.slices).passed)
    #expect(!OutputParity.compare(candidate: [], reference: [], slices: Self.slices).passed)
    #expect(!OutputParity.compare(candidate: reference, reference: reference, slices: [("pad", 1300..<1302)]).passed)
    #expect(!OutputParity.compare(candidate: reference, reference: reference, slices: []).passed)
  }

  static func report(_ p99: Double, over50: Int = 0, frames: Int = 3600, thermal: String = "nominal", cancelled: Bool = false) -> BenchmarkReport {
    let stats = BenchmarkStats(mean: p99 - 2, p50: p99 - 2, p90: p99 - 1, p99: p99, max: p99 + 1)
    return BenchmarkReport(
      sha256: "ab", device: "d", seconds: 180, frames: frames, frame: stats, accelerator: stats, queues: .empty, output: .empty, build: "", over35: 0,
      over50: over50, windows: [], thermalAtStart: "nominal", thermalAtEnd: thermal, cancelled: cancelled)
  }

  /// The three in order of preference, the Neural Engine alone needing its outputs checked.
  static func trials(_ only: BenchmarkReport?, _ split: BenchmarkReport?, _ gpu: BenchmarkReport?, match: Bool? = true) -> [ProcessorPick.Trial] {
    [
      ProcessorPick.Trial(id: "ane-both", title: "Neural Engine Only", report: only, outputsMatch: match, needsOutputCheck: true),
      ProcessorPick.Trial(id: "ane", title: "Neural Engine + GPU", report: split),
      ProcessorPick.Trial(id: "coreml", title: "GPU Only", report: gpu),
    ]
  }

  @Test("The first in order of preference that leaves about the fastest's room is taken")
  func pick() {
    // the iPad Air (M1) of PR 25: the Neural Engine alone is the fastest
    #expect(ProcessorPick.decide(Self.trials(Self.report(26.1), Self.report(41.9), Self.report(45))).pick == "ane-both")
    // within 2 ms of the fastest is the same room: the cooler one wins
    #expect(ProcessorPick.decide(Self.trials(Self.report(30), Self.report(28.5), Self.report(40))).pick == "ane-both")
    #expect(ProcessorPick.decide(Self.trials(Self.report(40), Self.report(28), Self.report(29))).pick == "ane")
    // this Mac's M1 Pro: 3 ms and more behind is not the same room
    let slower = ProcessorPick.decide(Self.trials(Self.report(35.1), Self.report(29.2), Self.report(44)))
    #expect(slower.pick == "ane" && slower.passedOver["ane-both"] == "5.9 ms slower at P99.")
    // the GPU alone only when it beats both, as with the Neural Engine busy elsewhere
    #expect(ProcessorPick.decide(Self.trials(Self.report(65), Self.report(52.5), Self.report(43.5))).pick == "coreml")
    // frames over the budget count before P99
    #expect(ProcessorPick.decide(Self.trials(Self.report(30, over50: 4), Self.report(31), Self.report(45))).pick == "ane")
    #expect(ProcessorPick.decide(Self.trials(Self.report(30, thermal: "serious"), Self.report(31), Self.report(45))).pick == "ane")
    #expect(
      ProcessorPick.decide(Self.trials(Self.report(30, thermal: "serious"), Self.report(31, thermal: "critical"), Self.report(45, thermal: "serious"))).pick
        == "ane-both")
  }

  @Test("Lower precision is taken only with its outputs checked, and nothing is picked from nothing")
  func pickGuards() {
    let differs = ProcessorPick.decide(Self.trials(Self.report(20), Self.report(30), Self.report(40), match: false))
    #expect(differs.pick == "ane" && differs.passedOver["ane-both"] == "Its outputs differ for this model.")
    #expect(ProcessorPick.decide(Self.trials(Self.report(20), Self.report(30), Self.report(40), match: nil)).pick == "ane")
    // one that could not be prepared, ran no frame or was stopped is passed over
    #expect(ProcessorPick.decide(Self.trials(nil, Self.report(30), Self.report(40))).pick == "ane")
    #expect(ProcessorPick.decide(Self.trials(Self.report(0, frames: 0), Self.report(30), Self.report(31))).pick == "ane")
    #expect(ProcessorPick.decide(Self.trials(Self.report(20, cancelled: true), Self.report(30), Self.report(40))).pick == "ane")
    #expect(ProcessorPick.decide(Self.trials(nil, nil, nil)).pick == nil)
    #expect(ProcessorPick.decide([]).pick == nil)
    let trials = Self.trials(Self.report(26.1), Self.report(41.9), nil)
    let text = ProcessorPick.text(
      trials, decision: ProcessorPick.decide(trials), parity: nil, chip: "Apple M1", model: "Light Speed 2", sha256: "909a140019050a9d7f00")
    #expect(text.contains("Pick: Neural Engine Only.") && text.contains("GPU Only: did not run") && text.contains("outputs match"))
  }
}
