import Foundation

/// Whether one processor's driving outputs match another's over the same
/// frames: the pooled check of scripts/verify_parity.py (`report_slices`), with
/// its constants and its reasons.
///
/// Each output slice is correlated with every frame pooled, and so is each of
/// its columns where openpilot's Parser reads it as [mu | std | selection]
/// blocks, because units mix across columns and the largest would set a
/// whole-slice correlation. What is too small or too quiet to correlate is
/// held to absolute error instead.
public enum OutputParity {
  public static let minCorrelation = 0.999
  /// Fewer values a frame than this cannot be correlated: one point
  /// correlates at 1.0 with anything.
  public static let minSamples = 16
  /// A column moving less than this fraction of its slice's spread is
  /// rounding against rounding.
  public static let quietFraction = 0.05
  /// What a slice or column held to absolute error may be off by, as a
  /// fraction of its own largest reference value.
  public static let tinyTolerance = 0.1
  /// The tolerance's floor, as a fraction of the slice's spread.
  public static let constantFraction = 1e-3

  /// parse_model_outputs.py's layouts: hypotheses, selection width, columns.
  static let layouts: [String: [(hypotheses: Int, selection: Int, width: Int)]] = [
    "plan": [(0, 0, 15), (5, 1, 15)],
    "lane_lines": [(0, 0, 2)],
    "road_edges": [(0, 0, 2)],
    "lead": [(0, 0, 4), (2, 3, 4)],
    "pose": [(0, 0, 6)],
    "wide_from_device_euler": [(0, 0, 3)],
    "road_transform": [(0, 0, 6)],
    "sim_pose": [(0, 0, 6)],
  ]

  public struct Slice: Sendable, Equatable {
    public let name: String
    public let passed: Bool
    /// The number that says so: a correlation, or an error against its bound.
    public let detail: String
  }

  public struct Result: Sendable, Equatable {
    public let slices: [Slice]
    public var passed: Bool { !slices.isEmpty && slices.allSatisfy(\.passed) }
    public var failed: [String] { slices.filter { !$0.passed }.map(\.name) }

    public var text: String {
      slices.map { "  \($0.name): \($0.passed ? "ok" : "FAIL") \($0.detail)" }.joined(separator: "\n")
    }
  }

  /// `candidate` against `reference`, a frame's whole driving output each,
  /// frame for frame, over `slices` of it.
  public static func compare(candidate: [[Float]], reference: [[Float]], slices: [(name: String, range: Range<Int>)]) -> Result {
    guard !reference.isEmpty, candidate.count == reference.count else {
      return Result(slices: [Slice(name: "frames", passed: false, detail: "\(candidate.count) frames against \(reference.count)")])
    }
    var out: [Slice] = []
    for (name, range) in slices.sorted(by: { $0.name < $1.name }) {
      guard !range.isEmpty, candidate.allSatisfy({ $0.count >= range.upperBound }), reference.allSatisfy({ $0.count >= range.upperBound }) else {
        out.append(Slice(name: name, passed: false, detail: "outside the output"))
        continue
      }
      let a = candidate.flatMap { $0[range].map(Double.init) }
      let b = reference.flatMap { $0[range].map(Double.init) }
      guard a.allSatisfy(\.isFinite), b.allSatisfy(\.isFinite) else {
        out.append(Slice(name: name, passed: false, detail: "not finite"))
        continue
      }
      var (ok, why) = judge(a, b, perFrame: range.count, spread: nil)
      let spread = deviation(b)
      var failed: [String] = []
      for (key, indices) in columns(name, count: range.count) ?? [] {
        let ca = candidate.flatMap { frame in indices.map { Double(frame[range.lowerBound + $0]) } }
        let cb = reference.flatMap { frame in indices.map { Double(frame[range.lowerBound + $0]) } }
        let (columnOK, columnWhy) = judge(ca, cb, perFrame: indices.count, spread: spread)
        if !columnOK { failed.append("\(key) \(columnWhy)") }
      }
      if !failed.isEmpty {
        ok = false
        why += ", columns: " + failed.joined(separator: ", ")
      }
      out.append(Slice(name: name, passed: ok, detail: why))
    }
    return Result(slices: out)
  }

  /// Correlation, unless there is too little to correlate: fewer than
  /// `minSamples` values a frame, or (a column, `spread` its slice's) moving
  /// less than `quietFraction` of its slice.
  static func judge(_ a: [Double], _ b: [Double], perFrame: Int, spread: Double?) -> (Bool, String) {
    let own = deviation(b)
    if perFrame < minSamples || (spread.map { own < quietFraction * $0 } ?? false) {
      let largest = b.reduce(0) { max($0, abs($1)) }
      let bound = max(tinyTolerance * largest, constantFraction * (spread ?? own))
      let error = zip(a, b).reduce(0) { max($0, abs($1.0 - $1.1)) }
      return (error <= bound, "by error, max \(number(error)) \(error <= bound ? "<=" : ">") \(number(bound))")
    }
    let c = correlation(a, b)
    return (c >= minCorrelation, "corr \(String(format: "%.6f", c))")
  }

  /// A head's unit-homogeneous columns as indices into its slice, or nil
  /// where its layout is unknown.
  static func columns(_ name: String, count: Int) -> [(key: String, indices: [Int])]? {
    for (hypotheses, selection, width) in layouts[name] ?? [] {
      let rows = max(hypotheses, 1)
      guard count % rows == 0 else { continue }
      let per = count / rows
      let n = (per - selection) / 2
      guard n > 0, (per - selection) % 2 == 0, n % width == 0 else { continue }
      var out: [(String, [Int])] = []
      for (label, offset) in [("mu", 0), ("std", n)] {
        for j in 0..<width {
          out.append(("\(label)[\(j)]", (0..<rows).flatMap { row in stride(from: j, to: n, by: width).map { row * per + offset + $0 } }))
        }
      }
      if selection > 0 {
        out.append(("sel", (0..<rows).flatMap { row in (2 * n..<per).map { row * per + $0 } }))
      }
      return out
    }
    return nil
  }

  /// The population standard deviation, as numpy's `std`.
  static func deviation(_ x: [Double]) -> Double {
    guard !x.isEmpty else { return 0 }
    let mean = x.reduce(0, +) / Double(x.count)
    return (x.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(x.count)).squareRoot()
  }

  static func correlation(_ a: [Double], _ b: [Double]) -> Double {
    let sa = deviation(a)
    let sb = deviation(b)
    // a head the model holds constant is a pass, not a division by zero
    if sa == 0 && sb == 0 { return 1 }
    if sa == 0 || sb == 0 {
      return zip(a, b).allSatisfy { abs($0 - $1) <= 1e-8 + 1e-5 * abs($1) } ? 1 : 0
    }
    let ma = a.reduce(0, +) / Double(a.count)
    let mb = b.reduce(0, +) / Double(b.count)
    let covariance = zip(a, b).reduce(0) { $0 + ($1.0 - ma) * ($1.1 - mb) } / Double(a.count)
    return covariance / (sa * sb)
  }

  private static func number(_ value: Double) -> String { String(format: "%.4g", value) }
}
