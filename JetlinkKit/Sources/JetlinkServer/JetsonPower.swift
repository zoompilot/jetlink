import Foundation

/// A Jetson's power modes as nvpmodel's files describe them, and a clock's
/// steps and limit: parsed here, applied by `JetsonClocks`, shown by the page.
public enum JetsonPower {
  /// One clock: the steps it takes, in its own unit (hertz, or kHz for
  /// cpufreq), and the power mode's limit; nil for none.
  public struct Clock: Equatable, Sendable {
    public let steps: [Int]
    public let cap: Int?

    public init(steps: [Int], cap: Int?) {
      self.steps = steps.sorted()
      self.cap = cap
    }

    /// The fastest step the cap allows, or the slowest when none fits.
    public var top: Int { steps.last(where: { $0 <= cap ?? .max }) ?? steps.first ?? 0 }
    public var bottom: Int { steps.first ?? 0 }
  }

  /// A mode's limits: `GPU MAX_FREQ` and `EMC MAX_FREQ` in hertz,
  /// `CPU_A78_0 MAX_FREQ` in kHz; nil where the mode says -1 or nothing.
  public struct ModeLimits: Equatable, Sendable {
    public var gpu: Int?
    public var memory: Int?
    public var cpu: Int?
    public init(gpu: Int? = nil, memory: Int? = nil, cpu: Int? = nil) {
      self.gpu = gpu
      self.memory = memory
      self.cpu = cpu
    }
  }

  /// The `< POWER_MODEL ID=2 NAME=MAXN_SUPER >` sections of an nvpmodel.conf,
  /// each with the lines up to the next section.
  static func sections(_ conf: String) -> [(id: Int, name: String, lines: [Substring])] {
    var out: [(id: Int, name: String, lines: [Substring])] = []
    var inMode = false
    for raw in conf.split(separator: "\n") {
      let line = raw.trimmingCharacters(in: .whitespaces)
      if line.hasPrefix("<") {
        var fields: [String: String] = [:]
        for field in line.trimmingCharacters(in: CharacterSet(charactersIn: "<> ")).split(separator: " ") {
          let pair = field.split(separator: "=", maxSplits: 1)
          if pair.count == 2 { fields[String(pair[0])] = String(pair[1]) }
        }
        inMode = line.contains("POWER_MODEL") && fields["ID"].flatMap { Int($0) } != nil
        if inMode, let id = fields["ID"].flatMap({ Int($0) }) { out.append((id, fields["NAME"] ?? String(id), [])) }
      } else if inMode, !line.hasPrefix("#") {
        out[out.count - 1].lines.append(Substring(line))
      }
    }
    return out
  }

  /// The modes' names by ID.
  public static func modes(_ conf: String?) -> [Int: String] {
    Dictionary(sections(conf ?? "").map { ($0.id, $0.name) }, uniquingKeysWith: { first, _ in first })
  }

  /// The limits of the mode named `name`; nil when the file has no such mode.
  public static func limits(_ conf: String, mode name: String) -> ModeLimits? {
    guard let section = sections(conf).first(where: { $0.name == name }) else { return nil }
    var limits = ModeLimits()
    for line in section.lines {
      let words = line.split(separator: " ")
      guard words.count == 3, words[1] == "MAX_FREQ", let value = Int(words[2]) else { continue }
      let limit = value > 0 ? value : nil
      switch words[0] {
      case "GPU": limits.gpu = limit
      case "EMC": limits.memory = limit
      case "CPU_A78_0": limits.cpu = limit
      default: break
      }
    }
    return limits
  }

  /// The mode ID in nvpmodel's status file: "pmode:0002 fmode:quiet".
  public static func modeID(status: String?) -> Int? {
    (status ?? "").split(separator: " ").first(where: { $0.hasPrefix("pmode:") }).flatMap { Int($0.dropFirst(6)) }
  }
}
