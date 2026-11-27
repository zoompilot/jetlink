#if canImport(SwiftUI)
  import JetlinkKit
  import SwiftUI

  /// How a model row reads in either app's list: one line of facts, what Use
  /// Model is about to do, and what the disk holds.
  public enum ModelFormatting {
    /// What Use Model is about to do. `device` is "this Mac", "this iPhone" or "this iPad".
    public static func useHelp(_ row: ModelRow, device: String) -> String {
      switch row.status {
      case .notDownloaded:
        let size = row.bytes.map { " \(ByteCount.string($0))" } ?? ""
        return "Downloads\(size), prepares it for \(device) and starts using it"
      case .prepared:
        return "Starts using it. It is prepared already, so this takes seconds."
      case .failed:
        return "Tries again"
      default:
        return "Prepares it for \(device) and starts using it"
      }
    }

    /// "Sep 1, 2026 · 766 MB · Prepared for CoreML": everything but the name and
    /// what is happening right now, on one line under the name.
    public static func detailLine(_ row: ModelRow) -> String {
      var parts: [String] = []
      let built = BuildTime.text(row.buildTime)
      if !built.isEmpty { parts.append(built) }
      if let bytes = row.bytes { parts.append(ByteCount.string(bytes)) }
      let prepared = preparedForText(row)
      if !prepared.isEmpty {
        parts.append("Prepared for \(prepared)")
      } else if row.status == .downloaded {
        parts.append("Downloaded")
      }
      if parts.isEmpty, row.isOrphan, let sha = row.sha256 {
        parts.append(String(sha.prefix(16)))
      }
      return parts.joined(separator: " · ")
    }

    /// "Downloads 2.3 GB · Prepared engines 6.9 GB · 13.6 GB available", with
    /// "Neural Engine copies 1.5 GB" after the engines where there are any.
    public static func diskSummary(models: Int64, engines: Int64, neuralEngine: Int64? = nil, free: Int64) -> String {
      let copies = neuralEngine.map { " · Neural Engine copies \(ByteCount.string($0))" } ?? ""
      return "Downloads \(ByteCount.string(models)) · Prepared engines \(ByteCount.string(engines))\(copies) · \(ByteCount.string(free)) available"
    }

    /// "CoreML, tinygrad": the backends a model already has an engine for.
    public static func preparedForText(_ row: ModelRow) -> String {
      var seen: [String] = []
      for artifact in row.preparedFor {
        let name = backendName(artifact.backend)
        if !seen.contains(name) { seen.append(name) }
      }
      return seen.joined(separator: ", ")
    }

    public static func backendName(_ backend: String) -> String {
      switch backend {
      case "ort": "CoreML"
      case "trt": "TensorRT"
      case "tinygrad": "tinygrad"
      default: backend
      }
    }

    /// "Downloading 42%, 41 MB/s".
    public static func downloadCaption(frac: Double, rateBps: Double) -> String {
      let percent = ModelStatusLabel.downloadingText(frac)
      return rateBps > 0 ? "\(percent), \(ByteCount.rate(rateBps))" : percent
    }
  }

  /// A small capsule label next to a model's name. On a selected row it turns
  /// to the selection's text colour, so an accent tag never sits on the accent.
  public struct ModelTag: View {
    let text: String
    let tone: Color

    public init(_ text: String, tone: Color) {
      self.text = text
      self.tone = tone
    }

    public var body: some View {
      Text(text)
        .font(.caption.weight(.medium))
        .padding(.horizontal, 6)
        .padding(.vertical, 1)
        .foregroundStyle(SelectableTint(tone))
        .background(Capsule().fill(SelectableTint(tone.opacity(0.14), selected: AnyShapeStyle(.white.opacity(0.2)))))
    }
  }
#endif
