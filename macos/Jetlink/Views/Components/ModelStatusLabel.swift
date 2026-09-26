import SwiftUI

/// One model's status, in a row of the Models list and in its inspector.
struct ModelStatusLabel: View {
  let status: ModelStatus
  /// A model whose checksum is not known yet reads as "Checking…" while the
  /// catalog is being fetched, and as "Unavailable" once it is not.
  var isCheckingCatalog: Bool = false

  init(_ status: ModelStatus, isCheckingCatalog: Bool = false) {
    self.status = status
    self.isCheckingCatalog = isCheckingCatalog
  }

  var body: some View {
    switch status {
    case .unresolved:
      Text(isCheckingCatalog ? "Checking…" : "Unavailable")
        .foregroundStyle(.secondary)
    case .notDownloaded:
      Text("Not downloaded")
        .foregroundStyle(.secondary)
    case let .downloading(frac, rateBps):
      VStack(alignment: .leading, spacing: 2) {
        Text(ModelStatusLabel.downloadingText(frac))
        ProgressView(value: min(max(frac, 0), 1))
          .controlSize(.small)
        if rateBps > 0 {
          Text(ByteCount.rate(rateBps))
            .font(.caption)
            .foregroundStyle(.secondary)
        }
      }
    case .downloaded:
      Text("Downloaded")
    case let .preparing(stage, frac, msg):
      VStack(alignment: .leading, spacing: 2) {
        Text(ProgressRow.stageName(stage))
        ProgressView(value: min(max(frac, 0), 1))
          .controlSize(.small)
        if !msg.isEmpty {
          Text(msg)
            .font(.caption)
            .foregroundStyle(.secondary)
            .lineLimit(1)
            .help(msg)
        }
      }
    case .prepared:
      Text("Prepared")
    case .loaded:
      Label {
        Text("In Use")
      } icon: {
        Image(systemName: "checkmark.circle.fill")
          .foregroundStyle(SelectableTint(.green))
      }
      .fontWeight(.medium)
      .help("Jetlink is using this model")
    case let .failed(detail):
      Label {
        Text("Failed")
      } icon: {
        Image(systemName: "exclamationmark.triangle.fill")
          .foregroundStyle(SelectableTint(.red))
      }
      .help(detail)
    }
  }

  /// "Downloading 42%".
  static func downloadingText(_ frac: Double) -> String {
    "Downloading \(Int((min(max(frac, 0), 1) * 100).rounded()))%"
  }
}

/// A colour that gives way to the selection's own text colour on a selected
/// row, where it would otherwise sit on the accent and vanish.
struct SelectableTint: ShapeStyle {
  let color: Color
  var selected: AnyShapeStyle = AnyShapeStyle(.primary)

  init(_ color: Color, selected: AnyShapeStyle = AnyShapeStyle(.primary)) {
    self.color = color
    self.selected = selected
  }

  func resolve(in environment: EnvironmentValues) -> AnyShapeStyle {
    environment.backgroundProminence == .increased ? selected : AnyShapeStyle(color)
  }
}

#Preview {
  VStack(alignment: .leading, spacing: 8) {
    ModelStatusLabel(.unresolved, isCheckingCatalog: true)
    ModelStatusLabel(.notDownloaded)
    ModelStatusLabel(.downloading(frac: 0.42, rateBps: 41_000_000))
    ModelStatusLabel(.downloaded)
    ModelStatusLabel(.preparing(stage: "build", frac: 0.3, msg: "compiling for CoreML, 3 min elapsed"))
    ModelStatusLabel(.prepared)
    ModelStatusLabel(.loaded)
    ModelStatusLabel(.failed("the model could not be parsed"))
  }
  .frame(width: 220, alignment: .leading)
  .padding()
}
