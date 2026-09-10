import SwiftUI

/// One model's status, compact enough for a table cell.
struct ModelStatusLabel: View {
  let status: ModelStatus
  /// A model whose size is not known yet reads as "Checking…" while the
  /// catalog is being fetched, and as "Unknown size" once it is not.
  var isCheckingCatalog: Bool = false

  init(_ status: ModelStatus, isCheckingCatalog: Bool = false) {
    self.status = status
    self.isCheckingCatalog = isCheckingCatalog
  }

  var body: some View {
    switch status {
    case .unresolved:
      Text(isCheckingCatalog ? "Checking…" : "Unknown size")
        .foregroundStyle(.secondary)
    case .notDownloaded:
      Text("Not downloaded")
        .foregroundStyle(.secondary)
    case let .downloading(frac, rateBps):
      VStack(alignment: .leading, spacing: 2) {
        Text("Downloading \(Int((frac * 100).rounded()))%")
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
      Text("Loaded")
        .fontWeight(.bold)
        .foregroundStyle(.green)
    case let .failed(detail):
      Text("Failed")
        .foregroundStyle(.red)
        .help(detail)
    }
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
