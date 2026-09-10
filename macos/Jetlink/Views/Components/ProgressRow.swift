import SwiftUI

/// The progress of a build or a load, with the server's own message underneath.
struct ProgressRow: View {
  let stage: String?
  let frac: Double
  let msg: String

  init(stage: String?, frac: Double, msg: String) {
    self.stage = stage
    self.frac = frac
    self.msg = msg
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 4) {
      Text(ProgressRow.stageName(stage))
      if frac > 0 {
        ProgressView(value: min(max(frac, 0), 1))
      } else {
        ProgressView()
          .progressViewStyle(.linear)
      }
      if !msg.isEmpty {
        Text(msg)
          .font(.callout)
          .foregroundStyle(.secondary)
          .fixedSize(horizontal: false, vertical: true)
      }
    }
    .accessibilityElement(children: .combine)
  }

  /// The server's stage names in plain English.
  static func stageName(_ stage: String?) -> String {
    switch stage {
    case "upload": "Receiving model"
    case "patch": "Preparing the model"
    case "parse": "Reading the model"
    case "build": "Building"
    case "save": "Saving"
    case "load": "Loading"
    case "failed": "Failed"
    default: "Working"
    }
  }
}

#Preview {
  VStack(alignment: .leading, spacing: 16) {
    ProgressRow(stage: "build", frac: 0.42, msg: "compiling for CoreML, 3 min elapsed; the big model takes 11 min on an M1 Pro")
    ProgressRow(stage: "load", frac: 0, msg: "")
  }
  .frame(width: 360)
  .padding()
}
