import SwiftUI

/// A byte count in the Finder's style, or nothing at all when the size is unknown.
struct ByteCount: View {
  let bytes: Int64?

  init(_ bytes: Int64?) {
    self.bytes = bytes
  }

  var body: some View {
    Text(bytes.map { ByteCount.string($0) } ?? "")
  }

  /// "766 MB", "1.8 GB": the Finder's decimal units, one decimal at most, and
  /// no decimal at all on a whole number of units.
  static func string(_ bytes: Int64) -> String {
    let units = ["bytes", "KB", "MB", "GB", "TB", "PB"]
    var value = Double(bytes)
    var unit = 0
    while abs(value) >= 1000, unit < units.count - 1 {
      value /= 1000
      unit += 1
    }
    if unit == 0 {
      return "\(bytes) bytes"
    }
    var rounded = (value * 10).rounded() / 10
    if abs(rounded) >= 1000, unit < units.count - 1 {
      rounded = ((rounded / 1000) * 10).rounded() / 10
      unit += 1
    }
    let fraction = rounded == rounded.rounded() ? 0 : 1
    return "\(rounded.formatted(.number.precision(.fractionLength(fraction)))) \(units[unit])"
  }

  /// A transfer rate, "41.2 MB/s". Rates below a byte a second read as "0 bytes/s".
  static func rate(_ bytesPerSecond: Double) -> String {
    let clamped = bytesPerSecond.isFinite && bytesPerSecond > 0 ? bytesPerSecond : 0
    return string(Int64(clamped.rounded())) + "/s"
  }
}

#Preview {
  VStack(alignment: .leading) {
    ByteCount(765_953_504)
    ByteCount(nil)
    Text(ByteCount.rate(41_200_000))
  }
  .padding()
}
