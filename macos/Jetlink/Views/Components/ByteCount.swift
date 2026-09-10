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

  /// "5.9 GB", the same decimal units the Finder shows.
  static func string(_ bytes: Int64) -> String {
    bytes.formatted(.byteCount(style: .file))
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
