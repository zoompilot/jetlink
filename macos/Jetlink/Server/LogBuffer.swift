import Foundation
import Observation

/// The server's output, as the Logs view sees it.
@MainActor
@Observable
final class LogBuffer {
  static let capacity = 5000
  static let trimChunk = 500

  private(set) var lines: [String] = []
  /// One integer a view can observe instead of the whole array.
  private(set) var revision: Int = 0

  init() {}

  func append(_ line: String) {
    lines.append(line)
    if lines.count > LogBuffer.capacity {
      lines.removeFirst(LogBuffer.trimChunk)
    }
    revision += 1
  }

  func clear() {
    lines.removeAll(keepingCapacity: true)
    revision += 1
  }

  /// The last `count` lines, for a crash report in the Status view.
  func tail(_ count: Int) -> [String] {
    Array(lines.suffix(count))
  }

  static func preview(lines: [String]) -> LogBuffer {
    let buffer = LogBuffer()
    for line in lines { buffer.append(line) }
    return buffer
  }
}
