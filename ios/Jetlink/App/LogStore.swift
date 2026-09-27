// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.

import Foundation
import JetlinkKit
import Observation

/// The last few hundred log lines, for the Logs tab and for sharing.
@MainActor
@Observable
final class LogStore {
  private(set) var lines: [LogLine] = []
  static let capacity = 1000

  func append(_ line: LogLine) {
    lines.append(line)
    if lines.count > LogStore.capacity { lines.removeFirst(lines.count - LogStore.capacity) }
  }

  func clear() { lines.removeAll() }

  /// Everything as plain text, one line each.
  var text: String {
    let f = DateFormatter()
    f.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
    return lines.map { "\(f.string(from: $0.date)) \($0.level) \($0.category): \($0.message)" }.joined(separator: "\n")
  }
}
