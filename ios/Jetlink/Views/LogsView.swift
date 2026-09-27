// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.

import JetlinkKit
import SwiftUI

struct LogsView: View {
  @Environment(LogStore.self) private var logs

  var body: some View {
    ScrollViewReader { proxy in
      List(logs.lines) { line in
        VStack(alignment: .leading, spacing: 2) {
          Text("\(line.date.formatted(date: .omitted, time: .standard))  \(line.category)")
            .font(.caption2).foregroundStyle(.secondary)
          Text(line.message)
            .font(.caption.monospaced())
            .foregroundStyle(line.level >= .error ? .red : (line.level == .warning ? .orange : .primary))
            .textSelection(.enabled)
        }
        .id(line.id)
      }
      .listStyle(.plain)
      .onChange(of: logs.lines.last?.id) { _, id in
        if let id { proxy.scrollTo(id, anchor: .bottom) }
      }
    }
    .navigationTitle("Logs")
    .toolbar {
      ToolbarItem(placement: .topBarTrailing) {
        ShareLink(item: logs.text) { Image(systemName: "square.and.arrow.up") }
      }
      ToolbarItem(placement: .topBarLeading) {
        Button("Clear") { logs.clear() }
      }
    }
  }
}
