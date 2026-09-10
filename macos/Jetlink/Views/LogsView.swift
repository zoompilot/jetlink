import AppKit
import SwiftUI

/// The server's stderr, line by line.
struct LogsView: View {
  @Environment(LogBuffer.self) private var logs
  @State private var filter = ""
  @State private var autoScroll = true

  private static let renderLimit = 5000
  private static let bottomAnchor = "logs.bottom"

  var body: some View {
    VStack(spacing: 0) {
      lines
      Divider()
      bottomBar
    }
    .searchable(text: $filter, placement: .toolbar, prompt: "Filter")
    .toolbar {
      ToolbarItem {
        Button("Copy all", systemImage: "doc.on.doc") { copyAll() }
          .help("Copy every shown line")
      }
      ToolbarItem {
        Button("Clear", systemImage: "trash") { logs.clear() }
          .help("Clear the lines shown here. The log file keeps them")
      }
      ToolbarItem {
        Button("Reveal log file", systemImage: "folder") {
          NSWorkspace.shared.activateFileViewerSelecting([LogsView.logFileURL])
        }
        .help("Show server.log in the Finder")
      }
    }
  }

  private var bottomBar: some View {
    HStack {
      Toggle("Follow new lines", isOn: $autoScroll)
        .toggleStyle(.checkbox)
      Spacer()
      Text("\(logs.lines.count.formatted()) lines")
        .foregroundStyle(.secondary)
    }
    .font(.callout)
    .padding(.horizontal, 12)
    .padding(.vertical, 6)
  }

  private var lines: some View {
    ScrollViewReader { proxy in
      ScrollView {
        LazyVStack(alignment: .leading, spacing: 1) {
          ForEach(Array(visibleLines.enumerated()), id: \.offset) { _, line in
            Text(line)
              .font(.system(.caption, design: .monospaced))
              .foregroundStyle(LogsView.tone(for: line))
              .textSelection(.enabled)
              .frame(maxWidth: .infinity, alignment: .leading)
          }
          Color.clear
            .frame(height: 1)
            .id(LogsView.bottomAnchor)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
      }
      .onScrollGeometryChange(for: Bool.self) { geometry in
        geometry.contentOffset.y + geometry.containerSize.height >= geometry.contentSize.height - 24
      } action: { _, atBottom in
        autoScroll = atBottom
      }
      .onChange(of: logs.revision) {
        if autoScroll {
          proxy.scrollTo(LogsView.bottomAnchor, anchor: .bottom)
        }
      }
      .onAppear {
        proxy.scrollTo(LogsView.bottomAnchor, anchor: .bottom)
      }
    }
  }

  private var visibleLines: [String] {
    let lines = filter.isEmpty ? logs.lines : logs.lines.filter { $0.localizedCaseInsensitiveContains(filter) }
    return lines.count > LogsView.renderLimit ? Array(lines.suffix(LogsView.renderLimit)) : lines
  }

  private func copyAll() {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(visibleLines.joined(separator: "\n"), forType: .string)
  }

  static var logFileURL: URL {
    FileManager.default.homeDirectoryForCurrentUser.appending(path: "Library/Logs/Jetlink/server.log")
  }

  /// The server logs as "%(asctime)s %(levelname)-7s %(name)s: %(message)s".
  static func tone(for line: String) -> Color {
    if line.contains(" ERROR ") { return .red }
    if line.contains(" WARNING") { return .orange }
    return .primary
  }
}

#Preview {
  LogsView()
    .environment(LogBuffer.preview(lines: PreviewData.logLines))
    .frame(width: 720, height: 400)
}
