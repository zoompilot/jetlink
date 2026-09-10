import SwiftUI

/// The three places in the app.
enum SidebarItem: String, CaseIterable, Identifiable, Hashable {
  case status, models, logs

  var id: String { rawValue }

  var title: String {
    switch self {
    case .status: "Status"
    case .models: "Models"
    case .logs: "Logs"
    }
  }

  var symbol: String {
    switch self {
    case .status: "gauge.with.dots.needle.33percent"
    case .models: "shippingbox"
    case .logs: "doc.text"
    }
  }
}

struct MainWindow: View {
  @Environment(ServerStore.self) private var server
  @State private var selection: SidebarItem? = .status

  var body: some View {
    NavigationSplitView {
      List(SidebarItem.allCases, selection: $selection) { item in
        Label(item.title, systemImage: item.symbol)
          .tag(item)
      }
      .navigationSplitViewColumnWidth(min: 160, ideal: 180)
    } detail: {
      detail
        .navigationTitle(selection?.title ?? "Jetlink")
        .toolbar {
          ToolbarItem(placement: .principal) {
            StatusBadge(text: summary.0, tone: summary.1)
              .lineLimit(1)
              .fixedSize()
          }
          ToolbarItem(placement: .primaryAction) {
            runButton
          }
        }
    }
    .frame(minWidth: 860, minHeight: 540)
  }

  @ViewBuilder
  private var detail: some View {
    switch selection ?? .status {
    case .status: StatusView(selection: $selection)
    case .models: ModelsView()
    case .logs: LogsView()
    }
  }

  private var summary: (String, StatusBadge.Tone) {
    StatusBadge.summary(runState: server.runState, link: server.link, engine: server.engine)
  }

  @ViewBuilder
  private var runButton: some View {
    switch server.runState {
    case .stopped, .failed:
      Button("Start server", systemImage: "play.fill") { server.start() }
        .help("Start the Jetlink server")
    case .serving:
      Button("Stop server", systemImage: "stop.fill") { server.stop() }
        .help("Stop the Jetlink server")
    case .starting, .stopping:
      Button { } label: {
        ProgressView()
          .controlSize(.small)
      }
      .disabled(true)
      .help(server.runState == .starting ? "The server is starting" : "The server is stopping")
    }
  }
}

#Preview("Serving") {
  MainWindow()
    .environment(ServerStore.preview(runState: .serving, info: PreviewData.serverInfo, link: PreviewData.linkWaiting, engine: PreviewData.engineReady))
    .environment(ModelStore.preview(catalog: PreviewData.catalog, inventory: PreviewData.inventory, engine: PreviewData.engineReady))
    .environment(LogBuffer.preview(lines: PreviewData.logLines))
    .environment(AppSettings.preview())
    .frame(width: 860, height: 560)
}
