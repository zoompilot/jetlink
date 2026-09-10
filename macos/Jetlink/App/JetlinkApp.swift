import AppKit
import SwiftUI

@main
struct JetlinkApp: App {
  @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
  @State private var appState = AppState()

  var body: some Scene {
    Window("Jetlink", id: "main") {
      MainWindow()
        .jetlinkEnvironment(appState)
        .onAppear { delegate.appState = appState }
    }
    .defaultSize(width: 860, height: 560)
    .commands { AppCommands(appState: appState) }

    MenuBarExtra("Jetlink", systemImage: menuBarSymbol) {
      MenuBarView()
        .jetlinkEnvironment(appState)
        .onAppear { delegate.appState = appState }
    }
    .menuBarExtraStyle(.menu)

    Settings {
      SettingsView()
        .jetlinkEnvironment(appState)
    }
  }

  /// Three symbols: nothing running, running, and a comma on the other end.
  private var menuBarSymbol: String {
    switch appState.server.runState {
    case .stopped, .failed:
      return "cable.connector.slash"
    case .starting, .stopping:
      return "cable.connector"
    case .serving:
      return appState.server.link.state == .connected ? "car.fill" : "cable.connector"
    }
  }
}

extension View {
  /// Every store the views read, from the one state the app owns.
  func jetlinkEnvironment(_ appState: AppState) -> some View {
    self
      .environment(appState)
      .environment(appState.settings)
      .environment(appState.server)
      .environment(appState.models)
      .environment(appState.logs)
  }
}

struct AppCommands: Commands {
  let appState: AppState

  var body: some Commands {
    CommandGroup(replacing: .newItem) { }
    CommandMenu("Server") {
      Button("Start server") { appState.server.start() }
        .keyboardShortcut("r", modifiers: .command)
        .disabled(!canStart)
      Button("Stop server") { appState.server.stop() }
        .keyboardShortcut(".", modifiers: .command)
        .disabled(appState.server.runState != .serving)
      Button("Restart server") { appState.server.restart() }
        .disabled(appState.server.runState != .serving)
      Divider()
      Button("Refresh model list") { appState.models.refreshCatalog() }
        .keyboardShortcut("r", modifiers: [.command, .shift])
        .disabled(appState.server.runState != .serving)
      Divider()
      Button("Reveal cache in Finder") {
        NSWorkspace.shared.activateFileViewerSelecting([cacheURL])
      }
    }
  }

  private var canStart: Bool {
    switch appState.server.runState {
    case .stopped, .failed: true
    case .starting, .serving, .stopping: false
    }
  }

  private var cacheURL: URL {
    if let cache = appState.server.info?.cache, !cache.isEmpty {
      return URL(fileURLWithPath: cache)
    }
    return appState.settings.cacheDirectory
  }
}
