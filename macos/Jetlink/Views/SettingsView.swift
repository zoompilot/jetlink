import AppKit
import JetlinkUI
import SwiftUI

struct SettingsView: View {
  var body: some View {
    TabView {
      GeneralSettingsView()
        .tabItem { Label("General", systemImage: "gear") }
      ServerSettingsView()
        .tabItem { Label("Server", systemImage: "cpu") }
    }
    .frame(width: 540)
  }
}

// MARK: - General

struct GeneralSettingsView: View {
  @Environment(AppSettings.self) private var settings
  @Environment(ServerStore.self) private var server
  @Environment(UpdateStore.self) private var updates
  @State private var loginItem = LoginItem()

  var body: some View {
    @Bindable var settings = settings
    @Bindable var updates = updates
    Form {
      Section {
        Toggle("Start server when Jetlink opens", isOn: $settings.startServerOnLaunch)
        VStack(alignment: .leading, spacing: 4) {
          Toggle("Open Jetlink at login", isOn: $loginItem.isEnabled)
          if loginItem.requiresApproval {
            Text("Approve Jetlink in System Settings > General > Login Items.")
              .font(.callout)
              .foregroundStyle(.secondary)
            Button("Open Login Items") { loginItem.openSystemSettings() }
          }
        }
        VStack(alignment: .leading, spacing: 4) {
          Toggle("Prevent sleep while server is running", isOn: $settings.keepAwakeWhileServing)
          Toggle("Prevent sleep with the lid closed", isOn: $settings.keepAwakeLidClosed)
            .disabled(!settings.keepAwakeWhileServing)
          Text(
            settings.keepAwakeLidClosed
              ? "On power or battery, with the lid open or closed. Normal sleep returns when the server stops."
              : "By default, sleep prevention applies only on power. Closing the lid may still put the Mac to sleep."
          )
          .font(.callout)
          .foregroundStyle(.secondary)
        }
        .onChange(of: [settings.keepAwakeWhileServing, settings.keepAwakeLidClosed]) { server.keepAwakeSettingChanged() }
      }

      Section("Cache folder") {
        VStack(alignment: .leading, spacing: 8) {
          Text(settings.cacheDirectory.path(percentEncoded: false))
            .font(.system(.callout, design: .monospaced))
            .textSelection(.enabled)
            .fixedSize(horizontal: false, vertical: true)
          HStack {
            Button("Choose…") { chooseCacheDirectory() }
            Button("Show in Finder") {
              NSWorkspace.shared.activateFileViewerSelecting([settings.cacheDirectory])
            }
            if needsRestart {
              Button("Restart Now") { server.restart() }
            }
          }
          Text("Models and prepared engines. A CoreML engine is about 2 GB. Takes effect when the server restarts.")
            .font(.callout)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
      }

      Section("Updates") {
        if updates.isAvailable {
          Toggle("Check for updates automatically", isOn: $updates.automaticallyChecks)
          VStack(alignment: .leading, spacing: 4) {
            Toggle("Download and install automatically", isOn: $updates.automaticallyDownloads)
              .disabled(!updates.automaticallyChecks)
            Text("Downloads in the background and installs when Jetlink quits.")
              .font(.callout)
              .foregroundStyle(.secondary)
          }
          LabeledContent("Jetlink \(ServerStore.appVersion)") {
            Button("Check Now") { updates.checkForUpdates() }
              .disabled(!updates.canCheckForUpdates)
          }
        } else {
          VStack(alignment: .leading, spacing: 4) {
            Text("Jetlink \(ServerStore.appVersion)")
            Text("This build does not update itself. Releases are on [GitHub](https://github.com/zoompilot/jetlink/releases).")
              .font(.callout)
              .foregroundStyle(.secondary)
          }
        }
      }
    }
    .formStyle(.grouped)
  }

  private var needsRestart: Bool {
    guard server.runState == .serving, let running = server.info?.cache else { return false }
    return running != settings.cacheDirectory.path(percentEncoded: false)
  }

  private func chooseCacheDirectory() {
    let panel = NSOpenPanel()
    panel.canChooseDirectories = true
    panel.canChooseFiles = false
    panel.allowsMultipleSelection = false
    panel.canCreateDirectories = true
    panel.directoryURL = settings.cacheDirectory
    panel.prompt = "Choose"
    if panel.runModal() == .OK, let url = panel.url {
      settings.cacheDirectory = url
    }
  }
}

// MARK: - Server

struct ServerSettingsView: View {
  @Environment(AppSettings.self) private var settings
  @Environment(ServerStore.self) private var server

  /// The server is still on the backend chosen before.
  private var pending: Bool {
    server.runState == .serving && server.info.map { $0.choice != settings.backend } ?? false
  }

  /// Restarts the server onto the backend chosen. A comma's frames come
  /// first: with one connected the change waits for Restart Server.
  private func applyBackend() {
    if pending, server.link.state != .connected { server.restart() }
  }

  var body: some View {
    @Bindable var settings = settings
    Form {
      Section {
        VStack(alignment: .leading, spacing: 4) {
          Picker("Backend", selection: $settings.backend) {
            ForEach(BackendChoice.allCases, id: \.self) { choice in
              Text(choice == .auto ? "Automatic (\(choice.title))" : choice.title).tag(choice)
            }
          }
          // Also once a restart lands, for a second change made during it.
          .onChange(of: settings.backend) { applyBackend() }
          .onChange(of: server.runState) { applyBackend() }
          Text(ServerSettingsView.backendCaption(settings.backend))
            .font(.callout)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
      } footer: {
        HStack {
          Text(
            pending
              ? "A comma is connected, so this applies when the server restarts."
              : "Changing the backend restarts the server, unless a comma is connected."
          )
          .font(.callout)
          .foregroundStyle(.secondary)
          Spacer()
          Button("Restart Server") { server.restart() }
            .disabled(server.runState != .serving)
        }
      }
    }
    .formStyle(.grouped)
  }

  static func backendCaption(_ backend: BackendChoice) -> String {
    switch backend {
    case .auto:
      "Recommended: most efficient on Apple silicon. Initial preparation takes about 20 seconds."
    case .aneBoth:
      "Nothing on the GPU: the least power and heat. Runs part of the model in lower precision, and nothing checks its outputs. Slower than the default on an M1 Pro."
    case .coreml:
      "Less efficient than the Neural Engine. Use it if another app keeps the Neural Engine busy."
    }
  }
}

#Preview("General") {
  GeneralSettingsView()
    .environment(AppSettings.preview())
    .environment(UpdateStore(isCommaConnected: { false }))
    .environment(ServerStore.preview(runState: .serving, info: PreviewData.serverInfo, link: PreviewData.linkWaiting, engine: PreviewData.engineReady))
    .frame(width: 540, height: 420)
}

#Preview("Server") {
  ServerSettingsView()
    .environment(AppSettings.preview())
    .environment(ServerStore.preview(runState: .serving, info: PreviewData.serverInfo, link: PreviewData.linkWaiting, engine: PreviewData.engineReady))
    .frame(width: 540, height: 420)
}
