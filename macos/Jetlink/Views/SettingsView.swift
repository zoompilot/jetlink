import AppKit
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
  @State private var loginItem = LoginItem()

  var body: some View {
    @Bindable var settings = settings
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
          Toggle("Keep the Mac awake while serving", isOn: $settings.keepAwakeWhileServing)
          Text("Only when connected to power. On battery, keep the lid open.")
            .font(.callout)
            .foregroundStyle(.secondary)
        }
      }

      Section("Cache folder") {
        VStack(alignment: .leading, spacing: 8) {
          Text(settings.cacheDirectory.path(percentEncoded: false))
            .font(.system(.callout, design: .monospaced))
            .textSelection(.enabled)
            .fixedSize(horizontal: false, vertical: true)
          HStack {
            Button("Choose…") { chooseCacheDirectory() }
            Button("Reveal") {
              NSWorkspace.shared.activateFileViewerSelecting([settings.cacheDirectory])
            }
            if needsRestart {
              Button("Restart now") { server.restart() }
            }
          }
          Text("Models and prepared engines. A CoreML engine is about 5.5 GB. Changing the folder takes effect when the server restarts.")
            .font(.callout)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
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
  @State private var advancedExpanded = false

  var body: some View {
    @Bindable var settings = settings
    Form {
      Section {
        VStack(alignment: .leading, spacing: 4) {
          Picker("Backend", selection: $settings.backend) {
            Text("Automatic (CoreML on the GPU)").tag(BackendChoice.auto)
            Text("CoreML on the GPU").tag(BackendChoice.coreml)
            Text("CoreML with the Neural Engine").tag(BackendChoice.ane)
            Text("tinygrad on Metal").tag(BackendChoice.tinygrad)
          }
          Text(ServerSettingsView.backendCaption(settings.backend))
            .font(.callout)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
        Picker("Connection", selection: $settings.transport) {
          Text("USB (the comma)").tag(TransportChoice.usb)
          Text("TCP (bench client)").tag(TransportChoice.tcp)
        }
        if settings.transport == .tcp {
          TextField("Port", value: $settings.tcpPort, format: .number.grouping(.never))
        }
        Picker("Log level", selection: $settings.logLevel) {
          Text("Normal (INFO)").tag("INFO")
          Text("Verbose (DEBUG)").tag("DEBUG")
        }
      } footer: {
        HStack {
          Text("Changes apply when the server restarts.")
            .font(.callout)
            .foregroundStyle(.secondary)
          Spacer()
          Button("Restart server") { server.restart() }
            .disabled(server.runState != .serving)
        }
      }

      Section {
        DisclosureGroup("Advanced", isExpanded: $advancedExpanded) {
          VStack(alignment: .leading, spacing: 6) {
            TextField(
              "Python interpreter override",
              text: Binding(
                get: { settings.pythonOverride ?? "" },
                set: { settings.pythonOverride = $0.isEmpty ? nil : $0 }
              ))
            Text("For development. Leave empty to use the bundled runtime.")
              .font(.callout)
              .foregroundStyle(.secondary)
            Text(bundledText)
              .font(.callout)
              .foregroundStyle(.secondary)
              .textSelection(.enabled)
          }
          .padding(.top, 4)
        }
      }
    }
    .formStyle(.grouped)
  }

  static func backendCaption(_ backend: BackendChoice) -> String {
    switch backend {
    case .auto:
      "Recommended. About 43 ms a frame on an M1 Pro. Preparing a model takes about 9 minutes, and loading one again takes as long, so keep Jetlink running."
    case .coreml:
      "About 43 ms a frame on an M1 Pro. Preparing a model takes about 9 minutes, and loading one again takes as long, so keep Jetlink running."
    case .ane:
      "Faster back to back, slower at the comma's 20 Hz on an M1 Pro. Measure on your Mac before using it in the car."
    case .tinygrad:
      "Loads in a second. About 66 ms a frame on an M1 Pro, which is over the 50 ms budget; a newer Mac may be under it."
    }
  }

  private var bundledText: String {
    guard let manifest = EmbeddedPython.manifest() else {
      return "This build has no bundled Python runtime."
    }
    var parts: [String] = []
    if let python = manifest["python"] { parts.append("Python \(python)") }
    if let ort = manifest["onnxruntime"] { parts.append("onnxruntime \(ort)") }
    if let tinygrad = manifest["tinygrad"] { parts.append("tinygrad \(String(tinygrad.prefix(8)))") }
    return parts.isEmpty ? "This build has no bundled Python runtime." : "Bundled: " + parts.joined(separator: ", ")
  }
}

#Preview("General") {
  GeneralSettingsView()
    .environment(AppSettings.preview())
    .environment(ServerStore.preview(runState: .serving, info: PreviewData.serverInfo, link: PreviewData.linkWaiting, engine: PreviewData.engineReady))
    .frame(width: 540, height: 420)
}

#Preview("Server") {
  ServerSettingsView()
    .environment(AppSettings.preview())
    .environment(ServerStore.preview(runState: .serving, info: PreviewData.serverInfo, link: PreviewData.linkWaiting, engine: PreviewData.engineReady))
    .frame(width: 540, height: 420)
}
