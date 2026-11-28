import JetlinkKit
import JetlinkORT
import JetlinkUI
import SwiftUI

/// Connection, performance, display, storage, help and versions, as a native form.
struct SettingsScreen: View {
  @Environment(AppModel.self) private var app
  @State private var portText = ""
  @State private var versionTaps = 0
  @State private var path: [Destination] = SettingsScreen.initialPath

  enum Destination: Hashable {
    case logs, connect
  }

  /// `-tab logs` or `-tab connect` opens Settings with that screen pushed,
  /// for screenshots.
  static var initialPath: [Destination] {
    switch UserDefaults.standard.string(forKey: "tab") {
    case "logs": [.logs]
    case "connect": [.connect]
    default: []
    }
  }

  var body: some View {
    @Bindable var settings = app.settings
    NavigationStack(path: $path) {
      Form {
        connection
        Section {
          Picker("Processor", selection: $settings.device) {
            Text(Processor.automatic.title).tag(Processor.automatic)
            Text(Processor.aneBoth.title).tag(Processor.aneBoth)
            Text(Processor.ane.title).tag(Processor.ane)
            Text(Processor.coreml.title).tag(Processor.coreml)
            #if targetEnvironment(simulator)
              // The simulator has no Neural Engine and runs CoreML on the CPU anyway.
              Text(Processor.cpu.title).tag(Processor.cpu)
            #endif
          }
          .disabled(app.comparison.running)
          if settings.device == .automatic {
            LabeledContent("Running On", value: app.server.profile.title)
          }
          Toggle("Keep CPU Awake", isOn: $settings.keepCPUWarm)
          Toggle("Keep GPU Awake", isOn: $settings.keepGPUAwake)
        } header: {
          Text("Performance")
        } footer: {
          Text(
            "Automatic uses Neural Engine Only, the coolest, unless Compare Processors on the Benchmark tab finds another best for a model. It runs in lower precision, and Compare Processors checks a model's outputs on it. Changing the processor prepares models again."
          )
        }
        Section {
          Toggle("Keep Screen On", isOn: $settings.keepScreenOn)
        } header: {
          Text("Display")
        } footer: {
          Text("Jetlink has to stay open while you drive.")
        }
        storage
        help
        about
        if settings.developer {
          developer
        }
      }
      .readableWidth()
      .navigationTitle("Settings")
      .navigationDestination(for: Destination.self) { destination in
        switch destination {
        case .logs: LogsScreen()
        case .connect: ConnectHelpScreen()
        }
      }
      .onAppear { portText = String(settings.port) }
      .onChange(of: settings.device) { app.server.restart() }
      .onChange(of: settings.keepGPUAwake) { app.server.restart() }
      .onChange(of: settings.keepCPUWarm) { app.server.restart() }
      .onChange(of: settings.listening) { app.server.restart() }
      .sensoryFeedback(.success, trigger: settings.developer)
    }
  }

  // MARK: connection

  private var connection: some View {
    Section("Connection") {
      LabeledContent("Link", value: app.server.linkMedium?.phoneTitle ?? (app.network.cable == nil ? "Not Connected" : "Connecting"))
    }
  }

  // MARK: developer

  /// Connect over Wi-Fi, and the port a comma on Wi-Fi or bench tools on a Mac
  /// reach the phone on. Driving over the cable never uses it: the phone dials the comma.
  private var developer: some View {
    @Bindable var settings = app.settings
    return Section {
      Toggle("Connect over Wi-Fi", isOn: $settings.overWiFi)
      if settings.overWiFi {
        wiFiDetails
      }
    } header: {
      Text("Developer")
    } footer: {
      Text(
        "Lets a comma with Jetlink set to Wi-Fi, and bench tools on a Mac, reach the phone over Wi-Fi. Use Personal Hotspot with Maximize Compatibility off; the comma dials port 5599. Driving over the cable does not need it."
      )
    }
  }

  @ViewBuilder
  private var wiFiDetails: some View {
    LabeledContent("Port") {
      TextField("5599", text: $portText)
        .keyboardType(.numberPad)
        .multilineTextAlignment(.trailing)
        .monospacedDigit()
        .onSubmit(applyPort)
    }
    if portChanged {
      Button("Use Port \(portText)", action: applyPort)
    }
    // The cable's address is the comma's to hand out and the phone's to
    // dial; nobody types it.
    ForEach(app.network.addresses.filter { $0.kind != .cable }) { address in
      LabeledContent {
        Text("\(address.address):\(app.server.port.map(String.init) ?? portText)")
          .monospacedDigit()
          .textSelection(.enabled)
      } label: {
        Label(address.kind.title, systemImage: address.kind.symbol)
      }
    }
  }

  private var portChanged: Bool {
    UInt16(portText).map { $0 != app.settings.port } ?? false
  }

  private func applyPort() {
    guard let port = UInt16(portText), port > 0 else {
      portText = String(app.settings.port)
      return
    }
    guard port != app.settings.port else { return }
    app.settings.port = port
    app.server.restart()
  }

  // MARK: storage

  @ViewBuilder
  private var storage: some View {
    if let disk = app.models.inventory?.disk {
      Section {
        LabeledContent("Downloaded", value: ByteCount.string(disk.modelsBytes))
        LabeledContent("Prepared", value: ByteCount.string(disk.enginesBytes))
        // CoreML's compiled copies in the app's caches, which iOS counts as Jetlink's data
        if let copies = disk.neuralEngineBytes {
          LabeledContent("Neural Engine Copies", value: ByteCount.string(copies))
        }
        LabeledContent("Available", value: ByteCount.string(disk.freeBytes))
      } header: {
        Text("Storage")
      } footer: {
        if let short = app.models.runningShortfall {
          Text(short.message)
        }
      }
    }
  }

  // MARK: help

  private var help: some View {
    Section("Help") {
      NavigationLink("Connecting the Comma", value: Destination.connect)
      NavigationLink("Logs", value: Destination.logs)
    }
  }

  // MARK: about

  private var about: some View {
    Section("About") {
      LabeledContent("Version", value: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "")
        .contentShape(Rectangle())
        .onTapGesture(perform: tapVersion)
      LabeledContent("Runtime", value: "onnxruntime \(app.server.state.server?.runtimeVersion ?? OrtRuntime.version)")
      if let device = app.server.state.server?.device {
        LabeledContent("Chip", value: SettingsScreen.chip(device))
      }
      LabeledContent("Server", value: serverText)
    }
  }

  /// Seven taps on the version turn the developer settings on or off, as a
  /// build number does on Android.
  private func tapVersion() {
    versionTaps += 1
    guard versionTaps >= SettingsScreen.developerTaps else { return }
    versionTaps = 0
    app.settings.developer.toggle()
  }

  static let developerTaps = 7

  /// "Apple A17 Pro" from "ane-Apple_A17_Pro".
  static func chip(_ device: String) -> String {
    let name = device.split(separator: "-", maxSplits: 1).last.map(String.init) ?? device
    return name.replacingOccurrences(of: "_", with: " ")
  }

  private var serverText: String {
    switch app.server.runState {
    case .stopped: "Stopped"
    case .starting: "Starting"
    case .serving: "Running"
    case .stopping: "Stopping"
    case .failed: "Failed"
    }
  }
}
