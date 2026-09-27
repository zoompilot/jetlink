// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.

import JetlinkKit
import SwiftUI

struct SettingsView: View {
  @Environment(AppModel.self) private var model
  @Environment(AppSettings.self) private var settings
  @State private var portText = ""

  private var connected: Bool {
    if case .connected = model.link { return true }
    return false
  }

  var body: some View {
    Form {
      Section {
        Picker(
          "Run models on",
          selection: Binding(get: { settings.compute }, set: { model.setCompute($0) })
        ) {
          ForEach([ComputePreference.auto, .ane, .coreml], id: \.self) { p in Text(p.label).tag(p) }
        }
        .disabled(connected || model.engine.isBusy)
      } header: {
        Text("Compute")
      } footer: {
        Text(computeFooter)
      }

      Section {
        Toggle("Keep the screen on while serving", isOn: Binding(get: { settings.keepScreenAwake }, set: { model.setKeepScreenAwake($0) }))
        Toggle("Start the server when Jetlink opens", isOn: Binding(get: { settings.startOnLaunch }, set: { settings.startOnLaunch = $0 }))
        Toggle("Keep the CPU ready between frames", isOn: Binding(get: { settings.cpuKeepWarm }, set: { model.setCPUKeepWarm($0) }))
        Toggle("Keep the GPU clocked between frames", isOn: Binding(get: { settings.gpuKeepAlive }, set: { model.setGPUKeepAlive($0) }))
      } footer: {
        Text(
          "iOS stops a background app, so Jetlink has to stay on screen while driving. Between frames the CPU slows down, and the Neural Engine path waits on it: keeping one CPU core busy while frames arrive took a 20 Hz run from 54 to 36 ms p99 on a Mac, for one core's power. The GPU keep-alive does the same job for the GPU path. Both stop a second after the last frame and apply from the next model load."
        )
      }

      Section {
        HStack {
          Text("Port")
          Spacer()
          TextField("5599", text: $portText)
            .keyboardType(.numberPad)
            .multilineTextAlignment(.trailing)
            .frame(width: 90)
            .onSubmit { applyPort() }
        }
        .disabled(connected)
        Button("Apply") { applyPort() }
          .disabled(Int(portText) == settings.port || Int(portText).map { !(1024...65535).contains($0) } ?? true)
      } header: {
        Text("Network")
      } footer: {
        Text("The comma's JetlinkEndpoint must use the same port. 5599 is the default.")
      }

      Section("Help") {
        NavigationLink("Connecting to the comma") { ConnectionHelpView() }
      }

      Section("About") {
        LabeledContent("Jetlink", value: Bundle.main.shortVersion)
        LabeledContent("onnxruntime", value: OrtRuntime.version)
        LabeledContent("Device", value: chipName())
        LabeledContent("Protocol", value: "v\(Wire.version)")
      }
    }
    .navigationTitle("Settings")
    .onAppear { portText = String(settings.port) }
  }

  private var computeFooter: String {
    switch settings.compute {
    case .auto:
      "The first time a model is prepared, Jetlink builds it for the Neural Engine and the GPU, times each at the comma's 20 frames a second, and keeps the Neural Engine if it runs a frame in \(Int(EngineHost.calibrationBudgetMs)) ms, since it uses far less power. Otherwise it keeps the faster one."
    case .ane:
      "CoreML with the Neural Engine allowed. The most efficient choice when it is fast enough."
    case .coreml:
      "CoreML on the GPU."
    case .cpu:
      "The CPU, for testing only. It will not keep up with driving."
    }
  }

  private func applyPort() {
    if let p = Int(portText) { model.setPort(p) }
    portText = String(settings.port)
  }
}

struct ConnectionHelpView: View {
  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 16) {
        Text("Jetlink needs a wired link to the comma. Wi-Fi is too slow for the 50 ms frame budget. The full steps, and the zoompilot patch the comma needs, are in ios/README.md.")
        Group {
          Text("Ethernet").font(.headline)
          Text(
            """
            1. Put a USB-C Ethernet adapter on the comma, and a USB-C hub with Ethernet and power pass-through on the iPhone, and join them with a network cable.
            2. On the comma, set JetlinkEndpoint to 192.168.60.2:5599, then turn Accelerator Link off, and give its adapter 192.168.60.1.
            3. On the iPhone, open Settings > Ethernet, choose the adapter, and set Configure IP to Manual: address 192.168.60.2, subnet mask 255.255.255.0, no router.
            4. Turn Accelerator Link back on.
            """)
        }
        Group {
          Text("One USB cable").font(.headline)
          Text(
            """
            The comma presents itself as a USB network adapter. Plugged straight into the iPhone, the comma tries to power the phone and reboots, so go through a hub: iPhone, USB-C hub, USB-A to USB-C cable, comma. With the zoompilot patch, turn on Settings > Models > Accelerator Link and Accelerator on iPhone on the comma, while parked; it switches its USB port over by itself, at every boot too. Set the phone's address as in step 3, the first time.
            """)
        }
        Group {
          Text("Power and heat").font(.headline)
          Text(
            """
            The iPhone's only port is taken by the link, so charge it with MagSafe, or use a hub with power pass-through. Running a large model continuously heats the phone; keep it out of the sun and watch the temperature on the Status tab.
            """)
        }
        Group {
          Text("Keep it on screen").font(.headline)
          Text("iOS suspends apps in the background. Leave Jetlink open while driving; it keeps the screen on while the server runs.")
        }
      }
      .padding()
    }
    .navigationTitle("Connecting")
  }
}
