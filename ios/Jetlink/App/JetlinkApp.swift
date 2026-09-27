// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.
//
// Jetlink for iPhone: runs openpilot's large driving models for a comma,
// on the phone's Neural Engine or GPU, over a wired link to the comma.

import SwiftUI

@main
struct JetlinkApp: App {
  @State private var model = AppModel()
  @Environment(\.scenePhase) private var scenePhase

  var body: some Scene {
    WindowGroup {
      RootView()
        .environment(model)
        .environment(model.settings)
        .environment(model.logs)
        .task { model.launch() }
    }
    .onChange(of: scenePhase) { _, phase in
      model.scenePhaseChanged(active: phase == .active, background: phase == .background)
    }
  }
}
