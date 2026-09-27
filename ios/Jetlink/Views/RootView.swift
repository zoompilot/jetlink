// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.

import SwiftUI

struct RootView: View {
  @Environment(AppModel.self) private var model

  var body: some View {
    TabView {
      NavigationStack { StatusView() }
        .tabItem { Label("Status", systemImage: "gauge.with.dots.needle.67percent") }
      NavigationStack { ModelsView() }
        .tabItem { Label("Models", systemImage: "square.stack.3d.up") }
      NavigationStack { LogsView() }
        .tabItem { Label("Logs", systemImage: "text.alignleft") }
      NavigationStack { SettingsView() }
        .tabItem { Label("Settings", systemImage: "gearshape") }
    }
    .alert(
      "The comma shut down",
      isPresented: Binding(get: { model.shutdownNotice != nil }, set: { if !$0 { model.dismissShutdownNotice() } })
    ) {
      Button("OK", role: .cancel) { model.dismissShutdownNotice() }
    } message: {
      Text(model.shutdownNotice ?? "")
    }
  }
}
