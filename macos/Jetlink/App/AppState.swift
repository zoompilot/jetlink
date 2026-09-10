import Foundation
import Observation
import os

/// The composition root. One of these exists for the life of the app.
@MainActor
@Observable
final class AppState {
  let settings: AppSettings
  let server: ServerStore
  let models: ModelStore
  let logs: LogBuffer
  let loginItem: LoginItem

  @ObservationIgnored private let log = Logger(subsystem: "io.zoompilot.jetlink", category: "app")
  @ObservationIgnored private var launched = false

  init() {
    let settings = AppSettings()
    let logs = LogBuffer()
    let server = ServerStore(settings: settings, logs: logs)
    self.settings = settings
    self.logs = logs
    self.server = server
    self.models = ModelStore(server: server)
    self.loginItem = LoginItem()
  }

  /// Called once, when the app has finished launching.
  func launch() {
    guard !launched else { return }
    launched = true
    guard settings.startServerOnLaunch else { return }
    log.info("starting the server on launch")
    server.start()
    Task { @MainActor in
      do {
        try await server.startIfNeeded()
        models.refreshCatalog()
      } catch {
        log.error("the server did not come up on launch: \(error.localizedDescription, privacy: .public)")
      }
    }
  }

  /// The app delegate awaits this before replying to applicationShouldTerminate.
  func applicationWillTerminate() async {
    log.info("stopping the server because the app is quitting")
    await server.stopAndWait()
  }
}
