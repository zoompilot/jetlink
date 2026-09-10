import AppKit
import OSLog
import SwiftUI

/// The bits of app lifecycle SwiftUI does not cover: stopping the server before
/// the app quits, and reopening the window from the Dock.
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
  private let log = Logger(subsystem: "io.zoompilot.jetlink", category: "app")

  /// Set by `JetlinkApp` once its state exists.
  var appState: AppState?

  /// How long the server gets to stop before the app quits anyway.
  private let shutdownTimeout = Duration.seconds(20)

  func applicationDidFinishLaunching(_ notification: Notification) {
    log.info("Jetlink launched")
  }

  func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
    guard let appState, appState.server.runState != .stopped else { return .terminateNow }
    log.info("stopping the server before quitting")
    let timeout = shutdownTimeout
    Task {
      await withTaskGroup(of: Void.self) { group in
        group.addTask { await appState.applicationWillTerminate() }
        group.addTask { try? await Task.sleep(for: timeout) }
        await group.next()
        group.cancelAll()
      }
      NSApp.reply(toApplicationShouldTerminate: true)
    }
    return .terminateLater
  }

  func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
    if !flag {
      NSApp.windows.first { $0.canBecomeMain }?.makeKeyAndOrderFront(nil)
      NSApp.activate(ignoringOtherApps: true)
    }
    return true
  }
}
