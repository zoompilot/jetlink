import AppKit
import Foundation
import Observation
import ServiceManagement
import os

/// Open at login, through the app's own login item registration. `isEnabled`
/// is settable so a SwiftUI Toggle can bind straight to it.
@MainActor
@Observable
final class LoginItem {
  private(set) var status: SMAppService.Status
  var lastError: String?

  @ObservationIgnored private let service = SMAppService.mainApp
  @ObservationIgnored private let log = Logger(subsystem: "io.zoompilot.jetlink", category: "loginitem")

  init() {
    status = SMAppService.mainApp.status
  }

  var isEnabled: Bool {
    get { status == .enabled }
    set {
      lastError = nil
      do {
        if newValue {
          if service.status != .enabled { try service.register() }
        } else {
          if service.status != .notRegistered { try service.unregister() }
        }
      } catch {
        log.error("could not change the login item: \(error.localizedDescription, privacy: .public)")
        lastError = error.localizedDescription
      }
      // Re-read the real status, so a toggle that did not take snaps back.
      status = service.status
    }
  }

  /// True when the user has to approve the login item in System Settings.
  var requiresApproval: Bool { status == .requiresApproval }

  func refresh() {
    status = service.status
  }

  /// Opens the Login Items pane so the user can approve a pending registration.
  func openSystemSettings() {
    guard let url = URL(string: "x-apple.systempreferences:com.apple.LoginItems-Settings.extension") else { return }
    NSWorkspace.shared.open(url)
  }
}
