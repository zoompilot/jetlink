import Foundation
import IOKit.ps
import IOKit.pwr_mgt
import os

/// Keeps the Mac awake while the server is serving on AC power, the same rule
/// caffeinate -s follows. No subprocess.
@MainActor
final class SleepAssertion {
  private var assertionID: IOPMAssertionID = IOPMAssertionID(0)
  private var isActive = false
  private var runLoopSource: CFRunLoopSource?
  private let log = Logger(subsystem: "io.zoompilot.jetlink", category: "sleep")

  /// Called on the main actor when the power source changes.
  var onPowerSourceChange: (@MainActor () -> Void)?

  var isOnACPower: Bool {
    IOPSCopyExternalPowerAdapterDetails()?.takeRetainedValue() != nil
  }

  var isHoldingAssertion: Bool { isActive }

  func setActive(_ active: Bool) {
    guard active != isActive else { return }
    if active {
      var identifier = IOPMAssertionID(0)
      let result = IOPMAssertionCreateWithName(
        kIOPMAssertionTypePreventUserIdleSystemSleep as CFString,
        IOPMAssertionLevel(kIOPMAssertionLevelOn),
        "Jetlink is serving the comma" as CFString,
        &identifier)
      guard result == kIOReturnSuccess else {
        log.error("could not take a sleep assertion, error \(result)")
        return
      }
      assertionID = identifier
      isActive = true
      log.info("holding a sleep assertion")
    } else {
      IOPMAssertionRelease(assertionID)
      assertionID = IOPMAssertionID(0)
      isActive = false
      log.info("released the sleep assertion")
    }
  }

  func startObservingPowerSource() {
    guard runLoopSource == nil else { return }
    let context = Unmanaged.passUnretained(self).toOpaque()
    let callback: IOPowerSourceCallbackType = { raw in
      guard let raw else { return }
      let assertion = Unmanaged<SleepAssertion>.fromOpaque(raw).takeUnretainedValue()
      Task { @MainActor in assertion.onPowerSourceChange?() }
    }
    guard let source = IOPSNotificationCreateRunLoopSource(callback, context)?.takeRetainedValue() else {
      log.error("could not observe power source changes")
      return
    }
    runLoopSource = source
    CFRunLoopAddSource(CFRunLoopGetMain(), source, CFRunLoopMode.defaultMode)
  }

  func stopObservingPowerSource() {
    guard let source = runLoopSource else { return }
    CFRunLoopRemoveSource(CFRunLoopGetMain(), source, CFRunLoopMode.defaultMode)
    runLoopSource = nil
  }

  /// Releases everything. Call before dropping the last reference.
  func invalidate() {
    setActive(false)
    stopObservingPowerSource()
    onPowerSourceChange = nil
  }
}
