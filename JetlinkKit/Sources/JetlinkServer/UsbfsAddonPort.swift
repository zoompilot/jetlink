// The add-on server: a fork's own models beside the driving model, each on its
// own interface of the comma's gadget. docs/addons.md. From Peter Nguyen's
// V-ASM and sign-vision servers (trung791997/openpilot).
import Foundation
import JetlinkKit

/// An add-on interface already claimed on a usbfs descriptor: the add-on's
/// `UsbfsGadget`. The claimer (JetlinkLinux's SysfsAddonSource, on a
/// descriptor of its own, or the Android app on the gadget's descriptor)
/// keeps the descriptor and calls `detach` before it closes it.
public final class UsbfsAddonPort: @unchecked Sendable {
  private struct Attached {
    let device: UsbfsDevice
    let inEndpoint: UInt8
    let outEndpoint: UInt8
    /// Made for this port's own descriptor, so this port ends it; a device
    /// shared with the driving link is the gadget's to end.
    let owned: Bool
  }

  private let lock = NSLock()
  private var attached: Attached?
  private var pipes: AddonPipes?

  public init() {}

  /// On another descriptor's device: the driving link's, where the add-on's
  /// interface was claimed on the same descriptor, as one device reaps for both.
  func attach(sharing device: UsbfsDevice, inEndpoint: UInt8, outEndpoint: UInt8) {
    let previous = swap(Attached(device: device, inEndpoint: inEndpoint, outEndpoint: outEndpoint, owned: false))
    end(previous)
  }

  public var isAttached: Bool {
    current() != nil
  }

  /// Ends this port's transfers, and its device too when it is its own;
  /// returns once nothing of the port's uses the descriptor.
  public func detach() {
    end(swap(nil))
  }

  public func open() throws -> AddonPipes {
    guard let current = current() else { throw LinkError.closed("the add-on interface is not attached") }
    let opened = AddonPipes(UsbfsPipes(device: current.device, inEndpoint: current.inEndpoint, outEndpoint: current.outEndpoint))
    lock.withLock { pipes = opened }
    return opened
  }

  private func end(_ previous: (Attached, AddonPipes?)?) {
    guard let (attached, pipes) = previous else { return }
    if attached.owned {
      attached.device.invalidate()
    } else {
      // the device goes on serving the driving link: only this port's pipes end
      pipes?.pipes.abort()
      pipes?.close()
    }
  }

  private func swap(_ next: Attached?) -> (Attached, AddonPipes?)? {
    lock.lock()
    defer { lock.unlock() }
    let previous = attached.map { ($0, pipes) }
    attached = next
    pipes = nil
    return previous
  }

  private func current() -> Attached? {
    lock.lock()
    defer { lock.unlock() }
    return attached.flatMap { $0.device.isGone ? nil : $0 }
  }
}

extension UsbfsAddonPort: AddonSource {
  public func present() -> Bool {
    isAttached
  }
}

#if canImport(CUsbfs)
  extension UsbfsAddonPort {
    /// Takes over `fd`, a descriptor of its own with the interface claimed.
    public func attach(fd: Int32, inEndpoint: UInt8, outEndpoint: UInt8) throws {
      let device = UsbfsDevice(kernel: try LinuxUsbfs(fd: fd))
      end(swap(Attached(device: device, inEndpoint: inEndpoint, outEndpoint: outEndpoint, owned: true)))
    }

    /// On the gadget's own descriptor, which the driving link reads: its device.
    public func attach(sharing gadget: UsbfsGadget, inEndpoint: UInt8, outEndpoint: UInt8) throws {
      guard let device = gadget.device else { throw LinkError.closed("the comma's gadget is not attached") }
      attach(sharing: device, inEndpoint: inEndpoint, outEndpoint: outEndpoint)
    }
  }
#endif
