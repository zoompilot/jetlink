// The add-on server: a fork's own models beside the driving model, each on its
// own interface of the comma's gadget. docs/addons.md. From Peter Nguyen's
// V-ASM and sign-vision servers (trung791997/openpilot).
#if canImport(CUsbfs)
  import Foundation
  import JetlinkKit

  /// An add-on interface already claimed on a usbfs descriptor: the add-on's
  /// `UsbfsGadget`. The claimer (JetlinkLinux's SysfsAddonSource) keeps the
  /// descriptor and calls `detach` before it closes it.
  public final class UsbfsAddonPort: @unchecked Sendable {
    private struct Attached {
      let device: UsbfsDevice
      let inEndpoint: UInt8
      let outEndpoint: UInt8
    }

    private let lock = NSLock()
    private var attached: Attached?

    public init() {}

    public func attach(fd: Int32, inEndpoint: UInt8, outEndpoint: UInt8) throws {
      let next = Attached(device: UsbfsDevice(kernel: try LinuxUsbfs(fd: fd)), inEndpoint: inEndpoint, outEndpoint: outEndpoint)
      swap(next)?.device.invalidate()
    }

    public func detach() {
      swap(nil)?.device.invalidate()
    }

    public var isAttached: Bool {
      current() != nil
    }

    public func open() throws -> AddonPipes {
      guard let current = current() else { throw LinkError.closed("the add-on interface is not attached") }
      return AddonPipes(UsbfsPipes(device: current.device, inEndpoint: current.inEndpoint, outEndpoint: current.outEndpoint))
    }

    private func swap(_ next: Attached?) -> Attached? {
      lock.lock()
      defer { lock.unlock() }
      let previous = attached
      attached = next
      return previous
    }

    private func current() -> Attached? {
      lock.lock()
      defer { lock.unlock() }
      return attached.flatMap { $0.device.isGone ? nil : $0 }
    }
  }
#endif
