// The add-on server: a fork's own models beside the driving model, each on its
// own interface of the comma's gadget. docs/addons.md. From Peter Nguyen's
// V-ASM and sign-vision servers (trung791997/openpilot).
#if canImport(CUsbfs)
  import Foundation
  import JetlinkKit

  /// An add-on interface already claimed on a usbfs descriptor: the add-on's
  /// `UsbfsGadget`. The claimer (JetlinkLinux's SysfsAddonSource, or the
  /// Android app on the gadget's own descriptor) keeps the descriptor and
  /// calls `detach` before it closes it.
  public final class UsbfsAddonPort: @unchecked Sendable {
    private struct Attached {
      let device: UsbfsDevice
      let inEndpoint: UInt8
      let outEndpoint: UInt8
      /// Its own descriptor's device, which it ends; a device shared with
      /// the driving link (Android) is the gadget's to end.
      let owned: Bool
    }

    private let lock = NSLock()
    private var attached: Attached?

    public init() {}

    public func attach(fd: Int32, inEndpoint: UInt8, outEndpoint: UInt8) throws {
      let next = Attached(device: UsbfsDevice(kernel: try LinuxUsbfs(fd: fd)), inEndpoint: inEndpoint, outEndpoint: outEndpoint, owned: true)
      end(swap(next))
    }

    /// On the gadget's own descriptor: the driving link's device, as usbfs has
    /// one completion queue per descriptor and only one device may reap it.
    public func attach(sharing gadget: UsbfsGadget, inEndpoint: UInt8, outEndpoint: UInt8) throws {
      guard let device = gadget.device else { throw LinkError.closed("the comma's gadget is not attached") }
      end(swap(Attached(device: device, inEndpoint: inEndpoint, outEndpoint: outEndpoint, owned: false)))
    }

    public func detach() {
      end(swap(nil))
    }

    private func end(_ previous: Attached?) {
      if let previous, previous.owned { previous.device.invalidate() }
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
  extension UsbfsAddonPort: AddonSource {
    public func present() -> Bool {
      isAttached
    }
  }
#endif
