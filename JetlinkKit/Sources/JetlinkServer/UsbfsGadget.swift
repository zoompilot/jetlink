#if canImport(CUsbfs)
  import Foundation
  import JetlinkKit

  /// The comma's USB gadget on a file descriptor the host opened, the
  /// server's `GadgetSource` where a platform hands USB devices to apps that
  /// way: Android gives an app a device only through UsbManager, so the app
  /// finds the gadget by its IDs, asks for permission, opens it, claims the
  /// vendor interface, and passes the descriptor and the two bulk endpoints'
  /// addresses to `attach`. The server opens pipes on that descriptor for each
  /// session and never closes it; the host calls `detach` before it does.
  public final class UsbfsGadget: @unchecked Sendable {
    private struct Attached {
      let device: UsbfsDevice
      let inEndpoint: UInt8
      let outEndpoint: UInt8
      let medium: LinkMedium?
      let buffers = LinkBuffers()
    }

    private let lock = NSLock()
    private var attached: Attached?

    public init() {}

    /// Takes over `fd`, the vendor interface already claimed, with its bulk IN
    /// and OUT endpoint addresses. Replaces any earlier device.
    public func attach(fd: Int32, inEndpoint: UInt8, outEndpoint: UInt8) throws {
      let kernel = try LinuxUsbfs(fd: fd)
      let next = Attached(device: UsbfsDevice(kernel: kernel), inEndpoint: inEndpoint, outEndpoint: outEndpoint, medium: kernel.medium)
      let previous = swap(next)
      previous?.device.invalidate()
    }

    /// The device is going: ends every transfer on it and waits for them to
    /// let go of the descriptor, so the host can close it once this returns.
    public func detach() {
      swap(nil)?.device.invalidate()
    }

    private func swap(_ next: Attached?) -> Attached? {
      lock.lock()
      defer { lock.unlock() }
      let previous = attached
      attached = next
      return previous
    }

    private var current: Attached? {
      lock.lock()
      defer { lock.unlock() }
      return attached.flatMap { $0.device.isGone ? nil : $0 }
    }

    /// The descriptor's one `UsbfsDevice`, for an add-on interface the host
    /// claimed on the same descriptor: usbfs has one completion queue per
    /// descriptor, and only one device may reap it (UsbfsPipes.swift).
    var device: UsbfsDevice? {
      current?.device
    }
  }

  extension UsbfsGadget: GadgetSource {
    public func present() -> Bool {
      current != nil
    }

    public func open() throws -> any MessageLink {
      guard let current else { throw LinkError.closed("no jetlink gadget is attached") }
      let pipes = UsbfsPipes(device: current.device, inEndpoint: current.inEndpoint, outEndpoint: current.outEndpoint)
      return USBTransport(pipes: pipes, medium: current.medium ?? .usb, buffers: current.buffers)
    }
  }
#endif
