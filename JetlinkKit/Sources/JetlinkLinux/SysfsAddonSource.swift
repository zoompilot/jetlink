// An add-on's interface on the comma's gadget, through sysfs and usbfs.
// docs/addons.md.
#if canImport(CUsbfs)
  import Foundation
  import JetlinkKit
  import JetlinkServer

  /// Finds the gadget as `SysfsGadget` does, then claims only the interface
  /// of `kind`, on a descriptor of its own: usbfs claims are per interface,
  /// so Jetlink's own claim is untouched.
  public final class SysfsAddonSource: AddonSource, @unchecked Sendable {
    private struct Claimed {
      let found: SysfsGadget.Found
      let fd: Int32
      let interface: UInt8
    }

    private let kind: [UInt8]
    private let root: HostRoot
    private let node: any UsbfsNode
    private let port = UsbfsAddonPort()
    private let lock = NSLock()
    private var claimed: Claimed?

    public convenience init(kind: [UInt8]) {
      self.init(kind: kind, root: .system, node: DevUsbfs())
    }

    init(kind: [UInt8], root: HostRoot, node: any UsbfsNode) {
      self.kind = kind
      self.root = root
      self.node = node
    }

    public func present() -> Bool {
      let found = SysfsGadget.scan(root: root)
      lock.withLock {
        if let claimed, found != claimed.found || !port.isAttached { release(claimed) }
      }
      return found != nil
    }

    public func open() throws -> AddonPipes {
      try lock.withLock {
        if let claimed, !port.isAttached { release(claimed) }
        if claimed == nil {
          guard let found = SysfsGadget.scan(root: root) else { throw LinkError.closed("no jetlink gadget is on the bus") }
          claimed = try claim(found)
        }
      }
      return try port.open()
    }

    private func claim(_ found: SysfsGadget.Found) throws -> Claimed {
      let path = root.path(String(format: "/dev/bus/usb/%03d/%03d", found.bus, found.device))
      let fd = try node.open(path)
      do {
        let active = Sysfs.readInt(root.path("/sys/bus/usb/devices/\(found.name)/bConfigurationValue"))
        let picked = try GadgetDescriptors.pick(node.descriptors(fd), configuration: active, kind: kind)
        if let driver = try node.driver(fd, interface: picked.interface) {
          guard driver != "usbfs" else { throw LinkError.closed("interface \(picked.interface) is claimed by another program") }
          try node.disconnect(fd, interface: picked.interface)
        }
        try node.claim(fd, interface: picked.interface)
        do {
          try port.attach(fd: fd, inEndpoint: picked.inEndpoint, outEndpoint: picked.outEndpoint)
        } catch {
          node.release(fd, interface: picked.interface)
          throw error
        }
        return Claimed(found: found, fd: fd, interface: picked.interface)
      } catch {
        node.close(fd)
        throw error
      }
    }

    /// Under the lock.
    private func release(_ claimed: Claimed) {
      port.detach()
      node.release(claimed.fd, interface: claimed.interface)
      node.close(claimed.fd)
      self.claimed = nil
    }
  }
#endif
