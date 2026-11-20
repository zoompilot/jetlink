#if os(Linux)
  import CUsbfs
  import Foundation
  import Glibc
  import JetlinkKit
  import JetlinkServer

  /// What `SysfsGadget` hands the claimed interface to: the server's
  /// `UsbfsGadget`, or a test's stand-in.
  protocol UsbfsTarget: GadgetSource {
    func attach(fd: Int32, inEndpoint: UInt8, outEndpoint: UInt8) throws
    func detach()
  }

  extension UsbfsGadget: UsbfsTarget {}

  /// The usbdevfs calls a claim makes, so the tests can stand in for the
  /// kernel. Each failure carries the errno.
  protocol UsbfsNode: Sendable {
    func open(_ path: String) throws(KernelError) -> Int32
    func descriptors(_ fd: Int32) throws(KernelError) -> [UInt8]
    /// The kernel driver bound to the interface, or nil.
    func driver(_ fd: Int32, interface: UInt8) throws(KernelError) -> String?
    func disconnect(_ fd: Int32, interface: UInt8) throws(KernelError)
    func claim(_ fd: Int32, interface: UInt8) throws(KernelError)
    func release(_ fd: Int32, interface: UInt8)
    func close(_ fd: Int32)
  }

  /// The comma's gadget on a Jetson or a Linux PC: found in sysfs, opened
  /// through usbdevfs, and served by the same `UsbfsGadget` Android's app
  /// feeds. Android's UsbManager finds, opens and claims for the app; here
  /// this does it (D7), with no libusb.
  ///
  /// `present()` is sysfs alone, as Python's was: a comma whose gadget cannot
  /// be claimed is still on the bus. `open()` claims on first sight and keeps
  /// the descriptor across the comma's per-run reopens, as on Android; the
  /// claim goes when the gadget leaves the bus, re-enumerates, or its
  /// transfers see ENODEV.
  public final class SysfsGadget: @unchecked Sendable {
    struct Found: Equatable {
      let name: String
      let bus: Int
      let device: Int
      /// Mb/s as sysfs says, for the log.
      let speed: String?
    }

    struct Claimed {
      let found: Found
      let fd: Int32
      let interface: UInt8
    }

    /// The comma's USB 3 link power files, found at the claim: its port's
    /// permit, by its real path since the port outlives the device's `port`
    /// link, and the device's own U1 and U2 states.
    struct LinkPower {
      let permit: String
      let states: [String]
    }

    let root: HostRoot
    private let node: any UsbfsNode
    private let target: any UsbfsTarget
    /// JETLINK_USB_LPM=1: USB 3 link power management stays on.
    private let keepLPM: Bool
    private let write: @Sendable (String, String) throws(KernelError) -> Void
    private let log: ServerLog
    private let lock = NSLock()
    private var claimed: Claimed?
    private var seen: Found?
    /// Link power management's writes and the state below, in order and off
    /// the session thread: a permit write makes the hub send control
    /// transfers, and a session's first message should not wait on them.
    let power = DispatchQueue(label: "jetlink-usb-lpm")
    /// On `power`: the claimed comma's files, when its link has any.
    private var linkPower: LinkPower?
    /// On `power`: a comma session is being served.
    private var serving = false
    /// On `power`: the files of a link whose power management is off.
    private var lpmOff: LinkPower?
    private var closed = false
    /// A session with nothing from the comma for this long has its link back
    /// at the default until the comma's next message. Three of the comma's
    /// 10 s keep-alive pings, so a modeld waiting to swap keeps it off.
    let quietAfter: TimeInterval
    /// Under `heardLock`, which the session thread takes once a message: when
    /// it last heard the comma (uptime, ns), and whether the link went back
    /// to the default for the quiet since.
    private let heardLock = NSLock()
    private var lastHeard: UInt64 = 0
    private var dozing = false
    /// On `power`: bumped by every (re)arm, so only the latest quiet check runs.
    private var quietCheck = 0

    public convenience init() {
      self.init(root: .system, node: DevUsbfs(), target: UsbfsGadget())
    }

    init(
      root: HostRoot, node: any UsbfsNode, target: any UsbfsTarget, environment: [String: String] = ProcessInfo.processInfo.environment,
      write: @escaping @Sendable (String, String) throws(KernelError) -> Void = Sysfs.write, log: ServerLog = ServerLog(category: "usb"),
      quietAfter: TimeInterval = 30
    ) {
      self.root = root
      self.node = node
      self.target = target
      // An A/B switch: with it set, sessions keep the kernel's default.
      keepLPM = environment["JETLINK_USB_LPM"] == "1"
      self.write = write
      self.log = log
      self.quietAfter = quietAfter
    }

    /// 1209:0001 in /sys/bus/usb/devices, the first by name.
    func scan() -> Found? {
      Self.scan(root: root)
    }

    static func scan(root: HostRoot) -> Found? {
      let devices = "/sys/bus/usb/devices"
      let vendor = String(format: "%04x", Pinned.usbVendorID)
      let product = String(format: "%04x", Pinned.usbProductID)
      for name in root.list(devices) where !name.contains(":") {
        let base = root.path("\(devices)/\(name)")
        guard Sysfs.read("\(base)/idVendor")?.lowercased() == vendor, Sysfs.read("\(base)/idProduct")?.lowercased() == product,
          let bus = Sysfs.readInt("\(base)/busnum"), let device = Sysfs.readInt("\(base)/devnum")
        else { continue }
        return Found(name: name, bus: bus, device: device, speed: Sysfs.read("\(base)/speed"))
      }
      return nil
    }

    /// Opens the node sysfs's bus and device numbers name, finds the vendor
    /// interface, and claims it for `target`.
    private func claim(_ found: Found) throws -> Claimed {
      let path = root.path(String(format: "/dev/bus/usb/%03d/%03d", found.bus, found.device))
      let fd = try node.open(path)
      do {
        let active = Sysfs.readInt(root.path("/sys/bus/usb/devices/\(found.name)/bConfigurationValue"))
        let picked = try GadgetDescriptors.pick(node.descriptors(fd), configuration: active)
        if let driver = try node.driver(fd, interface: picked.interface) {
          // "usbfs" is another program's claim, and taking the interface
          // from it would break that program mid-transfer: an old server
          // still running, most likely.
          guard driver != "usbfs" else {
            throw LinkError.closed("interface \(picked.interface) of \(path) is claimed by another program")
          }
          try node.disconnect(fd, interface: picked.interface)
          log.info("detached the \(driver) driver from the comma's interface \(picked.interface)")
        }
        try node.claim(fd, interface: picked.interface)
        do {
          try target.attach(fd: fd, inEndpoint: picked.inEndpoint, outEndpoint: picked.outEndpoint)
        } catch {
          node.release(fd, interface: picked.interface)
          throw error
        }
        let endpoints = String(format: "bulk in %02x out %02x", picked.inEndpoint, picked.outEndpoint)
        log.info(
          "claimed the comma's gadget at \(found.name) (\(path), \(found.speed.map { "\($0) Mb/s" } ?? "unknown speed")): interface \(picked.interface), \(endpoints)"
        )
        let files = linkPowerFiles(found)
        power.async { [self] in
          linkPower = files
          linkPowerAtClaim()
        }
        return Claimed(found: found, fd: fd, interface: picked.interface)
      } catch {
        node.close(fd)
        throw error
      }
    }

    // MARK: link power management

    /// USB 3 link power management (U1/U2) is off on the comma's own port
    /// while a session is served and the comma is talking, and the kernel's
    /// default otherwise. On it is the largest single cost of the link: every
    /// idle gap between frames ends in an exit from U1 or U2, 3.9 ms of a
    /// measured 7.6 ms onroad transport. Off only the comma's port: the
    /// hub then keeps its uplink in U0 by itself. Between sessions (parked,
    /// the comma holding the gadget) the link dozes in U2 again. Suspend is
    /// U3 and is unaffected.
    ///
    /// A session outlives the comma's modeld: it ends only when the gadget
    /// leaves the bus, and the comma's owner keeps it up 60 s after a drive,
    /// for good with sleep_after 0. Off, that idle link draws about 0.18 W
    /// more, so after `quietAfter` with no message the link goes back to
    /// the default, and the next message turns it off again. That write is
    /// queued like the others, so the message that brings it does not wait
    /// for it in user space. The kernel does hold the comma's device lock
    /// across the write's control transfers, and every usbfs call takes that
    /// lock, so its reply can wait a few ms, as at every session start. In
    /// the car that message is a hello or a keep-alive ping, not a frame.
    ///
    /// The kernel applies a permit to the attached device at once, but a
    /// write that cannot reach it (the bus suspended) reports success and
    /// changes nothing, so the device's own U1/U2 state is read back after
    /// each write. Nothing to do on a USB 2 link, or for a device the kernel
    /// runs without LPM (no such files).
    ///
    /// The server says when a session starts and ends over any link: a comma
    /// served over TCP while this gadget is claimed turns it off too.
    public func sessionStarted() {
      resetHeard()
      power.async { [self] in
        serving = true
        if let linkPower { linkPowerOff(linkPower) }
        armQuietCheck(after: quietAfter)
      }
    }

    public func sessionEnded() {
      resetHeard()
      power.async { [self] in
        serving = false
        quietCheck += 1
        linkPowerStock()
      }
    }

    /// For every message, on the session's thread, and for a finished
    /// build's push, on the job's: a clock read and a lock, and a queued
    /// write only for the first after a quiet spell (see above for what that
    /// write can hold up in the kernel).
    public func sessionHeard() {
      let now = DispatchTime.now().uptimeNanoseconds
      let woke = heardLock.withLock {
        let woke = dozing
        lastHeard = now
        dozing = false
        return woke
      }
      guard woke else { return }
      power.async { [self] in
        guard serving else { return }
        if let linkPower, !keepLPM { linkPowerOff(linkPower, " (the comma is sending again)") }
        armQuietCheck(after: quietAfter)
      }
    }

    /// The server is stopping: the comma's link goes back to the default
    /// before this returns.
    public func close() {
      power.sync {
        closed = true
        serving = false
        quietCheck += 1
        linkPowerStock()
      }
    }

    /// A session starts or ends: the quiet counts from now.
    private func resetHeard() {
      let now = DispatchTime.now().uptimeNanoseconds
      heardLock.withLock {
        lastHeard = now
        dozing = false
      }
    }

    /// On `power`: checks for quiet once `delay` has passed, instead of any
    /// check armed before.
    private func armQuietCheck(after delay: TimeInterval) {
      quietCheck += 1
      let armed = quietCheck
      power.asyncAfter(deadline: .now() + max(delay, 0)) { [weak self] in
        guard let self, armed == quietCheck, serving else { return }
        checkQuiet()
      }
    }

    /// On `power`: after `quietAfter` with nothing heard the link goes back to
    /// the default and stays there until `sessionHeard`; else the check is
    /// armed again for when that would be.
    private func checkQuiet() {
      let quiet = UInt64(quietAfter * 1e9)
      let left: UInt64? = heardLock.withLock {
        // Read under the lock, so no message can be stamped later than now.
        let now = DispatchTime.now().uptimeNanoseconds
        let since = now > lastHeard ? now - lastHeard : 0
        guard since < quiet else {
          dozing = true
          return nil
        }
        return quiet - since
      }
      if let left {
        armQuietCheck(after: Double(left) / 1e9)
      } else {
        linkPowerStock(" (nothing from the comma for \(String(format: "%g", quietAfter)) s)")
      }
    }

    private func linkPowerFiles(_ found: Found) -> LinkPower? {
      let device = root.path("/sys/bus/usb/devices/\(found.name)")
      let states = ["u1", "u2"].map { "\(device)/power/usb3_hardware_lpm_\($0)" }
      guard (found.speed.flatMap { Int($0) } ?? 5000) >= 5000, states.allSatisfy({ Sysfs.read($0) != nil }),
        let port = realpath("\(device)/port", nil)
      else { return nil }
      defer { free(port) }
      let permit = String(cString: port) + "/usb3_lpm_permit"
      return Sysfs.read(permit) == nil ? nil : LinkPower(permit: permit, states: states)
    }

    /// On `power`: a session starts, or the comma talks again after a quiet spell.
    private func linkPowerOff(_ files: LinkPower, _ why: String = "") {
      guard !closed else { return }
      if keepLPM {
        // A port an earlier session left off would stay so until a reboot.
        try? write(files.permit, "u1_u2")
        log.info("usb 3 link power management left on for the comma's session (JETLINK_USB_LPM=1)")
        return
      }
      do throws(KernelError) {
        try write(files.permit, "0")
      } catch {
        log.warning("could not turn usb 3 link power management off for the comma's session: \(error)")
        return
      }
      lpmOff = files
      let read = files.states.map { Sysfs.read($0) ?? "unreadable" }
      if read.allSatisfy({ $0 == "disabled" }) {
        log.info("usb 3 link power management off for the comma's session\(why)")
      } else {
        log.warning("usb 3 link power management is still on for the comma's session after writing 0 to \(files.permit): u1 \(read[0]), u2 \(read[1])")
      }
    }

    /// On `power`: the session ended or went quiet, or the server stops.
    /// Written to the port itself, which outlives a comma that left the bus.
    private func linkPowerStock(_ why: String = "") {
      guard let off = lpmOff else { return }
      lpmOff = nil
      do throws(KernelError) {
        try write(off.permit, "u1_u2")
      } catch {
        log.warning("could not put usb 3 link power management back to stock for the comma's link: \(error)")
        return
      }
      // A comma that left the bus took its state files with it.
      let read = off.states.map { Sysfs.read($0) }
      if read.contains(where: { $0 != nil && $0 != "enabled" }) {
        log.warning("usb 3 link power management back to stock for the comma's link, but u1 \(read[0] ?? "gone"), u2 \(read[1] ?? "gone")\(why)")
      } else {
        log.info("usb 3 link power management back to stock for the comma's link\(why)")
      }
    }

    /// On `power`, at a claim: a session in progress (the comma came back
    /// under it) gets its link off again, unless it has gone quiet; otherwise
    /// a port an earlier server left off, by crashing mid-session, goes back
    /// to stock for the park.
    private func linkPowerAtClaim() {
      guard let linkPower else { return }
      if serving {
        if !heardLock.withLock({ dozing }) { linkPowerOff(linkPower) }
      } else if !keepLPM, Sysfs.read(linkPower.permit) == "0" {
        lpmOff = linkPower
        linkPowerStock(" (an earlier server left it off)")
      }
    }

    /// Under `lock`.
    private func releaseClaim(_ reason: String) {
      guard let claimed else { return }
      self.claimed = nil
      target.detach()
      node.release(claimed.fd, interface: claimed.interface)
      node.close(claimed.fd)
      power.async { [self] in linkPower = nil }
      log.info("released the comma's gadget at \(claimed.found.name): \(reason)")
    }
  }

  extension SysfsGadget: GadgetSource {
    public func present() -> Bool {
      let found = scan()
      lock.withLock {
        seen = found
        guard let claimed else { return }
        guard let found else {
          releaseClaim("it left the bus")
          return
        }
        if found != claimed.found {
          releaseClaim("it came back as \(found.name) device \(found.device)")
        } else if !target.present() {
          releaseClaim("the device is gone")
        }
      }
      return found != nil
    }

    public func open() throws -> any MessageLink {
      try lock.withLock {
        if claimed != nil && !target.present() {
          releaseClaim("the device is gone")
        }
        guard claimed == nil else { return }
        guard let found = seen ?? scan() else { throw LinkError.closed("no jetlink gadget is on the bus") }
        claimed = try claim(found)
      }
      return try target.open()
    }
  }

  /// The vendor interface and its bulk pair, read from the descriptors the
  /// gadget sends: FunctionFS numbers endpoints at bind, so they are never
  /// the same two constants.
  struct GadgetDescriptors: Equatable {
    let interface: UInt8
    let inEndpoint: UInt8
    let outEndpoint: UInt8

    private struct Interface {
      let number: UInt8
      let alternate: UInt8
      let kind: [UInt8]
      var endpoints: [(address: UInt8, attributes: UInt8)] = []
    }

    /// From a usbfs node's bytes (the device descriptor, then every
    /// configuration): in configuration `configuration` (bConfigurationValue;
    /// nil takes the first), the FF/FF/FF interface, else interface 0; with
    /// `kind`, that class's interface and no other (an add-on's).
    static func pick(_ bytes: [UInt8], configuration: Int? = nil, kind: [UInt8]? = nil) throws -> GadgetDescriptors {
      var interfaces: [Interface] = []
      var inWanted = false
      var index = 0
      while index + 2 <= bytes.count {
        let length = Int(bytes[index])
        guard length >= 2, index + length <= bytes.count else { break }
        let field = { (offset: Int) in bytes[index + offset] }
        switch field(1) {
        case 2 where length >= 9:
          // A configuration: take this one, and stop at the next.
          if inWanted { index = bytes.count; continue }
          inWanted = configuration.map { Int(field(5)) == $0 } ?? true
        case 4 where length >= 9 && inWanted:
          interfaces.append(Interface(number: field(2), alternate: field(3), kind: [field(5), field(6), field(7)]))
        case 5 where length >= 7 && inWanted && !interfaces.isEmpty:
          interfaces[interfaces.count - 1].endpoints.append((field(2), field(3)))
        default:
          break
        }
        index += length
      }
      let usable = interfaces.filter { $0.alternate == 0 }
      let found =
        kind.map { kind in usable.first { $0.kind == kind } }
        ?? usable.first(where: { $0.kind == [0xff, 0xff, 0xff] }) ?? usable.first(where: { $0.number == 0 })
      guard let chosen = found else {
        let what = kind.map { "an interface of class " + $0.map { String(format: "%02x", $0) }.joined(separator: "/") } ?? "a vendor interface"
        throw LinkError.closed("the gadget's descriptors have no \(what)")
      }
      let bulk = chosen.endpoints.filter { $0.attributes & 0x03 == 0x02 }
      guard let input = bulk.first(where: { $0.address & 0x80 != 0 }), let output = bulk.first(where: { $0.address & 0x80 == 0 }) else {
        throw LinkError.closed("interface \(chosen.number) of the gadget has no bulk pair")
      }
      return GadgetDescriptors(interface: chosen.number, inEndpoint: input.address, outEndpoint: output.address)
    }
  }

  /// usbdevfs on /dev/bus/usb.
  struct DevUsbfs: UsbfsNode {
    func open(_ path: String) throws(KernelError) -> Int32 {
      let fd = Glibc.open(path, O_RDWR | O_CLOEXEC)
      guard fd >= 0 else { throw KernelError(what: path, errno: errno) }
      return fd
    }

    func descriptors(_ fd: Int32) throws(KernelError) -> [UInt8] {
      var bytes = [UInt8](repeating: 0, count: 8192)
      let n = bytes.withUnsafeMutableBytes { jl_usbfs_descriptors(fd, $0.baseAddress, Int32($0.count)) }
      guard n >= 0 else { throw KernelError(what: "the gadget's descriptors", errno: -n) }
      return Array(bytes[..<Int(n)])
    }

    func driver(_ fd: Int32, interface: UInt8) throws(KernelError) -> String? {
      var name = [CChar](repeating: 0, count: 256)
      let error = name.withUnsafeMutableBufferPointer { jl_usbfs_driver(fd, UInt32(interface), $0.baseAddress, Int32($0.count)) }
      if error == ENODATA { return nil }
      guard error == 0 else { throw KernelError(what: "the driver of interface \(interface)", errno: error) }
      return name.withUnsafeBufferPointer { String(cString: $0.baseAddress!) }
    }

    func disconnect(_ fd: Int32, interface: UInt8) throws(KernelError) {
      let error = jl_usbfs_disconnect(fd, UInt32(interface))
      guard error == 0 else { throw KernelError(what: "detaching the driver of interface \(interface)", errno: error) }
    }

    func claim(_ fd: Int32, interface: UInt8) throws(KernelError) {
      let error = jl_usbfs_claim(fd, UInt32(interface))
      guard error == 0 else { throw KernelError(what: "claiming interface \(interface)", errno: error) }
    }

    func release(_ fd: Int32, interface: UInt8) {
      // ENODEV once the gadget is gone, and the close lets go of it anyway.
      _ = jl_usbfs_release(fd, UInt32(interface))
    }

    func close(_ fd: Int32) {
      _ = Glibc.close(fd)
    }
  }
#endif
