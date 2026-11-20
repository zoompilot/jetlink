#if os(macOS)
  import Foundation
  import IOKit
  import IOUSBHost
  import JetlinkKit

  /// The comma's USB gadget as a Mac sees it, through IOUSBHost: the Swift
  /// form of `UsbBulkTransport.open` and `.present`, and the server's
  /// `GadgetSource` on a Mac.
  ///
  /// The gadget is found by the pinned vendor and product ID, and the link by
  /// its interface's class, FF/FF/FF, wherever the composite gadget put it (the
  /// comma also presents a network interface for a phone). A gadget with no
  /// vendor-class interface falls back to interface 0, where the gadget script
  /// links the link first. Endpoint addresses are read from the descriptors,
  /// never assumed: FunctionFS renumbers them at bind.
  ///
  /// No driver claims a vendor-class interface, so opening one needs no
  /// entitlement and no root. Another program holding it (the Python server,
  /// say) makes the open fail with exclusive access.
  public struct USBGadget: GadgetSource {
    public init() {}

    /// Is a jetlink gadget on the bus? Opens nothing; cheap enough to poll.
    public func present() -> Bool {
      guard let device = USBGadget.findDevice() else { return false }
      IOObjectRelease(device)
      return true
    }

    /// Opens the gadget's link interface and its bulk pair.
    public func open() throws -> any MessageLink {
      guard let device = USBGadget.findDevice() else {
        throw LinkError.closed(String(format: "no jetlink gadget at %04x:%04x", Pinned.usbVendorID, Pinned.usbProductID))
      }
      defer { IOObjectRelease(device) }
      guard let service = USBGadget.linkInterface(of: device) else {
        throw LinkError.closed("the gadget has no vendor interface yet")
      }
      // The pipes own the reference from here, and let it go when they close.
      return USBTransport(pipes: try IOUSBHostPipes(service: service), medium: USBGadget.medium(of: device) ?? .usb)
    }

    /// The USB generation the gadget enumerated at; nil without one.
    static func medium(of device: io_service_t) -> LinkMedium? {
      switch property(device, "USBSpeed") {
      case 1, 2: return .usb1
      case 3: return .usb2
      case 4, 5, 6: return .usb3
      default: return nil
      }
    }

    // MARK: the registry

    static func findDevice() -> io_service_t? {
      guard let matching = IOServiceMatching(kIOUSBHostDeviceClassName) as NSMutableDictionary? else { return nil }
      matching["idVendor"] = Int(Pinned.usbVendorID)
      matching["idProduct"] = Int(Pinned.usbProductID)
      let service = IOServiceGetMatchingService(kIOMainPortDefault, matching)
      return service == 0 ? nil : service
    }

    /// The vendor-class interface under `device`, else interface 0.
    private static func linkInterface(of device: io_service_t) -> io_service_t? {
      var iterator: io_iterator_t = 0
      guard IORegistryEntryGetChildIterator(device, kIOServicePlane, &iterator) == KERN_SUCCESS else { return nil }
      defer { IOObjectRelease(iterator) }
      let vendorClass = Pinned.usbVendorClass.map(Int.init)
      var vendor: io_service_t?
      var first: io_service_t?
      while case let child = IOIteratorNext(iterator), child != 0 {
        guard IOObjectConformsTo(child, kIOUSBHostInterfaceClassName) != 0 else {
          IOObjectRelease(child)
          continue
        }
        let triple = ["bInterfaceClass", "bInterfaceSubClass", "bInterfaceProtocol"].map { property(child, $0) }
        if vendor == nil && triple == vendorClass {
          vendor = child
        } else if first == nil && property(child, "bInterfaceNumber") == 0 {
          first = child
        } else {
          IOObjectRelease(child)
        }
      }
      if let vendor {
        if let first { IOObjectRelease(first) }
        return vendor
      }
      return first
    }

    static func property(_ entry: io_registry_entry_t, _ key: String) -> Int? {
      guard let value = IORegistryEntryCreateCFProperty(entry, key as CFString, kCFAllocatorDefault, 0)?.takeRetainedValue() else {
        return nil
      }
      return (value as? NSNumber)?.intValue
    }
  }

  /// An add-on's interface on the comma's gadget, found by its class triple.
  public struct USBAddonSource: AddonSource {
    let kind: [UInt8]

    public init(kind: [UInt8]) {
      self.kind = kind
    }

    public func present() -> Bool {
      USBGadget().present()
    }

    public func open() throws -> AddonPipes {
      guard let device = USBGadget.findDevice() else { throw LinkError.closed("no jetlink gadget on the bus") }
      defer { IOObjectRelease(device) }
      var iterator: io_iterator_t = 0
      guard IORegistryEntryGetChildIterator(device, kIOServicePlane, &iterator) == KERN_SUCCESS else {
        throw LinkError.closed("cannot list the gadget's interfaces")
      }
      defer { IOObjectRelease(iterator) }
      while case let child = IOIteratorNext(iterator), child != 0 {
        let triple = ["bInterfaceClass", "bInterfaceSubClass", "bInterfaceProtocol"].map { USBGadget.property(child, $0) }
        if IOObjectConformsTo(child, kIOUSBHostInterfaceClassName) != 0 && triple == kind.map(Int.init) {
          // the pipes own the reference from here
          return AddonPipes(try IOUSBHostPipes(service: child))
        }
        IOObjectRelease(child)
      }
      throw LinkError.closed("the gadget has no interface of class \(kind.map { String(format: "%02x", $0) }.joined(separator: "/"))")
    }
  }

  /// The link interface's bulk IN and OUT pipes, opened through IOUSBHost.
  ///
  /// Reads come from a `ReadRing` of 16 KB requests kept queued on the IN
  /// pipe, each landing in an NSMutableData of its own that the ring copies
  /// out of. A write is queued and waited for, because only a completion
  /// reports the bytes that went before a timeout, and those are part of the
  /// stream; it goes out of one NSMutableData the transport's buffer is copied
  /// into. Not NSMutableData(bytesNoCopy:) over the transport's buffers:
  /// mutable data copies the bytes into storage of its own, so a read landed
  /// in the copy and the transport read zeros ("bad magic 0x0"), which only a
  /// real gadget showed.
  final class IOUSBHostPipes: BulkPipes, ReadRingPipe, @unchecked Sendable {
    /// The interface's registry entry, held for as long as the interface is
    /// open rather than trusting IOUSBHost to hold its own.
    private let service: io_service_t
    private let interface: IOUSBHostInterface
    private let input: IOUSBHostPipe
    private let output: IOUSBHostPipe
    private let lock = NSLock()
    private var closing = false
    private var gone = false
    private var destroyed = false
    /// What the ring's reads land in, and their completion handlers, one per
    /// slot: made once, so a post allocates no closure.
    private var slots: [(data: NSMutableData, handler: IOUSBHostCompletionHandler)] = []
    /// Guards the ring, which completions reach on IOUSBHost's queue.
    private let ringLock = NSCondition()
    private var ring: ReadRing!
    /// What a write is sent from: writes are one at a time, under the
    /// transport's send lock.
    private let outData = NSMutableData()

    private static let queue = DispatchQueue(label: "io.zoompilot.jetlink.usb", qos: .userInteractive)
    /// kIOMessageServiceIsTerminated, a function-like macro Swift cannot see.
    private static let terminated: UInt32 = 0xE000_0010

    /// Takes ownership of `service`'s reference, also when it throws.
    init(service: io_service_t) throws {
      self.service = service
      let flag = TerminationFlag()
      do {
        interface = try IOUSBHostInterface(
          __ioService: service, options: [], queue: IOUSBHostPipes.queue,
          interestHandler: { _, message, _ in
            if message == IOUSBHostPipes.terminated { flag.set() }
          })
      } catch {
        IOObjectRelease(service)
        throw LinkError.closed("could not open the gadget's interface: \(IOUSBHostPipes.describe(error))")
      }
      var pair: (input: Int, output: Int) = (0, 0)
      let configuration = interface.configurationDescriptor
      let descriptor = interface.interfaceDescriptor
      var current = UnsafeRawPointer(descriptor).assumingMemoryBound(to: IOUSBDescriptorHeader.self)
      while let endpoint = IOUSBGetNextEndpointDescriptor(configuration, descriptor, current) {
        let address = Int(endpoint.pointee.bEndpointAddress)
        if endpoint.pointee.bmAttributes & 0x03 == 0x02 {
          if address & 0x80 != 0 { pair.input = address } else { pair.output = address }
        }
        current = UnsafeRawPointer(endpoint).assumingMemoryBound(to: IOUSBDescriptorHeader.self)
      }
      do {
        guard pair.input != 0, pair.output != 0 else { throw LinkError.closed("the gadget's interface has no bulk IN/OUT pair") }
        input = try interface.copyPipe(withAddress: pair.input)
        output = try interface.copyPipe(withAddress: pair.output)
      } catch {
        interface.destroy()
        IOObjectRelease(service)
        if let error = error as? LinkError { throw error }
        throw LinkError.closed("could not open the gadget's endpoints: \(IOUSBHostPipes.describe(error))")
      }
      ring = ReadRing(lock: ringLock, pipe: self)
      slots = (0..<ReadRing.depth).map { slot in
        let data = NSMutableData(length: ReadRing.slotSize)!
        // The request's buffer outlives it even if these pipes do not.
        return (
          data,
          { [weak self] status, transferred in
            withExtendedLifetime(data) {}
            self?.completed(slot, status, transferred)
          }
        )
      }
      flag.notify { [weak self] in self?.lost() }
    }

    deinit {
      close()
    }

    /// What the ring holds or has next, up to `count` bytes. A timeout leaves
    /// the ring queued, so what arrives later is kept for the next read.
    func read(into buffer: UnsafeMutableRawPointer, count: Int, timeout: TimeInterval) throws -> Int {
      let deadline = timeout > 0 ? MonotonicDeadline(in: timeout) : nil
      ringLock.lock()
      defer { ringLock.unlock() }
      try checkRunning()
      return try ring.read(into: buffer, count: count, deadline: deadline)
    }

    // MARK: ReadRingPipe, under ringLock

    /// The stop flags are checked where the reader waits, not per post.
    func post(_ slot: Int, size: Int) throws {
      do {
        try input.enqueueIORequest(with: slots[slot].data, completionTimeout: 0, completionHandler: slots[slot].handler)
      } catch {
        throw LinkError.closed("usb bulk read could not start: \(IOUSBHostPipes.describe(error))")
      }
    }

    func awaitCompletion(until deadline: MonotonicDeadline?) throws {
      try checkRunning()
      if let deadline {
        _ = ringLock.wait(until: deadline.wallClock())
      } else {
        ringLock.wait()
      }
    }

    func bytes(_ slot: Int) -> UnsafeRawPointer {
      slots[slot].data.bytes
    }

    /// On IOUSBHost's queue.
    private func completed(_ slot: Int, _ status: IOReturn, _ transferred: Int) {
      let completion: ReadRing.Completion
      switch UInt32(bitPattern: status) {
      case UInt32(bitPattern: kIOReturnSuccess), IOUSBHostPipes.underrun:
        // A short packet ends a request; it is not an error.
        completion = .data
      case IOUSBHostPipes.aborted:
        completion = .failed(.closed(stopReason ?? "usb bulk read aborted"))
      default:
        completion = .failed(failure(status, "read"))
      }
      ringLock.lock()
      ring.complete(slot, completion, count: transferred)
      ringLock.broadcast()
      ringLock.unlock()
    }

    /// Wakes a reader waiting on the ring, after a flag it checks has changed.
    private func wakeReader() {
      ringLock.lock()
      ringLock.broadcast()
      ringLock.unlock()
    }

    func write(from buffer: UnsafeRawPointer, count: Int, timeout: TimeInterval) throws -> Int {
      outData.length = 0  // keeps the capacity, and the append does not zero what it copies over
      outData.append(buffer, length: count)
      return try transfer(output, outData, timeout, "write")
    }

    private func transfer(_ pipe: IOUSBHostPipe, _ data: NSMutableData, _ timeout: TimeInterval, _ what: String) throws -> Int {
      try checkRunning()
      let completion = Completion()
      do {
        try pipe.enqueueIORequest(with: data, completionTimeout: timeout) { status, transferred in
          completion.finish(status, transferred)
        }
      } catch {
        throw LinkError.closed("usb bulk \(what) could not start: \(IOUSBHostPipes.describe(error))")
      }
      let (status, transferred) = completion.wait()
      switch UInt32(bitPattern: status) {
      case UInt32(bitPattern: kIOReturnSuccess), IOUSBHostPipes.underrun, IOUSBHostPipes.timeout, IOUSBHostPipes.transactionTimeout:
        // A short packet ends a transfer; whatever arrived before a deadline
        // is part of the stream.
        return transferred
      default:
        throw failure(status, what)
      }
    }

    /// Why transfers stop now, if they do: the link was closed from this
    /// end, or the gadget went away.
    private var stopReason: String? {
      lock.withLock { gone ? "the gadget went away" : closing ? "link closed" : nil }
    }

    private func checkRunning() throws {
      if let stopReason { throw LinkError.closed(stopReason) }
    }

    private func failure(_ status: IOReturn, _ what: String) -> LinkError {
      if UInt32(bitPattern: status) == IOUSBHostPipes.aborted {
        return .closed(stopReason ?? "usb bulk \(what) aborted")
      }
      return .closed("usb bulk \(what) failed: \(IOUSBHostPipes.describe(status))")
    }

    private func abortPipes(_ option: IOUSBHostAbortOption) {
      try? input.__abort(with: option)
      try? output.__abort(with: option)
    }

    func abort() {
      lock.withLock { closing = true }
      abortPipes(.asynchronous)
      wakeReader()
    }

    func close() {
      let first = lock.withLock {
        defer { destroyed = true; closing = true }
        return !destroyed
      }
      guard first else { return }
      abortPipes(.synchronous)
      wakeReader()
      interface.destroy()
      IOObjectRelease(service)
    }

    /// The device was unplugged or re-enumerated: whatever is in flight ends.
    private func lost() {
      lock.withLock { gone = true }
      abortPipes(.asynchronous)
      wakeReader()
    }

    // MARK: status codes

    private static let timeout: UInt32 = 0xE000_02D6  // kIOReturnTimeout
    private static let underrun: UInt32 = 0xE000_02E7  // kIOReturnUnderrun: a short packet, not an error
    private static let aborted: UInt32 = 0xE000_02EB  // kIOReturnAborted
    private static let transactionTimeout: UInt32 = 0xE000_4051  // kIOUSBTransactionTimeout

    static func describe(_ status: IOReturn) -> String {
      let names: [UInt32: String] = [
        0xE000_02C0: "no such device", 0xE000_02C5: "another program has the interface open",
        0xE000_02CA: "I/O error", 0xE000_02CD: "not open", 0xE000_02D7: "offline",
        0xE000_02D9: "not attached", 0xE000_02E8: "overrun", 0xE000_02ED: "not responding",
        0xE000_5000: "the endpoint stalled", timeout: "timed out", aborted: "aborted",
      ]
      let code = UInt32(bitPattern: status)
      let hex = String(format: "0x%08x", code)
      return names[code].map { "\($0) (\(hex))" } ?? hex
    }

    static func describe(_ error: any Error) -> String {
      let ns = error as NSError
      if ns.domain == IOUSBHostErrorDomain || ns.domain == NSMachErrorDomain || ns.domain == NSOSStatusErrorDomain {
        return describe(IOReturn(truncatingIfNeeded: ns.code))
      }
      return ns.localizedDescription
    }
  }

  /// One queued transfer's result, handed from IOUSBHost's queue to the
  /// waiting thread.
  private final class Completion: @unchecked Sendable {
    private let done = DispatchSemaphore(value: 0)
    private var status: IOReturn = 0
    private var transferred = 0

    func finish(_ status: IOReturn, _ transferred: Int) {
      self.status = status
      self.transferred = transferred
      done.signal()
    }

    func wait() -> (IOReturn, Int) {
      done.wait()
      return (status, transferred)
    }
  }

  /// Set from the interface's interest handler, which IOUSBHost wants before
  /// the pipes exist; `notify` is wired once they do, and fires at once if
  /// the gadget already went.
  private final class TerminationFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var isSet = false
    private var handler: (() -> Void)?

    func notify(_ handler: @escaping () -> Void) {
      let fire = lock.withLock {
        self.handler = handler
        return isSet
      }
      if fire { handler() }
    }

    func set() {
      let handler = lock.withLock {
        isSet = true
        return self.handler
      }
      handler?()
    }
  }
#endif
