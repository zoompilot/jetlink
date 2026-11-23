import Foundation
import JetlinkKit

#if canImport(CUsbfs)
  import CUsbfs
#endif

/// The comma's bulk pair through usbdevfs, Linux's user-space USB, on a file
/// descriptor someone else opened: the Linux server's, or on Android the
/// app's, through UsbDeviceConnection. The Android and Linux counterpart of
/// `IOUSBHostPipes`.
///
/// Reads come from a `ReadRing` of 16 KB URBs kept posted on the IN endpoint;
/// a write is one URB, sent and reaped before it returns. The URBs and their
/// memory are made once per descriptor and handed from one session's pipes to
/// the next (`UsbfsDevice.take`), so neither the steady state nor a reopen
/// allocates.
///
/// usbfs has one completion queue per file descriptor, and the kernel writes
/// a URB's status, length and any IN data only when the URB is reaped. So one
/// `UsbfsDevice` per descriptor owns the reaping: whichever transfer is
/// waiting polls the descriptor and reaps for everyone. The pipes free their
/// URBs and buffers only once the kernel is done with them (`close`), so a
/// later reap never writes into freed memory.
///
/// The kernel side is `UsbfsKernel`, so the tests run this against a fake on
/// any platform; `LinuxUsbfs` is the real one.
final class UsbfsPipes: BulkPipes, ReadRingPipe, @unchecked Sendable {
  let device: UsbfsDevice
  private let urbs: UsbfsURBs
  private var slots: [UsbfsURB] { urbs.slots }
  /// Writes go out one at a time, under the transport's send lock.
  private var outURB: UsbfsURB { urbs.out }
  private var ring: ReadRing!
  /// Under the device's lock.
  private var closed = false
  private var released = false

  /// `shortPackets`: an add-on's interface, whose messages end on a short packet (`ReadRing`).
  init(device: UsbfsDevice, inEndpoint: UInt8, outEndpoint: UInt8, shortPackets: Bool = false) {
    self.device = device
    urbs = device.take(inEndpoint: inEndpoint, outEndpoint: outEndpoint)
    ring = ReadRing(lock: device.condition, pipe: self, shortPackets: shortPackets)
    let owner = ObjectIdentifier(self)
    for urb in slots {
      urb.owner = owner
      urb.ring = ring
    }
    outURB.owner = owner
    device.register(urbs.all)
  }

  private var owner: ObjectIdentifier { ObjectIdentifier(self) }

  /// What the ring holds or has next, up to `count` bytes. A timeout leaves
  /// the ring posted, so what arrives later is kept for the next read.
  func read(into buffer: UnsafeMutableRawPointer, count: Int, timeout: TimeInterval) throws -> Int {
    let deadline = timeout > 0 ? MonotonicDeadline(in: timeout) : nil
    return try device.use(owner: owner) {
      try ring.read(into: buffer, count: count, deadline: deadline)
    }
  }

  func write(from buffer: UnsafeRawPointer, count: Int, timeout: TimeInterval) throws -> Int {
    try device.transfer(outURB, buffer: UnsafeMutableRawPointer(mutating: buffer), count: count, timeout: timeout > 0 ? timeout : nil)
  }

  func abort() {
    device.cancel(owner: owner)
  }

  /// Ends every URB in flight and waits for the kernel to give each back.
  /// The descriptor is the host's to close.
  func close() {
    let first = device.condition.withLock {
      defer { closed = true }
      return !closed
    }
    guard first else { return }
    released = device.retire(urbs.all, owner: owner, timeout: 2)
    if released {
      device.keep(urbs)
    }
  }

  deinit {
    close()
    device.forget(owner: owner)
    if !released {
      // The kernel may still write these at a reap: keep them for good
      // rather than hand their memory to something else.
      ServerLog(category: "usb").warning("the kernel kept this link's USB reads past its close; keeping their \(slots.count * ReadRing.slotSize >> 10) KB")
      _ = Unmanaged.passRetained(urbs)
    }
  }

  // MARK: ReadRingPipe, under the device's lock

  func post(_ slot: Int, size: Int) throws {
    try device.submit(slots[slot], count: size)
  }

  func awaitCompletion(until deadline: MonotonicDeadline?) throws {
    try device.awaitCompletion(owner: owner, until: deadline)
  }

  func bytes(_ slot: Int) -> UnsafeRawPointer {
    UnsafeRawPointer(slots[slot].buffer)
  }
}

/// A ring's URBs over 16 KB each of `memory`, and the write URB. Freed only
/// once the kernel has given every one back.
final class UsbfsURBs: @unchecked Sendable {
  let memory: UnsafeMutableRawPointer
  let slots: [UsbfsURB]
  let out: UsbfsURB

  init(inEndpoint: UInt8, outEndpoint: UInt8) {
    let size = ReadRing.slotSize
    memory = .allocate(byteCount: ReadRing.depth * size, alignment: 4096)
    slots = (0..<ReadRing.depth).map { [memory] in UsbfsURB(endpoint: inEndpoint, buffer: memory + $0 * size, slot: $0) }
    out = UsbfsURB(endpoint: outEndpoint, buffer: memory, slot: -1)
  }

  deinit {
    memory.deallocate()
  }

  var all: [UsbfsURB] { slots + [out] }

  func matches(inEndpoint: UInt8, outEndpoint: UInt8) -> Bool {
    slots[0].endpoint == inEndpoint && out.endpoint == outEndpoint
  }
}

/// One bulk URB, made once per descriptor and submitted over and over. On
/// Linux the kernel's `usbdevfs_urb` (`handle`) lives as long as this does,
/// and its user context points back here unretained: the pipes keep it alive.
final class UsbfsURB: @unchecked Sendable {
  let endpoint: UInt8
  var buffer: UnsafeMutableRawPointer
  var count = 0
  var owner = ObjectIdentifier(UsbfsURB.self)
  /// Its slot in the owner's ring, which hears of its completions; -1 for
  /// the write URB.
  let slot: Int
  weak var ring: ReadRing?
  /// Submitted and not reaped yet.
  var inFlight = false
  /// Ended early from this end since its last submit.
  var discarded = false
  /// Once reaped: 0, or the kernel's negative errno.
  var status: Int32 = 0
  var actual = 0
  /// The real kernel's URB, made at the first submit.
  var handle: OpaquePointer?

  init(endpoint: UInt8, buffer: UnsafeMutableRawPointer, slot: Int) {
    self.endpoint = endpoint
    self.buffer = buffer
    self.slot = slot
  }

  deinit {
    #if canImport(CUsbfs)
      if let handle { jl_urb_free(handle) }
    #endif
  }
}

/// usbfs as `UsbfsDevice` drives it. Calls come under the device's lock,
/// except `wait`, which one thread at a time makes without it, and `wake`.
protocol UsbfsKernel: AnyObject, Sendable {
  /// Queues `urb` for its `count` bytes: 0, or an errno.
  func submit(_ urb: UsbfsURB) -> Int32
  /// Ends `urb` early; it still has to be reaped.
  func discard(_ urb: UsbfsURB)
  /// A finished URB with its status and length filled in, or nil when none
  /// is ready. Throws once the device is gone and nothing is left to reap.
  func reap() throws -> UsbfsURB?
  /// Waits up to `timeout` (nil: forever) for a URB to reap or a `wake()`.
  /// False once the device is gone.
  func wait(timeout: TimeInterval?) -> Bool
  func wake()
}

/// Linux's errno values, which a URB's status carries whatever the platform
/// the code was built for: the fake kernel in the tests runs on a Mac.
enum LinuxErrno {
  static let noent: Int32 = 2
  static let again: Int32 = 11
  static let nomem: Int32 = 12
  static let nodev: Int32 = 19
  static let pipe: Int32 = 32
  static let proto: Int32 = 71
  static let overflow: Int32 = 75
  static let connreset: Int32 = 104
  static let shutdown: Int32 = 108

  static func describe(_ code: Int32) -> String {
    switch abs(code) {
    case noent, connreset: "cancelled"
    case nodev, shutdown: "the device is gone"
    case pipe: "the endpoint stalled"
    case proto: "a protocol error on the bus"
    case overflow: "the device sent more than was asked for"
    case nomem: "out of memory"
    default: "errno \(abs(code))"
    }
  }
}

/// One file descriptor's transfers, however many sessions' pipes use it.
final class UsbfsDevice: @unchecked Sendable {
  let kernel: any UsbfsKernel
  /// Guards all of this, and the rings of the pipes on this descriptor.
  let condition = NSCondition()
  /// A thread is in `kernel.wait` and reaps for everyone when it returns.
  private var polling = false
  private var gone = false
  /// Every URB of every open pipes, added and removed with the pipes.
  private var registered: [UsbfsURB] = []
  private var cancelled = Set<ObjectIdentifier>()
  /// Calls between their first and last touch of the kernel.
  private var active = 0
  /// The last pipes' URBs, which the kernel gave back, for the next pipes.
  private var spares: [UsbfsURBs] = []
  /// How long a discarded URB may take to come back before the wait for it
  /// polls again.
  static let discardPoll: TimeInterval = 0.05
  static let goneMessage = "the comma's USB device is gone"
  static let interruptedMessage = "the USB link was interrupted"

  init(kernel: any UsbfsKernel) {
    self.kernel = kernel
  }

  var isGone: Bool {
    condition.withLock { gone }
  }

  /// URBs for new pipes on `inEndpoint` and `outEndpoint`: the last pipes'
  /// on those endpoints, else new ones.
  func take(inEndpoint: UInt8, outEndpoint: UInt8) -> UsbfsURBs {
    condition.withLock {
      guard let i = spares.firstIndex(where: { $0.matches(inEndpoint: inEndpoint, outEndpoint: outEndpoint) }) else {
        return UsbfsURBs(inEndpoint: inEndpoint, outEndpoint: outEndpoint)
      }
      return spares.remove(at: i)
    }
  }

  /// Keeps closed pipes' URBs, every one back from the kernel, for the next
  /// on the same endpoints: one set per pair, as the driving link and an
  /// add-on's interface can share a descriptor (Android) and close in turn.
  func keep(_ urbs: UsbfsURBs) {
    condition.withLock {
      spares.removeAll { $0.matches(inEndpoint: urbs.slots[0].endpoint, outEndpoint: urbs.out.endpoint) }
      spares.append(urbs)
    }
  }

  func register(_ urbs: [UsbfsURB]) {
    condition.withLock { registered += urbs }
  }

  /// Runs `body` under the lock as a call that may touch the kernel, once
  /// `owner`'s pipes may still move data.
  func use<T>(owner: ObjectIdentifier, _ body: () throws -> T) throws -> T {
    condition.lock()
    defer { condition.unlock() }
    try check(owner)
    active += 1
    defer {
      active -= 1
      condition.broadcast()
    }
    return try body()
  }

  private func check(_ owner: ObjectIdentifier) throws {
    if gone { throw LinkError.closed(UsbfsDevice.goneMessage) }
    if cancelled.contains(owner) { throw LinkError.closed(UsbfsDevice.interruptedMessage) }
  }

  /// Runs one transfer on `urb` to its end and returns the bytes moved: all
  /// of them, or what arrived before `timeout` (nil: none) ended it, which is
  /// not an error. Throws when `cancel` ended it, the device went away, or
  /// the bus failed it.
  func transfer(_ urb: UsbfsURB, buffer: UnsafeMutableRawPointer, count: Int, timeout: TimeInterval?) throws -> Int {
    try use(owner: urb.owner) {
      urb.buffer = buffer
      try submit(urb, count: count)
      let deadline = timeout.map { MonotonicDeadline(in: $0) }
      while urb.inFlight {
        if gone {
          // Never reaped now: its pipes free it once no thread polls.
          throw LinkError.closed(UsbfsDevice.goneMessage)
        }
        if !urb.discarded && (cancelled.contains(urb.owner) || deadline?.passed() == true) {
          discard(urb)
        }
        waitOnce(until: urb.discarded ? nil : deadline, cap: urb.discarded ? UsbfsDevice.discardPoll : nil)
      }
      if urb.status == 0 {
        return urb.actual
      }
      if urb.status == -LinuxErrno.noent || urb.status == -LinuxErrno.connreset {
        if cancelled.contains(urb.owner) {
          throw LinkError.closed(UsbfsDevice.interruptedMessage)
        }
        if gone {
          throw LinkError.closed(UsbfsDevice.goneMessage)
        }
        // The deadline passed: what arrived is the caller's, as the pipes promise.
        return urb.actual
      }
      throw UsbfsDevice.failure(urb)
    }
  }

  private static func failure(_ urb: UsbfsURB) -> LinkError {
    .closed("USB transfer on endpoint \(String(format: "%02x", urb.endpoint)) failed: \(LinuxErrno.describe(urb.status))")
  }

  /// Under the lock: queues `urb` for `count` bytes.
  func submit(_ urb: UsbfsURB, count: Int) throws {
    if gone { throw LinkError.closed(UsbfsDevice.goneMessage) }
    urb.count = count
    urb.status = 0
    urb.actual = 0
    urb.discarded = false
    let error = kernel.submit(urb)
    if error != 0 {
      throw LinkError.closed("could not queue a USB transfer: \(LinuxErrno.describe(error))")
    }
    urb.inFlight = true
  }

  /// Under the lock: ends those of `urbs` still in flight.
  private func discard(_ urbs: [UsbfsURB]) {
    for urb in urbs where urb.inFlight {
      discard(urb)
    }
  }

  private func discard(_ urb: UsbfsURB) {
    guard !gone, !urb.discarded else { return }
    urb.discarded = true
    kernel.discard(urb)
  }

  /// Under the lock: waits once for a completion, until `deadline` (nil: none).
  func awaitCompletion(owner: ObjectIdentifier, until deadline: MonotonicDeadline?) throws {
    try check(owner)
    waitOnce(until: deadline, cap: nil)
  }

  /// Under the lock. Polls the descriptor and reaps for everyone if no other
  /// thread is, else waits for that one's broadcast. Never polls once the
  /// device is gone: the host may already have closed the descriptor. May
  /// return early; every caller checks its deadline again.
  private func waitOnce(until deadline: MonotonicDeadline?, cap: TimeInterval?) {
    var until = deadline
    if let cap {
      let capped = MonotonicDeadline(in: cap)
      if until.map({ capped < $0 }) ?? true {
        until = capped
      }
    }
    if polling || gone {
      if let until {
        _ = condition.wait(until: until.wallClock())
      } else {
        condition.wait()
      }
      return
    }
    polling = true
    condition.unlock()
    let alive = kernel.wait(timeout: until.map { $0.remaining() })
    condition.lock()
    polling = false
    if !gone {
      reapAll()
    }
    if !alive {
      gone = true
    }
    condition.broadcast()
  }

  /// Under the lock: every finished URB marked, and its ring told.
  private func reapAll() {
    while true {
      let urb: UsbfsURB?
      do {
        urb = try kernel.reap()
      } catch {
        gone = true
        return
      }
      guard let urb else { return }
      urb.inFlight = false
      urb.ring?.complete(urb.slot, completion(of: urb), count: urb.actual)
    }
  }

  private func completion(of urb: UsbfsURB) -> ReadRing.Completion {
    if urb.status == 0 {
      return .data
    }
    if urb.discarded && (urb.status == -LinuxErrno.noent || urb.status == -LinuxErrno.connreset) {
      return .failed(.closed(gone ? UsbfsDevice.goneMessage : UsbfsDevice.interruptedMessage))
    }
    return .failed(UsbfsDevice.failure(urb))
  }

  /// Ends `owner`'s transfers, and any it starts later, with an error.
  func cancel(owner: ObjectIdentifier) {
    condition.withLock {
      cancelled.insert(owner)
      for urb in registered where urb.owner == owner && urb.inFlight {
        discard(urb)
      }
    }
    kernel.wake()
  }

  /// Forgets a closed owner, so a later one at the same address starts clean.
  func forget(owner: ObjectIdentifier) {
    condition.withLock { _ = cancelled.remove(owner) }
  }

  /// Cancels `owner` and waits up to `timeout` for the kernel to let go of
  /// `urbs`: until none is in flight, or the device is gone and no thread is
  /// left that could reap one. True when it has, and the owner may free them
  /// and their buffers.
  func retire(_ urbs: [UsbfsURB], owner: ObjectIdentifier, timeout: TimeInterval) -> Bool {
    condition.lock()
    defer { condition.unlock() }
    cancelled.insert(owner)
    active += 1
    defer {
      active -= 1
      condition.broadcast()
    }
    discard(urbs)
    let deadline = MonotonicDeadline(in: timeout)
    while urbs.contains(where: { $0.inFlight }) {
      if gone && !polling {
        break
      }
      if deadline.passed() {
        return false
      }
      waitOnce(until: deadline, cap: UsbfsDevice.discardPoll)
    }
    registered.removeAll { $0.owner == owner }
    return true
  }

  /// The device is going: every transfer ends with an error, and this waits
  /// up to `timeout` for them to leave the kernel alone, so the host can close
  /// the descriptor after. True when they all did.
  @discardableResult
  func invalidate(timeout: TimeInterval = 1.0) -> Bool {
    condition.lock()
    defer { condition.unlock() }
    discard(registered)
    gone = true
    kernel.wake()
    condition.broadcast()
    let deadline = MonotonicDeadline(in: timeout)
    while active > 0 {
      if deadline.passed() {
        return false
      }
      _ = condition.wait(until: deadline.wallClock())
    }
    return true
  }
}

#if canImport(CUsbfs)
  /// usbfs on a real file descriptor, which it does not own or close.
  final class LinuxUsbfs: UsbfsKernel, @unchecked Sendable {
    let fd: Int32
    private let wakeFD: Int32

    init(fd: Int32) throws {
      self.fd = fd
      wakeFD = jl_usbfs_wake_create()
      guard wakeFD >= 0 else { throw LinkError.closed("could not create the USB wake event") }
    }

    deinit {
      jl_usbfs_wake_close(wakeFD)
    }

    /// The bus speed `USBDEVFS_GET_SPEED` reports, by the kernel's names
    /// for it (usb_speed_string), as the comma's hello names its own.
    var medium: LinkMedium? {
      let names = [1: "low-speed", 2: "full-speed", 3: "high-speed", 4: "wireless", 5: "super-speed", 6: "super-speed-plus"]
      return names[Int(jl_usbfs_speed(fd))].flatMap { Pinned.usbSpeedMedia[$0] }.flatMap(LinkMedium.init(rawValue:))
    }

    func submit(_ urb: UsbfsURB) -> Int32 {
      if let handle = urb.handle {
        jl_urb_set(handle, urb.buffer, Int32(urb.count))
      } else {
        guard let handle = jl_urb_create(urb.endpoint, urb.buffer, Int32(urb.count), Unmanaged.passUnretained(urb).toOpaque()) else {
          return LinuxErrno.nomem
        }
        urb.handle = handle
      }
      return jl_usbfs_submit(fd, urb.handle)
    }

    func discard(_ urb: UsbfsURB) {
      if let handle = urb.handle {
        _ = jl_usbfs_discard(fd, handle)
      }
    }

    func reap() throws -> UsbfsURB? {
      var done: OpaquePointer?
      let error = jl_usbfs_reap(fd, &done)
      if error == LinuxErrno.again { return nil }
      guard error == 0, let done, let context = jl_urb_context(done) else {
        throw LinkError.closed("USB reap failed: \(LinuxErrno.describe(error))")
      }
      let urb = Unmanaged<UsbfsURB>.fromOpaque(context).takeUnretainedValue()
      urb.status = jl_urb_status(done)
      urb.actual = Int(jl_urb_actual(done))
      return urb
    }

    func wait(timeout: TimeInterval?) -> Bool {
      let ms = timeout.map { Int32(min(($0 * 1000).rounded(.up), Double(Int32.max))) } ?? -1
      return jl_usbfs_wait(fd, wakeFD, ms) != LinuxErrno.nodev
    }

    func wake() {
      jl_usbfs_wake(wakeFD)
    }
  }
#endif
