import Foundation
import JetlinkKit
import JetlinkTestSupport
import Testing

@testable import JetlinkServer

/// usbfs in memory, as the kernel and the bus behave for one file descriptor.
/// What the comma writes arrives in 1024-byte packets. An IN URB fills with
/// them and completes when it is full or when a short packet (or a
/// zero-length one) ends one of the comma's writes; until then it waits,
/// holding what it has. OUT URBs complete at once, with at most `writeLimit`
/// bytes taken. A discarded URB comes back with -ENOENT and whatever it got.
/// Every finished URB waits in one completion queue until it is reaped.
final class FakeUsbfs: UsbfsKernel, @unchecked Sendable {
  static let packet = Pinned.usbMaxPacket
  private let condition = NSCondition()
  private var inbound: [UInt8] = []
  private var cursor = 0
  /// Offsets into `inbound` where a short packet ended a write.
  private var shortEnds: [Int] = []
  private var pending: [UsbfsURB] = []
  private var completed: [UsbfsURB] = []
  private var woken = false
  private(set) var alive = true
  private var outbound = Data()
  private(set) var discards = 0
  private(set) var submits = 0
  /// Every URB ever submitted, by identity.
  private(set) var urbs = Set<ObjectIdentifier>()
  /// An errno every submit fails with, when set.
  var failSubmit: Int32 = 0
  /// Most IN URBs in flight; a submit past it fails with ENOMEM, as usbfs's
  /// memory limit makes it.
  var maxPending = Int.max
  /// The status the next IN URB completes with, once, when set.
  var failNext: Int32 = 0
  /// Reaps hand finished URBs back in any order.
  var shuffleReaps = false
  /// The most an OUT URB takes; 0 is a comma that stopped reading.
  var writeLimit = Int.max
  private(set) var writes = 0

  /// One message as the gadget frames it: header, payload, and zeros to the
  /// next 16 KB, never the PADDED flag (`FfsTransport`, `tx_align`).
  static func gadgetFrame(_ type: Wire.Msg, seq: UInt32, payload: Data = Data(), flags: Wire.Flag = []) -> [UInt8] {
    var bytes = [UInt8](repeating: 0, count: Wire.headerSize)
    bytes.withUnsafeMutableBytes {
      Wire.packHeader(Wire.Header(msgType: type.rawValue, seq: seq, flags: flags.rawValue, length: UInt32(payload.count)), into: $0.baseAddress!)
    }
    bytes += payload
    bytes += [UInt8](repeating: 0, count: USBTransport.gadgetPad(bytes.count))
    return bytes
  }

  /// A gadget on the bus that nothing on the comma serves yet: every read
  /// fails at once, as the endpoints do before a run borrows the link.
  static func unserved() -> FakeUsbfs {
    let kernel = FakeUsbfs()
    kernel.unplug()
    return kernel
  }

  /// The comma writes `bytes`: whole packets, then a short one when the
  /// length is not a multiple of a packet, or a zero-length one when empty.
  func feed(_ bytes: [UInt8]) {
    condition.withLock {
      inbound += bytes
      if bytes.count % FakeUsbfs.packet != 0 || bytes.isEmpty {
        shortEnds.append(inbound.count)
      }
      serve()
      condition.broadcast()
    }
  }

  /// The cable comes out: pending URBs die with -ESHUTDOWN, and the
  /// descriptor reports the device gone once they are reaped.
  func unplug() {
    condition.withLock {
      alive = false
      for urb in pending {
        urb.status = -LinuxErrno.shutdown
        completed.append(urb)
      }
      pending.removeAll()
      condition.broadcast()
    }
  }

  var pendingCount: Int { condition.withLock { pending.count } }
  /// The byte counts of the IN URBs in flight, oldest first.
  var pendingSizes: [Int] { condition.withLock { pending.filter { $0.endpoint & 0x80 != 0 }.map(\.count) } }
  /// What the comma sent that no URB has taken yet.
  var buffered: Int { condition.withLock { inbound.count - cursor } }
  var written: Data { condition.withLock { outbound } }

  /// Waits until `written` holds at least `count` bytes.
  func waitForWritten(_ count: Int, timeout: TimeInterval = 30) -> Data? {
    condition.lock()
    defer { condition.unlock() }
    let deadline = Date().addingTimeInterval(timeout)
    while outbound.count < count {
      if !condition.wait(until: deadline) { return nil }
    }
    return outbound
  }

  private func finish(_ urb: UsbfsURB) {
    if failNext != 0 {
      urb.status = failNext
      failNext = 0
    }
    completed.append(urb)
  }

  private func serve() {
    while let index = pending.firstIndex(where: { $0.endpoint & 0x80 != 0 }) {
      let urb = pending[index]
      var n = min(urb.count - urb.actual, inbound.count - cursor)
      var ends = false
      if let short = shortEnds.first, short <= cursor + n {
        n = short - cursor
        ends = true
        shortEnds.removeFirst()
      }
      if n > 0 {
        inbound.withUnsafeBytes { (urb.buffer + urb.actual).copyMemory(from: $0.baseAddress! + cursor, byteCount: n) }
        urb.actual += n
        cursor += n
      }
      guard urb.actual == urb.count || ends else { break }
      finish(pending.remove(at: index))
    }
    if cursor > 1 << 20 {
      inbound.removeFirst(cursor)
      shortEnds = shortEnds.map { $0 - cursor }
      cursor = 0
    }
  }

  func submit(_ urb: UsbfsURB) -> Int32 {
    condition.lock()
    defer { condition.unlock() }
    if failSubmit != 0 { return failSubmit }
    if !alive { return LinuxErrno.nodev }
    submits += 1
    urbs.insert(ObjectIdentifier(urb))
    if urb.endpoint & 0x80 == 0 {
      writes += 1
      urb.actual = min(urb.count, writeLimit)
      outbound.append(urb.buffer.assumingMemoryBound(to: UInt8.self), count: urb.actual)
      completed.append(urb)
    } else {
      if pending.count >= maxPending { return LinuxErrno.nomem }
      urb.actual = 0
      pending.append(urb)
      serve()
    }
    condition.broadcast()
    return 0
  }

  func discard(_ urb: UsbfsURB) {
    condition.lock()
    defer { condition.unlock() }
    guard let index = pending.firstIndex(where: { $0 === urb }) else { return }
    discards += 1
    pending.remove(at: index)
    urb.status = -LinuxErrno.noent
    completed.append(urb)
    condition.broadcast()
  }

  func reap() throws -> UsbfsURB? {
    condition.lock()
    defer { condition.unlock() }
    if !completed.isEmpty {
      return completed.remove(at: shuffleReaps ? Int.random(in: 0..<completed.count) : 0)
    }
    if !alive { throw LinkError.closed("the device is gone") }
    return nil
  }

  func wait(timeout: TimeInterval?) -> Bool {
    condition.lock()
    defer { condition.unlock() }
    let deadline = timeout.map { Date().addingTimeInterval($0) }
    while completed.isEmpty && alive && !woken {
      if let deadline {
        if !condition.wait(until: deadline) { break }
      } else {
        condition.wait()
      }
    }
    woken = false
    return alive || !completed.isEmpty
  }

  func wake() {
    condition.withLock {
      woken = true
      condition.broadcast()
    }
  }
}

/// Runs `body` on a thread of its own; `join` waits for its result.
final class Background<T>: @unchecked Sendable {
  private let done = DispatchSemaphore(value: 0)
  private var result: Result<T, any Error>?

  init(_ body: @escaping @Sendable () throws -> T) {
    let thread = Thread { [self] in
      result = Result { try body() }
      done.signal()
    }
    thread.start()
  }

  func join(timeout: TimeInterval = 10) -> Result<T, any Error>? {
    guard done.wait(timeout: .now() + timeout) == .success else { return nil }
    return result
  }
}

/// `count` bytes that differ from their neighbours, starting at `seed`.
func pattern(_ count: Int, seed: Int = 0) -> [UInt8] {
  (0..<count).map { UInt8(truncatingIfNeeded: ($0 + seed) &* 131 &+ ($0 + seed) >> 8) }
}

/// A scratch buffer for a read.
final class Scratch {
  let pointer: UnsafeMutableRawPointer
  let count: Int

  init(_ count: Int) {
    self.count = count
    pointer = .allocate(byteCount: count, alignment: 64)
  }

  deinit {
    pointer.deallocate()
  }

  func bytes(_ n: Int) -> [UInt8] {
    Array(UnsafeRawBufferPointer(start: pointer, count: n))
  }
}

@Suite("usbfs pipes")
struct UsbfsPipesTests {
  let kernel = FakeUsbfs()
  var device: UsbfsDevice { UsbfsDevice(kernel: kernel) }
  let slot = ReadRing.slotSize

  func pipes(_ device: UsbfsDevice) -> UsbfsPipes {
    UsbfsPipes(device: device, inEndpoint: 0x81, outEndpoint: 0x01)
  }

  /// Reads until `count` bytes arrived.
  func readAll(_ pipes: UsbfsPipes, _ count: Int) throws -> [UInt8] {
    let scratch = Scratch(count)
    var got = 0
    while got < count {
      got += try pipes.read(into: scratch.pointer + got, count: count - got, timeout: 5)
    }
    return scratch.bytes(count)
  }

  @Test("A read returns what arrived, at most what it asked for, and a write sends it all in one URB")
  func readAndWrite() throws {
    let pipes = pipes(device)
    let sent = pattern(slot)
    kernel.feed(sent)
    let scratch = Scratch(slot)
    #expect(try pipes.read(into: scratch.pointer, count: 4096, timeout: 0) == 4096)
    #expect(try pipes.read(into: scratch.pointer + 4096, count: slot, timeout: 0) == slot - 4096)
    #expect(scratch.bytes(slot) == sent)
    let bytes: [UInt8] = [1, 2, 3, 4, 5]
    #expect(try bytes.withUnsafeBytes { try pipes.write(from: $0.baseAddress!, count: 5, timeout: 2) } == 5)
    #expect(kernel.written == Data(bytes))
  }

  @Test("The ring posts its reads before the comma sends, so a whole request lands with no read asked for in between")
  func ringIsPostedAhead() throws {
    let pipes = pipes(device)
    let scratch = Scratch(64)
    #expect(try pipes.read(into: scratch.pointer, count: 64, timeout: 0.01) == 0)
    #expect(kernel.pendingSizes == Array(repeating: slot, count: ReadRing.depth))
    // A 766 MB model's request, as the comma pads it: 29 slots.
    let request = pattern(475_136)
    kernel.feed(request)
    #expect(kernel.buffered == 0, "the comma waited for a read")
    #expect(try readAll(pipes, request.count) == request)
    #expect(kernel.discards == 0)
  }

  @Test("Bytes come out in the order the reads were posted, however the reaps are ordered, around the ring many times")
  func outOfOrderReaps() throws {
    kernel.shuffleReaps = true
    let pipes = pipes(device)
    var expected: [UInt8] = []
    var got: [UInt8] = []
    for i in 0..<40 {
      let chunk = pattern(slot * (1 + i % 7), seed: i)
      expected += chunk
      let reader = Background { [self] in try readAll(pipes, chunk.count) }
      kernel.feed(chunk)
      got += try #require(reader.join()).get()
    }
    #expect(got == expected)
    #expect(kernel.discards == 0)
  }

  @Test("A short packet takes the stream off the grid: the reads before it are delivered, then the link fails")
  func shortPacketFails() throws {
    let pipes = pipes(device)
    let first = pattern(slot, seed: 1)
    kernel.feed(first)
    kernel.feed(pattern(5000, seed: 2))
    #expect(try readAll(pipes, slot) == first)
    let scratch = Scratch(slot)
    #expect(throws: LinkError.self) { try pipes.read(into: scratch.pointer, count: slot, timeout: 0) }
  }

  @Test("An add-on's pipes take messages that end on a short packet, each whole")
  func addonShortPackets() throws {
    let pipes = UsbfsPipes(device: device, inEndpoint: 0x81, outEndpoint: 0x01, shortPackets: true)
    let first = pattern(24 + slot + 5000, seed: 1)
    let second = pattern(24 + 700, seed: 2)
    kernel.feed(first)
    kernel.feed(second)
    #expect(try readAll(pipes, first.count) == first)
    #expect(try readAll(pipes, second.count) == second)
  }

  @Test("A zero-length packet keeps the ring on the grid")
  func zeroLengthPacket() throws {
    let pipes = pipes(device)
    let scratch = Scratch(8)
    #expect(try pipes.read(into: scratch.pointer, count: 8, timeout: 0.01) == 0)
    kernel.feed([])
    let sent = pattern(slot * 2)
    kernel.feed(sent)
    #expect(try readAll(pipes, sent.count) == sent)
    #expect(kernel.discards == 0)
    _ = try pipes.read(into: scratch.pointer, count: 8, timeout: 0.01)
    #expect(kernel.pendingSizes == Array(repeating: slot, count: ReadRing.depth))
  }

  @Test("A bus error on a slot fails the read, and every read after")
  func slotFailure() throws {
    let pipes = pipes(device)
    kernel.failNext = -LinuxErrno.proto
    kernel.feed(pattern(slot))
    let scratch = Scratch(slot)
    #expect(throws: LinkError.self) { try pipes.read(into: scratch.pointer, count: slot, timeout: 0) }
    #expect(throws: LinkError.self) { try pipes.read(into: scratch.pointer, count: slot, timeout: 0) }
  }

  @Test("A submit refused while reads are posted leaves the ring shallower, not failed")
  func shallowRing() throws {
    kernel.maxPending = 3
    let pipes = pipes(device)
    let sent = pattern(slot * 10)
    let reader = Background { [self] in try readAll(pipes, sent.count) }
    #expect(eventually { kernel.pendingCount == 3 })
    kernel.feed(sent)
    #expect(try #require(reader.join()).get() == sent)
  }

  @Test("abort ends a read blocked without a deadline, from another thread")
  func abortWakesRead() throws {
    let pipes = pipes(device)
    let reader = Background {
      let buffer = UnsafeMutableRawPointer.allocate(byteCount: 1024, alignment: 16)
      defer { buffer.deallocate() }
      return try pipes.read(into: buffer, count: 1024, timeout: 0)
    }
    Thread.sleep(forTimeInterval: 0.05)
    pipes.abort()
    let result = try #require(reader.join())
    #expect(throws: LinkError.self) { try result.get() }
    #expect(kernel.pendingCount == 0)
    // and it stays aborted
    let buffer = UnsafeMutableRawPointer.allocate(byteCount: 1024, alignment: 16)
    defer { buffer.deallocate() }
    #expect(throws: LinkError.self) { try pipes.read(into: buffer, count: 1024, timeout: 0) }
  }

  @Test("A write completes while another thread waits in the kernel for a read")
  func oneReaperForBoth() throws {
    let pipes = pipes(device)
    let reader = Background {
      let buffer = UnsafeMutableRawPointer.allocate(byteCount: 1024, alignment: 16)
      defer { buffer.deallocate() }
      return try pipes.read(into: buffer, count: 1024, timeout: 0)
    }
    Thread.sleep(forTimeInterval: 0.05)
    let bytes = [UInt8](repeating: 9, count: 100)
    let writer = Background { try bytes.withUnsafeBytes { try pipes.write(from: $0.baseAddress!, count: 100, timeout: 2) } }
    #expect(try #require(writer.join()).get() == 100)
    kernel.feed(pattern(slot))
    #expect(try #require(reader.join()).get() == 1024)
  }

  @Test("Closing ends the ring's reads and waits for the kernel to give them back")
  func closeRetiresRing() throws {
    let pipes = pipes(device)
    let scratch = Scratch(8)
    #expect(try pipes.read(into: scratch.pointer, count: 8, timeout: 0.01) == 0)
    #expect(kernel.pendingCount == ReadRing.depth)
    pipes.close()
    #expect(kernel.pendingCount == 0)
    #expect(throws: LinkError.self) { try pipes.read(into: scratch.pointer, count: 8, timeout: 0) }
  }

  @Test("A new session's pipes on the same device start clean after the last one aborted, on its URBs")
  func pipesPerSession() throws {
    let device = device
    let first = pipes(device)
    let buffer = UnsafeMutableRawPointer.allocate(byteCount: 1024, alignment: 16)
    defer { buffer.deallocate() }
    #expect(try first.read(into: buffer, count: 1024, timeout: 0.01) == 0)
    first.abort()
    first.close()
    let second = pipes(device)
    kernel.feed(pattern(slot))
    #expect(try second.read(into: buffer, count: 1024, timeout: 0) == 1024)
    #expect(kernel.urbs.count == ReadRing.depth)
  }

  @Test("Pipes on two endpoint pairs of one descriptor each get their own last URBs back")
  func sparesPerEndpoints() {
    let device = device
    let driving = device.take(inEndpoint: 0x81, outEndpoint: 0x01)
    let addon = device.take(inEndpoint: 0x82, outEndpoint: 0x02)
    device.keep(driving)
    device.keep(addon)
    #expect(device.take(inEndpoint: 0x81, outEndpoint: 0x01) === driving)
    #expect(device.take(inEndpoint: 0x82, outEndpoint: 0x02) === addon)
  }

  @Test("Unplugging with a message half in the ring fails the read, and closing does not wait on the dead device")
  func unplugMidRing() throws {
    let device = device
    let pipes = pipes(device)
    let transport = USBTransport(pipes: pipes)
    kernel.feed(Array(FakeUsbfs.gadgetFrame(.inferReq, seq: 1, payload: Data(pattern(200_000))).prefix(slot * 5)))
    let reader = Background { try transport.recv() }
    // Five slots emptied and posted again: the reader waits for the rest.
    #expect(eventually { kernel.submits == ReadRing.depth + 5 })
    kernel.unplug()
    let result = try #require(reader.join())
    #expect(throws: LinkError.self) { try result.get() }
    #expect(device.isGone)
    let started = Date()
    transport.close()
    #expect(Date().timeIntervalSince(started) < 1)
  }

  @Test("invalidate ends every transfer and returns once none touches the descriptor")
  func invalidateWaitsForTransfers() throws {
    let device = device
    let sessions = [pipes(device), pipes(device)]
    let readers = sessions.map { pipes in
      Background {
        let buffer = UnsafeMutableRawPointer.allocate(byteCount: 1024, alignment: 16)
        defer { buffer.deallocate() }
        return try pipes.read(into: buffer, count: 1024, timeout: 0)
      }
    }
    Thread.sleep(forTimeInterval: 0.05)
    #expect(device.invalidate(timeout: 2))
    for reader in readers {
      #expect(throws: LinkError.self) { try #require(reader.join()).get() }
    }
    #expect(kernel.pendingCount == 0)
  }

  @Test("A submit the kernel refuses is a link error")
  func submitFailure() {
    kernel.failSubmit = LinuxErrno.nomem
    let pipes = pipes(device)
    let buffer = UnsafeMutableRawPointer.allocate(byteCount: 16, alignment: 16)
    defer { buffer.deallocate() }
    #expect(throws: LinkError.self) { try pipes.read(into: buffer, count: 16, timeout: 0) }
  }

  @Test("Messages queued back to back come out one at a time, each at the start of the buffer")
  func backToBack() throws {
    kernel.shuffleReaps = true
    let transport = USBTransport(pipes: pipes(device))
    let payloads = [0, 1, 16_352, 16_353, 475_104, 300_000, 4].map { Data(pattern($0, seed: $0)) }
    for (i, payload) in payloads.enumerated() {
      kernel.feed(FakeUsbfs.gadgetFrame(.inferReq, seq: UInt32(i + 1), payload: payload))
    }
    var starts = Set<UnsafeRawPointer>()
    for (i, payload) in payloads.enumerated() {
      let message = try transport.recv()
      #expect(message.seq == UInt32(i + 1))
      #expect(Data(message.payload) == payload)
      #expect(Int(bitPattern: message.payload.baseAddress) % 16 == 0, "the payload is not aligned")
      starts.insert(message.payload.baseAddress! - Wire.headerSize)
    }
    #expect(starts.count == 1, "a message did not land at the start of the buffer")
    #expect(kernel.discards == 0)
    #expect(USBTransport.gadgetPad(32) == 16_352 && USBTransport.gadgetPad(16_384) == 0 && USBTransport.gadgetPad(16_385) == 16_383)
  }

  @Test("A corrupt header latches the link desynced, drain empties what the comma still sends, and the next session reads cleanly")
  func desyncDrainReopen() throws {
    let device = device
    let first = USBTransport(pipes: pipes(device))
    kernel.feed([UInt8](repeating: 0xEE, count: slot))
    #expect(throws: LinkError.self) { try first.recv() }
    #expect(first.desynced)
    kernel.feed([UInt8](repeating: 0x11, count: 300 * 1024))
    first.drain(0.3)
    #expect(kernel.buffered == 0)
    // Drained already: close without the five seconds' drain it would add.
    first.shutdown()
    first.close()
    #expect(kernel.pendingCount == 0)
    let second = USBTransport(pipes: pipes(device))
    kernel.feed(FakeUsbfs.gadgetFrame(.ping, seq: 1, payload: Data()))
    #expect(try second.recv().msgType == Wire.Msg.ping.rawValue)
  }

  @Test("The steady state reuses the pipes' URBs: none is made per transfer")
  func urbsArePooled() throws {
    let pipes = pipes(device)
    let transport = USBTransport(pipes: pipes)
    let payload = Data(pattern(475_104))
    let reply = Data(pattern(73_860))
    for seq in 1...100 {
      kernel.feed(FakeUsbfs.gadgetFrame(.inferReq, seq: UInt32(seq), payload: payload))
      #expect(try transport.recv().seq == UInt32(seq))
      try reply.withUnsafeBytes { try transport.send(.inferResp, seq: UInt32(seq), parts: [$0]) }
    }
    #expect(kernel.urbs.count == ReadRing.depth + 1)
    #expect(kernel.submits > 100 * 29)
  }
}

/// The USB waits' deadlines, on a clock the tests set.
@Suite("Monotonic deadlines")
struct MonotonicDeadlineTests {
  @Test("Time left is read off the monotonic clock, and never goes below zero")
  func remaining() {
    let deadline = MonotonicDeadline(in: 2, now: 100)
    #expect(deadline.remaining(now: 100.5) == 1.5)
    #expect(!deadline.passed(now: 101.999))
    #expect(deadline.passed(now: 102) && deadline.remaining(now: 102) == 0)
    #expect(deadline.remaining(now: 250) == 0)
    #expect(MonotonicDeadline(in: 0.05, now: 100) < deadline)
  }

  @Test("A wall-clock step moves where a wait ends, not when: the wait is for the time left")
  func wallClockSteps() {
    let deadline = MonotonicDeadline(in: 2, now: 100)
    let before = Date(timeIntervalSince1970: 1_000_000)
    #expect(deadline.wallClock(now: 100.5, date: before) == before.addingTimeInterval(1.5))
    // timesyncd stepped the clock an hour forward half a second in: the wait
    // still has 1.5 s to go, not none
    let stepped = before.addingTimeInterval(3600.5)
    #expect(deadline.wallClock(now: 100.5, date: stepped) == stepped.addingTimeInterval(1.5))
    #expect(!deadline.passed(now: 100.5))
  }
}
