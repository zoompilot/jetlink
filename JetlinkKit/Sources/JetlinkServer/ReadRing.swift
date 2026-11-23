import Foundation
import JetlinkKit

/// The bulk IN pipe under a `ReadRing`: usbfs on Linux and Android, IOUSBHost
/// on a Mac. Every call is made under the ring's lock.
protocol ReadRingPipe: AnyObject {
  /// Posts a read of `size` bytes into `slot`'s buffer. Its completion comes
  /// back through `ReadRing.complete`. Throws when the pipe refuses it.
  func post(_ slot: Int, size: Int) throws
  /// Waits until a read may have completed or `deadline` (nil: none) passes,
  /// releasing the ring's lock meanwhile. Throws once the link is gone or
  /// closed.
  func awaitCompletion(until deadline: MonotonicDeadline?) throws
  /// Where `slot`'s bytes are.
  func bytes(_ slot: Int) -> UnsafeRawPointer
}

/// Reads kept posted ahead of the reader on the comma's bulk IN pipe, so the
/// comma's stream never waits for the host to ask for the next piece of a
/// message: `depth` reads of `slotSize`, posted again as the reader empties
/// them.
///
/// Why reading ahead is safe. The comma pads every message to
/// `Wire.gadgetTxAlign` (16 KB) and writes it in whole multiples of 16 KB
/// (`StreamTransport.send`, `FfsTransport.write_chunk`), so it never sends a
/// short packet and every message ends on the stream's 16 KB grid. A 16 KB
/// read posted on that grid ends at a message's end or inside the message,
/// never past it: it cannot sit holding the end of one message while it waits
/// for bytes of the next. Bytes go to the reader in the order the reads were
/// posted, whatever order they are reaped in, and never more than it asks
/// for, so a message still lands whole at the start of its buffer.
///
/// A short packet takes the stream off the grid, which only a broken stream
/// does: the read fails, the session ends, and the comma reconnects. A
/// zero-length packet carries nothing and leaves the grid alone. An add-on's
/// interface (`shortPackets`) has no grid: its messages end on a short
/// packet, which ends the read there, so a read still never spans two.
///
/// One reader at a time. Every call is made under `lock`, which the pipe also
/// holds when it calls `complete`, and releases while it waits.
final class ReadRing: @unchecked Sendable {
  enum Completion {
    /// Bytes arrived, or none did (a zero-length packet).
    case data
    case failed(LinkError)
  }

  private enum Slot {
    case idle, posted, done
  }

  static let slotSize = Wire.gadgetTxAlign
  /// 512 KB: a whole 400 KB inference request streams without the reader
  /// posting anything, with slots to spare.
  static let depth = 32

  let lock: NSCondition
  private unowned let pipe: any ReadRingPipe
  private let shortPackets: Bool
  private var state: [Slot]
  private var actual: [Int]
  private var failures: [LinkError?]
  /// The oldest slot the reader has not emptied, and how much of it it took.
  private var head = 0
  private var offset = 0
  /// Slots from `head` on that are posted or done, in the order posted.
  private var used = 0

  init(lock: NSCondition, pipe: any ReadRingPipe, shortPackets: Bool = false) {
    self.lock = lock
    self.pipe = pipe
    self.shortPackets = shortPackets
    state = Array(repeating: .idle, count: ReadRing.depth)
    actual = Array(repeating: 0, count: ReadRing.depth)
    failures = Array(repeating: nil, count: ReadRing.depth)
  }

  /// Copies up to `count` bytes of the stream into `buffer` and returns how
  /// many: whatever has arrived, once anything has. Returns 0 when `deadline`
  /// (nil: none) passes first, leaving the reads posted, so nothing that
  /// arrives later is lost. Throws the first failed read's error, and again
  /// on every call after.
  func read(into buffer: UnsafeMutableRawPointer, count: Int, deadline: MonotonicDeadline?) throws -> Int {
    guard count > 0 else { return 0 }
    while true {
      let copied = try take(into: buffer, count: count)
      if copied > 0 {
        return copied
      }
      // Emptied slots go back only now, when the reader would wait: the
      // read that ends a message returns without posting on its way out.
      try refill()
      if let deadline, deadline.passed() {
        return 0
      }
      try pipe.awaitCompletion(until: deadline)
    }
  }

  /// Records that `slot`'s read ended with `count` bytes. The pipe wakes the
  /// reader after.
  func complete(_ slot: Int, _ completion: Completion, count: Int) {
    guard state[slot] == .posted else { return }
    state[slot] = .done
    actual[slot] = 0
    switch completion {
    case .failed(let error):
      failures[slot] = error
    case .data where count > 0 && count < ReadRing.slotSize && !shortPackets:
      failures[slot] = .desynced("a short packet from the comma: the stream is off the 16 KB grid")
    case .data:
      actual[slot] = min(max(count, 0), ReadRing.slotSize)
    }
  }

  /// Empties done slots in the order they were posted, into `buffer`.
  private func take(into buffer: UnsafeMutableRawPointer, count: Int) throws -> Int {
    var copied = 0
    while copied < count && used > 0 && state[head] == .done {
      if let failure = failures[head] {
        throw failure
      }
      let n = min(actual[head] - offset, count - copied)
      if n > 0 {
        (buffer + copied).copyMemory(from: pipe.bytes(head) + offset, byteCount: n)
        copied += n
        offset += n
      }
      if offset == actual[head] {
        state[head] = .idle
        head = (head + 1) % ReadRing.depth
        used -= 1
        offset = 0
      }
    }
    return copied
  }

  /// Posts every free slot. A read the pipe refuses while others are in hand
  /// leaves the ring shallower until the next refill; with none in hand it
  /// is the reader's error.
  private func refill() throws {
    while used < ReadRing.depth {
      let slot = (head + used) % ReadRing.depth
      do {
        try pipe.post(slot, size: ReadRing.slotSize)
      } catch {
        if used > 0 { return }
        throw error
      }
      state[slot] = .posted
      failures[slot] = nil
      used += 1
    }
  }
}
