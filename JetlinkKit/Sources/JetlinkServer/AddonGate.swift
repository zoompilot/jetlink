// The add-on server: a fork's own models beside the driving model, each on its
// own interface of the comma's gadget. docs/addons.md. From Peter Nguyen's
// V-ASM and sign-vision servers (trung791997/openpilot).
import Foundation

/// The driving model first: an add-on pass runs only in the gap after a
/// driving frame, when it ends before the next is due. Frames come at a steady
/// rate (20 Hz from a comma), so the next one is due a period after the last
/// one started. With no frame for `idleAfter`, nothing is driving and a pass
/// may run at once. The pass holds the host's lock as a frame does, so the two
/// never share the GPU; a frame that comes early waits at most one pass.
/// Admission and the engine are one decision: a pass admitted holds the gap
/// until `passEnded()`, so a second add-on never runs on a gap the first
/// already spent, and decides again from the clock once the first is done.
public final class AddonGate: @unchecked Sendable {
  static let margin: UInt64 = 3_000_000
  static let idleAfter: UInt64 = 1_000_000_000
  /// Hellos say yielding while `yieldAfter` requests in `yieldFor` found no gap.
  static let yieldFor: UInt64 = 2_000_000_000
  static let yieldAfter = 3

  private let condition = NSCondition()
  private let clock: @Sendable () -> UInt64
  private var lastStart: UInt64 = 0
  private var period: UInt64 = 50_000_000
  private var inFrame = false
  /// An add-on's pass was admitted and has not ended.
  private var passing = false
  private var misses: [UInt64] = []

  public init(clock: @escaping @Sendable () -> UInt64 = { DispatchTime.now().uptimeNanoseconds }) {
    self.clock = clock
  }

  /// A driving frame arrived.
  public func frameStarted() {
    condition.withLock {
      let now = clock()
      if lastStart > 0, now > lastStart, now - lastStart < Self.idleAfter {
        period = (period * 7 + (now - lastStart)) / 8
      }
      lastStart = now
      inFrame = true
    }
  }

  /// It was answered.
  public func frameEnded() {
    condition.withLock {
      inFrame = false
      condition.broadcast()
    }
  }

  /// Is there a gap of `cost` before the next frame, with no other add-on's
  /// pass in it? Waits up to `wait` for one: the end of the frame or the pass
  /// in flight, most likely. True admits the pass, which holds the gap until
  /// the caller's `passEnded()`.
  public func admit(cost: UInt64, wait: TimeInterval) -> Bool {
    condition.withLock {
      let deadline = Date().addingTimeInterval(wait)
      while true {
        if !passing && fits(cost) {
          passing = true
          return true
        }
        if Date() >= deadline || !condition.wait(until: deadline) {
          if !passing && fits(cost) {
            passing = true
            return true
          }
          let now = clock()
          misses = misses.filter { now - $0 < Self.yieldFor } + [now]
          return false
        }
      }
    }
  }

  /// The admitted pass is over, run or failed: the next add-on decides from now.
  public func passEnded() {
    condition.withLock {
      passing = false
      condition.broadcast()
    }
  }

  /// Found no gap again and again lately: hellos tell the comma to keep to
  /// its own model. A single miss is answered with a yield and nothing more.
  public var yielding: Bool {
    condition.withLock {
      let now = clock()
      return misses.filter { now - $0 < Self.yieldFor }.count >= Self.yieldAfter
    }
  }

  private func fits(_ cost: UInt64) -> Bool {
    let now = clock()
    if lastStart == 0 || now - lastStart > Self.idleAfter { return true }
    return !inFrame && now + cost + Self.margin <= lastStart + period
  }
}
