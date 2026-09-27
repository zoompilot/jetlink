// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.
//
// The model's history buffers, as jetlink/queues.py keeps them.
//
// openpilot folds these into the tinygrad JIT; shipping the history across
// the link would cost ~10 MB a frame, so they live on the server and the
// comma sends only the newest warped frame and the packed scalars. Ring
// buffers in float16, as the Python stores them, so only the new row is
// converted. ios/scripts/queue_fixtures.py writes what the Python produces
// for a run long enough to wrap every ring; the tests hold this to it bit for
// bit.

import Foundation

/// One of the model's inputs as the engine stages it: a buffer of float16 or
/// float32 values.
public struct TensorBuffer: @unchecked Sendable {
  public enum Kind: Sendable { case float16, float32 }
  public let base: UnsafeMutableRawPointer
  public let count: Int
  public let kind: Kind

  public init(base: UnsafeMutableRawPointer, count: Int, kind: Kind) {
    self.base = base
    self.count = count
    self.kind = kind
  }

  var byteCount: Int { count * (kind == .float16 ? 2 : 4) }

  /// Copies float16 values in at element `offset`, converting if the buffer is float32.
  @inline(__always)
  func store(_ src: UnsafePointer<Float16>, count n: Int, at offset: Int) {
    switch kind {
    case .float16:
      (base.assumingMemoryBound(to: Float16.self) + offset).update(from: src, count: n)
    case .float32:
      Convert.f16ToF32(src, base.assumingMemoryBound(to: Float.self) + offset, count: n)
    }
  }

  /// Copies float32 values in, rounding to float16 as numpy's assignment does.
  @inline(__always)
  func store(_ src: UnsafePointer<Float>, count n: Int, at offset: Int) {
    switch kind {
    case .float16:
      Convert.f32ToF16(src, base.assumingMemoryBound(to: Float16.self) + offset, count: n)
    case .float32:
      (base.assumingMemoryBound(to: Float.self) + offset).update(from: src, count: n)
    }
  }
}

/// A fixed-length FIFO of rows over one preallocated array. Logical row i
/// (0 = oldest) lives at physical (head + i) % rows.
final class RingQueue {
  let rows: Int
  let rowElems: Int
  let buf: UnsafeMutablePointer<Float16>
  private(set) var head = 0

  init(rows: Int, rowElems: Int) {
    self.rows = rows
    self.rowElems = rowElems
    buf = .allocate(capacity: rows * rowElems)
    buf.initialize(repeating: 0, count: rows * rowElems)
  }

  deinit { buf.deallocate() }

  func reset() {
    buf.update(repeating: 0, count: rows * rowElems)
    head = 0
  }

  private var slot: UnsafeMutablePointer<Float16> { buf + head * rowElems }

  private func advance() { head = (head + 1) % rows }

  /// The oldest row becomes the newest, from uint8.
  func push(_ src: UnsafePointer<UInt8>) {
    Convert.u8ToF16(src, slot, count: rowElems)
    advance()
  }

  /// The oldest row becomes the newest, from float32 (round to nearest even).
  func push(_ src: UnsafePointer<Float>) {
    Convert.f32ToF16(src, slot, count: rowElems)
    advance()
  }

  /// Logical rows 0, step, 2*step, ... into `out` from element `offset`,
  /// oldest first: openpilot's buf[::step].
  func gather(step: Int, into out: TensorBuffer) {
    let m = (rows + step - 1) / step
    for i in 0..<m {
      let physical = (head + i * step) % rows
      out.store(buf + physical * rowElems, count: rowElems, at: i * rowElems)
    }
  }

  /// openpilot's sample_desire: every frame_skip consecutive rows reduced by
  /// an elementwise max, as numpy's max(axis=1) does, NaN propagating.
  func maxOfGroups(of group: Int, into out: TensorBuffer, scratch: UnsafeMutablePointer<Float16>) {
    let groups = rows / group
    for j in 0..<groups {
      let first = buf + ((head + j * group) % rows) * rowElems
      scratch.update(from: first, count: rowElems)
      for k in 1..<max(group, 1) {
        let row = buf + ((head + j * group + k) % rows) * rowElems
        for e in 0..<rowElems {
          let a = scratch[e], b = row[e]
          // numpy's maximum: (a >= b || isnan(a)) ? a : b
          scratch[e] = (a >= b || a.isNaN) ? a : b
        }
      }
      out.store(scratch, count: rowElems, at: j * rowElems)
    }
  }
}

/// Everything run_policy owned in the JIT, for one model.
public final class PolicyQueues {
  public let spec: ModelSpec
  let frameSkip: Int
  let imgQ: RingQueue
  let bigImgQ: RingQueue
  let featQ: RingQueue
  let desireQ: RingQueue
  private let scratch: UnsafeMutablePointer<Float16>
  /// (offset, count) of desire, traffic_convention, action_t, prev_feat in the packed floats.
  private let layout: [(Int, Int)]
  private let warpedRow: Int

  public init(spec: ModelSpec) throws {
    try spec.validate()
    self.spec = spec
    frameSkip = spec.frameSkip
    let ib = spec.imgBufShape
    imgQ = RingQueue(rows: ib[0], rowElems: product(Array(ib.dropFirst())))
    bigImgQ = RingQueue(rows: ib[0], rowElems: product(Array(ib.dropFirst())))
    let fq = spec.featQShape
    featQ = RingQueue(rows: fq[0], rowElems: product(Array(fq.dropFirst())))
    let dq = spec.desireQShape
    guard dq[0] % spec.frameSkip == 0 else { throw SpecError("desire history \(dq[0]) is not a multiple of frame_skip") }
    desireQ = RingQueue(rows: dq[0], rowElems: product(Array(dq.dropFirst())))
    scratch = .allocate(capacity: max(desireQ.rowElems, 1))
    var offset = 0
    var layout: [(Int, Int)] = []
    for size in spec.packedSizes {
      layout.append((offset, size))
      offset += size
    }
    self.layout = layout
    warpedRow = spec.warpedNbytes / 2
    guard warpedRow == imgQ.rowElems else { throw SpecError("a warped frame does not fill an image row") }
    guard layout[0].1 == desireQ.rowElems, layout[3].1 == featQ.rowElems else {
      throw SpecError("the packed inputs do not match the history rows")
    }
  }

  deinit { scratch.deallocate() }

  public func reset() {
    for q in [imgQ, bigImgQ, featQ, desireQ] { q.reset() }
  }

  /// The names of the inputs `step` writes, which must be the engine's.
  public static let inputNames = ["img", "big_img", "desire_pulse", "traffic_convention", "action_t", "features_buffer"]

  /// Advances the queues one frame and writes the model's inputs.
  ///
  /// `warped` is (2, 6, H, W) uint8 off openpilot's warp; `packed` the flat
  /// float32 scalars laid out per ModelSpec.packedShapes. Both are sized by
  /// the caller against the spec before this is reached.
  public func step(warped: UnsafePointer<UInt8>, packed: UnsafePointer<Float>, into dest: [String: TensorBuffer]) {
    let desire = packed + layout[0].0
    let traffic = packed + layout[1].0
    let action = packed + layout[2].0
    let prevFeat = packed + layout[3].0
    imgQ.push(warped)
    bigImgQ.push(warped + warpedRow)
    desireQ.push(desire)
    featQ.push(prevFeat)

    imgQ.gather(step: frameSkip, into: dest["img"]!)
    bigImgQ.gather(step: frameSkip, into: dest["big_img"]!)
    featQ.gather(step: frameSkip, into: dest["features_buffer"]!)
    desireQ.maxOfGroups(of: frameSkip, into: dest["desire_pulse"]!, scratch: scratch)
    dest["traffic_convention"]!.store(traffic, count: layout[1].1, at: 0)
    dest["action_t"]!.store(action, count: layout[2].1, at: 0)
  }
}
