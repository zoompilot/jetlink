// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.
//
// Keeps the GPU clocked up between frames on the GPU path, as
// jetlink/server/backends/ort/metal.py does on a Mac.
//
// On an M2 Pro the gaps in a 20 Hz stream let GPU clocks fall and frames
// miss 50 ms that back-to-back inference made easily; a tiny independent
// kernel, one command at a time on its own queue, kept them up (docs/
// backends.md). It touches no model buffer, stops a second after the last
// frame, and never runs for the Neural Engine path. On a phone it is a
// power-for-latency trade the same way; the app can turn it off.

import Foundation
import Metal

final class MetalKeepAlive: @unchecked Sendable {
  static let idleSeconds = 1.0
  // About 0.3 ms at full clock on an M2 Pro; finite, so a stalled submitter
  // bounds the work.
  static let rounds: UInt32 = 10_000
  static let shader = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void keep_active(device uint *out [[buffer(0)]],
                            constant uint &rounds [[buffer(1)]],
                            uint tid [[thread_position_in_grid]]) {
      uint x = out[tid];
      for (uint i = 0; i < rounds; i++) x = x * 1664525u + 1013904223u;
      out[tid] = x;
    }
    """

  /// Off with JETLINK_METAL_KEEPALIVE=0, as on the Mac, or from the app.
  nonisolated(unsafe) static var enabled = ProcessInfo.processInfo.environment["JETLINK_METAL_KEEPALIVE"] != "0"
  static let log = JLogger("jetlink.ort.metal")

  private let cond = NSCondition()
  private var deadline = 0.0
  private var closed = false
  private var failed = false
  private var thread: Thread?

  private let queue: MTLCommandQueue
  private let pipeline: MTLComputePipelineState
  private let buffer: MTLBuffer

  static func make() -> MetalKeepAlive? {
    guard enabled else { return nil }
    do {
      return try MetalKeepAlive()
    } catch {
      log.warning("Metal keep-alive unavailable; continuing inference without it: \(error)")
      return nil
    }
  }

  private init() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("no Metal device") }
    let library = try device.makeLibrary(source: MetalKeepAlive.shader, options: nil)
    guard let function = library.makeFunction(name: "keep_active") else { throw EngineError("no keep_active function") }
    pipeline = try device.makeComputePipelineState(function: function)
    guard let q = device.makeCommandQueue(), let b = device.makeBuffer(length: 128, options: .storageModeShared) else {
      throw EngineError("Metal queue or buffer unavailable")
    }
    queue = q
    buffer = b
    let t = Thread { [weak self] in self?.loop() }
    t.name = "jetlink-metal-keepalive"
    t.qualityOfService = .userInitiated
    thread = t
    t.start()
  }

  func pulse() {
    cond.lock()
    if !closed && !failed {
      deadline = monotonic() + MetalKeepAlive.idleSeconds
      cond.signal()
    }
    cond.unlock()
  }

  func pause() {
    cond.lock()
    deadline = 0
    cond.signal()
    cond.unlock()
  }

  func close() {
    cond.lock()
    closed = true
    cond.signal()
    cond.unlock()
  }

  private func loop() {
    while true {
      cond.lock()
      while !closed && monotonic() >= deadline { cond.wait() }
      let stop = closed
      cond.unlock()
      if stop { return }
      autoreleasepool {
        do {
          try runOnce()
        } catch {
          cond.lock()
          failed = true
          deadline = 0
          cond.unlock()
          MetalKeepAlive.log.warning("Metal keep-alive stopped; continuing inference without it: \(error)")
        }
      }
      cond.lock()
      let giveUp = failed
      cond.unlock()
      if giveUp { return }
    }
  }

  private func runOnce() throws {
    guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
      throw EngineError("Metal command buffer unavailable")
    }
    var rounds = MetalKeepAlive.rounds
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(buffer, offset: 0, index: 0)
    encoder.setBytes(&rounds, length: MemoryLayout<UInt32>.size, index: 1)
    encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    // Never a backlog: one command in flight at a time.
    command.waitUntilCompleted()
    if command.status != .completed {
      // An app sent to the background may not submit GPU work at all.
      throw command.error ?? EngineError("keep-alive command did not complete")
    }
  }
}
