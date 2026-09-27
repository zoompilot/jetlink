// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.
//
// The loaded engine, run on the phone the way the car will run it, for
// deciding at home whether a model is fast enough and stays fast enough.
//
// Frames come at 20 a second with the accelerator idle between them, as
// scripts/bench_link.py paces them, because a paced stream and not a
// back-to-back one is what the comma asks for (docs/backends.md). Each frame
// is the server's whole share of it: the queues, the model, and reading the
// output back, with the hidden state fed back as modeld does. The link is
// not in it; run bench_link.py on the comma for that. The run is reported
// in windows, so a phone that slows as it heats shows it.

import Foundation

public struct BenchmarkStats: Sendable, Equatable {
  public var mean = 0.0, p50 = 0.0, p90 = 0.0, p99 = 0.0, max = 0.0

  init(_ ms: [Double]) {
    guard !ms.isEmpty else { return }
    let s = ms.sorted()
    func pct(_ q: Double) -> Double { s[Int((q * Double(s.count - 1)).rounded())] }
    mean = ms.reduce(0, +) / Double(ms.count)
    p50 = pct(0.5)
    p90 = pct(0.9)
    p99 = pct(0.99)
    max = s.last!
  }
}

public struct BenchmarkWindow: Sendable, Equatable {
  public var startSecond: Int
  public var frame: BenchmarkStats
  public var thermal: ProcessInfo.ThermalState
}

public struct BenchmarkReport: Sendable, Equatable {
  public var modelSha256: String
  public var units: ComputeUnits
  public var seconds: Double
  public var frames: Int
  /// The server's share of each frame: queues, model, output.
  public var frame: BenchmarkStats
  /// The model alone: the gpu_us the comma is told.
  public var accelerator: BenchmarkStats
  /// The history queues building the model's inputs, and the output read back.
  public var queues: BenchmarkStats
  public var output: BenchmarkStats
  /// How this build was compiled and what ran beside it, for comparing reports.
  public var build: String
  public var over35: Int
  public var over50: Int
  public var windows: [BenchmarkWindow]
  public var thermalAtStart: ProcessInfo.ThermalState
  public var thermalAtEnd: ProcessInfo.ThermalState
  public var cancelled: Bool

  /// The report as text, to paste into an issue or a note.
  public var text: String {
    func f(_ s: BenchmarkStats) -> String {
      String(format: "mean %.1f  p50 %.1f  p90 %.1f  p99 %.1f  max %.1f ms", s.mean, s.p50, s.p90, s.p99, s.max)
    }
    var lines = [
      "Jetlink benchmark, \(chipName()), onnxruntime \(OrtRuntime.version), \(units.label)",
      "model \(modelSha256.prefix(16)), \(frames) frames at 20 Hz over \(Int(seconds)) s\(cancelled ? " (stopped early)" : "")",
      "\(build)",
      "frame        \(f(frame))",
      "accelerator  \(f(accelerator))",
      "queues       \(f(queues))",
      "output       \(f(output))",
      "over 35 ms: \(over35)   over 50 ms: \(over50)",
      "temperature: \(thermalLabel(thermalAtStart)) at start, \(thermalLabel(thermalAtEnd)) at end",
      "by \(BenchmarkRun.window) s window:",
    ]
    for w in windows {
      lines.append(
        String(format: "  %4d s  mean %5.1f  p99 %5.1f  max %5.1f ms  ", w.startSecond, w.frame.mean, w.frame.p99, w.frame.max)
          + thermalLabel(w.thermal))
    }
    return lines.joined(separator: "\n")
  }
}

extension BenchmarkReport {
  static func buildLine(threadQoS: QualityOfService) -> String {
    #if DEBUG
      let config = "Debug build (unoptimized)"
    #else
      let config = "Release build"
    #endif
    let qos: String
    switch threadQoS {
    case .userInteractive: qos = "user-interactive"
    case .userInitiated: qos = "user-initiated"
    case .utility: qos = "utility"
    case .background: qos = "background"
    default: qos = "default"
    }
    return "\(config), frame thread \(qos), CPU keep-warm \(OrtRuntime.cpuKeepWarm.rawValue)"
  }
}

public func thermalLabel(_ t: ProcessInfo.ThermalState) -> String {
  switch t {
  case .nominal: "nominal"
  case .fair: "fair"
  case .serious: "serious"
  case .critical: "critical"
  @unknown default: "unknown"
  }
}

/// What a running benchmark has done so far.
public struct BenchmarkProgress: Sendable {
  public var elapsed: Double
  public var total: Double
  public var frames: Int
  public var lastWindow: BenchmarkWindow?

  public init(elapsed: Double, total: Double, frames: Int, lastWindow: BenchmarkWindow?) {
    self.elapsed = elapsed
    self.total = total
    self.frames = frames
    self.lastWindow = lastWindow
  }
}

/// A benchmark in flight: cancel() stops it at the next frame.
public final class BenchmarkRun: @unchecked Sendable {
  public static let window = 10
  private let lock = NSLock()
  private var stop = false

  public init() {}

  public func cancel() {
    lock.lock()
    stop = true
    lock.unlock()
  }

  var cancelled: Bool {
    lock.lock()
    defer { lock.unlock() }
    return stop
  }
}

extension EngineHost {
  /// Runs the loaded engine at 20 Hz for `seconds`. Refuses while a comma is
  /// connected; a comma that connects meanwhile is told the engine is not
  /// ready until it ends, rather than have its frames mixed into the
  /// benchmark's history. Blocks; call it off the main thread.
  public func benchmark(seconds: Double, run: BenchmarkRun, progress: @escaping @Sendable (BenchmarkProgress) -> Void) throws
    -> BenchmarkReport
  {
    lock.lock()
    guard let l = loaded else {
      lock.unlock()
      throw EngineError("no model is loaded; prepare one on the Models tab first")
    }
    if session != nil {
      lock.unlock()
      throw EngineError("a comma is connected; disconnect it to benchmark, or watch the live numbers on the Status tab")
    }
    if benchmarking {
      lock.unlock()
      throw EngineError("a benchmark is already running")
    }
    benchmarking = true
    lock.unlock()
    defer {
      lock.lock()
      benchmarking = false
      loaded?.queues.reset()
      lock.unlock()
    }
    log.info("benchmark: \(l.units.label), \(Int(seconds)) s at 20 Hz")

    let spec = l.spec
    var rng = SystemRandomNumberGenerator()
    let warped = (0..<spec.warpedNbytes).map { _ in UInt8.random(in: 0...255, using: &rng) }
    var packed = [Float](repeating: 0, count: spec.packedNelem)
    let hidden = spec.outputSlices["hidden_state"]
    let feedback = hidden.map { $0.count <= spec.packedNelem && $0.stop <= spec.outputNelem } ?? false

    var frameMs: [Double] = [], accelMs: [Double] = [], queueMs: [Double] = [], outMs: [Double] = []
    var windows: [BenchmarkWindow] = []
    var windowFrames: [Double] = []
    let thermalAtStart = ProcessInfo.processInfo.thermalState
    let period = 1.0 / Double(modelRunFreq)
    let warmup = 5
    let t0 = monotonic()
    var next = t0
    var windowStart = 0
    var i = 0
    l.queues.reset()
    while !run.cancelled {
      let elapsed = monotonic() - t0
      if elapsed >= seconds + Double(warmup) * period { break }
      let wait = next - monotonic()
      if wait > 0 { Thread.sleep(forTimeInterval: wait) }
      next += period

      lock.lock()
      guard loaded === l else {
        lock.unlock()
        throw EngineError("the engine was unloaded during the benchmark")
      }
      let f0 = monotonic()
      warped.withUnsafeBufferPointer { w in
        packed.withUnsafeBufferPointer { p in l.queues.step(warped: w.baseAddress!, packed: p.baseAddress!, into: l.hostInputs) }
      }
      let q = (monotonic() - f0) * 1e3
      do {
        try l.engine.run()
      } catch {
        lock.unlock()
        throw error
      }
      let r0 = monotonic()
      let out = l.engine.output()
      let finite = Convert.allFinite(out.baseAddress!, count: out.count, fromFloat16: l.engine.outputIsFloat16)
      let end = monotonic()
      let ms = (end - f0) * 1e3
      let o = (end - r0) * 1e3
      let accel = Double(l.engine.lastRunUs) / 1e3
      if feedback, let h = hidden, finite {
        // prev_feat is the last of the packed inputs, as modeld feeds it.
        let dst = spec.packedNelem - h.count
        for k in 0..<h.count { packed[dst + k] = out[h.start + k] }
      }
      lock.unlock()

      i += 1
      if i <= warmup { continue }
      frameMs.append(ms)
      accelMs.append(accel)
      queueMs.append(q)
      outMs.append(o)
      windowFrames.append(ms)
      let second = Int(monotonic() - t0)
      if second - windowStart >= BenchmarkRun.window {
        let w = BenchmarkWindow(startSecond: windowStart, frame: BenchmarkStats(windowFrames), thermal: ProcessInfo.processInfo.thermalState)
        windows.append(w)
        windowFrames.removeAll(keepingCapacity: true)
        windowStart = second
        progress(BenchmarkProgress(elapsed: monotonic() - t0, total: seconds, frames: frameMs.count, lastWindow: w))
      } else if frameMs.count % 20 == 0 {
        progress(BenchmarkProgress(elapsed: monotonic() - t0, total: seconds, frames: frameMs.count, lastWindow: windows.last))
      }
    }
    if !windowFrames.isEmpty {
      windows.append(
        BenchmarkWindow(startSecond: windowStart, frame: BenchmarkStats(windowFrames), thermal: ProcessInfo.processInfo.thermalState))
    }
    let report = BenchmarkReport(
      modelSha256: l.sha256, units: l.units, seconds: monotonic() - t0, frames: frameMs.count, frame: BenchmarkStats(frameMs),
      accelerator: BenchmarkStats(accelMs), queues: BenchmarkStats(queueMs), output: BenchmarkStats(outMs),
      build: BenchmarkReport.buildLine(threadQoS: Thread.current.qualityOfService), over35: frameMs.filter { $0 > 35 }.count, over50: frameMs.filter { $0 > 50 }.count,
      windows: windows, thermalAtStart: thermalAtStart, thermalAtEnd: ProcessInfo.processInfo.thermalState,
      cancelled: run.cancelled)
    log.info("benchmark done:\n\(report.text)")
    return report
  }
}
