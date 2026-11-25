import Foundation
import JetlinkKit

/// The loaded engine, run the way the car will run it, for deciding at home
/// whether a model is fast enough and stays fast enough.
///
/// Frames come at 20 a second with the accelerator idle between them, as
/// scripts/bench_link.py paces them, because a paced stream and not a
/// back-to-back one is what the comma asks for (docs/backends.md). Each frame
/// is the server's whole share of it: the queues, the model, and reading the
/// output back, with the hidden state fed back as modeld does. The link is
/// not in it; run bench_link.py on the comma for that. The run is reported
/// in windows, so a phone that slows as it heats shows it.

/// A benchmark in flight: `cancel()` stops it at the next frame.
public final class BenchmarkRun: @unchecked Sendable {
  private let lock = NSLock()
  private var stop = false

  public init() {}

  public func cancel() {
    lock.lock()
    stop = true
    lock.unlock()
  }

  public var cancelled: Bool {
    lock.lock()
    defer { lock.unlock() }
    return stop
  }
}

extension BenchmarkStats {
  /// Percentiles by nearest rank over the sorted values, as the Python
  /// benchmark prints them.
  package static func of(_ ms: [Double]) -> BenchmarkStats {
    guard !ms.isEmpty else { return .empty }
    let sorted = ms.sorted()
    func percentile(_ q: Double) -> Double { sorted[Int((q * Double(sorted.count - 1)).rounded())] }
    return BenchmarkStats(
      mean: round2(ms.reduce(0, +) / Double(ms.count)), p50: round2(percentile(0.5)), p90: round2(percentile(0.9)), p99: round2(percentile(0.99)),
      max: round2(sorted.last!))
  }
}

/// The device's thermal state as the benchmark reports it: "nominal",
/// "fair", "serious" or "critical", from ProcessInfo on Apple platforms. The
/// Android app passes `ServerHooks.thermal` from PowerManager instead.
public func platformThermal() -> String {
  #if canImport(Darwin)
    thermalLabel(ProcessInfo.processInfo.thermalState)
  #else
    "unknown"
  #endif
}

#if canImport(Darwin)
  func thermalLabel(_ state: ProcessInfo.ThermalState) -> String {
    switch state {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
  }
#endif

extension EngineHost {
  /// Windows the run is reported in, seconds.
  public static let benchmarkWindow = 10
  /// Frames dropped at the start: CoreML settles over the first few.
  public static let benchmarkWarmup = 5
  /// The warm-up ends after this long however few frames it saw, so an
  /// engine that takes seconds a frame is measured rather than dropped.
  public static let benchmarkWarmupSeconds = 1.0
  /// The longest the screen goes without a progress event, for an engine
  /// too slow to finish a second's worth of frames in one.
  static let benchmarkProgressInterval = 1.0

  /// How this build was compiled and what ran beside it, for comparing reports.
  static func buildLine(_ engine: any Engine) -> String {
    #if DEBUG
      let config = "Debug build (unoptimized)"
    #else
      let config = "Release build"
    #endif
    let qos: String
    switch Thread.current.qualityOfService {
    case .userInteractive: qos = "user-interactive"
    case .userInitiated: qos = "user-initiated"
    case .utility: qos = "utility"
    case .background: qos = "background"
    default: qos = "default"
    }
    let notes = engine.notes
    return "\(config), frame thread \(qos)" + (notes.isEmpty ? "" : ", \(notes)")
  }

  /// Runs the loaded engine at 20 Hz for `seconds`, through the real queues
  /// on random camera bytes and zeroed scalars, and emits `.benchmark`
  /// events as it goes: running progress, then done, cancelled or failed.
  /// Refuses while a comma is connected; a comma that connects meanwhile is
  /// told the engine is not ready until the run ends, rather than have its
  /// frames mixed into the benchmark's history. Blocks; call it off the
  /// main thread. The report is the caller's to show; `logsReport` is
  /// ignored.
  public func benchmark(seconds: Double, run: BenchmarkRun, logsReport: Bool = false) throws -> BenchmarkReport {
    do {
      let report = try withEngineHeld(to: "benchmark") { l in
        // The benchmark's own GPU times, before a comma's frames join the block.
        defer { lock.withLock { if loaded === l { l.engine.flushTiming() } } }
        return try measure(l, seconds: seconds, run: run)
      }
      emit(
        .benchmark(
          BenchmarkEvent(
            state: report.cancelled ? "cancelled" : "done", elapsed: report.seconds, total: seconds, frames: report.frames, frame: report.frame,
            report: report, detail: "")))
      return report
    } catch {
      emit(.benchmark(BenchmarkEvent(state: "failed", elapsed: 0, total: seconds, frames: 0, frame: nil, report: nil, detail: "\(error)")))
      throw error
    }
  }

  /// Takes the loaded engine for `body`, a benchmark or a probe: refused
  /// while a comma is connected or another holds it, and the history is
  /// reset when it is given back.
  func withEngineHeld<T>(to purpose: String, _ body: (Loaded) throws -> T) throws -> T {
    let l = try lock.withLock {
      guard let l = loaded else { throw HostError.invalid("no model is loaded; prepare one first") }
      if session != nil { throw HostError.invalid("a comma is connected; disconnect it to \(purpose)") }
      if benchmarking { throw HostError.invalid("a benchmark is already running") }
      benchmarking = true
      return l
    }
    defer {
      lock.withLock {
        benchmarking = false
        loaded?.staging.reset()
      }
    }
    return try body(l)
  }

  /// One frame's times, and whether its output was all finite.
  struct FrameRun {
    let queueUs: UInt32
    let acceleratorUs: UInt32
    let outputUs: UInt32
    let totalUs: UInt32
    let finite: Bool
  }

  /// One frame on a held engine, as a frame from the comma runs: staged
  /// through the queues, run, read back into `output` as float32, and its
  /// hidden state kept for the next when it is finite.
  func runFrame(_ l: Loaded, warped: [UInt8], packed: [Float], into output: inout [Float]) throws -> FrameRun {
    guard let io = l.engine.outputs[ModelConstants.drivingOutput], io.type == .float || io.type == .float16 else {
      throw HostError.failed("the engine's driving output is not float")
    }
    let count = l.spec.outputCount
    return try lock.withLock {
      guard loaded === l else { throw HostError.failed("the engine was unloaded during the run") }
      let started = DispatchTime.now().uptimeNanoseconds
      try warped.withUnsafeBytes { w in
        try packed.withUnsafeBytes { p in try l.staging.stage(warped: w.baseAddress!, packed: p.baseAddress!) }
      }
      let queueUs = microseconds(since: started)
      try l.engine.run()
      let readStarted = DispatchTime.now().uptimeNanoseconds
      let out = l.staging.layout.output!
      output.withUnsafeMutableBytes { o in
        if io.type == .float16 {
          Convert.f16ToF32(out, o.baseAddress!, count: count)
        } else {
          o.baseAddress!.copyMemory(from: out, byteCount: count * 4)
        }
      }
      let finite = output.withUnsafeBytes { Convert.allFinite($0.baseAddress!, count: count) }
      let outputUs = microseconds(since: readStarted)
      let totalUs = microseconds(since: started)
      if finite {
        // the hidden state back into the queues, as a frame from the comma does
        l.staging.keep(outputs: out, type: io.type)
      }
      return FrameRun(queueUs: queueUs, acceleratorUs: l.engine.lastGpuUs, outputUs: outputUs, totalUs: totalUs, finite: finite)
    }
  }

  private func measure(_ l: Loaded, seconds: Double, run: BenchmarkRun) throws -> BenchmarkReport {
    let spec = l.spec
    let device = backend.describe()["device"] ?? ""
    log.info("benchmark: \(device), \(Int(seconds)) s at \(ModelConstants.runFrequency) Hz")

    var generator = SystemRandomNumberGenerator()
    let warped = (0..<spec.warpedBytes).map { _ in UInt8.random(in: 0...255, using: &generator) }
    let packed = [Float](repeating: 0, count: spec.packedCount)
    var output = [Float](repeating: 0, count: spec.outputCount)

    var frameMs: [Double] = [], accelMs: [Double] = [], queueMs: [Double] = [], outMs: [Double] = []
    var windows: [BenchmarkWindow] = []
    var windowFrames: [Double] = []
    let thermalAtStart = hooks.thermal()
    let period = 1.0 / Double(ModelConstants.runFrequency)
    let warmup = EngineHost.benchmarkWarmup
    let build = EngineHost.buildLine(l.engine)
    let t0 = ProcessInfo.processInfo.systemUptime
    var next = t0
    var windowStart = 0
    var i = 0
    var progressAt = t0
    lock.lock()
    l.staging.reset()
    lock.unlock()
    func progress(_ last: BenchmarkWindow?) {
      progressAt = ProcessInfo.processInfo.systemUptime
      emit(
        .benchmark(
          BenchmarkEvent(
            state: "running", elapsed: round2(ProcessInfo.processInfo.systemUptime - t0), total: seconds, frames: frameMs.count,
            frame: BenchmarkStats.of(frameMs), report: nil, detail: last.map { "window \($0.startSecond) s: \($0.thermal)" } ?? "")))
    }
    while !run.cancelled {
      let elapsed = ProcessInfo.processInfo.systemUptime - t0
      if elapsed >= seconds + Double(warmup) * period { break }
      let wait = next - ProcessInfo.processInfo.systemUptime
      if wait > 0 { Thread.sleep(forTimeInterval: wait) }
      next += period

      let frameAt = ProcessInfo.processInfo.systemUptime - t0
      let frame = try runFrame(l, warped: warped, packed: packed, into: &output)

      i += 1
      if i <= warmup && frameAt < EngineHost.benchmarkWarmupSeconds { continue }
      frameMs.append(Double(frame.totalUs) / 1000)
      accelMs.append(Double(frame.acceleratorUs) / 1000)
      queueMs.append(Double(frame.queueUs) / 1000)
      outMs.append(Double(frame.outputUs) / 1000)
      windowFrames.append(Double(frame.totalUs) / 1000)
      let second = Int(ProcessInfo.processInfo.systemUptime - t0)
      if second - windowStart >= EngineHost.benchmarkWindow {
        let window = BenchmarkWindow(
          startSecond: windowStart, frame: BenchmarkStats.of(windowFrames), thermal: hooks.thermal())
        windows.append(window)
        windowFrames.removeAll(keepingCapacity: true)
        windowStart = second
        progress(window)
      } else if frameMs.count % ModelConstants.runFrequency == 0
        || ProcessInfo.processInfo.systemUptime - progressAt >= EngineHost.benchmarkProgressInterval
      {
        progress(windows.last)
      }
    }
    if !windowFrames.isEmpty {
      windows.append(
        BenchmarkWindow(startSecond: windowStart, frame: BenchmarkStats.of(windowFrames), thermal: hooks.thermal()))
    }
    return BenchmarkReport(
      sha256: l.sha256, device: device, seconds: round2(ProcessInfo.processInfo.systemUptime - t0), frames: frameMs.count,
      frame: BenchmarkStats.of(frameMs), accelerator: BenchmarkStats.of(accelMs), queues: BenchmarkStats.of(queueMs), output: BenchmarkStats.of(outMs),
      build: build, over35: frameMs.filter { $0 > 35 }.count, over50: frameMs.filter { $0 > 50 }.count, windows: windows,
      thermalAtStart: thermalAtStart, thermalAtEnd: hooks.thermal(), cancelled: run.cancelled)
  }
}
