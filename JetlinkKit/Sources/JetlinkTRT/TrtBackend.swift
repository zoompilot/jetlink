import CTrt
import Foundation
import JetlinkONNX
import JetlinkServer

#if canImport(Android)
  import Android
#endif

/// TensorRT on an NVIDIA GPU: the Jetson's backend, and a PC's. The Swift
/// form of the Python server's trt/__init__.py and trt/build.py.
///
/// Artifacts are plans, `engines/<sha16>.<tag>.plan`, with the tag Python
/// wrote (`trt<version>.<device>-sm<cc>`) so the plans already on a Jetson
/// load without a rebuild.
///
/// Two TensorRT generations build from this one file, as in Python. JetPack
/// ships 10.x, which builds a weakly typed network with the FP16 flag. PCs run 11.x, which dropped weak typing and the precision flags
/// with it: precision follows the ONNX, which is fp16 end to end.
public final class TrtBackend: EngineBackend {
  public let name = "trt"
  public let suffix = ".plan"
  public let trt: TensorRT
  let gpuTiming: Bool
  let faultAfter: Int?
  /// MemAvailable, which the workspace is sized from.
  let available: @Sendable () -> Int
  private let log = ServerLog(category: "builder")

  /// A ceiling, not an allocation: TensorRT picks tactics that fit inside
  /// it. A flat 4 GB on an 8 GB Orin met the OOM killer on a 1.76 GB model,
  /// so it is sized from what is free.
  static let maxWorkspace = 4 << 30
  static let minWorkspace = 256 << 20
  static let workspaceFraction = 0.4
  static let optimizationLevel = 3

  /// `gpuTiming` times every launch with CUDA events and logs the spread
  /// (section 7's pure GPU time); `faultAfter` makes every frame after the
  /// first N throw a sticky CUDA error, to exercise the fatal exit and
  /// restart (H6, D15). `available` is MemAvailable in bytes, 0 for unknown.
  public init(trt: TensorRT, gpuTiming: Bool = false, faultAfter: Int? = nil, available: @escaping @Sendable () -> Int) {
    self.trt = trt
    self.gpuTiming = gpuTiming
    self.faultAfter = faultAfter
    self.available = available
  }

  public var runtimeVersion: String { trt.version }

  /// The hardware a plan is valid for. The compute capability is what
  /// matters; the name makes the file readable.
  public func deviceTag() -> String {
    sanitize("\(trt.deviceName)-sm\(trt.computeCapability.major)\(trt.computeCapability.minor)")
  }

  /// The hello's and the status page's; the comma logs trt_version.
  public func describe() -> [String: String] {
    ["backend": name, "runtime_version": runtimeVersion, "device": deviceTag(), "trt_version": trt.fullVersion]
  }

  /// `engines/timing.<tag>.cache`: kernel timings mostly do not depend on
  /// the model (a warm cache cut a Lebowski build from 254 s to 173 s), but
  /// a timing from another TensorRT build or chip is not one.
  func timingCache(beside artifact: URL) -> URL {
    artifact.deletingLastPathComponent().appending(path: "timing.\(tag()).cache")
  }

  public func deriveSpec(model: URL, sha256: String, nbytes: Int64, frameSkip: Int) throws -> ModelSpec {
    try ONNXPreparer().readSpec(model: model, sha256: sha256, nbytes: nbytes, frameSkip: frameSkip)
  }

  static func workspaceBytes(available: Int) -> Int {
    guard available > 0 else { return maxWorkspace }
    return max(minWorkspace, min(maxWorkspace, Int(Double(available) * workspaceFraction)))
  }

  // MARK: build

  /// Patch, parse and build, with Python's progress stages and messages;
  /// the plan is staged and moved into place, the sidecar beside it.
  public func build(model: URL, artifact: URL, report: @escaping ProgressFn, metaExtra: [String: Any]) throws {
    try build(model: model, artifact: artifact, report: report, metaExtra: metaExtra, shouldStop: { false })
  }

  /// TensorRT asks its progress monitor at every step whether to go on, so
  /// `shouldStop` ends a build within a step: an add-on's, for the driving model's.
  public func build(
    model: URL, artifact: URL, report: @escaping ProgressFn, metaExtra: [String: Any], shouldStop: @escaping @Sendable () -> Bool
  ) throws {
    let started = Date()
    let free = available()
    let workspace = Self.workspaceBytes(available: free)
    log.info("building with a \(workspace >> 20) MB workspace (\(free >> 20) MB available)")
    let cacheURL = timingCache(beside: artifact)
    try Artifact.build(artifact, metaExtra: metaExtra, report: report) { staged in
      report("patch", 0, "retyping uint8 image inputs to fp16")
      let prepared = try CoreMLPreparation.prepare(source: model, into: staged.deletingLastPathComponent(), layout: .plain, cacheKey: { _ in "" })
      report("patch", 1, "patched")

      var build: OpaquePointer?
      try trt.check { jl_trt_build_create(trt.handle, &build, $0, $1) }
      defer { jl_trt_build_destroy(build) }
      report("parse", 0, "parsing onnx")
      do {
        try trt.check(capacity: 16 << 10) { jl_trt_build_parse(build, prepared.parts[0].url.path, $0, $1) }
      } catch let error as TrtError where !error.isFatal {
        throw TrtError(code: error.code, "onnx parse failed:\n\(error.description)")
      }
      report("parse", 1, "\(jl_trt_build_layers(build)) layers")

      let precision: String
      if trt.stronglyTyped {
        precision = "strongly typed network; precision follows the ONNX"
      } else {
        try trt.check { jl_trt_build_set_fp16(build, $0, $1) }
        precision = "fp16 enabled"
      }
      log.info("tensorrt \(runtimeVersion): \(precision)")
      jl_trt_build_set_optimization_level(build, Int32(Self.optimizationLevel))
      jl_trt_build_set_workspace(build, workspace)
      let monitor = BuildMonitor(report, shouldStop: shouldStop) { [build] in jl_trt_build_cancel(build) }
      let cached = attachTimingCache(build!, cacheURL)

      report("build", 0, "building engine")
      try withExtendedLifetime(monitor) {
        jl_trt_build_set_progress(build, forwardProgress, Unmanaged.passUnretained(monitor).toOpaque())
        defer { jl_trt_build_set_progress(build, nil, nil) }
        do {
          try trt.check { jl_trt_build_write_plan(build, staged.path, $0, $1) }
        } catch let error as TrtError where monitor.cancelled {
          throw TrtError(code: error.code, "the build was stopped")
        } catch let error as TrtError where !error.isFatal && !error.description.hasPrefix("cannot write") {
          log.error("\(error.description)")
          throw TrtError(code: error.code, "TensorRT returned no engine; see the build log")
        }
      }
      if cached {
        saveTimingCache(build!, cacheURL)
      }

      var meta = Artifact.meta(self, runtimeKey: "trt_version", model: model, started: started)
      meta["fp16"] = true
      meta["strongly_typed"] = trt.stronglyTyped
      meta["optimization_level"] = Self.optimizationLevel
      return meta
    }
  }

  /// Seeds the tactic timings from the last build. Fails open: another
  /// build's cache or a killed build's truncated one is replaced by an empty
  /// one, which this build fills and saves. (Python kept it detached, so
  /// every later build stayed cold.) False when not even that attaches.
  private func attachTimingCache(_ build: OpaquePointer, _ url: URL) -> Bool {
    let data = (try? Data(contentsOf: url)) ?? Data()
    do {
      try data.withUnsafeBytes { bytes in
        try trt.check { jl_trt_build_set_timing_cache(build, bytes.baseAddress, bytes.count, $0, $1) }
      }
      return true
    } catch {
      log.warning("timing cache unusable (\(error)), building cold")
    }
    do {
      try trt.check { jl_trt_build_set_timing_cache(build, nil, 0, $0, $1) }
      return true
    } catch {
      log.warning("no timing cache for this build: \(error)")
      return false
    }
  }

  /// Atomic like the plan: a build killed mid-write would leave a truncated
  /// cache for the next one to read.
  private func saveTimingCache(_ build: OpaquePointer, _ url: URL) {
    let temp = url.deletingLastPathComponent().appending(path: url.lastPathComponent + ".tmp")
    do {
      try trt.check { jl_trt_build_write_timing_cache(build, temp.path, $0, $1) }
      guard rename(temp.path, url.path) == 0 else {
        throw TrtError("cannot rename \(temp.lastPathComponent): \(String(cString: strerror(errno)))")
      }
    } catch {
      try? FileManager.default.removeItem(at: temp)
      log.warning("could not write the timing cache: \(error)")
    }
  }

  // MARK: load

  /// The host reports "deserializing engine" before and "ready" after, as
  /// Python's did; a plan TensorRT refuses is `ArtifactInvalid`.
  public func load(artifact: URL, report: @escaping ProgressFn) throws -> any Engine {
    try TrtEngine(plan: artifact, trt: trt, gpuTiming: gpuTiming, faultAfter: faultAfter)
  }
}

/// TensorRT's build phases as one fraction, as trt/build.py's _Monitor made
/// them: the root phase's step over its step count, named by the phase, sent
/// again at every nested phase's start and step too. Never stops the build.
final class BuildMonitor {
  private let report: ProgressFn
  private let shouldStop: @Sendable () -> Bool
  private let cancel: () -> Void
  private var root: (phase: String, step: Int, total: Int)?
  /// The build was told to stop: TensorRT ends it at its next step.
  private(set) var cancelled = false

  init(_ report: @escaping ProgressFn, shouldStop: @escaping @Sendable () -> Bool = { false }, cancel: @escaping () -> Void = {}) {
    self.report = report
    self.shouldStop = shouldStop
    self.cancel = cancel
  }

  func event(_ kind: Int32, phase: String, parent: String?, value: Int) {
    if !cancelled && shouldStop() {
      cancelled = true
      cancel()
    }
    switch kind {
    case Int32(JL_TRT_PHASE_START):
      if parent == nil { root = (phase, 0, value) }
    case Int32(JL_TRT_PHASE_STEP):
      if phase == root?.phase { root?.step = value }
    default:
      if phase == root?.phase { root = nil }
      return
    }
    guard let root else { return }
    let frac = root.total != 0 ? Double(root.step) / Double(root.total) : 0
    report("build", min(max(frac, 0), 1), root.phase)
  }
}

/// The shim's progress calls, one at a time, into the build's monitor.
private func forwardProgress(
  _ ctx: UnsafeMutableRawPointer?, _ event: Int32, _ phase: UnsafePointer<CChar>?, _ parent: UnsafePointer<CChar>?, _ value: Int32
) {
  guard let ctx, let phase else { return }
  Unmanaged<BuildMonitor>.fromOpaque(ctx).takeUnretainedValue()
    .event(event, phase: String(cString: phase), parent: parent.map { String(cString: $0) }, value: Int(value))
}
