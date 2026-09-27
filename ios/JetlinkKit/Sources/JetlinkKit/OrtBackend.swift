// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.
//
// onnxruntime with CoreML: the Python OrtBackend's build and load, on a phone.
//
// An artifact is a directory, as on the Mac: the prepared ONNX, a manifest
// naming the session, and onnxruntime's CoreML cache. The model carries a
// COREML_CACHE_KEY so the compile done under the build's temporary
// directory is found again from the artifact's final path; without it
// onnxruntime keys the cache on the path and compiles everything twice.

import Foundation

public typealias ProgressFn = @Sendable (_ stage: String, _ frac: Double, _ msg: String) -> Void

/// The artifact on disk is not one this backend can load; the host deletes
/// it and rebuilds from the ONNX. See backends.base.ArtifactInvalid.
public struct ArtifactInvalid: Error, CustomStringConvertible {
  public let description: String
  init(_ d: String) { description = d }
}

/// The accelerator's name for cache keys and the hello. A Mac says what the
/// Python says ("Apple M1 Pro"), so artifact names line up; a phone gives its
/// model identifier ("iPhone17,1"), which pins the chip exactly.
public func chipName() -> String {
  func sysctlString(_ name: String) -> String? {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
    var buf = [CChar](repeating: 0, count: size)
    guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return nil }
    let s = cText(buf).trimmingCharacters(in: .whitespaces)
    return s.isEmpty ? nil : s
  }
  #if os(macOS)
    return sysctlString("machdep.cpu.brand_string") ?? sysctlString("hw.model") ?? "unknown"
  #else
    // hw.machine is the model identifier on iOS; the simulator reports the Mac.
    if let sim = ProcessInfo.processInfo.environment["SIMULATOR_MODEL_IDENTIFIER"] { return "\(sim)-simulator" }
    return sysctlString("hw.machine") ?? "unknown"
  #endif
}

public final class OrtBackend: Sendable {
  public let units: ComputeUnits
  public let name = "ort"
  /// Bumped whenever OnnxPrepare changes what it writes, so an artifact
  /// prepared by an older app is rebuilt rather than trusted.
  public static let prepareVersion = 3
  static let manifestName = "sessions.json"
  /// The last resort for the compile stage's fraction; see the Python backend.
  static let expectedCoreMLSeconds = 10.0
  static let compiledDir = "compiled_model.mlmodelc"
  static let log = JLogger("jetlink.ort")

  public init(units: ComputeUnits) { self.units = units }

  public var runtimeVersion: String { OrtRuntime.version }
  public var deviceTag: String { sanitize("\(units.rawValue)-\(chipName())") }
  public var tag: String { "ort\(sanitize(runtimeVersion)).\(deviceTag)" }
  public var onCoreML: Bool { units != .cpu }

  public func describe() -> [String: JSON] {
    ["backend": "ort", "runtime_version": .string(runtimeVersion), "device": .string(deviceTag)]
  }

  /// onnxruntime wants the key alphanumeric and under 64 characters.
  static func cacheKey(_ artifact: URL, _ part: String) -> String {
    let stem = artifact.deletingPathExtension().lastPathComponent + part
    return String(String(stem.unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) && $0.isASCII }).prefix(63))
  }

  static func sidecar(_ artifact: URL) -> [String: JSON] {
    (try? JSON.parse(Data(contentsOf: artifact.deletingPathExtension().appendingPathExtension("json"))))?.object ?? [:]
  }

  // MARK: build

  public func build(onnx: URL, out: URL, report: ProgressFn?, metaExtra: [String: JSON]) throws {
    let report = report ?? { _, _, _ in }
    let t0 = monotonic()
    let fm = FileManager.default
    let parent = out.deletingLastPathComponent()
    try fm.createDirectory(at: parent, withIntermediateDirectories: true)
    var template = Array(parent.appendingPathComponent("tmpXXXXXX").path.utf8CString)
    guard mkdtemp(&template) != nil else { throw EngineError("cannot make a build directory in \(parent.path)") }
    let tmp = URL(fileURLWithPath: cText(template), isDirectory: true)
    defer { try? fm.removeItem(at: tmp) }
    let staged = tmp.appendingPathComponent("artifact", isDirectory: true)
    try fm.createDirectory(at: staged, withIntermediateDirectories: false)

    report("patch", 0.0, "preparing the model for onnxruntime")
    let prep = try OnnxPrepare.prepare(
      src: onnx.path, dst: staged.appendingPathComponent("model.onnx").path, forANE: units == .ane, forCoreML: onCoreML,
      cacheKey: OrtBackend.cacheKey(out, "model"))
    OrtBackend.log.info("prepared \(onnx.lastPathComponent): \(prep)")
    let manifest: JSON = [
      [
        "model": "model.onnx", "units": .str(units.mlComputeUnits), "cache": onCoreML ? "coreml" : .null,
      ]
    ]
    let cache = staged.appendingPathComponent("coreml", isDirectory: true)
    if onCoreML { try fm.createDirectory(at: cache, withIntermediateDirectories: false) }
    try atomicWrite(manifest.data, to: staged.appendingPathComponent(OrtBackend.manifestName))
    report("patch", 1.0, "prepared")

    var stages: [String: JSON] = [:]
    let tEngine = monotonic()
    let engine: OrtEngine
    if onCoreML {
      let progress = CoreMLProgress(caches: [cache], weightsBytes: prep.weightsBytes, expect: OrtBackend.sidecar(out))
      report("convert", 0.0, "converting for CoreML")
      let ticker = CoreMLTicker(report: report, progress: progress, initial: "convert")
      engine = try Ticking.during(ticker.tick) {
        try OrtEngine(model: staged.appendingPathComponent("model.onnx").path, units: units, cacheDirectory: cache.path, keepAlive: false)
      }
      let (converted, compiled) = progress.measure()
      stages = [
        "convert_bytes": .int(converted), "compile_bytes": .int(compiled),
        "compile_seconds": .double(pyRound(monotonic() - tEngine, 1)),
      ]
      if ticker.stage != "compile" { report("convert", 1.0, "converted \(sizeText(converted))") }
      report("compile", 1.0, "compiled in \(secondsText(monotonic() - tEngine))")
    } else {
      report("build", 0.0, "creating the onnxruntime session (\(units.rawValue))")
      engine = try OrtEngine(model: staged.appendingPathComponent("model.onnx").path, units: units, cacheDirectory: nil)
      report("build", 1.0, "sessions created in \(Int(monotonic() - t0)) s")
    }
    // Prove it runs before calling it built.
    defer { engine.close() }
    try engine.run()
    let providers = engine.providers
    engine.close()
    OrtBackend.log.info("onnxruntime providers in use: \(providers)")
    if onCoreML {
      let freed = OrtBackend.dropConvertedModels(cache)
      if freed > 0 { OrtBackend.log.info("removed \(sizeText(freed)) of converted model the compiled one replaces") }
    }

    if fm.fileExists(atPath: out.path) { try fm.removeItem(at: out) }
    try fm.moveItem(at: staged, to: out)

    var meta: [String: JSON] = [
      "backend": "ort", "onnxruntime": .string(runtimeVersion), "device": .string(deviceTag),
      "sessions": manifest, "providers": .array([.array(providers.split(separator: ",").map { .string(String($0)) })]),
      "build_seconds": .double(pyRound(monotonic() - t0, 1)), "onnx": .string(onnx.lastPathComponent),
      "built_at": .string(isoNow()), "swift_prepare": .int(OrtBackend.prepareVersion),
      "prepare": .string(prep.description),
    ]
    meta.merge(stages) { _, b in b }
    meta.merge(metaExtra) { _, b in b }
    try atomicWrite(JSON.object(meta).data, to: out.deletingPathExtension().appendingPathExtension("json"))
    report("build", 1.0, "done in \(meta["build_seconds"]!.double!)s")
  }

  /// Deletes the MLProgram onnxruntime converted each partition to, once the
  /// compile of it exists beside it. A load from the cache reads only the
  /// compiled model: measured on an M1 Pro with 1.29.0, the artifact went
  /// from 2.2 GB to 1.4 GB and loaded in 0.5 s with outputs bit for bit the
  /// same. Returns the bytes freed.
  static func dropConvertedModels(_ cache: URL) -> Int {
    let fm = FileManager.default
    var freed = 0
    guard let walk = fm.enumerator(at: cache, includingPropertiesForKeys: [.isDirectoryKey]) else { return 0 }
    var converted: [URL] = []
    for case let u as URL in walk where u.lastPathComponent == "Data" {
      let compiled = u.deletingLastPathComponent().appendingPathComponent(compiledDir)
      if fm.fileExists(atPath: compiled.path) { converted.append(u) }
      walk.skipDescendants()
    }
    for u in converted {
      freed += treeBytes(u).0
      try? fm.removeItem(at: u)
    }
    return freed
  }

  // MARK: load

  public func load(artifact: URL, report: ProgressFn?) throws -> OrtEngine {
    let manifest: [JSON]
    do {
      manifest = try JSON.parse(Data(contentsOf: artifact.appendingPathComponent(OrtBackend.manifestName))).array ?? []
    } catch {
      throw ArtifactInvalid("\(artifact.lastPathComponent): no readable \(OrtBackend.manifestName) inside (\(error))")
    }
    guard manifest.count == 1, let entry = manifest.first, let model = entry["model"]?.string else {
      throw ArtifactInvalid("\(artifact.lastPathComponent): \(OrtBackend.manifestName) names no single session")
    }
    var sidecar = OrtBackend.sidecar(artifact)
    if sidecar["swift_prepare"]?.int != OrtBackend.prepareVersion {
      throw ArtifactInvalid("\(artifact.lastPathComponent): prepared by another version of the model preparation")
    }
    if onCoreML && sidecar["compile_bytes"] == nil {
      throw ArtifactInvalid("\(artifact.lastPathComponent): the build never recorded a CoreML compile")
    }
    let modelPath = artifact.appendingPathComponent(model)
    guard FileManager.default.fileExists(atPath: modelPath.path) else {
      throw ArtifactInvalid("\(artifact.lastPathComponent): no \(model) inside")
    }
    var cache: URL?
    if let c = entry["cache"]?.string {
      let dir = artifact.appendingPathComponent(c, isDirectory: true)
      let key = OrtBackend.cacheKey(artifact, (model as NSString).deletingPathExtension)
      // A missing or empty compile would make onnxruntime recompile for
      // minutes under a "loading" that never moves. A rebuild reports progress.
      guard FileManager.default.fileExists(atPath: dir.appendingPathComponent(key).path) else {
        throw ArtifactInvalid("\(artifact.lastPathComponent): the CoreML cache has no compile for \(key)")
      }
      cache = dir
    }
    let t0 = monotonic()
    let expected = sidecar["load_seconds"]?.double ?? 0
    report?("load", 0.0, "loading the CoreML model")
    let tick: @Sendable (Double) -> Void = { elapsed in
      if expected > 0 {
        report?("load", min(0.95, elapsed / expected), "loading the CoreML model, \(Int(elapsed)) s of about \(Int(expected.rounded())) s")
      } else {
        report?("load", 0.0, "loading the CoreML model, \(Int(elapsed)) s elapsed")
      }
    }
    let engine = try Ticking.during(tick) {
      try OrtEngine(model: modelPath.path, units: units, cacheDirectory: cache?.path)
    }
    let took = monotonic() - t0
    OrtBackend.log.info("onnxruntime session on \(units.label) in \(String(format: "%.1f", took)) s, providers \(engine.providers)")
    report?("load", 1.0, "loaded in \(secondsText(took))")
    sidecar["load_seconds"] = .double(pyRound(took, 1))
    try? atomicWrite(JSON.object(sidecar).data, to: artifact.deletingPathExtension().appendingPathExtension("json"))
    return engine
  }
}

// MARK: - progress while CoreML works

/// What a CoreML session creation is doing, read off the cache directory:
/// onnxruntime writes the converted MLProgram, then compiles it into
/// compiled_model.mlmodelc. Which is growing is the stage; the bytes written
/// against the last build's are the fraction. See the Python CoreMLProgress.
final class CoreMLProgress: @unchecked Sendable {
  let caches: [URL]
  let weights: Int
  let expect: [String: JSON]

  init(caches: [URL], weightsBytes: Int, expect: [String: JSON]) {
    self.caches = caches
    weights = weightsBytes
    self.expect = expect
  }

  func measure() -> (Int, Int) {
    var converted = 0, compiled = 0
    for c in caches {
      let (out, inside) = treeBytes(c, split: OrtBackend.compiledDir)
      converted += out
      compiled += inside
    }
    return (converted, compiled)
  }

  func tick(_ elapsed: Double) -> (String, Double, String) {
    let (converted, compiled) = measure()
    if compiled == 0 {
      let total = weights > 0 ? weights : (expect["convert_bytes"]?.int ?? 0)
      let frac = total > 0 ? min(0.95, Double(converted) / Double(total)) : 0
      let of = total > 0 ? " of \(sizeText(total))" : ""
      return ("convert", frac, "converting for CoreML, \(sizeText(converted))\(of) written")
    }
    if let total = expect["compile_bytes"]?.int, total > 0 {
      return (
        "compile", min(0.95, Double(compiled) / Double(total)),
        "compiling for CoreML, \(sizeText(compiled)) of \(sizeText(total)) written"
      )
    }
    let took = expect["compile_seconds"]?.double ?? OrtBackend.expectedCoreMLSeconds
    return (
      "compile", min(0.95, elapsed / max(took, 0.1)),
      "compiling for CoreML, \(sizeText(compiled)) written, \(secondsText(elapsed)) elapsed"
    )
  }
}

/// Reports every tick, and closes a stage with its 100 % line when the next
/// one starts, as the Python coreml_ticker does.
final class CoreMLTicker: @unchecked Sendable {
  let report: ProgressFn
  let progress: CoreMLProgress
  private(set) var stage: String?
  private var since = 0.0
  private let lock = NSLock()

  init(report: @escaping ProgressFn, progress: CoreMLProgress, initial: String?) {
    self.report = report
    self.progress = progress
    stage = initial
  }

  func tick(_ elapsed: Double) {
    let (s, frac, msg) = progress.tick(elapsed)
    lock.lock()
    if let prev = stage, prev != s {
      let done = elapsed - since
      OrtBackend.log.info("\(prev) finished in \(Int(done)) s")
      report(prev, 1.0, "\(prev) done in \(Int(done)) s")
      since = elapsed
    }
    stage = s
    lock.unlock()
    OrtBackend.log.debug("\(s): \(msg)")
    report(s, frac, msg)
  }
}

enum Ticking {
  /// Runs `body`, calling `tick(elapsed)` every two seconds on another thread
  /// until it returns.
  static func during<T>(_ tick: @escaping @Sendable (Double) -> Void, every: Double = 2.0, _ body: () throws -> T) rethrows -> T {
    let cond = NSCondition()
    let exited = DispatchSemaphore(value: 0)
    nonisolated(unsafe) var done = false
    let t0 = monotonic()
    let thread = Thread {
      cond.lock()
      while !done {
        if !cond.wait(until: Date(timeIntervalSinceNow: every)) && !done {
          cond.unlock()
          tick(monotonic() - t0)
          cond.lock()
        }
      }
      cond.unlock()
      exited.signal()
    }
    thread.name = "jetlink-progress"
    thread.start()
    defer {
      cond.lock()
      done = true
      cond.signal()
      cond.unlock()
      // No tick may land after the caller has moved on to its next stage.
      exited.wait()
    }
    return try body()
  }
}

func sizeText(_ n: Int) -> String {
  Double(n) >= 1e9 ? String(format: "%.1f GB", Double(n) / 1e9) : String(format: "%.0f MB", Double(n) / 1e6)
}

func secondsText(_ s: Double) -> String { s >= 90 ? String(format: "%.0f min", s / 60) : String(format: "%.0f s", s) }

func isoNow() -> String {
  let f = ISO8601DateFormatter()
  f.formatOptions = [.withInternetDateTime]
  return f.string(from: Date())
}
