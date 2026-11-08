import Foundation
import JetlinkKit
import JetlinkONNX
import JetlinkRegistry
import JetlinkServer

/// Where onnxruntime runs a model: the sessions the graph is prepared into
/// and what runs each, CoreML's units on Apple platforms, QNN's on a
/// Snapdragon, the CPU provider anywhere. The raw value is what `--device`
/// and the apps' settings name, and the device part of the artifact's tag.
public enum OrtProfile: String, CaseIterable, Sendable {
  /// The vision trunk on the Neural Engine, everything after it on the GPU.
  case ane
  /// The whole graph on the GPU, for when something else holds the Neural Engine.
  case coreml
  /// The whole graph as one CoreML program with every compute unit allowed,
  /// prepared for the Neural Engine (the policy's norms prescaled, the vision
  /// heads in fp32): what a phone, whose GPU is far weaker than its Neural
  /// Engine, wants.
  case aneWhole = "ane-whole"
  /// The vision trunk on the Hexagon NPU in fp16, everything after it on the
  /// Adreno GPU.
  case htp
  /// The whole graph on the NPU, prepared as `ane-whole`. The NPU computes in
  /// fp16 whatever the graph says, so its fp32 heads are fp16 there too, and
  /// only the parity check can say whether that is close enough.
  case htpWhole = "htp-whole"
  /// Same-frame causal work: HTP vision and GPU history, then HTP decode.
  case htpParallel = "htp-parallel"
  /// The whole graph on the Adreno GPU.
  case gpu
  /// onnxruntime's CPU provider alone: for tests, the emulator and Linux,
  /// and nowhere near the frame budget with a real model.
  case cpu

  /// What this build's onnxruntime runs, the default first.
  public static var available: [OrtProfile] {
    #if canImport(Metal)
      [.ane, .aneWhole, .coreml, .cpu]
    #elseif os(Android)
      [.htp, .htpWhole, .htpParallel, .gpu, .cpu]
    #else
      [.cpu]
    #endif
  }

  /// (session name, unit) in run order.
  var sessions: [(name: String, unit: OrtUnit)] {
    switch self {
    case .ane: [("vision", .coreML("CPUAndNeuralEngine")), ("policy", .coreML("CPUAndGPU"))]
    case .coreml: [("model", .coreML("CPUAndGPU"))]
    case .aneWhole: [("model", .coreML("ALL"))]
    case .htp: [("vision", .htp), ("policy", .qnnGPU)]
    case .htpWhole: [("model", .htp)]
    case .htpParallel: [("vision", .htp), ("history", .qnnGPU), ("decode", .htp), ("join", .cpu)]
    case .gpu: [("model", .qnnGPU)]
    case .cpu: [("model", .cpu)]
    }
  }

  /// How the preparation lays the graph out for those sessions. On Android
  /// the CPU's is the graph without CoreML's rewrites: onnxruntime's Android
  /// build runs the rewritten fp16 Gemm on one thread, a hundred times slower
  /// than the MatMul it replaces. Apple's build runs it as fast, and the
  /// goldens hold the Mac's CPU to Python's outputs bit for bit.
  var layout: CoreMLPreparation.Layout {
    switch self {
    case .ane, .htp: .split
    case .aneWhole, .htpWhole: .aneWhole
    case .htpParallel: .qnnPrefillBase
    case .coreml, .gpu: .whole
    case .cpu:
      #if os(Android)
        .plain
      #else
        .whole
      #endif
    }
  }

  /// What a build writes: 5 is every graph split on `ane` and Expand as Tile
  /// on both, 6 the plain graph. An artifact prepared under another rebuilds.
  public var prepareVersion: Int {
    self == .htpParallel ? 11 : (layout == .plain ? 6 : 5)
  }

  var usesCoreML: Bool {
    sessions.contains { if case .coreML = $0.unit { true } else { false } }
  }
}

/// What runs one session, and how the artifact's manifest names it.
enum OrtUnit: Equatable {
  /// CoreML on these compute units, its compiled model kept in the artifact.
  case coreML(String)
  /// QNN on the Hexagon NPU, compiled into onnxruntime's EP context at build.
  case htp
  /// QNN on the Adreno GPU.
  case qnnGPU
  case cpu

  /// A manifest entry's unit. QNN and CPU sessions name theirs; a CoreML one
  /// names its compute units, or null for the CPU in older artifacts.
  init?(entry: [String: Any]) {
    switch entry["unit"] as? String {
    case "htp": self = .htp
    case "gpu": self = .qnnGPU
    case "cpu": self = .cpu
    case nil where entry.keys.contains("units"): self = (entry["units"] as? String).map(OrtUnit.coreML) ?? .cpu
    default: return nil
    }
  }

  func entry(model file: String, session: String) -> [String: Any] {
    switch self {
    case .coreML(let units): ["model": file, "units": units, "cache": "coreml-\(session)"]
    case .htp: ["model": file, "unit": "htp"]
    case .qnnGPU: ["model": file, "unit": "gpu"]
    case .cpu: ["model": file, "unit": "cpu"]
    }
  }
}

/// onnxruntime in process, on one of its profiles: the Swift form of the
/// Python ort backend. The artifact is a directory of the sessions' models
/// and their manifest (OrtArtifact), CoreML's compiled models or QNN's EP
/// context beside them.
///
/// The causal profile uses a model-specific contract and generic QNN provider
/// discovery. A phone/runtime still needs accuracy and sustained qualification.
public final class OrtBackend: EngineBackend {
  /// A first NPU compile with no earlier build to go by. The QNN graph
  /// finalization of a big model is minutes on a phone (unmeasured).
  static let expectedCompileSeconds = 180.0

  public let name = "ort"
  /// A directory: the prepared model and the compiled caches.
  public let suffix = ".ortcache"
  public let profile: OrtProfile
  /// Keep the GPU clocked up between frames (Metal), or the NPU in burst
  /// mode rather than sustained (QNN).
  public let keepAlive: Bool
  /// Keep a CPU core warm while the Neural Engine or the NPU runs a frame.
  public let keepCPUWarm: Bool
  public let qnn: QnnPolicy
  private let preparer: any ModelPreparer
  private let chip: String
  private let causalBundle: URL?
  private let log = ServerLog(category: "ort")

  /// `chip` names what the artifacts are valid for: on Android the SoC's
  /// model, "SM8650" (Build.SOC_MODEL), which a QNN context is compiled for.
  /// Nil is the Apple chip on Apple platforms and "cpu" elsewhere.
  public init(
    profile: OrtProfile, preparer: any ModelPreparer, keepAlive: Bool = true, keepCPUWarm: Bool = true,
    chip: String? = nil, qnn: QnnPolicy = .init(), causalBundle: URL? = nil
  ) {
    self.profile = profile
    self.preparer = preparer
    self.keepAlive = keepAlive
    self.keepCPUWarm = keepCPUWarm
    self.chip = HostChip.resolve(chip)
    self.qnn = profile == .htpParallel ? QnnPolicy(
      sharedIO: qnn.sharedIO, strict: true, disableAdvancedActivationFusion: qnn.disableAdvancedActivationFusion,
      libraryDirectory: qnn.libraryDirectory, runtimeIdentity: qnn.runtimeIdentity) : qnn
    self.causalBundle = causalBundle
  }

  public var runtimeVersion: String { OrtRuntime.version }

  public func deviceTag() -> String {
    let contract = profile == .htpParallel ? "-\(CausalPrefillBundle.identity)" : ""
    return sanitize("\(profile.rawValue)-\(chip)\(contract)\(profile.sessions.contains(where: { $0.unit == .htp }) ? qnn.cacheSuffix : "")")
  }

  public func deriveSpec(model: URL, sha256: String, nbytes: Int64, frameSkip: Int) throws -> ModelSpec {
    try preparer.readSpec(model: model, sha256: sha256, nbytes: nbytes, frameSkip: frameSkip)
  }

  /// QNN's options for a unit, as onnxruntime 1.29 names them.
  func providerOptions(_ unit: OrtUnit) -> [String: String] {
    switch unit {
    case .htp:
      var options = qnn.htpBackend.merging([
        // burst holds the clocks up between frames, the counterpart of the
        // Mac's GPU keep-alive; sustained trades speed for less heat.
        "htp_performance_mode": keepAlive ? "burst" : "sustained_high_performance",
        // the slowest finalization and the fastest graph: it runs once, at build
        "htp_graph_finalization_optimization_mode": "3",
        "enable_htp_fp16_precision": "1",
        "qnn_context_priority": "high",
      ]) { _, value in value }
      if qnn.sharedIO { options["enable_htp_shared_memory_allocator"] = "1" }
      return options
    case .qnnGPU: return ["backend_type": "gpu"]
    case .coreML, .cpu: return [:]
    }
  }

  /// A session of `unit` on `model`, with today's options: the performance
  /// mode is a setting, not part of the artifact.
  func plan(_ unit: OrtUnit, model: URL, cache: URL?) -> SessionPlan {
    switch unit {
    case .coreML(let units): SessionPlan(model: model, computeUnits: units, cacheDirectory: cache)
    case .htp: SessionPlan(
      model: model, provider: "QNN", options: providerOptions(unit), config: qnn.sessionConfig,
      label: "QNN(htp)", usesNeuralEngine: true)
    case .qnnGPU: SessionPlan(
      model: model, provider: "QNN", options: providerOptions(unit),
      config: profile == .htpParallel ? qnn.sessionConfig : [:], label: "QNN(gpu)", usesGPU: true)
    case .cpu: SessionPlan(model: model, provider: nil, threads: HostChip.cpuThreads, label: "CPU")
    }
  }

  private func plans(_ artifact: URL, _ manifest: [[String: Any]]) -> [SessionPlan] {
    manifest.map { entry in
      plan(
        OrtUnit(entry: entry)!, model: artifact.appending(path: entry["model"] as! String),
        cache: (entry["cache"] as? String).map { artifact.appending(path: $0, directoryHint: .isDirectory) })
    }
  }

  /// Choose before preparation, not after an arbitrary inference error.
  func effectiveProfile(for sha256: String) -> OrtProfile {
    profile == .htpParallel && sha256 != CausalPrefillBundle.sourceSHA ? .htpWhole : profile
  }

  private func wholeFallback() -> OrtBackend {
    OrtBackend(profile: .htpWhole, preparer: preparer, keepAlive: keepAlive, keepCPUWarm: keepCPUWarm, chip: chip, qnn: qnn)
  }

  /// The configured cache remains isolated; the sidecar and engine report the
  /// route actually used for this model. Unknown/old route metadata rebuilds.
  func artifactProfile(_ meta: [String: Any]) throws -> OrtProfile {
    guard let source = meta["causal_prefill_source"] as? String, CacheLayout.isSHA256(source),
      let route = (meta["causal_prefill_route"] as? String).flatMap(OrtProfile.init(rawValue:)),
      route == effectiveProfile(for: source)
    else { throw ArtifactInvalid("causal-prefill artifact has no matching source and execution route") }
    return route
  }

  private func engine(_ artifact: URL, _ manifest: [[String: Any]], keepAlive: Bool, keepCPUWarm: Bool) throws -> OrtEngine {
    try OrtEngine(
      plans: plans(artifact, manifest), device: deviceTag(), keepAlive: keepAlive, keepCPUWarm: keepCPUWarm,
      dataflow: profile == .htpParallel,
      accelerator: profile == .htpParallel ? "NPU + GPU" : (profile == .htpWhole ? "NPU" : ""))
  }

  // MARK: build

  public func build(model: URL, artifact: URL, report: @escaping ProgressFn, metaExtra: [String: Any]) throws {
    try qnn.validate()
    var extra = metaExtra
    if profile == .htpParallel {
      let source = try Registry.hashFile(model).0
      let route = effectiveProfile(for: source)
      extra["causal_prefill_source"] = source
      extra["causal_prefill_route"] = route.rawValue
      if route == .htpWhole {
        log.info("this model has no causal split; preparing the existing whole-NPU route")
        return try wholeFallback().build(model: model, artifact: artifact, report: report, metaExtra: extra)
      }
    }
    let started = Date()
    let expect = Artifact.sidecar(artifact)
    try Artifact.build(artifact, metaExtra: extra, report: report) { staged in
      try FileManager.default.createDirectory(at: staged, withIntermediateDirectories: true)
      report("patch", 0, profile.usesCoreML ? "preparing the model for CoreML" : "preparing the model")
      let prepared: PreparedModel
      if profile == .htpParallel {
        prepared = try CausalPrefillBundle.prepare(model: model, into: staged, bundle: causalBundle)
      } else {
        prepared = try preparer.prepare(model: model, into: staged, layout: profile.layout) {
          CoreMLPreparation.cacheKey(stem: artifact.deletingPathExtension().lastPathComponent, part: $0)
        }
      }
      log.info("prepared \(model.lastPathComponent): \(prepared.summary)")
      let sessions = profile.sessions
      guard prepared.parts.map(\.name) == sessions.map(\.name) else {
        throw HostError.failed("the preparation wrote \(prepared.parts.map(\.name)), expected \(sessions.map(\.name))")
      }
      report("patch", 1, "prepared")

      let took = (expect["compile_seconds"] as? NSNumber)?.doubleValue ?? OrtBackend.expectedCompileSeconds
      let compileStarted = Date()
      var manifest: [[String: Any]] = []
      for (session, part) in zip(sessions, prepared.parts) {
        let entry = session.unit.entry(model: part.file, session: session.name)
        switch session.unit {
        case .coreML:
          let cache = staged.appending(path: entry["cache"] as! String, directoryHint: .isDirectory)
          try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
          manifest.append(entry)
        case .htp:
          manifest.append(entry.merging(["model": try compileContext(session.name, file: part.file, in: staged, took: took, report: report)]) { $1 })
        case .qnnGPU, .cpu:
          manifest.append(entry)
        }
      }
      let compileSeconds = Date().timeIntervalSince(compileStarted)
      if sessions.contains(where: { $0.unit == .htp }) {
        report("compile", 1, "compiled in \(Int(compileSeconds.rounded())) s")
      }
      try OrtArtifact.writeManifest(manifest, in: staged)

      if profile.usesCoreML {
        return try checkCoreML(
          staged, manifest, weightBytes: prepared.parts.reduce(0) { $0 + $1.weightBytes }, expect: expect, model: model, started: started, report: report)
      }
      // Prove it runs before calling it built: minutes on a CPU, so it ticks.
      let providers = try Ticker.during(interval: 1, Ticker.paced("load", "loading the model to check it runs", took: 0, report: report)) {
        let engine = try self.engine(staged, manifest, keepAlive: false, keepCPUWarm: false)
        defer { engine.close() }
        try engine.run()
        return engine.providers
      }
      var meta = OrtArtifact.meta(self, manifest: manifest, providers: providers, model: model, started: started)
      meta["compile_seconds"] = pythonRound(compileSeconds, 1)
      meta["artifact_bytes"] = Files.size(of: staged)
      return meta
    }
  }

  /// Compiles an NPU session into its EP context, which then replaces its
  /// prepared model: the context carries the compiled graph and any nodes
  /// left to the CPU with their weights. Returns the file the session loads.
  private func compileContext(_ session: String, file: String, in staged: URL, took: Double, report: @escaping ProgressFn) throws -> String {
    let source = staged.appending(path: file)
    let context = staged.appending(path: "\(session)_ctx.onnx")
    report("compile", 0, "compiling \(session) for the NPU")
    try Ticker.during(interval: 2, Ticker.paced("compile", "compiling \(session) for the NPU", took: took, report: report)) {
      let config = contextConfiguration(context: context)
      _ = try OrtSession(model: source, provider: "QNN", options: providerOptions(.htp), config: config)
    }
    guard FileManager.default.fileExists(atPath: context.path) else {
      log.warning("onnxruntime wrote no EP context for \(session); each load will compile it again")
      return file
    }
    try? FileManager.default.removeItem(at: source)
    return context.lastPathComponent
  }

  /// The same strict provider contract applies during compilation and reload.
  func contextConfiguration(context: URL) -> [String: String] {
    qnn.sessionConfig.merging([
        "ep.context_enable": "1",
        "ep.context_file_path": context.path,
        // the context binary in a file of its own beside the model, not base64 inside it
        "ep.context_embed_mode": "0",
    ]) { _, value in value }
  }

  /// A CoreML build's first session creation is the conversion and CoreML's
  /// compile, so its progress is read off the cache directories as they fill.
  private func checkCoreML(
    _ staged: URL, _ manifest: [[String: Any]], weightBytes: Int64, expect: [String: Any], model: URL, started: Date, report: @escaping ProgressFn
  ) throws -> [String: Any] {
    let caches = manifest.compactMap { ($0["cache"] as? String).map { staged.appending(path: $0, directoryHint: .isDirectory) } }
    let progress = CoreMLProgress(caches: caches, weightBytes: weightBytes, expect: expect)
    report("convert", 0, "converting for CoreML")
    let engineStarted = Date()
    let tick: @Sendable (TimeInterval) -> Void = { elapsed in
      let (stage, frac, msg) = progress.tick(elapsed: elapsed)
      report(stage, frac, msg)
    }
    let engine = try Ticker.during(interval: 2, tick) {
      try self.engine(staged, manifest, keepAlive: false, keepCPUWarm: false)
    }
    defer { engine.close() }
    let (converted, compiled) = progress.measure()
    let compileSeconds = Date().timeIntervalSince(engineStarted)
    report("convert", 1, "converted \(formatBytes(converted))")
    report("compile", 1, "compiled in \(Int(compileSeconds.rounded())) s")
    // Prove it runs before calling it built.
    try engine.run()
    engine.close()
    // A load reads only the compiled model, so the MLProgram onnxruntime
    // converted each session to is dead weight beside it: on an M1 Pro with
    // 1.29.0 the artifact went from 2.2 GB to 1.4 GB, loading in 0.5 s with
    // outputs bit for bit the same. A phone has no room for both.
    let freed = CoreMLProgress.dropConvertedModels(caches)
    if freed > 0 {
      log.info("removed \(formatBytes(freed)) of converted model the compiled one replaces")
    }
    var meta = OrtArtifact.meta(self, manifest: manifest, providers: engine.providers, model: model, started: started)
    meta["convert_bytes"] = converted
    meta["compile_bytes"] = compiled
    meta["compile_seconds"] = pythonRound(compileSeconds, 1)
    meta["freed_bytes"] = freed
    return meta
  }

  // MARK: load

  public func load(artifact: URL, report: @escaping ProgressFn) throws -> any Engine {
    try qnn.validate()
    if profile == .htpParallel, try artifactProfile(Artifact.sidecar(artifact)) == .htpWhole {
      let (manifest, _) = try OrtArtifact.open(artifact, version: OrtProfile.htpWhole.prepareVersion)
      guard manifest.count == 1, OrtUnit(entry: manifest[0]) == .htp,
        ["model.onnx", "model_ctx.onnx"].contains(manifest[0]["model"] as? String ?? "") else {
        throw ArtifactInvalid("causal-prefill fallback artifact is not the whole-NPU route")
      }
      return try wholeFallback().load(artifact: artifact, report: report)
    }
    let (manifest, meta) = try OrtArtifact.open(artifact, version: profile.prepareVersion) { entry in
      guard let unit = OrtUnit(entry: entry) else { return "a session names no unit" }
      // Without its compile, onnxruntime would recompile under a "loading"
      // that never moves. Rebuild instead, which reports progress. Only the
      // compile is looked for: the converted MLProgram beside it is dropped
      // after the build (dropConvertedModels).
      guard case .coreML = unit, let cache = entry["cache"] as? String else { return nil }
      let contents = (try? FileManager.default.contentsOfDirectory(atPath: artifact.appending(path: cache, directoryHint: .isDirectory).path)) ?? []
      return contents.isEmpty ? "the CoreML cache for \(entry["model"] as? String ?? cache) is empty" : nil
    }
    if profile == .htpParallel {
      guard manifest.count == 4,
        manifest.compactMap(OrtUnit.init(entry:)) == [.htp, .qnnGPU, .htp, .cpu],
        zip(manifest, ["vision", "history", "decode", "join"]).allSatisfy({ entry, name in
          let file = entry["model"] as? String
          return file == "\(name).onnx" || ((name == "vision" || name == "decode") && file == "\(name)_ctx.onnx")
        })
      else { throw ArtifactInvalid("causal-prefill artifact has a different session contract") }
      try CausalPrefillBundle.verifyRetainedParts(in: artifact)
    }
    let what = profile.usesCoreML ? "the CoreML model" : "the model"
    let (engine, seconds) = try Artifact.load(artifact, meta: meta, what: what, report: report) {
      try self.engine(artifact, manifest, keepAlive: keepAlive, keepCPUWarm: keepCPUWarm)
    }
    log.info("onnxruntime sessions on \(profile.rawValue) in \(String(format: "%.1f", seconds)) s: \(engine.providers.joined(separator: " then "))")
    return engine
  }
}

/// What a CoreML session creation is doing, read off its cache directory:
/// onnxruntime writes the converted MLProgram first, then CoreML compiles it
/// into `compiled_model.mlmodelc`. Which one is growing is the stage.
final class CoreMLProgress: @unchecked Sendable {
  static let compiledDir = "compiled_model.mlmodelc"
  /// The last resort for the compile's fraction, only on a first build.
  static let expectedCompileSeconds = 10.0

  let caches: [URL]
  let weightBytes: Int64
  let expect: [String: Any]

  init(caches: [URL], weightBytes: Int64, expect: [String: Any]) {
    self.caches = caches
    self.weightBytes = weightBytes
    self.expect = expect
  }

  func measure() -> (converted: Int64, compiled: Int64) {
    var converted: Int64 = 0
    var compiled: Int64 = 0
    for cache in caches {
      guard let walker = FileManager.default.enumerator(at: cache, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey]) else { continue }
      for case let file as URL in walker {
        guard let values = try? file.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey]), values.isRegularFile == true else { continue }
        let size = Int64(values.fileSize ?? 0)
        if file.pathComponents.contains(CoreMLProgress.compiledDir) {
          compiled += size
        } else {
          converted += size
        }
      }
    }
    return (converted, compiled)
  }

  func tick(elapsed: TimeInterval) -> (String, Double, String) {
    let (converted, compiled) = measure()
    if compiled == 0 {
      let total = weightBytes > 0 ? weightBytes : ((expect["convert_bytes"] as? NSNumber)?.int64Value ?? 0)
      let frac = total > 0 ? min(0.95, Double(converted) / Double(total)) : 0
      let of = total > 0 ? " of \(formatBytes(total))" : ""
      return ("convert", frac, "converting for CoreML, \(formatBytes(converted))\(of) written")
    }
    if let total = (expect["compile_bytes"] as? NSNumber)?.int64Value, total > 0 {
      return ("compile", min(0.95, Double(compiled) / Double(total)), "compiling for CoreML, \(formatBytes(compiled)) of \(formatBytes(total)) written")
    }
    let took = (expect["compile_seconds"] as? NSNumber)?.doubleValue ?? CoreMLProgress.expectedCompileSeconds
    return ("compile", min(0.95, elapsed / took), "compiling for CoreML, \(formatBytes(compiled)) written, \(Int(elapsed)) s elapsed")
  }

  /// Deletes each session's MLProgram `Data` directory once CoreML's
  /// compile of it exists beside it. Returns the bytes freed.
  static func dropConvertedModels(_ caches: [URL]) -> Int64 {
    let fm = FileManager.default
    var freed: Int64 = 0
    for cache in caches {
      guard let walker = fm.enumerator(at: cache, includingPropertiesForKeys: [.isDirectoryKey]) else { continue }
      var converted: [URL] = []
      for case let url as URL in walker where url.lastPathComponent == "Data" {
        guard (try? url.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true else { continue }
        if fm.fileExists(atPath: url.deletingLastPathComponent().appending(path: compiledDir, directoryHint: .isDirectory).path) {
          converted.append(url)
        }
        walker.skipDescendants()
      }
      for url in converted {
        freed += Files.size(of: url)
        try? fm.removeItem(at: url)
      }
    }
    return freed
  }
}
