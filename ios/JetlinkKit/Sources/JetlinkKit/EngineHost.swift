// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.
//
// The engine and the build in flight, for the life of the server: EngineHost
// in jetlink/server/session.py, method for method.
//
// The engine outlives the connection. The comma reconnects at every handover
// and reloading costs seconds of modeld's budget, so the host owns the loaded
// engine and whatever build is running, and a session only borrows them.
//
// On top of the Python: the compute units can be chosen per model. With
// `auto`, the first time a model is prepared the host builds it for the
// Neural Engine and for the GPU, runs each at the comma's own pace, and keeps
// the one that makes the frame budget, preferring the Neural Engine, which
// does the same work for far less power.

import Foundation

public enum ComputePreference: String, Sendable, CaseIterable, Codable {
  case auto, ane, coreml, cpu

  public var label: String {
    switch self {
    case .auto: "Automatic"
    case .ane: "Neural Engine"
    case .coreml: "GPU"
    case .cpu: "CPU (testing only)"
    }
  }
}

/// What a client asked for: enough to identify the model without the file.
public struct EngineRequest: Sendable, Equatable {
  public let sha256: String
  public let nbytes: Int
  public let frameSkip: Int

  public init(sha256: String, nbytes: Int, frameSkip: Int) throws {
    try EngineCache.validate(sha256)
    if nbytes < 0 || frameSkip <= 0 { throw SpecError("invalid model size or frame skip") }
    self.sha256 = sha256
    self.nbytes = nbytes
    self.frameSkip = frameSkip
  }
}

final class Job: @unchecked Sendable {
  let sha256: String
  let loadOnly: Bool
  let calibrate: Bool
  var state = "building"  // building | ready | failed
  var detail = ""

  init(sha256: String, loadOnly: Bool, calibrate: Bool = false) {
    self.sha256 = sha256
    self.loadOnly = loadOnly
    self.calibrate = calibrate
  }
}

/// An engine resident on the accelerator, with the state that goes with it.
final class Loaded: @unchecked Sendable {
  let sha256: String
  let spec: ModelSpec
  let engine: OrtEngine
  let queues: PolicyQueues
  let hostInputs: [String: TensorBuffer]
  let units: ComputeUnits

  init(sha256: String, spec: ModelSpec, engine: OrtEngine, queues: PolicyQueues, units: ComputeUnits) {
    self.sha256 = sha256
    self.spec = spec
    self.engine = engine
    self.queues = queues
    self.hostInputs = engine.hostInputs
    self.units = units
  }
}

/// What a calibration run measured for one compute unit, in milliseconds.
public struct BenchResult: Sendable, Equatable {
  public var mean: Double
  public var p99: Double
  public var max: Double
  public var frames: Int

  var json: JSON {
    ["mean_ms": .double(pyRound(mean, 2)), "p99_ms": .double(pyRound(p99, 2)), "max_ms": .double(pyRound(max, 2)), "frames": .int(frames)]
  }
}

public final class EngineHost: @unchecked Sendable {
  public let cache: EngineCache
  let log = JLogger("jetlink.server")
  let lock = NSLock()

  private var preferenceValue: ComputePreference
  private let backends: [ComputeUnits: OrtBackend]
  var loaded: Loaded?
  var job: Job?
  var session: Session?
  /// A benchmark owns the engine (Benchmark.swift); frames get NOT_READY.
  var benchmarking = false
  private var lastProgress = 0.0
  private var lastStage: (String?, Double, String) = (nil, 0, "")
  private var lastEngine: JSON?
  private var listeners: [@Sendable (String, JSON) -> Void] = []
  public let frameStats = FrameStats()

  static let progressMinInterval = 0.25
  /// A calibration candidate makes the budget when 99 in 100 paced frames
  /// run in this long: 35 of the 50 ms, leaving the rest for the link and
  /// the queues.
  public static let calibrationBudgetMs = 35.0
  static let calibrationFrames = 60
  static let choicesName = "compute-choice.json"

  public init(cache: EngineCache, preference: ComputePreference) {
    self.cache = cache
    self.preferenceValue = preference
    var b: [ComputeUnits: OrtBackend] = [:]
    for u in ComputeUnits.allCases { b[u] = OrtBackend(units: u) }
    backends = b
  }

  public var preference: ComputePreference {
    lock.lock()
    defer { lock.unlock() }
    return preferenceValue
  }

  /// Changes the compute units for what is prepared next. A loaded engine on
  /// other units is released, so the next request loads the right one.
  public func setPreference(_ p: ComputePreference) {
    lock.lock()
    preferenceValue = p
    let stale = loaded.map { l in unitsFor(l.sha256).map { $0 != l.units } ?? true } ?? false
    lock.unlock()
    if stale { unload() }
  }

  // MARK: listeners

  /// Hear about progress, engine changes and the link. Called from the job
  /// thread, the session thread and the accept thread; must not block.
  public func subscribe(_ fn: @escaping @Sendable (String, JSON) -> Void) {
    lock.lock()
    listeners.append(fn)
    lock.unlock()
  }

  func emit(_ kind: String, _ payload: JSON) {
    lock.lock()
    if kind == "engine" {
      if payload == lastEngine {
        lock.unlock()
        return
      }
      lastEngine = payload
    }
    let fns = listeners
    lock.unlock()
    for fn in fns { fn(kind, payload) }
  }

  // MARK: which backend serves which model

  /// The units every model runs on when the preference names one; nil for auto.
  /// Call with the lock held.
  var forcedUnits: ComputeUnits? {
    switch preferenceValue {
    case .ane: .ane
    case .coreml: .coreml
    case .cpu: .cpu
    case .auto: nil
    }
  }

  /// The compute units a model runs on, or nil when `auto` has not measured
  /// it yet. Call with the lock held.
  func unitsFor(_ sha256: String) -> ComputeUnits? {
    forcedUnits ?? choice(sha256)?.units
  }

  public func backend(for units: ComputeUnits) -> OrtBackend { backends[units]! }

  /// The candidates `auto` chooses between, in order of preference.
  static let autoCandidates: [ComputeUnits] = [.ane, .coreml]

  /// Entries for every backend a model might be served by: the one it runs
  /// on, or while `auto` has not measured it, each candidate.
  func candidateEntries(_ sha256: String, units: ComputeUnits?) -> [(ComputeUnits, CacheEntry)] {
    let all: [ComputeUnits] = units.map { [$0] } ?? EngineHost.autoCandidates
    return all.compactMap { u in (try? cache.entry(sha256, tag: backends[u]!.tag)).map { (u, $0) } }
  }

  public struct Choice: Sendable {
    public let units: ComputeUnits
    public let measured: [ComputeUnits: BenchResult]
  }

  /// The compute units `auto` settled on for a model on this device.
  public func choice(_ sha256: String) -> Choice? {
    guard let d = try? JSON.parse(Data(contentsOf: cache.root.appendingPathComponent(EngineHost.choicesName))),
      let e = d[sha256], e["chip"]?.string == chipName(), let u = e["units"]?.string.flatMap(ComputeUnits.init(rawValue:))
    else { return nil }
    var measured: [ComputeUnits: BenchResult] = [:]
    for (k, v) in e["measured"]?.object ?? [:] {
      if let unit = ComputeUnits(rawValue: k), let mean = v["mean_ms"]?.double, let p99 = v["p99_ms"]?.double,
        let mx = v["max_ms"]?.double
      {
        measured[unit] = BenchResult(mean: mean, p99: p99, max: mx, frames: v["frames"]?.int ?? 0)
      }
    }
    return Choice(units: u, measured: measured)
  }

  func saveChoice(_ sha256: String, _ units: ComputeUnits, _ measured: [ComputeUnits: BenchResult]) {
    let url = cache.root.appendingPathComponent(EngineHost.choicesName)
    var d = (try? JSON.parse(Data(contentsOf: url)))?.object ?? [:]
    d[sha256] = [
      "units": .string(units.rawValue), "chip": .string(chipName()), "at": .string(isoNow()),
      "measured": .object(Dictionary(uniqueKeysWithValues: measured.map { ($0.key.rawValue, $0.value.json) })),
    ]
    try? atomicWrite(JSON.object(d).data, to: url)
  }

  /// Forgets what `auto` chose for a model, so the next prepare measures again.
  public func forgetChoice(_ sha256: String) {
    let url = cache.root.appendingPathComponent(EngineHost.choicesName)
    guard var d = (try? JSON.parse(Data(contentsOf: url)))?.object else { return }
    d[sha256] = nil
    try? atomicWrite(JSON.object(d).data, to: url)
  }

  // MARK: what a client sees

  public func snapshot() -> JSON {
    lock.lock()
    defer { lock.unlock() }
    let (stage, frac, msg) = lastStage
    if let loaded {
      return [
        "state": "ready", "sha256": .string(loaded.sha256), "detail": "", "stage": .null, "frac": 1.0, "msg": .string(msg),
        "load_only": .bool(job?.loadOnly ?? false), "units": .string(loaded.units.rawValue),
      ]
    }
    if let job, job.state == "building" {
      return [
        "state": job.loadOnly ? "loading" : "building", "sha256": .string(job.sha256), "detail": .string(job.detail),
        "stage": .str(stage), "frac": .double(frac), "msg": .string(msg), "load_only": .bool(job.loadOnly), "units": .null,
      ]
    }
    if let job, job.state == "failed" {
      return [
        "state": "failed", "sha256": .string(job.sha256), "detail": .string(job.detail), "stage": "failed",
        "frac": .double(frac), "msg": .string(msg), "load_only": .bool(job.loadOnly), "units": .null,
      ]
    }
    return ["state": "none", "sha256": .null, "detail": "", "stage": .null, "frac": 0.0, "msg": "", "load_only": false, "units": .null]
  }

  /// The engine state for one model, in the shape ENGINE_RESP carries.
  public func status(_ sha256: String?, _ frameSkip: Int?) -> JSON {
    lock.lock()
    defer { lock.unlock() }
    return statusLocked(sha256, frameSkip)
  }

  private func statusLocked(_ sha256: String?, _ frameSkip: Int?) -> JSON {
    guard let sha256 else { return ["state": "none", "detail": "", "sha256": .null, "chunk": .int(uploadChunk)] }
    if let loaded, loaded.sha256 == sha256, frameSkip == nil || loaded.spec.frameSkip == frameSkip {
      return ["state": "ready", "detail": "", "sha256": .string(sha256), "chunk": .int(uploadChunk), "spec": loaded.spec.json]
    }
    if let job, job.sha256 == sha256, job.state != "ready" {
      return ["state": .string(job.state), "detail": .string(job.detail), "sha256": .string(sha256), "chunk": .int(uploadChunk)]
    }
    if let job, job.state == "building" {
      return [
        "state": "building", "sha256": .string(sha256), "chunk": .int(uploadChunk),
        "detail": .string("another build is in progress (\(job.sha256.prefix(16)))"),
      ]
    }
    if candidateEntries(sha256, units: unitsFor(sha256)).contains(where: { cachedSpec($0.1) != nil }) {
      // Built already, just not loaded: say what a request will do, or
      // modeld, which never carries the ONNX, reads need_upload as gone.
      return ["state": "building", "sha256": .string(sha256), "chunk": .int(uploadChunk), "detail": "engine cached, not loaded yet"]
    }
    return [
      "state": "need_upload", "sha256": .string(sha256), "chunk": .int(uploadChunk),
      "detail": .string("have \(modelBytes(sha256)) of the model"),
    ]
  }

  public func loadedSha() -> String? {
    lock.lock()
    defer { lock.unlock() }
    return loaded?.sha256
  }

  public var loadedUnits: ComputeUnits? {
    lock.lock()
    defer { lock.unlock() }
    return loaded?.units
  }

  /// Model identities with an engine this host can load, for the hello.
  public func inventory() -> [String] {
    lock.lock()
    let units = forcedUnits.map { [$0] } ?? EngineHost.autoCandidates
    lock.unlock()
    return Array(Set(units.flatMap { cache.inventory(tag: backends[$0]!.tag) })).sorted()
  }

  /// backend, runtime_version, device for the hello.
  public func describe() -> [String: JSON] {
    lock.lock()
    let units = loaded?.units ?? forcedUnits ?? EngineHost.autoCandidates[0]
    lock.unlock()
    return backends[units]!.describe()
  }

  // MARK: requests

  /// Makes `req` the model being served, starting whatever that takes.
  @discardableResult
  public func request(_ req: EngineRequest, session: Session?) -> JSON {
    lock.lock()
    if let session { self.session = session }
    if let loaded, loaded.sha256 == req.sha256, loaded.spec.frameSkip == req.frameSkip {
      let r = ready(loaded)
      lock.unlock()
      return r
    }
    if let job, job.state == "building" {
      let r = statusLocked(req.sha256, req.frameSkip)
      lock.unlock()
      return r
    }
    let units = unitsFor(req.sha256)
    lock.unlock()

    let modelPath = (try? cache.modelPath(req.sha256))!
    let complete = modelComplete(modelPath, req.nbytes)
    if let units {
      let entry = (try? cache.entry(req.sha256, tag: backends[units]!.tag))!
      let spec = specOnDisk(entry, modelPath, req)
      if entry.exists && spec != nil {
        start(Job(sha256: req.sha256, loadOnly: true), req, spec)
      } else if complete {
        start(Job(sha256: req.sha256, loadOnly: false), req, spec)
      } else {
        clearFailed(req.sha256)
      }
    } else {
      // auto, not measured yet: anything to measure is enough to start
      let anyBuilt = candidateEntries(req.sha256, units: nil).contains { $0.1.exists && cachedSpec($0.1) != nil }
      if complete || anyBuilt {
        start(Job(sha256: req.sha256, loadOnly: false, calibrate: true), req, nil)
      } else {
        clearFailed(req.sha256)
      }
    }
    return status(req.sha256, req.frameSkip)
  }

  private func clearFailed(_ sha256: String) {
    lock.lock()
    // Whatever failed has left the disk; what the client needs now is need_upload.
    if let job, job.sha256 == sha256, job.state == "failed" { self.job = nil }
    lock.unlock()
  }

  private func ready(_ loaded: Loaded) -> JSON {
    ["state": "ready", "detail": "", "sha256": .string(loaded.sha256), "chunk": .int(uploadChunk), "spec": loaded.spec.json]
  }

  func cachedSpec(_ entry: CacheEntry) -> JSON? {
    guard entry.exists else { return nil }
    guard let meta = try? entry.meta() else {
      log.warning("unreadable sidecar for \(entry.path.lastPathComponent)")
      return nil
    }
    return meta["spec"]
  }

  private func modelBytes(_ sha256: String) -> Int {
    guard let p = try? cache.modelPath(sha256) else { return 0 }
    return ((try? FileManager.default.attributesOfItem(atPath: p.path))?[.size] as? Int) ?? 0
  }

  /// Starts loading whatever was loaded last, before a client asks for it.
  public func preload() {
    guard let (sha256, frameSkip, _) = cache.lastLoaded() else { return }
    lock.lock()
    let busy = loaded != nil || job != nil
    let units = unitsFor(sha256)
    lock.unlock()
    guard !busy, let units, let entry = try? cache.entry(sha256, tag: backends[units]!.tag),
      let d = cachedSpec(entry), let spec = try? ModelSpec(json: d)
    else { return }
    log.info("preloading the engine loaded last: \(entry.path.lastPathComponent)")
    guard let req = try? EngineRequest(sha256: sha256, nbytes: 0, frameSkip: frameSkip) else { return }
    start(Job(sha256: sha256, loadOnly: true), req, spec.with(frameSkip: frameSkip))
  }

  /// The spec for a cached artifact, from its sidecar or failing that the ONNX.
  func specOnDisk(_ entry: CacheEntry, _ modelPath: URL, _ req: EngineRequest) -> ModelSpec? {
    if let d = cachedSpec(entry), let s = try? ModelSpec(json: d) { return s.with(frameSkip: req.frameSkip) }
    if modelComplete(modelPath, req.nbytes) {
      do {
        return try deriveSpec(modelPath, req.frameSkip)
      } catch {
        log.error("could not derive a spec from \(modelPath.lastPathComponent): \(error)")
      }
    }
    return nil
  }

  // MARK: the worker

  private func start(_ job: Job, _ req: EngineRequest, _ spec: ModelSpec?) {
    job.detail = job.calibrate ? "measuring the Neural Engine and the GPU" : (job.loadOnly ? "loading engine" : "building engine")
    lock.lock()
    self.job = job
    lastStage = (nil, 0, "")
    lock.unlock()
    let t = Thread { [self] in run(job, req, spec) }
    t.name = "jetlink-build"
    t.qualityOfService = .userInitiated
    t.stackSize = 4 << 20
    t.start()
    emit("engine", snapshot())
  }

  private func run(_ job: Job, _ req: EngineRequest, _ given: ModelSpec?) {
    var engine: OrtEngine?
    defer {
      lock.lock()
      let s = session
      lock.unlock()
      if let s {
        servePending(job, s)
        s.engineUpdate()
      }
      emit("engine", snapshot())
    }
    do {
      // One engine resident at a time: a build needs the memory.
      unload()
      let modelPath = try cache.modelPath(req.sha256)
      var spec = given
      let units: ComputeUnits
      if job.calibrate {
        units = try calibrate(req, modelPath)
      } else {
        lock.lock()
        let u = unitsFor(req.sha256)
        lock.unlock()
        guard let u else { throw EngineError("no compute units chosen for \(req.sha256.prefix(16))") }
        units = u
      }
      let backend = backends[units]!
      let entry = try cache.entry(req.sha256, tag: backend.tag)
      if job.calibrate {
        guard let d = cachedSpec(entry), let s = try? ModelSpec(json: d) else {
          throw EngineError("calibration chose the \(units.label) but left no engine for it")
        }
        spec = s.with(frameSkip: req.frameSkip)
      } else if !job.loadOnly {
        spec = try buildJob(req, entry, modelPath, spec, backend)
      }
      guard let s = spec else { throw EngineError("no model spec") }
      writeSpec(entry, s)

      progress("load", 0.0, "deserializing engine", force: true)
      do {
        engine = try backend.load(artifact: entry.path, report: progressFn)
      } catch let e as ArtifactInvalid {
        // Wrong on disk, not wrong here: replace it from the ONNX when there
        // is one, else let the client upload again.
        log.warning("discarding \(entry.path.lastPathComponent): \(e)")
        entry.remove()
        let have = FileManager.default.fileExists(atPath: modelPath.path) && (req.nbytes == 0 || fileSize(modelPath) == req.nbytes)
        if job.loadOnly && !have { throw EngineError("artifact invalid and the model is not on disk: \(e)") }
        spec = try buildJob(req, entry, modelPath, spec, backend)
        writeSpec(entry, spec!)
        progress("load", 0.0, "deserializing engine", force: true)
        engine = try backend.load(artifact: entry.path, report: progressFn)
      }
      let l = try warm(engine!, spec!, units)
      engine = nil  // owned by `l` from here
      lock.lock()
      loaded = l
      job.state = "ready"
      job.detail = ""
      lock.unlock()
      cache.rememberLoaded(req.sha256, frameSkip: req.frameSkip, units: units)
      progress("load", 1.0, "ready", force: true)
      log.info("engine ready: \(entry.path.lastPathComponent) on the \(units.label)")
    } catch {
      log.error("engine preparation failed: \(error)")
      engine?.close()
      lock.lock()
      job.state = "failed"
      job.detail = "\(type(of: error)): \(error)"
      let detail = job.detail
      lock.unlock()
      progress("failed", 1.0, detail, force: true)
    }
  }

  /// Starts what the client is still waiting for, now the accelerator is
  /// free: a preload that guessed wrong must not leave it waiting.
  private func servePending(_ done: Job, _ session: Session) {
    guard let req = session.currentRequest, done.sha256 != req.sha256 else { return }
    request(req, session: session)
  }

  private func buildJob(_ req: EngineRequest, _ entry: CacheEntry, _ modelPath: URL, _ spec: ModelSpec?, _ backend: OrtBackend)
    throws -> ModelSpec
  {
    var spec = spec
    if spec == nil {
      progress("parse", 0.0, "reading model metadata", force: true)
      spec = try deriveSpec(modelPath, req.frameSkip)
    }
    try backend.build(onnx: modelPath, out: entry.path, report: progressFn, metaExtra: ["spec": spec!.json])
    cache.prune(tagSuffix: backend.tag, protect: entry.path)
    cache.sweepTemp()
    return spec!
  }

  private func writeSpec(_ entry: CacheEntry, _ spec: ModelSpec) {
    var meta = (try? entry.meta())?.object ?? [:]
    if meta["spec"] == nil {
      meta["spec"] = spec.json
      try? entry.writeMeta(.object(meta))
    }
  }

  func deriveSpec(_ modelPath: URL, _ frameSkip: Int) throws -> ModelSpec {
    let meta = try OnnxMeta(path: modelPath.path)
    let (sha, n) = try sha256File(modelPath.path)
    return try ModelSpec(meta: meta, sha256: sha, nbytes: n, frameSkip: frameSkip)
  }

  private func warm(_ engine: OrtEngine, _ spec: ModelSpec, _ units: ComputeUnits) throws -> Loaded {
    try checkShapes(engine, spec)
    let queues = try PolicyQueues(spec: spec)
    // Warm on zeros, so the first real frame pays for nothing lazy.
    let warped = [UInt8](repeating: 0, count: spec.warpedNbytes)
    let packed = [Float](repeating: 0, count: spec.packedNelem)
    warped.withUnsafeBufferPointer { w in
      packed.withUnsafeBufferPointer { p in queues.step(warped: w.baseAddress!, packed: p.baseAddress!, into: engine.hostInputs) }
    }
    let line = try engine.warm()
    log.info(line)
    queues.reset()
    return Loaded(sha256: spec.sha256, spec: spec, engine: engine, queues: queues, units: units)
  }

  public func unload() {
    lock.lock()
    let l = loaded
    loaded = nil
    lock.unlock()
    if let l {
      l.engine.close()
      log.info("engine \(l.sha256.prefix(16)) unloaded")
      emit("engine", snapshot())
    }
  }

  public func close() { unload() }

  // MARK: calibration

  /// Builds (or loads) the model for each candidate, runs it at 20 Hz, and
  /// returns the one to keep. Leaves nothing loaded.
  private func calibrate(_ req: EngineRequest, _ modelPath: URL) throws -> ComputeUnits {
    var measured: [ComputeUnits: BenchResult] = [:]
    var failures: [String] = []
    var derived: ModelSpec?  // hashing and parsing the model once is enough
    let candidates = EngineHost.autoCandidates
    for (i, units) in candidates.enumerated() {
      let backend = backends[units]!
      let label = units.label
      progress("build", Double(i) / Double(candidates.count), "preparing for the \(label)", force: true)
      do {
        let entry = try cache.entry(req.sha256, tag: backend.tag)
        var spec = cachedSpec(entry).flatMap { try? ModelSpec(json: $0) }?.with(frameSkip: req.frameSkip)
        if !entry.exists || spec == nil {
          guard modelComplete(modelPath, req.nbytes) || (req.nbytes == 0 && FileManager.default.fileExists(atPath: modelPath.path))
          else { throw EngineError("no model on disk to build for the \(label)") }
          spec = try buildJob(req, entry, modelPath, spec ?? derived, backend)
        }
        derived = derived ?? spec
        writeSpec(entry, spec!)
        var engine: OrtEngine
        do {
          engine = try backend.load(artifact: entry.path, report: progressFn)
        } catch let e as ArtifactInvalid {
          log.warning("discarding \(entry.path.lastPathComponent): \(e)")
          entry.remove()
          spec = try buildJob(req, entry, modelPath, spec, backend)
          writeSpec(entry, spec!)
          engine = try backend.load(artifact: entry.path, report: progressFn)
        }
        let l: Loaded
        do {
          l = try warm(engine, spec!, units)
        } catch {
          engine.close()
          throw error
        }
        defer { l.engine.close() }
        progress("build", (Double(i) + 0.5) / Double(candidates.count), "timing the \(label) at 20 Hz", force: true)
        let r = try bench(l, frames: EngineHost.calibrationFrames)
        measured[units] = r
        log.info(
          "\(label): mean \(String(format: "%.1f", r.mean)) ms, p99 \(String(format: "%.1f", r.p99)) ms, max \(String(format: "%.1f", r.max)) ms over \(r.frames) paced frames"
        )
      } catch {
        log.warning("the \(label) could not run this model: \(error)")
        failures.append("\(label): \(error)")
      }
    }
    guard let chosen = EngineHost.choose(measured) else {
      throw EngineError("neither the Neural Engine nor the GPU could run the model (\(failures.joined(separator: "; ")))")
    }
    saveChoice(req.sha256, chosen, measured)
    let m = measured[chosen]!
    progress(
      "build", 1.0, "chose the \(chosen.label): p99 \(String(format: "%.1f", m.p99)) ms at 20 Hz", force: true)
    // The loser's artifact is gigabytes a phone can use for something else.
    for u in candidates where u != chosen {
      if let e = try? cache.entry(req.sha256, tag: backends[u]!.tag) { e.remove() }
    }
    return chosen
  }

  /// Prefer the Neural Engine whenever it makes the budget; otherwise
  /// whichever does; otherwise the faster.
  static func choose(_ measured: [ComputeUnits: BenchResult]) -> ComputeUnits? {
    for u in autoCandidates {
      if let r = measured[u], r.p99 <= calibrationBudgetMs { return u }
    }
    return measured.min { $0.value.p99 < $1.value.p99 }?.key
  }

  /// Runs the engine at the comma's pace, 20 frames a second with the
  /// accelerator idle in between, because that and not back-to-back is what
  /// the car asks of it. Returns the run times after five warm-up frames.
  func bench(_ l: Loaded, frames: Int) throws -> BenchResult {
    var rng = SystemRandomNumberGenerator()
    let warped = (0..<l.spec.warpedNbytes).map { _ in UInt8.random(in: 0...255, using: &rng) }
    let packed = [Float](repeating: 0, count: l.spec.packedNelem)
    var times: [Double] = []
    let period = 1.0 / Double(modelRunFreq)
    var next = monotonic()
    for i in 0..<(frames + 5) {
      let wait = next - monotonic()
      if wait > 0 { Thread.sleep(forTimeInterval: wait) }
      next += period
      let t0 = monotonic()
      warped.withUnsafeBufferPointer { w in
        packed.withUnsafeBufferPointer { p in l.queues.step(warped: w.baseAddress!, packed: p.baseAddress!, into: l.hostInputs) }
      }
      try l.engine.run()
      _ = l.engine.output()
      if i >= 5 { times.append((monotonic() - t0) * 1e3) }
    }
    l.queues.reset()
    let sorted = times.sorted()
    return BenchResult(
      mean: times.reduce(0, +) / Double(times.count), p99: sorted[Int(0.99 * Double(sorted.count - 1))], max: sorted.last!,
      frames: times.count)
  }

  // MARK: talking back

  var progressFn: ProgressFn { { [weak self] s, f, m in self?.progress(s, f, m) } }

  func progress(_ stage: String, _ frac: Double, _ msg: String, force: Bool = false) {
    let now = monotonic()
    lock.lock()
    if !force && frac < 1.0 && now - lastProgress < EngineHost.progressMinInterval {
      lock.unlock()
      return
    }
    lastProgress = now
    lastStage = (stage, frac, msg)
    let s = session
    lock.unlock()
    let payload: JSON = ["stage": .string(stage), "frac": .double(pyRound(frac, 4)), "msg": .string(msg)]
    emit("progress", payload)
    s?.progress(stage, frac, msg)
  }
}

/// Uploads land in place chunk by chunk, so the file is only the model once
/// it is the size the client declared.
func modelComplete(_ path: URL, _ nbytes: Int) -> Bool {
  FileManager.default.fileExists(atPath: path.path) && fileSize(path) == nbytes
}

func fileSize(_ url: URL) -> Int {
  ((try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? NSNumber)?.intValue ?? -1
}

/// The engine is what will execute and the spec came from the file, so a
/// disagreement means the artifact on disk is not this model's.
func checkShapes(_ engine: OrtEngine, _ spec: ModelSpec) throws {
  for io in engine.inputs {
    guard let want = spec.inputShapes[io.name] else {
      throw EngineError("engine input \(io.name) is not in the model spec")
    }
    if io.count != product(want) { throw EngineError("input \(io.name): engine \(io.shape) vs spec \(want)") }
  }
  let have = Set(engine.inputs.map(\.name))
  let missing = Set(spec.inputShapes.keys).subtracting(have)
  if !missing.isEmpty { throw EngineError("engine has no input(s) \(missing.sorted()) the model spec declares") }
  let queueInputs = Set(PolicyQueues.inputNames).subtracting(have)
  if !queueInputs.isEmpty { throw EngineError("the model has no \(queueInputs.sorted()) input(s) for the history queues") }
  guard let out = engine.outputs.first, out.count == spec.outputNelem else {
    throw EngineError("output: engine \(engine.outputs.first?.shape ?? []) vs spec \(spec.outputNelem)")
  }
}

/// A rolling window of served frames, for the app's live numbers.
public final class FrameStats: @unchecked Sendable {
  private let lock = NSLock()
  private var samples: [(Double, Int, Int)] = []  // (time, total_us, gpu_us)
  private var head = 0
  static let capacity = 2000
  static let slowFrameUs = 60_000

  public init() {}

  func record(totalUs: Int, gpuUs: Int) {
    lock.lock()
    let s = (monotonic(), totalUs, gpuUs)
    if samples.count < FrameStats.capacity {
      samples.append(s)
    } else {
      samples[head] = s
      head = (head + 1) % FrameStats.capacity
    }
    lock.unlock()
  }

  public struct Summary: Sendable, Equatable {
    public var fps: Double
    public var meanMs: Double
    public var p99Ms: Double
    public var maxMs: Double
    public var gpuMeanMs: Double
    public var slow: Int
  }

  /// The frames of the last `seconds`, or nil when there were none.
  public func summary(seconds: Double = 1.0) -> Summary? {
    let cutoff = monotonic() - seconds
    lock.lock()
    let rows = samples.filter { $0.0 >= cutoff }
    lock.unlock()
    guard !rows.isEmpty else { return nil }
    let totals = rows.map(\.1).sorted()
    let n = Double(rows.count)
    return Summary(
      fps: pyRound(n / seconds, 2), meanMs: Double(totals.reduce(0, +)) / n / 1e3,
      p99Ms: Double(totals[Int(0.99 * Double(totals.count - 1))]) / 1e3, maxMs: Double(totals.last!) / 1e3,
      gpuMeanMs: Double(rows.map(\.2).reduce(0, +)) / n / 1e3, slow: totals.filter { $0 > FrameStats.slowFrameUs }.count)
  }
}
