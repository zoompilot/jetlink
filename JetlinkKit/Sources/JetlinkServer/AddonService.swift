// The add-on server: a fork's own models beside the driving model, each on its
// own interface of the comma's gadget. docs/addons.md. From Peter Nguyen's
// V-ASM and sign-vision servers (trung791997/openpilot).
import Foundation
import JetlinkKit
import JetlinkRegistry

/// Where an add-on's interface is: sysfs and usbfs on Linux, IOKit on a Mac,
/// a fake in the tests. Each claims only its own interface, by class.
public protocol AddonSource: Sendable {
  /// Is the comma's gadget on the bus? Cheap enough to poll every second.
  func present() -> Bool
  /// Claims the add-on's interface and opens its bulk pair.
  func open() throws -> AddonPipes
}

/// An add-on interface's bulk pair, a byte stream each way.
public final class AddonPipes: @unchecked Sendable {
  let pipes: any BulkPipes

  init(_ pipes: any BulkPipes) {
    self.pipes = pipes
  }

  func close() {
    pipes.close()
  }
}

/// Serves the add-on models a comma provisions over the main link (an
/// ENGINE_REQ with `addon`), the ones built here before, and any --addons
/// names. One build at a time on a thread of its own, after the driving
/// model's; each interface with a model is answered on a thread of its own,
/// a pass only in a gap the driving model leaves (`AddonGate`).
public final class AddonService: @unchecked Sendable {
  static let readTimeout: TimeInterval = 1
  static let writeTimeout: TimeInterval = 1
  static let helloTimeout: TimeInterval = 0.1
  /// The comma waits 50 ms for an answer: a request may wait for a gap, the
  /// end of the next frame most often, as long as the pass still fits after.
  static let answerBudget: TimeInterval = 0.040
  static let reportEvery: TimeInterval = 60
  /// A hello at least this often, busy or not: it carries what is ready.
  static let helloEvery: TimeInterval = 2
  /// Add-on artifacts kept for this backend.
  static let keepArtifacts = 8

  struct Job {
    let kind: [UInt8]
    let name: String
    let file: URL
    let sha256: String?
  }

  private let configs: [AddonConfig]
  private let host: EngineHost
  private let backend: any EngineBackend
  private let cacheRoot: URL
  private let source: @Sendable ([UInt8]) -> any AddonSource
  private let log = ServerLog(category: "addons")
  private let lock = NSLock()
  private var stopped = false
  private var open: [AddonPipes] = []
  private var served: [[UInt8]: Served] = [:]
  /// Queued or building, by sha256, with the session waiting on each, if any.
  private var pending: [String: Session?] = [:]
  /// What this backend could not build or run, by sha256: kept, so a comma is not asked for it on every park.
  private var failed: [String: String] = [:]
  private let jobs = NSCondition()
  private var queue: [Job] = []

  /// `backend` builds the add-on models: the host's own, or one profile of it
  /// that takes a whole model where the host's splits the driving model.
  public init(
    configs: [AddonConfig], host: EngineHost, backend: (any EngineBackend)? = nil, cacheRoot: URL,
    source: @escaping @Sendable ([UInt8]) -> any AddonSource
  ) {
    self.configs = configs
    self.host = host
    self.backend = backend ?? host.backend
    self.cacheRoot = cacheRoot
    self.source = source
  }

  public var gate: AddonGate { host.addonGate }

  private var store: EngineCache { AddonModel.store(root: cacheRoot, backend: backend) }
  private var failedURL: URL { store.layout.root.appending(path: "failed.json") }

  /// Takes add-on requests from now on: before the server starts, so its first hello says add-ons are served.
  public func attach() {
    loadFailed()
    host.addons = self
  }

  /// Loads what was built before, then what --addons names; after the server
  /// starts, so the driving model's preload goes first.
  public func start() {
    if host.addons !== self { attach() }
    Thread.detachNewThread { [self] in builder() }
    for (sha256, kind, name) in stored() {
      enqueue(Job(kind: kind, name: name, file: (try? host.cache.modelPath(sha256)) ?? cacheRoot, sha256: sha256), session: nil)
    }
    for config in configs {
      for file in config.models { enqueue(Job(kind: config.interfaceClass, name: file.lastPathComponent, file: file, sha256: nil), session: nil) }
    }
  }

  public func stop() {
    let pipes = lock.withLock {
      stopped = true
      return open
    }
    host.addons = nil
    jobs.withLock { jobs.broadcast() }
    for pipe in pipes { pipe.pipes.abort() }
  }

  private var isStopped: Bool { lock.withLock { stopped } }

  // MARK: provisioning

  /// For the hello: the models ready here, and those this backend cannot build or run.
  func helloState() -> [String: Any] {
    lock.withLock {
      ["ready": served.values.flatMap { $0.models.values.map(\.sha256) }.sorted(), "failed": failed.keys.sorted()]
    }
  }

  /// An add-on ENGINE_REQ, answered as the driving model's is: ready, building,
  /// need_upload or failed. A model whose bytes are here is queued to build,
  /// and `session` hears its progress and its end.
  func request(sha256: String, nbytes: Int64, interface kind: [UInt8], name: String?, session: Session?) -> [String: Any] {
    func state(_ state: String, _ detail: String = "") -> [String: Any] {
      ["state": state, "detail": detail, "sha256": sha256, "chunk": ModelConstants.chunk, "addon": true]
    }
    let (ready, why, queued) = lock.withLock { () -> (Bool, String?, Bool) in
      let queued = pending[sha256] != nil
      if queued, let session { pending[sha256] = session }
      return (served.values.contains { $0.has(sha256) }, failed[sha256], queued)
    }
    if ready { return state("ready") }
    if let why { return state("failed", why) }
    if queued { return state("building", "queued behind another build") }
    guard let file = try? host.cache.modelPath(sha256) else { return state("failed", "not a model identity") }
    let have = Files.size(of: file)
    guard have == nbytes else { return state("need_upload", "have \(have) of the model") }
    enqueue(Job(kind: kind, name: name ?? String(sha256.prefix(16)), file: file, sha256: sha256), session: session)
    return state("building", "building")
  }

  private func enqueue(_ job: Job, session: Session?) {
    if let sha256 = job.sha256 {
      let queued = lock.withLock { () -> Bool in
        if let waiting = pending[sha256] {
          pending[sha256] = .some(session ?? waiting)
          return true
        }
        pending[sha256] = .some(session)
        return false
      }
      if queued { return }
    }
    jobs.withLock {
      queue.append(job)
      jobs.signal()
    }
  }

  private func waiting(_ sha256: String) -> Session? {
    lock.withLock { pending[sha256] ?? nil }
  }

  private func builder() {
    while !isStopped {
      let next = jobs.withLock { () -> Job? in
        while queue.isEmpty && !isStopped { jobs.wait() }
        return queue.isEmpty ? nil : queue.removeFirst()
      }
      guard let job = next else { return }
      var sha256 = job.sha256 ?? ""
      do {
        if sha256.isEmpty { sha256 = try Files.sha256(of: job.file) }
        if lock.withLock({ served.values.contains { $0.has(sha256) } }) {
          finish(sha256, ["state": "ready", "detail": ""])
          continue
        }
        let entry = try store.entry(sha256)
        // the bytes are checked once more before a build: a model named by a sha it does not have would lie forever
        if !entry.exists, job.sha256 != nil, try Files.sha256(of: job.file) != sha256 {
          throw HostError.failed("\(job.name) does not hash to \(sha256.prefix(16))")
        }
        let known = sha256
        let model = try AddonModel.prepare(
          job.file, backend: backend, root: cacheRoot, log: log, host: host, sha256: known, name: job.name, interface: job.kind
        ) { [self] stage, frac, message in waiting(known)?.progress(stage, frac, message) }
        serve(job.kind, name: configs.first { $0.interfaceClass == job.kind }?.name).add(model)
        store.prune(keep: Self.keepArtifacts, protect: entry.path)
        finish(sha256, ["state": "ready", "detail": ""])
      } catch {
        log.error("\(job.name): cannot serve it: \(error)")
        if sha256.isEmpty { continue }
        lock.withLock { failed[sha256] = "\(error)" }
        saveFailed()
        finish(sha256, ["state": "failed", "detail": "\(error)"])
      }
    }
  }

  private func finish(_ sha256: String, _ state: [String: Any]) {
    let session = lock.withLock { pending.removeValue(forKey: sha256) ?? nil }
    session?.addonUpdate(state.merging(["sha256": sha256, "chunk": ModelConstants.chunk, "addon": true]) { a, _ in a })
  }

  /// The artifacts built here before, with the interface and name each was built for.
  private func stored() -> [(String, [UInt8], String)] {
    store.inventory().compactMap { sha256 in
      guard let entry = try? store.entry(sha256) else { return nil }
      let meta = Artifact.sidecar(entry.path)
      guard let kind = (meta["interface"] as? String).flatMap(AddonConfig.interfaceClass) else { return nil }
      return (sha256, kind, meta["addon"] as? String ?? String(sha256.prefix(16)))
    }
  }

  private func loadFailed() {
    let all = (try? JSONSerialization.jsonObject(with: Data(contentsOf: failedURL))) as? [String: Any]
    let known = all?[backend.tag()] as? [String: String] ?? [:]
    lock.withLock { failed = known }
  }

  private func saveFailed() {
    var all = (try? JSONSerialization.jsonObject(with: Data(contentsOf: failedURL))) as? [String: Any] ?? [:]
    all[backend.tag()] = lock.withLock { failed }
    try? FileManager.default.createDirectory(at: store.layout.root, withIntermediateDirectories: true)
    if let data = try? JSONSerialization.data(withJSONObject: all, options: [.sortedKeys]) { try? data.write(to: failedURL, options: .atomic) }
  }

  // MARK: serving

  /// The interface's add-on, its thread started the first time a model is ready for it.
  private func serve(_ kind: [UInt8], name: String?) -> Served {
    let (addon, new) = lock.withLock { () -> (Served, Bool) in
      if let addon = served[kind] { return (addon, false) }
      let addon = Served(kind: kind, name: name ?? AddonConfig.triple(kind))
      served[kind] = addon
      return (addon, true)
    }
    if new { Thread.detachNewThread { [self] in loop(addon) } }
    return addon
  }

  final class Served: @unchecked Sendable {
    let kind: [UInt8]
    let name: String
    private let lock = NSLock()
    private var ready: [UInt32: AddonModel]
    var passes = 0
    var yields = 0
    /// Answers of any kind, ever: whether a link carried anything.
    var answered = 0
    var failures = 0
    var lastReport = Date()

    init(kind: [UInt8], name: String, models: [UInt32: AddonModel] = [:]) {
      self.kind = kind
      self.name = name
      ready = models
    }

    convenience init(config: AddonConfig, models: [UInt32: AddonModel]) {
      self.init(kind: config.interfaceClass, name: config.name, models: models)
    }

    var models: [UInt32: AddonModel] { lock.withLock { ready } }

    func add(_ model: AddonModel) {
      lock.withLock { ready[model.tag] = model }
    }

    func has(_ sha256: String) -> Bool {
      lock.withLock { ready.values.contains { $0.sha256 == sha256 } }
    }
  }

  private func loop(_ addon: Served) {
    let source = source(addon.kind)
    let name = addon.name
    log.info("\(name): serving interface class \(AddonConfig.triple(addon.kind))")
    var lastError = ""
    var quiet = false
    while !isStopped {
      guard source.present() else {
        Thread.sleep(forTimeInterval: 1)
        continue
      }
      let pipes: AddonPipes
      do {
        pipes = try source.open()
      } catch {
        // the comma has not linked this add-on's interface, most often
        if "\(error)" != lastError { log.info("\(name): \(error)") }
        lastError = "\(error)"
        Thread.sleep(forTimeInterval: 2)
        continue
      }
      lastError = ""
      lock.withLock { open.append(pipes) }
      let before = addon.answered
      do {
        try serve(addon, pipes: pipes)
      } catch {
        // before the comma opens its end, every read fails at once: said once
        if addon.answered > before {
          log.info("\(name): link ended after \(addon.answered - before) answers: \(error)")
          quiet = false
        } else if !quiet {
          log.info("\(name): claimed its interface; waiting for the comma to serve it (\(error))")
          quiet = true
        }
      }
      lock.withLock { open.removeAll { $0 === pipes } }
      pipes.close()
      Thread.sleep(forTimeInterval: quiet ? 2 : 1)
    }
  }

  /// Answers requests until the link fails, a hello whenever none comes for
  /// a second and every `helloEvery` regardless, with the models ready now.
  /// A malformed header cannot be skipped in a byte stream, so it ends the
  /// link, which the loop opens again.
  func serve(_ addon: Served, pipes: AddonPipes) throws {
    var header = [UInt8](repeating: 0, count: AddonWire.headerBytes)
    var body = [UInt8](repeating: 0, count: AddonWire.maxRequest + 1)
    var lastHello = Date()
    while !isStopped {
      let first = try header.withUnsafeMutableBytes { try pipes.pipes.read(into: $0.baseAddress!, count: $0.count, timeout: Self.readTimeout) }
      if first == 0 || Date().timeIntervalSince(lastHello) >= Self.helloEvery {
        let hello = AddonWire.hello(tags: Array(addon.models.keys), yielding: gate.yielding)
        _ = try hello.withUnsafeBytes { try pipes.pipes.write(from: $0.baseAddress!, count: $0.count, timeout: Self.helloTimeout) }
        lastHello = Date()
        report(addon)
        if first == 0 { continue }
      }
      try readFully(pipes, &header, from: first, to: header.count)
      let request = try header.withUnsafeBytes { try AddonWire.request(header: $0) }
      try readFully(pipes, &body, from: 0, to: request.bodyBytes)
      let answer = body.withUnsafeBytes { handle(addon, request, image: UnsafeRawBufferPointer(rebasing: $0[0..<request.nbytes])) }
      try writeFully(pipes, answer)
      addon.answered += 1
    }
  }

  func handle(_ addon: Served, _ request: AddonWire.Request, image: UnsafeRawBufferPointer) -> [UInt8] {
    let started = DispatchTime.now().uptimeNanoseconds
    guard let model = addon.models[request.tag], model.shape == request.shape else {
      return AddonWire.answer(seq: request.seq, tag: request.tag, status: .unknownModel)
    }
    guard gate.gap(cost: model.costNs, wait: max(0, Self.answerBudget - Double(model.costNs) / 1e9)) else {
      addon.yields += 1
      return AddonWire.answer(seq: request.seq, tag: request.tag, status: .yield)
    }
    host.lock.lock()
    defer { host.lock.unlock() }
    do {
      let floats = try model.run(request, image: image)
      addon.passes += 1
      let us = UInt32(min(UInt64(UInt32.max), (DispatchTime.now().uptimeNanoseconds - started) / 1000))
      return AddonWire.answer(seq: request.seq, tag: request.tag, status: .ok, serverUs: us, floats: floats)
    } catch {
      addon.failures += 1
      if addon.failures == 1 || addon.failures % 100 == 0 { log.warning("\(addon.name): \(model.file) failed: \(error)") }
      return AddonWire.answer(seq: request.seq, tag: request.tag, status: .inferFailed)
    }
  }

  private func report(_ addon: Served) {
    guard Date().timeIntervalSince(addon.lastReport) >= Self.reportEvery, addon.passes + addon.yields + addon.failures > 0 else { return }
    let costs = addon.models.values.map { "\($0.file) \(String(format: "%.1f", Double($0.costNs) / 1e6)) ms" }.joined(separator: ", ")
    log.info("\(addon.name): \(addon.passes) passes, \(addon.yields) yielded, \(addon.failures) failed; \(costs)")
    addon.passes = 0
    addon.yields = 0
    addon.failures = 0
    addon.lastReport = Date()
  }

  private func readFully(_ pipes: AddonPipes, _ buffer: inout [UInt8], from start: Int, to end: Int) throws {
    var have = start
    let deadline = Date().addingTimeInterval(Self.readTimeout)
    while have < end {
      let got = try buffer.withUnsafeMutableBytes {
        try pipes.pipes.read(into: $0.baseAddress! + have, count: end - have, timeout: Self.readTimeout)
      }
      have += got
      if got == 0 && Date() >= deadline { throw LinkError.closed("a request stopped part way, \(have) of \(end) bytes") }
    }
  }

  private func writeFully(_ pipes: AddonPipes, _ bytes: [UInt8]) throws {
    var sent = 0
    let deadline = Date().addingTimeInterval(Self.writeTimeout)
    while sent < bytes.count {
      let wrote = try bytes.withUnsafeBytes {
        try pipes.pipes.write(from: $0.baseAddress! + sent, count: bytes.count - sent, timeout: Self.writeTimeout)
      }
      sent += wrote
      if wrote == 0 && Date() >= deadline { throw LinkError.closed("the comma stopped reading, \(sent) of \(bytes.count) bytes") }
    }
  }
}
