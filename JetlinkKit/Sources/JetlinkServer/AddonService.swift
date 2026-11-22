// The add-on server: a fork's own models beside the driving model, each on its
// own interface of the comma's gadget. docs/addons.md. From Peter Nguyen's
// V-ASM and sign-vision servers (trung791997/openpilot).
import Foundation
import JetlinkKit
import JetlinkRegistry

/// An add-on as the comma provisions it: its interface's class triple on the
/// gadget, and the port the comma listens on for it where the cable is a
/// network (an iPhone), from its latest ENGINE_REQ that named one.
public final class AddonInterface: @unchecked Sendable {
  public let kind: [UInt8]
  private let lock = NSLock()
  private var learned: UInt16?

  public init(kind: [UInt8], port: UInt16? = nil) {
    self.kind = kind
    learned = port
  }

  public var port: UInt16? { lock.withLock { learned } }

  func learn(port: UInt16) {
    lock.withLock { learned = port }
  }
}

/// Where an add-on's interface is: sysfs and usbfs on Linux, IOKit on a Mac,
/// the app's descriptor on Android, TCP to the comma on an iPhone, a fake in
/// the tests. Each claims only its own interface, by class.
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
/// ENGINE_REQ with `addon`) and the ones built here before. One build at a time on a thread of its own, after the driving
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

  /// A model served on one interface: the artifact is the sha256's, shared by
  /// every interface that serves it, but each interface is ready on its own.
  struct Key: Hashable {
    let kind: [UInt8]
    let sha256: String
  }

  struct Job {
    let kind: [UInt8]
    let port: UInt16?
    let name: String
    let file: URL
    let sha256: String
  }

  private let host: EngineHost
  private let backend: any EngineBackend
  private let cacheRoot: URL
  private let source: @Sendable (AddonInterface) -> any AddonSource
  private let log = ServerLog(category: "addons")
  private let lock = NSLock()
  private var stopped = false
  private var open: [AddonPipes] = []
  private var served: [[UInt8]: Served] = [:]
  /// Queued or building, by interface and sha256, with the session waiting on each, if any.
  private var pending: [Key: Session?] = [:]
  /// What this backend could not build or run, by sha256: kept, so a comma is not asked for it on every park.
  private var failed: [String: String] = [:]
  /// Each model's name, by sha256, for the web page.
  private var names: [String: String] = [:]
  /// Interfaces and models serving.json holds or is about to.
  private var remembered = Set<Key>()
  private let servingLock = NSLock()
  private let jobs = NSCondition()
  private var queue: [Job] = []

  /// `backend` builds the add-on models: the host's own, or one profile of it
  /// that takes a whole model where the host's splits the driving model.
  public init(
    host: EngineHost, backend: (any EngineBackend)? = nil, cacheRoot: URL,
    source: @escaping @Sendable (AddonInterface) -> any AddonSource
  ) {
    self.host = host
    self.backend = backend ?? host.backend
    self.cacheRoot = cacheRoot
    self.source = source
  }

  public var gate: AddonGate { host.addonGate }

  private var store: EngineCache { AddonModel.store(root: cacheRoot, backend: backend) }
  private var failedURL: URL { store.layout.root.appending(path: "failed.json") }
  /// What each interface serves, by backend tag: an artifact's sidecar names one
  /// interface only, and a model two add-ons share is served on both.
  private var servingURL: URL { store.layout.root.appending(path: "serving.json") }

  /// Takes add-on requests from now on: before the server starts, so its first hello says add-ons are served.
  public func attach() {
    loadFailed()
    host.addons = self
  }

  /// Loads what was built before; after the server
  /// starts, so the driving model's preload goes first.
  public func start() {
    if host.addons !== self { attach() }
    Thread.detachNewThread { [self] in builder() }
    for (sha256, kind, port, name) in stored() {
      _ = lock.withLock { remembered.insert(Key(kind: kind, sha256: sha256)) }
      enqueue(Job(kind: kind, port: port, name: name, file: (try? host.cache.modelPath(sha256)) ?? cacheRoot, sha256: sha256), session: nil)
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

  /// For the hello: the models ready on each interface (`serving`, by class
  /// triple), all of them (`ready`, for a comma too old to read `serving`),
  /// and those this backend cannot build or run.
  func helloState() -> [String: Any] {
    lock.withLock {
      var serving: [String: [String]] = [:]
      for (kind, addon) in served { serving[AddonWire.triple(kind)] = Set(addon.models.values.map(\.sha256)).sorted() }
      return ["ready": Set(serving.values.joined()).sorted(), "serving": serving, "failed": failed.keys.sorted()]
    }
  }

  /// An add-on ENGINE_REQ, answered as the driving model's is: ready, building,
  /// need_upload or failed. A model whose bytes are here is queued to build,
  /// and `session` hears its progress and its end.
  func request(sha256: String, nbytes: Int64, kind: [UInt8], port: UInt16?, name: String?, session: Session?) -> [String: Any] {
    func state(_ state: String, _ detail: String = "") -> [String: Any] {
      ["state": state, "detail": detail, "sha256": sha256, "chunk": ModelConstants.chunk, "addon": true]
    }
    let key = Key(kind: kind, sha256: sha256)
    let (ready, why, queued, elsewhere) = lock.withLock { () -> (Bool, String?, Bool, Bool) in
      let queued = pending[key] != nil
      if queued, let session { pending[key] = session }
      if let port { served[kind]?.interface.learn(port: port) }
      return (served[kind]?.has(sha256) == true, failed[sha256], queued, served.values.contains { $0.has(sha256) })
    }
    if ready { return state("ready") }
    if let why { return state("failed", why) }
    if queued { return state("building", "queued behind another build") }
    // loaded for another interface: served here too, without its bytes again
    if elsewhere {
      enqueue(Job(kind: kind, port: port, name: name ?? String(sha256.prefix(16)), file: cacheRoot, sha256: sha256), session: session)
      return state("building", "serving it on this interface too")
    }
    guard let file = try? host.cache.modelPath(sha256) else { return state("failed", "not a model identity") }
    let have = Files.size(of: file)
    guard have == nbytes else { return state("need_upload", "have \(have) of the model") }
    enqueue(
      Job(kind: kind, port: port, name: name ?? String(sha256.prefix(16)), file: file, sha256: sha256), session: session)
    return state("building", "building")
  }

  private func enqueue(_ job: Job, session: Session?) {
    let key = Key(kind: job.kind, sha256: job.sha256)
    let queued = lock.withLock { () -> Bool in
      names[job.sha256] = job.name
      if let waiting = pending[key] {
        pending[key] = .some(session ?? waiting)
        return true
      }
      pending[key] = .some(session)
      return false
    }
    if queued { return }
    jobs.withLock {
      queue.append(job)
      jobs.signal()
    }
  }

  private func waiting(_ key: Key) -> Session? {
    lock.withLock { pending[key] ?? nil }
  }

  private func builder() {
    while !isStopped {
      let next = jobs.withLock { () -> Job? in
        while queue.isEmpty && !isStopped { jobs.wait() }
        return queue.isEmpty ? nil : queue.removeFirst()
      }
      guard let job = next else { return }
      let sha256 = job.sha256
      let key = Key(kind: job.kind, sha256: sha256)
      do {
        if lock.withLock({ served[job.kind]?.has(sha256) == true }) {
          finish(key, ["state": "ready", "detail": ""])
          continue
        }
        // the one engine serves every interface that asks for it: passes take the host's lock
        if let loaded = lock.withLock({ served.values.lazy.compactMap { $0.model(sha256) }.first }) {
          serve(job.kind, port: job.port).add(loaded)
          remember(key, name: job.name, port: job.port)
          finish(key, ["state": "ready", "detail": ""])
          continue
        }
        let entry = try store.entry(sha256)
        // the bytes are checked once more before a build: a model named by a sha it does not have would lie forever
        if !entry.exists, try Files.sha256(of: job.file) != sha256 {
          throw HostError.failed("\(job.name) does not hash to \(sha256.prefix(16))")
        }
        let model = try AddonModel.prepare(
          job.file, backend: backend, root: cacheRoot, log: log, host: host, sha256: sha256, name: job.name, interface: job.kind,
          port: job.port
        ) { [self] stage, frac, message in
          waiting(key)?.progress(stage, frac, message, extra: ["addon": true, "sha256": sha256])
        }
        serve(job.kind, port: job.port).add(model)
        remember(key, name: job.name, port: job.port)
        store.prune(keep: Self.keepArtifacts, protect: entry.path)
        finish(key, ["state": "ready", "detail": ""])
      } catch {
        log.error("\(job.name): cannot serve it: \(error)")
        lock.withLock { failed[sha256] = "\(error)" }
        saveFailed()
        finish(key, ["state": "failed", "detail": "\(error)"])
      }
    }
  }

  private func finish(_ key: Key, _ state: [String: Any]) {
    let session = lock.withLock { pending.removeValue(forKey: key) ?? nil }
    session?.addonUpdate(state.merging(["sha256": key.sha256, "chunk": ModelConstants.chunk, "addon": true]) { a, _ in a })
  }

  /// The artifacts built here before, on every interface each was served on:
  /// serving.json's, then the interface an older server wrote in the sidecar.
  private func stored() -> [(String, [UInt8], UInt16?, String)] {
    let built = Set(store.inventory())
    var out: [(String, [UInt8], UInt16?, String)] = []
    var seen = Set<Key>()
    for (triple, models) in loadServing() {
      guard let kind = AddonWire.interfaceClass(triple) else { continue }
      for (sha256, meta) in models where built.contains(sha256) && seen.insert(Key(kind: kind, sha256: sha256)).inserted {
        let port = (meta["port"] as? NSNumber).flatMap { UInt16(exactly: $0.intValue) }
        out.append((sha256, kind, port, meta["name"] as? String ?? String(sha256.prefix(16))))
      }
    }
    for sha256 in built.sorted() {
      guard let entry = try? store.entry(sha256) else { continue }
      let meta = Artifact.sidecar(entry.path)
      guard let kind = (meta["interface"] as? String).flatMap(AddonWire.interfaceClass),
        seen.insert(Key(kind: kind, sha256: sha256)).inserted
      else { continue }
      let port = (meta["port"] as? NSNumber).flatMap { UInt16(exactly: $0.intValue) }
      out.append((sha256, kind, port, meta["addon"] as? String ?? String(sha256.prefix(16))))
    }
    return out
  }

  /// This backend's serving.json: interface triple, then sha256, then its name and port.
  private func loadServing() -> [String: [String: [String: Any]]] {
    let all = (try? JSONSerialization.jsonObject(with: Data(contentsOf: servingURL))) as? [String: Any]
    return all?[backend.tag()] as? [String: [String: [String: Any]]] ?? [:]
  }

  /// Notes that `key` is served, so a restart serves it on that interface again.
  private func remember(_ key: Key, name: String, port: UInt16?) {
    let new = lock.withLock { remembered.insert(key).inserted }
    guard new else { return }
    servingLock.withLock {
      var all = (try? JSONSerialization.jsonObject(with: Data(contentsOf: servingURL))) as? [String: Any] ?? [:]
      var mine = all[backend.tag()] as? [String: [String: [String: Any]]] ?? [:]
      var meta: [String: Any] = ["name": name]
      if let port { meta["port"] = Int(port) }
      mine[AddonWire.triple(key.kind), default: [:]][key.sha256] = meta
      all[backend.tag()] = mine
      try? FileManager.default.createDirectory(at: store.layout.root, withIntermediateDirectories: true)
      if let data = try? JSONSerialization.data(withJSONObject: all, options: [.sortedKeys]) { try? data.write(to: servingURL, options: .atomic) }
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

  /// For the web page (`GET /api/addons`): each interface served, its models
  /// and how long a pass takes, what the comma had from it since the start,
  /// and how long ago its last answer was; the models building, and those
  /// this backend cannot build or run, with why.
  public func pageState() -> [String: Any] {
    let snapshot: ([Served], [Key], [String: String], [String: String]) = lock.withLock {
      (Array(self.served.values), Array(self.pending.keys), self.failed, self.names)
    }
    let (addons, building, refusals, known) = snapshot
    let now = Date()
    func name(_ sha256: String) -> String { known[sha256] ?? String(sha256.prefix(16)) }
    var rows: [[String: Any]] = []
    for addon in addons.sorted(by: { $0.name < $1.name }) {
      var models: [[String: Any]] = []
      for model in addon.models.values.sorted(by: { $0.file < $1.file }) {
        let ms: Double = (Double(model.costNs) / 1e5).rounded() / 10
        models.append(["name": model.file, "sha256": model.sha256, "pass_ms": ms])
      }
      var row: [String: Any] = ["interface": addon.name, "models": models, "answers": addon.answered]
      row["passes"] = addon.allPasses
      row["yields"] = addon.allYields
      row["failures"] = addon.allFailures
      if let last = addon.lastAnswer {
        let ago: Double = (now.timeIntervalSince(last) * 10).rounded() / 10
        row["last_answer_s"] = ago
      }
      rows.append(row)
    }
    var queued: [[String: Any]] = []
    for key in building.sorted(by: { $0.sha256 < $1.sha256 }) {
      queued.append(["interface": AddonWire.triple(key.kind), "sha256": key.sha256, "name": name(key.sha256)])
    }
    var refused: [[String: Any]] = []
    for sha256 in refusals.keys.sorted() {
      refused.append(["sha256": sha256, "name": name(sha256), "detail": refusals[sha256] ?? ""])
    }
    return ["served": true, "addons": rows, "building": queued, "failed": refused]
  }

  // MARK: serving

  /// The interface's add-on, its thread started the first time a model is ready for it.
  private func serve(_ kind: [UInt8], port: UInt16?) -> Served {
    let (addon, new) = lock.withLock { () -> (Served, Bool) in
      if let addon = served[kind] {
        if let port { addon.interface.learn(port: port) }
        return (addon, false)
      }
      let addon = Served(kind: kind, name: AddonWire.triple(kind), port: port)
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
    /// Since the server started, for the web page: the counts above are a report's.
    var allPasses = 0
    var allYields = 0
    var allFailures = 0
    /// When the comma last had an answer here.
    var lastAnswer: Date?
    /// The one its source is made for, its port learned in place.
    let interface: AddonInterface

    init(kind: [UInt8], name: String, port: UInt16? = nil, models: [UInt32: AddonModel] = [:]) {
      self.kind = kind
      self.name = name
      interface = AddonInterface(kind: kind, port: port)
      ready = models
    }

    var models: [UInt32: AddonModel] { lock.withLock { ready } }

    func add(_ model: AddonModel) {
      lock.withLock { ready[model.tag] = model }
    }

    func has(_ sha256: String) -> Bool {
      model(sha256) != nil
    }

    func model(_ sha256: String) -> AddonModel? {
      lock.withLock { ready.values.first { $0.sha256 == sha256 } }
    }
  }

  private func loop(_ addon: Served) {
    // made once: a source may hold its claim on the interface between opens
    let source = source(addon.interface)
    let name = addon.name
    log.info("\(name): serving interface class \(AddonWire.triple(addon.kind))")
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
        // a hello the comma is not reading is skipped, but one begun is finished or the link ends: the stream has no other framing
        let hello = AddonWire.hello(tags: Array(addon.models.keys), yielding: gate.yielding)
        let sent = try hello.withUnsafeBytes { try pipes.pipes.write(from: $0.baseAddress!, count: $0.count, timeout: Self.helloTimeout) }
        if sent > 0 && sent < hello.count { try writeFully(pipes, Array(hello[sent...])) }
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
      addon.lastAnswer = Date()
    }
  }

  func handle(_ addon: Served, _ request: AddonWire.Request, image: UnsafeRawBufferPointer) -> [UInt8] {
    let started = DispatchTime.now().uptimeNanoseconds
    guard let model = addon.models[request.tag], model.shape == request.shape else {
      return AddonWire.answer(seq: request.seq, tag: request.tag, status: .unknownModel)
    }
    // admitted, the gap is this pass's until it ends: no other add-on spends it too
    guard gate.admit(cost: model.costNs, wait: max(0, Self.answerBudget - Double(model.costNs) / 1e9)) else {
      addon.yields += 1
      addon.allYields += 1
      return AddonWire.answer(seq: request.seq, tag: request.tag, status: .yield)
    }
    defer { gate.passEnded() }
    host.lock.lock()
    defer { host.lock.unlock() }
    do {
      let floats = try model.run(request, image: image)
      addon.passes += 1
      addon.allPasses += 1
      let us = UInt32(min(UInt64(UInt32.max), (DispatchTime.now().uptimeNanoseconds - started) / 1000))
      return AddonWire.answer(seq: request.seq, tag: request.tag, status: .ok, serverUs: us, floats: floats)
    } catch {
      addon.failures += 1
      addon.allFailures += 1
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
