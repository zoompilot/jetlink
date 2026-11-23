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
  /// Failures of a model, the kind that may not last, before it is kept as failed.
  static let strikesToFail = 2

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
  /// Failures that may not last (out of memory, a read error), by sha256: the
  /// comma tries again, and only a second is kept in `failed`.
  private var strikes: [String: Int] = [:]
  /// Each model's name, by sha256, for the web page.
  private var names: [String: String] = [:]
  /// What a comma last said it wants on each interface it named (keep(_:)):
  /// a build that ends for anything else is not served.
  private var keeping: [[UInt8]: Set<String>] = [:]
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
      for (kind, addon) in served {
        let shas = Set(addon.models.values.map(\.sha256)).sorted()
        if !shas.isEmpty { serving[AddonWire.triple(kind)] = shas }
      }
      // forgets: a hello's addons_keep is heard, and what it leaves out is deleted
      return ["ready": Set(serving.values.joined()).sorted(), "serving": serving, "failed": failed.keys.sorted(), "forgets": true]
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
      // asked for: wanted, whatever the last hello said
      keeping[kind]?.insert(sha256)
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

  private func builder() {
    while !isStopped {
      let next = jobs.withLock { () -> Job? in
        while queue.isEmpty && !isStopped { jobs.wait() }
        return queue.isEmpty ? nil : queue.removeFirst()
      }
      guard let job = next else { return }
      let sha256 = job.sha256
      let key = Key(kind: job.kind, sha256: sha256)
      if lock.withLock({ unwantedLocked(key) }) {
        finish(key, ["state": "failed", "detail": "the comma no longer wants it"])
        continue
      }
      do {
        // the one engine serves every interface that asks for it: passes take
        // the host's lock. Found and added in one hold of the lock, so keep()
        // cannot close it in between for want of another interface serving it
        let shared = lock.withLock { () -> Served?? in
          guard !unwantedLocked(key), let loaded = served.values.lazy.compactMap({ $0.model(sha256) }).first else { return nil }
          return .some(addLocked(loaded, kind: job.kind, port: job.port))
        }
        if let started = shared {
          if let started { startLoop(started) }
          remember(key, name: job.name, port: job.port)
          finish(key, ["state": "ready", "detail": ""])
          continue
        }
        let entry = try store.entry(sha256)
        // the bytes are checked once more before a build: a model named by a sha it does not have would lie forever
        if !entry.exists, (try Registry.hashFile(job.file)).0 != sha256 {
          throw AddonRefused("\(job.name) does not hash to \(sha256.prefix(16))")
        }
        let model = try AddonModel.prepare(
          job.file, backend: backend, root: cacheRoot, log: log, host: host, sha256: sha256, name: job.name, interface: job.kind,
          port: job.port
        ) { [self] stage, frac, message in
          lock.withLock { pending[key] ?? nil }?.progress(stage, frac, message, extra: ["addon": true, "sha256": sha256])
        }
        // turned off while it built: kept for nothing. Checked and added in
        // one hold of the lock, so a keep() in between cannot miss it
        let added = lock.withLock { () -> Served?? in
          unwantedLocked(key) ? nil : .some(addLocked(model, kind: job.kind, port: job.port))
        }
        guard let started = added else {
          close([model])
          finish(key, ["state": "failed", "detail": "the comma no longer wants it"])
          deleteIfUnused([sha256])
          continue
        }
        if let started { startLoop(started) }
        lock.withLock { strikes[sha256] = nil }
        remember(key, name: job.name, port: job.port)
        store.prune(keep: Self.keepArtifacts, protect: entry.path)
        finish(key, ["state": "ready", "detail": ""])
      } catch let deferred as AddonDeferred {
        // the driving model went first: the comma sends it again next park
        log.info("\(job.name): \(deferred.description)")
        finish(key, ["state": "failed", "detail": deferred.description, "retry": true])
      } catch {
        // a model this backend can never take is kept as failed at once;
        // anything else only when it fails a second time
        let lasting = lock.withLock { () -> Bool in
          let strike = (strikes[sha256] ?? 0) + 1
          strikes[sha256] = strike
          let lasting = error is AddonRefused || strike >= Self.strikesToFail
          if lasting { failed[sha256] = "\(error)" }
          return lasting
        }
        if lasting {
          log.error("\(job.name): cannot serve it: \(error)")
          saveFailed()
          finish(key, ["state": "failed", "detail": "\(error)"])
        } else {
          log.warning("\(job.name): could not serve it, the comma tries again: \(error)")
          finish(key, ["state": "failed", "detail": "\(error)", "retry": true])
        }
      }
    }
  }

  /// Every model this service serves, builds, cannot run or means to serve
  /// after a restart: their files share the driving models' cache, and the
  /// inventory leaves them out (ServerController).
  public func modelSHAs() -> Set<String> {
    lock.withLock {
      var shas = Set(served.values.flatMap { $0.models.values.map(\.sha256) })
      shas.formUnion(pending.keys.map(\.sha256))
      shas.formUnion(failed.keys)
      shas.formUnion(remembered.map(\.sha256))
      return shas
    }
  }

  /// Under the lock.
  private func unwantedLocked(_ key: Key) -> Bool {
    keeping[key.kind].map { !$0.contains(key.sha256) } ?? false
  }

  /// A comma's word on what it wants, its hello's `addons_keep` (class
  /// triple, then the sha256s turned on): every other model on an interface it
  /// names is unloaded and forgotten, and once nothing serves or builds one,
  /// its artifact and its bytes are deleted. An interface it does not name is
  /// left as it is, and a model it turns on again is built again.
  func keep(_ said: [String: Any]) {
    var wanted: [[UInt8]: Set<String>] = [:]
    for (triple, value) in said {
      guard let kind = AddonWire.interfaceClass(triple), let shas = value as? [String] else { continue }
      wanted[kind] = Set(shas)
    }
    guard !wanted.isEmpty else { return }
    let union = Set(wanted.values.joined())
    let (dropped, unloaded, unfailed) = lock.withLock { () -> ([Key], [AddonModel], [String]) in
      var dropped: [Key] = []
      var models: [AddonModel] = []
      for (kind, shas) in wanted {
        keeping[kind] = shas
        for model in served[kind]?.drop(keeping: shas) ?? [] {
          dropped.append(Key(kind: kind, sha256: model.sha256))
          models.append(model)
        }
      }
      for key in dropped { remembered.remove(key) }
      let unfailed = failed.keys.filter { !union.contains($0) }
      for sha256 in unfailed { failed[sha256] = nil }
      for sha256 in strikes.keys where !union.contains(sha256) { strikes[sha256] = nil }
      // an engine another interface still serves stays open
      let unused = models.filter { model in !served.values.contains { $0.model(model.sha256) === model } }
      return (dropped, unused, unfailed)
    }
    close(unloaded)
    // queued and not started: never built
    func notWanted(_ job: Job) -> Bool { wanted[job.kind].map { !$0.contains(job.sha256) } ?? false }
    let unqueued = jobs.withLock { () -> [Job] in
      let gone = queue.filter(notWanted)
      queue.removeAll(where: notWanted)
      return gone
    }
    for job in unqueued { finish(Key(kind: job.kind, sha256: job.sha256), ["state": "failed", "detail": "the comma no longer wants it"]) }
    for key in dropped {
      let name = lock.withLock { names[key.sha256] } ?? String(key.sha256.prefix(16))
      log.info("\(AddonWire.triple(key.kind)): the comma turned \(name) off, no longer serving it")
    }
    if !unfailed.isEmpty { saveFailed() }
    let unserved = forgetServing(wanted)
    let droppedShas: Set<String> = Set(dropped.map { $0.sha256 })
    deleteIfUnused(droppedShas.union(unserved).union(unqueued.map { $0.sha256 }))
  }

  /// Closes engines nothing serves any more. A TensorRT engine frees its
  /// memory only on close; under the host's lock, as no pass runs then, and
  /// one waiting for the lock finds its model gone (handle).
  private func close(_ models: [AddonModel]) {
    guard !models.isEmpty else { return }
    host.lock.lock()
    defer { host.lock.unlock() }
    var closed = Set<ObjectIdentifier>()
    for model in models where closed.insert(ObjectIdentifier(model)).inserted { model.engine.close() }
  }

  /// Leaves only `wanted`'s models in serving.json on the interfaces it names; the sha256s taken out.
  private func forgetServing(_ wanted: [[UInt8]: Set<String>]) -> Set<String> {
    servingLock.withLock {
      var mine = loadServing()
      var gone = Set<String>()
      for (kind, shas) in wanted {
        let triple = AddonWire.triple(kind)
        guard let models = mine[triple] else { continue }
        let kept = models.filter { shas.contains($0.key) }
        gone.formUnion(models.keys.filter { !shas.contains($0) })
        mine[triple] = kept.isEmpty ? nil : kept
      }
      guard !gone.isEmpty else { return gone }
      writeTagged(servingURL, mine)
      return gone
    }
  }

  /// Deletes the artifact and the uploaded bytes of each model nothing serves,
  /// builds or means to serve after a restart; never the driving model's.
  private func deleteIfUnused(_ shas: Set<String>) {
    guard !shas.isEmpty else { return }
    let listed = Set(loadServing().values.flatMap(\.keys))
    let loaded = host.loadedSHA()
    for sha256 in shas.sorted() {
      let busy = lock.withLock {
        served.values.contains { $0.has(sha256) } || pending.keys.contains { $0.sha256 == sha256 }
      }
      guard !busy, !listed.contains(sha256), sha256 != loaded else { continue }
      if let entry = try? store.entry(sha256) { entry.remove() }
      if let model = try? host.cache.modelPath(sha256) {
        try? FileManager.default.removeItem(at: model)
        try? FileManager.default.removeItem(at: model.deletingLastPathComponent().appending(path: model.lastPathComponent + ".part"))
      }
      lock.withLock { names[sha256] = nil }
      log.info("deleted \(sha256.prefix(16)): no add-on wants it")
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
    let serving = loadServing()
    // the sidecar's interface counts only for an artifact serving.json does not know
    let listed = Set(serving.values.flatMap(\.keys))
    for (triple, models) in serving {
      guard let kind = AddonWire.interfaceClass(triple) else { continue }
      for (sha256, meta) in models where built.contains(sha256) && seen.insert(Key(kind: kind, sha256: sha256)).inserted {
        let port = (meta["port"] as? NSNumber).flatMap { UInt16(exactly: $0.intValue) }
        out.append((sha256, kind, port, meta["name"] as? String ?? String(sha256.prefix(16))))
      }
    }
    for sha256 in built.sorted() where !listed.contains(sha256) {
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
    readTagged(servingURL) as? [String: [String: [String: Any]]] ?? [:]
  }

  /// This backend's entry in `url`, one of the store's files keyed by backend tag.
  private func readTagged(_ url: URL) -> Any? {
    ((try? JSONSerialization.jsonObject(with: Data(contentsOf: url))) as? [String: Any])?[backend.tag()]
  }

  /// Replaces this backend's entry in `url`, leaving the other backends'.
  private func writeTagged(_ url: URL, _ value: Any) {
    var all = (try? JSONSerialization.jsonObject(with: Data(contentsOf: url))) as? [String: Any] ?? [:]
    all[backend.tag()] = value
    try? FileManager.default.createDirectory(at: store.layout.root, withIntermediateDirectories: true)
    if let data = try? JSONSerialization.data(withJSONObject: all, options: [.sortedKeys]) { try? data.write(to: url, options: .atomic) }
  }

  /// Notes that `key` is served, so a restart serves it on that interface again.
  private func remember(_ key: Key, name: String, port: UInt16?) {
    let new = lock.withLock { remembered.insert(key).inserted }
    guard new else { return }
    servingLock.withLock {
      var mine = loadServing()
      var meta: [String: Any] = ["name": name]
      if let port { meta["port"] = Int(port) }
      mine[AddonWire.triple(key.kind), default: [:]][key.sha256] = meta
      writeTagged(servingURL, mine)
    }
  }

  private func loadFailed() {
    let known = readTagged(failedURL) as? [String: String] ?? [:]
    lock.withLock { failed = known }
  }

  private func saveFailed() {
    writeTagged(failedURL, lock.withLock { failed })
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
    for addon in addons.sorted(by: { $0.name < $1.name }) where !addon.models.isEmpty {
      var models: [[String: Any]] = []
      for model in addon.models.values.sorted(by: { $0.file < $1.file }) {
        let ms: Double = (Double(model.costNs) / 1e5).rounded() / 10
        models.append(["name": model.file, "sha256": model.sha256, "pass_ms": ms])
      }
      var row: [String: Any] = [
        "interface": addon.name, "models": models, "answers": addon.answered, "passes": addon.passes, "yields": addon.yields,
        "failures": addon.failures,
      ]
      if let last = addon.lastAnswer {
        let ago: Double = (now.timeIntervalSince(last) * 10).rounded() / 10
        row["last_answer_s"] = ago
      }
      rows.append(row)
    }
    let queued = building.sorted(by: { $0.sha256 < $1.sha256 }).map {
      ["interface": AddonWire.triple($0.kind), "sha256": $0.sha256, "name": name($0.sha256)]
    }
    let refused = refusals.keys.sorted().map { ["sha256": $0, "name": name($0), "detail": refusals[$0] ?? ""] }
    return ["served": true, "addons": rows, "building": queued, "failed": refused]
  }

  // MARK: serving

  /// Under the lock: `model` served on the interface's add-on, made the
  /// first time a model is ready for it. That new add-on, whose thread the
  /// caller starts once the lock is let go (`startLoop`), or nil.
  private func addLocked(_ model: AddonModel, kind: [UInt8], port: UInt16?) -> Served? {
    if let addon = served[kind] {
      if let port { addon.interface.learn(port: port) }
      addon.add(model)
      return nil
    }
    let addon = Served(kind: kind, name: AddonWire.triple(kind), port: port)
    served[kind] = addon
    addon.add(model)
    return addon
  }

  private func startLoop(_ addon: Served) {
    Thread.detachNewThread { [self] in loop(addon) }
  }

  final class Served: @unchecked Sendable {
    let kind: [UInt8]
    let name: String
    private let lock = NSLock()
    private var ready: [UInt32: AddonModel]
    /// Since the server started; `reported` holds them as the last report left them.
    var passes = 0
    var yields = 0
    var failures = 0
    var reported = (0, 0, 0)
    /// Answers of any kind, ever: whether a link carried anything.
    var answered = 0
    var lastReport = Date()
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

    /// Unloads every model but `shas`; the sha256s unloaded. A pass in
    /// flight holds its own: its engine goes when the pass ends.
    func drop(keeping shas: Set<String>) -> [AddonModel] {
      lock.withLock {
        let gone = ready.filter { !shas.contains($0.value.sha256) }
        for tag in gone.keys { ready[tag] = nil }
        return Array(gone.values)
      }
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
    let cost = model.costNs
    guard gate.admit(cost: cost, wait: max(0, Self.answerBudget - Double(cost) / 1e9)) else {
      model.relax()
      addon.yields += 1
      return AddonWire.answer(seq: request.seq, tag: request.tag, status: .yield)
    }
    defer { gate.passEnded() }
    host.lock.lock()
    defer { host.lock.unlock() }
    // turned off while this pass waited: its engine is closed
    guard addon.models[request.tag] === model else {
      return AddonWire.answer(seq: request.seq, tag: request.tag, status: .unknownModel)
    }
    do {
      let floats = try model.run(request, image: image)
      addon.passes += 1
      let us = UInt32(min(UInt64(UInt32.max), (DispatchTime.now().uptimeNanoseconds - started) / 1000))
      return AddonWire.answer(seq: request.seq, tag: request.tag, status: .ok, serverUs: us, floats: floats)
    } catch {
      addon.failures += 1
      let since = addon.failures - addon.reported.2
      if since == 1 || since % 100 == 0 { log.warning("\(addon.name): \(model.file) failed: \(error)") }
      return AddonWire.answer(seq: request.seq, tag: request.tag, status: .inferFailed)
    }
  }

  private func report(_ addon: Served) {
    let (passes, yields, failures) = (addon.passes - addon.reported.0, addon.yields - addon.reported.1, addon.failures - addon.reported.2)
    guard Date().timeIntervalSince(addon.lastReport) >= Self.reportEvery, passes + yields + failures > 0 else { return }
    let costs = addon.models.values.map { "\($0.file) \(String(format: "%.1f", Double($0.costNs) / 1e6)) ms" }.joined(separator: ", ")
    log.info("\(addon.name): \(passes) passes, \(yields) yielded, \(failures) failed; \(costs)")
    addon.reported = (addon.passes, addon.yields, addon.failures)
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
