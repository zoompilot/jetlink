import Foundation
import JetlinkKit
import JetlinkRegistry

/// What the control layer needs from the model registry: the model index,
/// the bytes, and what is on disk. JetlinkRegistry's `Registry` is the one the
/// apps use; the seam keeps this file testable without the network.
public protocol ModelRegistry: Sendable {
  func cachedCatalog() -> CatalogEvent?
  func catalog(refresh: Bool, maxAge: TimeInterval) async -> CatalogEvent
  /// The ONNX the model index gives a ref: the model's sha256 and size.
  func resolvePointer(ref: String) async throws -> (sha256: String, size: Int64)
  /// The size the model index gives a model, nil for one it does not know.
  func size(for sha256: String) -> Int64?
  func ref(for sha256: String) -> String?
  /// The index's name and ref for a model, as far as it knows them.
  func name(for sha256: String) -> (name: String?, ref: String?)
  /// The model behind an LFS pointer: one the index gave, or one the comma
  /// named from its own copy of it.
  func fetch(sha256: String, size: Int64, progress: @escaping @Sendable (Double) -> Void, shouldStop: @escaping @Sendable () -> Bool) async throws -> URL
  /// Copies a model in and returns its sha256.
  func importModelFile(at url: URL, name: String?, progress: @escaping @Sendable (Double) -> Void) async throws -> String
  func inventory(artifactTag: String?, artifactSuffix: String, loaded: String?) -> InventoryEvent
  func remove(sha256: String, artifacts: Bool, model: Bool) throws
}

/// The control channel's semantics in process: the Swift form of
/// `server/control.py`, without the socket. It turns the host's news into the
/// same events the Mac app hears, and runs the same commands.
///
/// Nothing here is on the frame path. Downloads run one at a time, in order,
/// and `prepare` goes through the host's own request, so there is exactly one
/// piece of code that loads an engine, as in Python.
public final class ServerController: @unchecked Sendable {
  static let catalogMaxAge: TimeInterval = 3600
  static let inventoryTTL: TimeInterval = 2
  static let progressInterval: TimeInterval = 0.25
  static let rateInterval: TimeInterval = 0.5

  public let server: Server

  private let registry: any ModelRegistry
  private let log = ServerLog(category: "control")
  private let lock = NSLock()
  private var link: LinkEvent = .waiting
  private var active: [String: Download] = [:]
  private var queue: [Download] = []
  private var downloading = false
  private var catalogKicked = false
  private var inventoryPayload: InventoryEvent?
  private var inventoryAt: TimeInterval = 0
  private var benchmark: BenchmarkRun?
  /// When a download last failed for want of the internet: `canFetch` says no
  /// for a while after, so a parked comma fetches a new pick itself.
  private var offlineAt: TimeInterval?
  private let observersLock = NSLock()
  private var observers: [(id: Int, listener: @Sendable (ControlEvent) -> Void)] = []
  private var nextObserver = 0
  private var stream: (events: AsyncStream<ControlEvent>, continuation: AsyncStream<ControlEvent>.Continuation)?
  private var finished = false

  /// One download, queued or running, and the last event it published.
  final class Download: @unchecked Sendable {
    let sha256: String
    let ref: String?
    let total: Int64
    var cancel = false
    var frac = 0.0
    var last: DownloadEvent?
    var lastEvent: TimeInterval = 0
    var rate = 0.0
    var rateBytes: Int64 = 0
    var rateAt: TimeInterval = 0
    /// The frame_skip to prepare with once the download lands: asked for by
    /// a `prepare` that found no model file, or by a comma (`forComma`).
    var prepare: Int?
    /// A `prepare` here asked for it, and whether a comma was connected then;
    /// nil when only a comma did.
    var askedHere: Bool?
    /// A comma asked for it and has not got it either (`ModelFetching`): it
    /// waits on this download, and hears how it ends.
    var forComma = false

    init(sha256: String, ref: String?, total: Int64) {
      self.sha256 = sha256
      self.ref = ref
      self.total = total
    }
  }

  /// `streaming` is ignored, until the daemon stops passing it.
  public init(server: Server, registry: any ModelRegistry, streaming: Bool = true) {
    self.server = server
    self.registry = registry
    server.host.subscribe { [weak self] event in self?.onHost(event) }
    server.host.fetcher = self
  }

  /// Every event published from the first time this is read, for one
  /// reader, until `finish`. A host that only observes never makes it, so
  /// nothing piles up.
  public var events: AsyncStream<ControlEvent> {
    observersLock.withLock {
      if let stream { return stream.events }
      let made = AsyncStream.makeStream(of: ControlEvent.self)
      if finished { made.continuation.finish() }
      stream = (made.stream, made.continuation)
      return made.stream
    }
  }

  /// Calls `listener` with every event published from now on, on the thread
  /// that publishes it, beside `events`: any number may listen, and nothing
  /// is kept for them. Listening sends no command and starts nothing, so the
  /// status page can never cause a catalog fetch, a download or a build. The
  /// listener must return quickly: it runs on the server's threads.
  @discardableResult
  public func observe(_ listener: @escaping @Sendable (ControlEvent) -> Void) -> Int {
    observersLock.withLock {
      nextObserver += 1
      observers.append((nextObserver, listener))
      return nextObserver
    }
  }

  public func stopObserving(_ id: Int) {
    observersLock.withLock { observers.removeAll { $0.id == id } }
  }

  /// A first screen as events, from memory and the disk: server, link, engine
  /// and inventory. Unlike `publishInitialState` it publishes nothing and
  /// never fetches the catalog. It takes the host's lock, which a frame holds,
  /// so it is not for a low-priority thread.
  public func currentState() -> [ControlEvent] {
    [.server(serverEvent("serving")), .link(currentLink), .engine(server.host.snapshot()), .inventory(inventory())]
  }

  /// Everything a client needs for its first screen, before it asks.
  public func publishInitialState() {
    publish(.server(serverEvent("serving")))
    publish(.link(currentLink))
    publish(.engine(server.host.snapshot()))
    publish(.inventory(inventory()))
    let catalog = catalogPayload()
    publish(.catalog(catalog))
    lock.lock()
    let pending = active.values.compactMap(\.last)
    let kick = catalog.fetchedAt == nil && !catalogKicked
    if kick { catalogKicked = true }
    lock.unlock()
    for event in pending { publish(.download(event)) }
    if kick {
      Task { await self.refreshCatalog(false) }
    }
  }

  public func finish() {
    lock.lock()
    for download in active.values { download.cancel = true }
    lock.unlock()
    publish(.server(serverEvent("stopping")))
    observersLock.withLock {
      finished = true
      return stream
    }?.continuation.finish()
  }

  // MARK: publishing

  private func publish(_ event: ControlEvent) {
    let (continuation, listeners) = observersLock.withLock { (stream?.continuation, observers) }
    continuation?.yield(event)
    for observer in listeners {
      observer.listener(event)
    }
  }

  private var currentLink: LinkEvent {
    lock.lock()
    defer { lock.unlock() }
    return link
  }

  private func serverEvent(_ state: String) -> ServerEvent {
    let info = server.backend.describe()
    return ServerEvent(state: state, detail: "", backend: info["backend"], runtimeVersion: info["runtime_version"], device: info["device"])
  }

  private func catalogPayload(error: String? = nil) -> CatalogEvent {
    let cached = registry.cachedCatalog() ?? CatalogEvent(fetchedAt: nil, url: "", defaultRef: "", error: nil, models: [])
    guard let error else { return cached }
    return CatalogEvent(fetchedAt: cached.fetchedAt, url: cached.url, defaultRef: cached.defaultRef, error: error, models: cached.models)
  }

  private func inventory(fresh: Bool = false) -> InventoryEvent {
    let now = ProcessInfo.processInfo.systemUptime
    lock.lock()
    if !fresh, let payload = inventoryPayload, now - inventoryAt < ServerController.inventoryTTL {
      lock.unlock()
      return payload
    }
    lock.unlock()
    let payload = registry.inventory(artifactTag: server.backend.tag(), artifactSuffix: server.backend.suffix, loaded: server.host.loadedSHA())
    lock.lock()
    inventoryPayload = payload
    inventoryAt = now
    lock.unlock()
    return payload
  }

  private func publishInventory() {
    publish(.inventory(inventory(fresh: true)))
  }

  private func onHost(_ event: HostEvent) {
    switch event {
    case .progress:
      // The snapshot already carries the stage this progress set.
      publish(.engine(server.host.snapshot()))
    case .engine:
      let snapshot = server.host.snapshot()
      publish(.engine(snapshot))
      if snapshot.state == .ready || snapshot.state == .failed {
        publishInventory()  // a build or a load changed the disk
      }
    case .link(let value):
      lock.lock()
      link = value
      lock.unlock()
      publish(.link(value))
    case .stats(let stats):
      publish(.stats(stats))
    case .benchmark(let event):
      publish(.benchmark(event))
    case .shutdownRequested(let reason):
      publish(.shutdownRequest(ShutdownRequestEvent(reason: reason)))
    }
  }

  // MARK: commands

  enum ControlError: Error, CustomStringConvertible {
    case refused(String)

    var description: String {
      switch self {
      case .refused(let text): return text
      }
    }
  }

  /// Runs one command and replies as the Python server would.
  public func handle(_ command: ControlCommand) async -> ReplyEvent {
    do {
      let extras = try await run(command)
      return ReplyEvent(id: nil, ok: true, error: nil, extras: extras)
    } catch let error as ControlError {
      return ReplyEvent(id: nil, ok: false, error: error.description)
    } catch {
      log.error("the \(command.name) command failed: \(String(describing: error))")
      return ReplyEvent(id: nil, ok: false, error: "\(type(of: error)): \(error)")
    }
  }

  private func run(_ command: ControlCommand) async throws -> [String: JSONValue] {
    switch command {
    case .status:
      publish(.server(serverEvent("serving")))
      publish(.link(currentLink))
      publish(.engine(server.host.snapshot()))
      publish(.inventory(inventory()))
      publish(.catalog(catalogPayload()))
      return [:]
    case .catalog(let refresh):
      Task { await self.refreshCatalog(refresh) }
      return ["queued": .bool(true)]
    case .download(let ref, let sha256):
      return try await download(ref: ref, sha256: sha256)
    case .cancelDownload(let sha256):
      try cancelDownload(sha256)
      return [:]
    case .importModel(let path):
      let url = URL(fileURLWithPath: path)
      guard FileManager.default.fileExists(atPath: url.path) else {
        throw ControlError.refused("\(path) is not a file")
      }
      Task { await self.runImport(url) }
      return ["queued": .bool(true)]
    case .prepare(let sha256, let frameSkip, let ref):
      return try await prepare(sha256, frameSkip: frameSkip, ref: ref)
    case .unload:
      server.host.unload()
      return [:]
    case .forget(let sha256, let artifacts, let model):
      try forget(sha256, artifacts: artifacts, model: model)
      return [:]
    case .inventory:
      publish(.inventory(inventory()))
      return [:]
    case .shutdown:
      publish(.server(serverEvent("stopping")))
      return [:]
    case .benchmark(let seconds):
      try startBenchmark(seconds: seconds)
      return ["queued": .bool(true)]
    case .cancelBenchmark:
      let run = lock.withLock { benchmark }
      guard let run else { throw ControlError.refused("no benchmark is running") }
      run.cancel()
      return [:]
    }
  }

  // MARK: benchmark

  /// The run itself emits its events through the host; this only refuses
  /// what cannot run and gives it a thread.
  private func startBenchmark(seconds: Double) throws {
    guard seconds > 0 && seconds <= 3600 else { throw ControlError.refused("a benchmark runs for 1 to 3600 seconds") }
    if currentLink.state == .connected {
      throw ControlError.refused("a comma is connected; disconnect it to benchmark, or watch the live numbers")
    }
    guard server.host.loadedSHA() != nil else { throw ControlError.refused("no model is loaded; prepare one first") }
    let run = BenchmarkRun()
    let started = lock.withLock {
      if benchmark != nil { return false }
      benchmark = run
      return true
    }
    guard started else { throw ControlError.refused("a benchmark is already running") }
    let thread = Thread { [self] in
      do {
        _ = try server.host.benchmark(seconds: seconds, run: run)
      } catch {
        log.warning("benchmark failed: \(String(describing: error))")
      }
      lock.withLock {
        if benchmark === run { benchmark = nil }
      }
    }
    thread.name = "jetlink-benchmark"
    // The frame path's priority, so the numbers are the car's.
    thread.qualityOfService = .userInteractive
    thread.start()
  }

  private func refreshCatalog(_ refresh: Bool) async {
    publish(.catalog(await registry.catalog(refresh: refresh, maxAge: ServerController.catalogMaxAge)))
  }

  // MARK: downloads

  private func download(ref: String?, sha256: String?) async throws -> [String: JSONValue] {
    guard (ref == nil) != (sha256 == nil) else {
      throw ControlError.refused("a download needs exactly one of ref and sha256")
    }
    var ref = ref
    let pointer: (sha256: String, size: Int64)
    if let sha256 {
      guard CacheLayout.isSHA256(sha256) else { throw ControlError.refused("sha256 must be a lowercase SHA-256 digest") }
      // An LFS object is an oid plus a size, and the index has the size.
      guard let size = registry.size(for: sha256) else {
        throw ControlError.refused("model \(sha256.prefix(16)) is not in the model index")
      }
      pointer = (sha256, size)
      ref = registry.ref(for: sha256)
    } else {
      do {
        pointer = try await registry.resolvePointer(ref: ref!)
      } catch {
        throw ControlError.refused("could not resolve \(ref!): \(error)")
      }
    }
    let path = try server.cache.modelPath(pointer.sha256)
    if FileManager.default.fileExists(atPath: path.path) && (pointer.size == 0 || Files.size(of: path) == pointer.size) {
      throw ControlError.refused("model \(pointer.sha256.prefix(16)) is already downloaded")
    }
    let queued = lock.withLock {
      if active[pointer.sha256] != nil { return false }
      _ = enqueue(Download(sha256: pointer.sha256, ref: ref, total: pointer.size))
      return true
    }
    guard queued else {
      throw ControlError.refused("model \(pointer.sha256.prefix(16)) is already downloading")
    }
    return ["sha256": .string(pointer.sha256)]
  }

  /// Hold `lock`. `download`, set to prepare with `frameSkip` once it lands.
  private func preparing(_ download: Download, frameSkip: Int) -> Download {
    download.prepare = frameSkip
    return download
  }

  /// Hold `lock`, and check `active` first.
  private func enqueue(_ download: Download) -> Download {
    active[download.sha256] = download
    queue.append(download)
    if !downloading {
      downloading = true
      Task { await self.drainDownloads() }
    }
    return download
  }

  /// One download at a time, so two can never fight over the disk.
  private func drainDownloads() async {
    while true {
      let next: Download? = lock.withLock {
        if queue.isEmpty {
          downloading = false
          return nil
        }
        return queue.removeFirst()
      }
      guard let next else { return }
      await runDownload(next)
    }
  }

  private func runDownload(_ state: Download) async {
    if state.cancel {
      downloadEvent(state, "cancelled")
      forgetDownload(state)
      return
    }
    downloadEvent(state, "started")
    do {
      _ = try await registry.fetch(
        sha256: state.sha256, size: state.total,
        progress: { [weak self] frac in self?.downloadProgress(state, frac) },
        shouldStop: { state.cancel })
    } catch {
      if state.cancel {
        downloadEvent(state, "cancelled")
      } else {
        log.warning("downloading \(state.sha256.prefix(16)) failed: \(String(describing: error))")
        downloadEvent(state, "failed", detail: "\(type(of: error)): \(error)")
        if let error = error as? RegistryError, error.isNetwork {
          lock.withLock { offlineAt = ProcessInfo.processInfo.systemUptime }
        }
      }
      forgetDownload(state)
      // after the forget: the comma's next look at the model finds no download
      server.host.fetchEnded(state.sha256)
      return
    }
    lock.withLock { offlineAt = nil }
    if state.cancel {
      downloadEvent(state, "cancelled")
      forgetDownload(state)
      server.host.fetchEnded(state.sha256)
      return
    }
    downloadEvent(state, "done", frac: 1)
    forgetDownload(state)
    publishInventory()
    if state.prepare != nil {
      prepareDownloaded(state)
    }
  }

  /// The second half of a `prepare` that had to download first. A comma that
  /// connected during the download and is driving on another model keeps it.
  /// One only a comma asked for is built only while the comma still asks for
  /// it: a build unloads the engine, and between two sessions of a drive the
  /// engine loaded is the one driving. Its next request builds it.
  private func prepareDownloaded(_ state: Download) {
    let (frameSkip, askedHere) = lock.withLock { (state.prepare, state.askedHere) }
    guard let frameSkip else { return }
    server.host.lock.lock()
    let wanted = server.host.session?.request?.sha256
    server.host.lock.unlock()
    let skip: Bool
    if let overComma = askedHere {
      skip = currentLink.state == .connected && wanted != nil && wanted != state.sha256 && !overComma
    } else {
      skip = wanted != state.sha256
    }
    if skip {
      log.info("downloaded \(state.sha256.prefix(16)); not preparing it over the model the comma is using")
      return
    }
    do {
      let request = try Request(sha256: state.sha256, nbytes: nbytes(state.sha256), frameSkip: frameSkip)
      _ = server.host.request(request, session: nil)
    } catch {
      log.error("preparing \(state.sha256.prefix(16)) after its download failed: \(String(describing: error))")
    }
  }

  private func downloadProgress(_ state: Download, _ frac: Double) {
    if frac < 1 && ProcessInfo.processInfo.systemUptime - state.lastEvent < ServerController.progressInterval {
      return
    }
    downloadEvent(state, "progress", frac: frac)
  }

  private func downloadEvent(_ state: Download, _ kind: String, frac: Double? = nil, detail: String = "") {
    let now = ProcessInfo.processInfo.systemUptime
    lock.lock()
    if let frac {
      state.frac = min(max(frac, 0), 1)
    }
    let done = Int64(state.frac * Double(state.total))
    let elapsed = now - state.rateAt
    if state.rateAt > 0 && elapsed >= ServerController.rateInterval {
      state.rate = Double(done - state.rateBytes) / elapsed
    }
    if state.rateAt == 0 || elapsed >= ServerController.rateInterval {
      state.rateBytes = done
      state.rateAt = now
    }
    state.lastEvent = now
    let event = DownloadEvent(
      sha256: state.sha256, ref: state.ref, state: kind, frac: pythonRound(state.frac, 4),
      bytes: done, total: state.total, rateBps: pythonRound(state.rate, 1), detail: detail, source: nil)
    state.last = event
    let forComma = state.forComma
    lock.unlock()
    publish(.download(event))
    if forComma && (kind == "started" || kind == "progress") {
      server.host.fetchProgress(state.sha256, state.frac, bytes: done, total: state.total)
    }
  }

  private func forgetDownload(_ state: Download) {
    lock.lock()
    if active[state.sha256] === state {
      active[state.sha256] = nil
    }
    lock.unlock()
  }

  private func cancelDownload(_ sha256: String) throws {
    lock.lock()
    guard let state = active[sha256] else {
      lock.unlock()
      throw ControlError.refused("no download of \(sha256.isEmpty ? "that model" : String(sha256.prefix(16))) is running")
    }
    state.cancel = true
    let queued = queue.firstIndex { $0 === state }
    if let queued {
      // Never started, so nothing will publish for it but us.
      queue.remove(at: queued)
    }
    lock.unlock()
    if queued != nil {
      downloadEvent(state, "cancelled")
      forgetDownload(state)
    }
  }

  // MARK: imports

  private func runImport(_ url: URL) async {
    let path = url.path
    importEvent(path, "hashing")
    let seen = Locked(-1.0)
    do {
      let sha = try await registry.importModelFile(at: url, name: nil) { [weak self] frac in
        // hashing over the first half, copying over the second, as in Python
        if frac - seen.value >= 0.05 || frac >= 1 {
          seen.value = frac
          self?.importEvent(path, frac < 0.5 ? "hashing" : "copying", frac: frac)
        }
      }
      importEvent(path, "done", frac: 1, sha256: sha)
      publishInventory()
    } catch {
      log.warning("importing \(path) failed: \(String(describing: error))")
      importEvent(path, "failed", frac: 1, detail: "\(type(of: error)): \(error)")
    }
  }

  private func importEvent(_ path: String, _ state: String, frac: Double = 0, sha256: String? = nil, detail: String = "") {
    publish(.importEvent(ImportEvent(path: path, state: state, frac: pythonRound(frac, 4), sha256: sha256, detail: detail)))
  }

  // MARK: prepare, forget

  /// Use this model: a pick made here, which the comma takes at its next
  /// hello unless it has picked another since (PickStore). `ref` is the
  /// catalog row the user chose, for a model two rows share.
  private func prepare(_ sha256: String, frameSkip: Int, ref: String? = nil) async throws -> [String: JSONValue] {
    guard CacheLayout.isSHA256(sha256) else { throw ControlError.refused("sha256 must be a lowercase SHA-256 digest") }
    let entry = try server.cache.entry(sha256)
    let modelPath = try server.cache.modelPath(sha256)
    if !entry.exists && !FileManager.default.fileExists(atPath: modelPath.path) {
      let (reply, size) = try await downloadThenPrepare(sha256, frameSkip: frameSkip)
      pickHere(sha256, ref: ref, nbytes: size)
      return reply
    }
    // The comma's own path, with no comma: one piece of code loads an engine.
    let request = try Request(sha256: sha256, nbytes: nbytes(sha256), frameSkip: frameSkip)
    pickHere(sha256, ref: ref, nbytes: request.nbytes)
    _ = server.host.request(request, session: nil)
    return ["state": .string(server.host.snapshot().state.rawValue)]
  }

  private func pickHere(_ sha256: String, ref: String?, nbytes: Int64) {
    let known = registry.name(for: sha256)
    server.host.picks.pickHere(sha256: sha256, ref: ref ?? known.ref, nbytes: nbytes, name: known.name)
  }

  /// `prepare` for a model with nothing on disk: download it, prepare it after.
  /// A download already running for it is joined rather than refused. The
  /// reply, and the model's size.
  private func downloadThenPrepare(_ sha256: String, frameSkip: Int) async throws -> ([String: JSONValue], Int64) {
    let overComma = currentLink.state == .connected
    let joined: Int64? = lock.withLock {
      guard let state = active[sha256] else { return nil }
      preparing(state, frameSkip: frameSkip).askedHere = overComma
      return state.total
    }
    if let joined {
      return (["state": .string("downloading"), "sha256": .string(sha256)], joined)
    }
    guard let size = registry.size(for: sha256) else {
      throw ControlError.refused("model \(sha256.prefix(16)) is not downloaded, and is not in the model index")
    }
    let ref = registry.ref(for: sha256)
    lock.withLock {
      // Joined by the lock with forgetDownload, so a download still listed
      // here is one that will read `prepare` when it finishes.
      preparing(active[sha256] ?? enqueue(Download(sha256: sha256, ref: ref, total: size)), frameSkip: frameSkip)
        .askedHere = overComma
    }
    return (["state": .string("downloading"), "sha256": .string(sha256)], size)
  }

  /// What the model weighs, for a request that has no client behind it.
  private func nbytes(_ sha256: String) throws -> Int64 {
    let modelPath = try server.cache.modelPath(sha256)
    if FileManager.default.fileExists(atPath: modelPath.path) {
      return Files.size(of: modelPath)
    }
    if let spec = Artifact.sidecar(try server.cache.entry(sha256).path)["spec"] as? [String: Any],
      let bytes = (spec["nbytes"] as? NSNumber)?.int64Value, bytes > 0
    {
      return bytes
    }
    return 0
  }

  private func forget(_ sha256: String, artifacts: Bool, model: Bool) throws {
    guard CacheLayout.isSHA256(sha256) else { throw ControlError.refused("sha256 must be a lowercase SHA-256 digest") }
    let snapshot = server.host.snapshot()
    if snapshot.sha256 == sha256 && (snapshot.state == .building || snapshot.state == .loading) {
      throw ControlError.refused("a build for this model is running")
    }
    if server.host.loadedSHA() == sha256 {
      server.host.unload()
    }
    try registry.remove(sha256: sha256, artifacts: artifacts, model: model)
    if let remembered = server.cache.lastLoaded(), remembered.sha256 == sha256, try !server.cache.entry(sha256).exists {
      // Preloading an engine that is no longer there costs a start-up failure.
      server.cache.forgetLastLoaded()
    }
    publishInventory()
  }
}

// MARK: downloads for the comma

extension ServerController: ModelFetching {
  /// How long a download that could not reach the internet keeps `canFetch` no.
  static let offlineFor: TimeInterval = 600

  /// A model the comma asked for that neither end has: downloaded here by the
  /// oid and size the comma named, then prepared for it (prepareDownloaded),
  /// joining a download already running for it. Nothing when this server
  /// cannot reach the internet: the comma hears need_upload and fetches the
  /// model itself at once.
  func fetch(_ request: Request) {
    guard canFetch else { return }
    let joined = lock.withLock {
      let running = active[request.sha256]
      preparing(running ?? enqueue(Download(sha256: request.sha256, ref: request.ref, total: request.nbytes)), frameSkip: request.frameSkip)
        .forComma = true
      return running != nil
    }
    log.info("\(joined ? "the comma waits on the download of" : "downloading") \(request.sha256.prefix(16)) (\(request.nbytes >> 20) MB) for the comma")
  }

  func downloading(_ sha256: String) -> (bytes: Int64, total: Int64)? {
    lock.withLock {
      guard let state = active[sha256], state.forComma else { return nil }
      return (state.last?.bytes ?? 0, state.total)
    }
  }

  var canFetch: Bool {
    let offline = lock.withLock { offlineAt.map { ProcessInfo.processInfo.systemUptime - $0 < Self.offlineFor } ?? false }
    return !offline && Reachability.hasDefaultRoute()
  }
}
