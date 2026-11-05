import Foundation
import JetlinkKit
import JetlinkRegistry
import JetlinkTestSupport
import Testing

@testable import JetlinkServer

/// A registry with one listed model, the tiny queued graph, whose "download"
/// is a copy from the fixtures. Nothing touches the network.
final class FakeRegistry: ModelRegistry, @unchecked Sendable {
  static let ref = String(repeating: "a", count: 40)
  let cache: URL
  let model: URL
  let sha256: String
  let size: Int64
  private let lock = NSLock()
  private(set) var fetches = 0
  /// What the next fetches throw, as a registry with no internet does.
  var failing: RegistryError?

  init(cache: URL, golden: Golden) throws {
    self.cache = cache
    model = golden.model
    sha256 = golden.sha256
    size = Int64(try Data(contentsOf: golden.model).count)
  }

  var catalogEvent: CatalogEvent {
    CatalogEvent(
      fetchedAt: 1, url: "fake", defaultRef: FakeRegistry.ref, error: nil,
      models: [
        CatalogModel(name: "Tiny", shortName: "T", ref: FakeRegistry.ref, buildTime: "", index: 1, sha256: sha256, bytes: size)
      ])
  }

  func cachedCatalog() -> CatalogEvent? { catalogEvent }
  func catalog(refresh: Bool, maxAge: TimeInterval) async -> CatalogEvent { catalogEvent }
  func size(for sha256: String) -> Int64? { sha256 == self.sha256 ? size : nil }
  func ref(for sha256: String) -> String? { sha256 == self.sha256 ? FakeRegistry.ref : nil }
  func name(for sha256: String) -> (name: String?, ref: String?) { sha256 == self.sha256 ? ("Tiny", FakeRegistry.ref) : (nil, nil) }

  func resolvePointer(ref: String) async throws -> (sha256: String, size: Int64) {
    guard ref == FakeRegistry.ref else { throw TestError("unknown ref") }
    return (sha256, size)
  }

  func fetch(sha256: String, size: Int64, progress: @escaping @Sendable (Double) -> Void, shouldStop: @escaping @Sendable () -> Bool) async throws
    -> URL
  {
    lock.withLock { fetches += 1 }
    if let failing { throw failing }
    let destination = cache.appending(path: "models/\(sha256.prefix(16)).onnx")
    progress(0.5)
    try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
    try FileManager.default.copyItem(at: model, to: destination)
    progress(1)
    return destination
  }

  func importModelFile(at url: URL, name: String?, progress: @escaping @Sendable (Double) -> Void) async throws -> String {
    throw TestError("not in this fake")
  }

  func inventory(artifactTag: String?, artifactSuffix: String, loaded: String?) -> InventoryEvent {
    let path = cache.appending(path: "models/\(sha256.prefix(16)).onnx")
    let models =
      FileManager.default.fileExists(atPath: path.path)
      ? [InventoryModel(sha256: sha256, bytes: size, path: path.path, name: "Tiny", ref: FakeRegistry.ref)] : []
    return InventoryEvent(loaded: loaded, lastLoaded: nil, models: models, artifacts: [], disk: InventoryDisk(modelsBytes: 0, enginesBytes: 0, freeBytes: 0))
  }

  func remove(sha256: String, artifacts: Bool, model: Bool) throws {
    if model {
      try? FileManager.default.removeItem(at: cache.appending(path: "models/\(sha256.prefix(16)).onnx"))
    }
    if artifacts {
      let engines = cache.appending(path: "engines")
      for file in (try? FileManager.default.contentsOfDirectory(atPath: engines.path)) ?? [] where file.hasPrefix(String(sha256.prefix(16))) {
        try? FileManager.default.removeItem(at: engines.appending(path: file))
      }
    }
  }
}

/// Events collected off the controller's stream, waited on with a deadline.
final class EventLog: @unchecked Sendable {
  private let lock = NSLock()
  private var events: [ControlEvent] = []

  init(_ stream: AsyncStream<ControlEvent>) {
    Task {
      for await event in stream {
        self.lock.withLock { self.events.append(event) }
      }
    }
  }

  var all: [ControlEvent] { lock.withLock { events } }

  func wait(timeout: TimeInterval = 60, _ predicate: @escaping (ControlEvent) -> Bool) async throws -> ControlEvent {
    let deadline = Date().addingTimeInterval(timeout)
    while Date() < deadline {
      if let found = all.first(where: predicate) { return found }
      try await Task.sleep(for: .milliseconds(20))
    }
    throw TestError("no such event in \(timeout) s; saw \(all.count)")
  }
}

@Suite("Control layer", .serialized)
struct ControllerTests {
  func withController(_ body: (ServerController, FakeRegistry, EventLog) async throws -> Void) async throws {
    let cache = try TemporaryDirectory()
    let server = try Server(
      configuration: Server.Configuration(host: "127.0.0.1", port: 0, cacheRoot: cache.url, preload: false), backend: cpuBackend())
    let registry = try FakeRegistry(cache: cache.url, golden: Golden("tiny_queued"))
    let controller = ServerController(server: server, registry: registry)
    let log = EventLog(controller.events)
    try server.start()
    defer {
      controller.finish()
      server.stop()
    }
    controller.publishInitialState()
    try await body(controller, registry, log)
  }

  @Test("A client's first screen needs no command")
  func initialState() async throws {
    try await withController { _, _, log in
      _ = try await log.wait { if case .server = $0 { true } else { false } }
      _ = try await log.wait { if case .link(let link) = $0 { link.state == .waiting } else { false } }
      _ = try await log.wait { if case .engine(let engine) = $0 { engine.state == .none } else { false } }
      _ = try await log.wait { if case .catalog(let catalog) = $0 { catalog.models.count == 1 } else { false } }
      _ = try await log.wait { if case .inventory = $0 { true } else { false } }
    }
  }

  @Test("Use Model with nothing on disk downloads, prepares and loads it")
  func prepareDownloadsFirst() async throws {
    try await withController { controller, registry, log in
      let reply = await controller.handle(.prepare(sha256: registry.sha256, frameSkip: 4))
      #expect(reply.ok)
      #expect(reply.extras["state"]?.stringValue == "downloading")
      _ = try await log.wait { if case .download(let d) = $0 { d.state == "done" } else { false } }
      _ = try await log.wait { if case .engine(let e) = $0 { e.state == .ready && e.sha256 == registry.sha256 } else { false } }
      #expect(registry.fetches == 1)
      #expect(controller.server.host.loadedSHA() == registry.sha256)

      // Using it again is a load of what is already there.
      let again = await controller.handle(.prepare(sha256: registry.sha256, frameSkip: 4))
      #expect(again.ok)
      #expect(registry.fetches == 1)
    }
  }

  @Test("Forgetting the loaded model unloads it first")
  func forgetUnloads() async throws {
    try await withController { controller, registry, log in
      _ = await controller.handle(.prepare(sha256: registry.sha256, frameSkip: 4))
      _ = try await log.wait { if case .engine(let e) = $0 { e.state == .ready } else { false } }
      let reply = await controller.handle(.forget(sha256: registry.sha256, artifacts: true, model: true))
      #expect(reply.ok)
      #expect(controller.server.host.loadedSHA() == nil)
      #expect(try !FileManager.default.fileExists(atPath: controller.server.cache.modelPath(registry.sha256).path))
    }
  }

  @Test(
    "Commands the server cannot carry out say why",
    arguments: [
      ControlCommand.prepare(sha256: String(repeating: "b", count: 64), frameSkip: 4),
      ControlCommand.prepare(sha256: "not a sha", frameSkip: 4),
      ControlCommand.download(ref: nil, sha256: nil),
      ControlCommand.cancelDownload(sha256: String(repeating: "c", count: 64)),
      ControlCommand.importModel(path: "/no/such/file.onnx"),
      ControlCommand.benchmark(seconds: 10),
      ControlCommand.benchmark(seconds: 0),
      ControlCommand.cancelBenchmark,
    ])
  func refusals(_ command: ControlCommand) async throws {
    try await withController { controller, _, _ in
      let reply = await controller.handle(command)
      #expect(!reply.ok)
      #expect(!(reply.error ?? "").isEmpty)
    }
  }

  // MARK: a model neither end has

  @Test("Use Model is a pick here, by the row chosen")
  func useIsAPick() async throws {
    try await withController { controller, registry, log in
      _ = await controller.handle(.prepare(sha256: registry.sha256, frameSkip: 4, ref: FakeRegistry.ref))
      let pick = try #require(controller.server.host.picks.current())
      #expect(pick.sha256 == registry.sha256 && pick.ref == FakeRegistry.ref && pick.by == "here")
      #expect(pick.nbytes == registry.size)
    }
  }

  @Test("A comma's request for a model it has not got either is downloaded here, then prepared for it")
  func fetchesForTheComma() async throws {
    try await withController { controller, registry, log in
      let port = try #require(controller.server.port)
      let (first, stages, last, served) = try await offThePool {
        let client = try TestClient(port: port)
        defer { client.close() }
        _ = try client.hello()
        try client.sendJSON(
          .engineReq, ["sha256": registry.sha256, "nbytes": registry.size, "frame_skip": 4, "fetch": true, "ref": FakeRegistry.ref])
        let first = try client.recv(.engineResp).json["state"] as? String
        var stages: [String] = []
        while true {
          let reply = try client.recv()
          if reply.type == Wire.Msg.progress.rawValue {
            stages.append(reply.json["stage"] as? String ?? "")
            continue
          }
          let state = reply.json["state"] as? String
          if state != "building" && state != "downloading" {
            return (first, stages, state, (reply.json["spec"] as? [String: Any])?["sha256"] as? String)
          }
        }
      }
      #expect(first == "downloading")
      #expect(stages.contains("download") && stages.contains("build"))
      #expect(last == "ready")
      #expect(served == registry.sha256)
      #expect(registry.fetches == 1)
      _ = log
    }
  }

  @Test("A download only a comma asked for is built only while it still asks: a build unloads the engine")
  func notBuiltForAnAbsentComma() async throws {
    try await withController { controller, registry, log in
      let request = try Request(sha256: registry.sha256, nbytes: registry.size, frameSkip: 4, fetch: true)
      #expect(controller.server.host.request(request, session: nil)["state"] as? String == "downloading")
      _ = try await log.wait { if case .download(let d) = $0 { d.state == "done" } else { false } }
      try await Task.sleep(for: .milliseconds(300))
      #expect(controller.server.host.snapshot().state == .none)
      // its next request builds it from the file
      let again = try Request(sha256: registry.sha256, nbytes: registry.size, frameSkip: 4, fetch: true)
      _ = controller.server.host.request(again, session: nil)
      _ = try await log.wait { if case .engine(let e) = $0 { e.state == .ready && e.sha256 == registry.sha256 } else { false } }
      #expect(registry.fetches == 1)
    }
  }

  @Test("A comma that did not ask for the download hears need_upload, as it always did")
  func noFetchUnasked() async throws {
    try await withController { controller, registry, _ in
      let request = try Request(sha256: registry.sha256, nbytes: registry.size, frameSkip: 4)
      #expect(controller.server.host.request(request, session: nil)["state"] as? String == "need_upload")
      #expect(registry.fetches == 0)
    }
  }

  @Test("A download with no internet ends in need_upload, and the hello says not to count on one for a while")
  func offline() async throws {
    try await withController { controller, registry, log in
      registry.failing = RegistryError(.network, "no route to host")
      let request = try Request(sha256: registry.sha256, nbytes: registry.size, frameSkip: 4, fetch: true)
      #expect(controller.server.host.request(request, session: nil)["state"] as? String == "downloading")
      _ = try await log.wait { if case .download(let d) = $0 { d.state == "failed" } else { false } }
      #expect(controller.server.host.status(registry.sha256)["state"] as? String == "need_upload")
      #expect(!controller.canFetch)
      // the comma uploads it or fetches it itself; asking again starts nothing
      #expect(controller.server.host.request(request, session: nil)["state"] as? String == "need_upload")
      #expect(registry.fetches == 1)
    }
  }

  /// A blocking client's exchange, on a thread of its own rather than one
  /// the server's downloads need.
  func offThePool<T: Sendable>(_ body: @escaping @Sendable () throws -> T) async throws -> T {
    try await withCheckedThrowingContinuation { continuation in
      Thread { continuation.resume(with: Result { try body() }) }.start()
    }
  }
}
