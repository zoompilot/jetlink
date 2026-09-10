import Foundation
import Observation
import os

/// The catalog, what is on disk, and what is happening to it right now.
@MainActor
@Observable
final class ModelStore {
  private(set) var catalog: CatalogEvent?
  private(set) var inventory: InventoryEvent?
  private(set) var downloads: [String: DownloadEvent] = [:]
  private(set) var imports: [ImportEvent] = []
  private(set) var rows: [ModelRow] = []
  /// The last action that failed, for the view to show and clear.
  var lastError: String?

  static let defaultFrameSkip = 4
  /// How long a finished download stays on screen before its row goes quiet.
  static let terminalDownloadLinger: Duration = .seconds(3)

  let server: ServerStore

  @ObservationIgnored private let isLive: Bool
  @ObservationIgnored private let log = Logger(subsystem: "io.zoompilot.jetlink", category: "models")
  @ObservationIgnored private var consumeTask: Task<Void, Never>?

  init(server: ServerStore, isLive: Bool = true) {
    self.server = server
    self.isLive = isLive
    if isLive {
      consumeTask = Task { [weak self] in
        for await event in server.modelEvents {
          if Task.isCancelled { break }
          self?.apply(event)
        }
      }
      observeServerState()
    }
    rebuild()
  }

  // MARK: events

  func apply(_ event: ControlEvent) {
    switch event {
    case .inventory(let value):
      inventory = value
    case .catalog(let value):
      catalog = value
    case .download(let value):
      downloads[value.sha256] = value
      scheduleTerminalDownloadRemoval(value)
    case .importEvent(let value):
      if let index = imports.firstIndex(where: { $0.path == value.path }) {
        imports[index] = value
      } else {
        imports.append(value)
      }
    default:
      return
    }
    rebuild()
  }

  /// Rebuilds the rows after the engine or the link changed.
  func engineChanged() {
    rebuild()
  }

  private func rebuild() {
    rows = ModelRowBuilder.build(
      catalog: catalog,
      inventory: inventory,
      downloads: downloads,
      engine: server.engine,
      link: server.link)
  }

  private func observeServerState() {
    withObservationTracking {
      _ = server.engine
      _ = server.link
    } onChange: { [weak self] in
      Task { @MainActor in
        guard let self else { return }
        self.rebuild()
        self.observeServerState()
      }
    }
  }

  private func scheduleTerminalDownloadRemoval(_ event: DownloadEvent) {
    guard ["done", "failed", "cancelled"].contains(event.state) else { return }
    let sha = event.sha256
    let state = event.state
    Task { @MainActor [weak self] in
      try? await Task.sleep(for: ModelStore.terminalDownloadLinger)
      guard let self else { return }
      guard self.downloads[sha]?.state == state else { return }
      self.downloads[sha] = nil
      self.rebuild()
    }
  }

  // MARK: actions

  func refreshCatalog() {
    perform("refresh the catalog") { try await $0.send(.catalog(refresh: true)) }
  }

  func download(_ row: ModelRow) {
    guard row.sha256 != nil || row.ref != nil else {
      lastError = "That model has no reference to download."
      return
    }
    let sha = row.sha256
    let ref = sha == nil ? row.ref : nil
    perform("download that model") { try await $0.send(.download(ref: ref, sha256: sha)) }
  }

  func cancelDownload(_ row: ModelRow) {
    guard let sha = row.sha256 else { return }
    perform("cancel that download") { try await $0.send(.cancelDownload(sha256: sha)) }
  }

  /// True when preparing this model would interrupt the comma that is driving:
  /// a comma is connected and a different model is loaded or being prepared.
  func prepareNeedsConfirmation(_ row: ModelRow) -> Bool {
    guard server.link.state == .connected else { return false }
    guard let inFlight = server.engine.sha256, server.engine.state != .none else { return false }
    return inFlight != row.sha256
  }

  func prepare(_ row: ModelRow) {
    prepare(row, confirmedInterruption: false)
  }

  func prepare(_ row: ModelRow, confirmedInterruption: Bool) {
    guard let sha = row.sha256 else {
      lastError = "That model has to be downloaded before it can be prepared."
      return
    }
    if prepareNeedsConfirmation(row) && !confirmedInterruption {
      log.debug("prepare needs confirmation while the comma is connected")
      return
    }
    perform("prepare that model") { try await $0.send(.prepare(sha256: sha, frameSkip: ModelStore.defaultFrameSkip)) }
  }

  func unload() {
    perform("unload the engine") { try await $0.send(.unload) }
  }

  func forget(_ row: ModelRow, artifacts: Bool, model: Bool) {
    guard let sha = row.sha256 else { return }
    perform("delete those files") { try await $0.send(.forget(sha256: sha, artifacts: artifacts, model: model)) }
  }

  func importModel(at url: URL) {
    let path = url.path(percentEncoded: false)
    perform("import that model") { try await $0.send(.importModel(path: path)) }
  }

  func clearError() {
    lastError = nil
  }

  private func perform(_ what: String, _ body: @escaping @MainActor (ServerStore) async throws -> ReplyEvent) {
    guard isLive else { return }
    Task { @MainActor [weak self] in
      guard let self else { return }
      do {
        try await self.server.startIfNeeded()
        let reply = try await body(self.server)
        if !reply.ok {
          self.lastError = reply.error ?? "Jetlink could not \(what)."
        }
      } catch {
        self.log.error("could not \(what, privacy: .public): \(error.localizedDescription, privacy: .public)")
        self.lastError = error.localizedDescription
      }
    }
  }
}

extension ModelStore {
  /// A store with fixed state and nothing behind it. Actions are no-ops.
  static func preview(
    catalog: CatalogEvent?,
    inventory: InventoryEvent?,
    downloads: [String: DownloadEvent] = [:],
    engine: EngineEvent
  ) -> ModelStore {
    let server = ServerStore.preview(link: .waiting, engine: engine)
    let store = ModelStore(server: server, isLive: false)
    store.catalog = catalog
    store.inventory = inventory
    store.downloads = downloads
    store.engineChanged()
    return store
  }
}
