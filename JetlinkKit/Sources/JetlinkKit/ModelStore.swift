import Foundation
import Observation

#if canImport(os)
  import os
#else
  import JetlinkLog
#endif

/// The catalog, what is on disk, and what is happening to it right now.
@MainActor
@Observable
public final class ModelStore {
  public private(set) var catalog: CatalogEvent?
  public private(set) var inventory: InventoryEvent?
  public private(set) var downloads: [String: DownloadEvent] = [:]
  public private(set) var imports: [ImportEvent] = []
  public private(set) var rows: [ModelRow] = []
  /// sha256 to the ref of the catalog model last used for it, so of several
  /// models on one export the one picked shows its progress.
  private var picks: [String: String] = [:]
  /// The last action that failed, for the view to show and clear.
  public var lastError: String?

  public static let defaultFrameSkip = Pinned.defaultFrameSkip
  /// How long a finished download stays on screen before its row goes quiet.
  public nonisolated static let terminalDownloadLinger: Duration = .seconds(3)

  public let server: any ServerControlling

  @ObservationIgnored private let isLive: Bool
  @ObservationIgnored private let log = Logger(subsystem: "io.zoompilot.jetlink", category: "models")
  @ObservationIgnored private var consumeTask: Task<Void, Never>?

  public init(server: any ServerControlling, isLive: Bool = true) {
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

  /// The row for a model, by its sha256: of several on one export, the one
  /// that shows it (`ModelRowBuilder.owners`).
  public func row(for sha256: String?) -> ModelRow? {
    guard let sha256 else { return nil }
    let owner = ModelRowBuilder.owners(of: catalog?.models ?? [], picks: picks)[sha256]
    return rows.first { $0.sha256 == sha256 && (owner == nil || $0.ref == owner) }
  }

  public func apply(_ event: ControlEvent) {
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
  public func engineChanged() {
    rebuild()
  }

  private func rebuild() {
    rows = ModelRowBuilder.build(
      catalog: catalog,
      inventory: inventory,
      downloads: downloads,
      engine: server.engine,
      link: server.link,
      picks: picks)
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
    guard event.isTerminal else { return }
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

  public func refreshCatalog() {
    perform("refresh the catalog") { try await $0.send(.catalog(refresh: true)) }
  }

  public func cancelDownload(_ row: ModelRow) {
    guard let sha = row.sha256 else { return }
    perform("cancel that download") { try await $0.send(.cancelDownload(sha256: sha)) }
  }

  /// True when Use Model has something to do: download the model, prepare
  /// it, load it, or try again after a failure.
  public nonisolated static func canUse(_ row: ModelRow) -> Bool {
    guard row.sha256 != nil else { return false }
    switch row.status {
    case .notDownloaded, .downloaded, .prepared, .failed: return true
    case .unresolved, .downloading, .preparing, .loaded: return false
    }
  }

  /// True when using this model would interrupt the comma that is driving:
  /// a comma is connected and a different model is in use or being prepared.
  public func useNeedsConfirmation(_ row: ModelRow) -> Bool {
    guard server.link.state == .connected else { return false }
    guard let inFlight = server.engine.sha256, server.engine.state != .none else { return false }
    return inFlight != row.sha256
  }

  public func use(_ row: ModelRow) {
    use(row, confirmedInterruption: false)
  }

  /// Use Model, in one step: the server downloads the model when it is not on
  /// disk, prepares it when it has no engine, and loads it.
  public func use(_ row: ModelRow, confirmedInterruption: Bool) {
    guard let sha = row.sha256 else {
      lastError = "The model index does not know that model's checksum. Refresh the model list and try again."
      return
    }
    if useNeedsConfirmation(row) && !confirmedInterruption {
      log.debug("using another model needs confirmation while the comma is connected")
      return
    }
    if let ref = row.ref, picks[sha] != ref {
      picks[sha] = ref
      rebuild()
    }
    perform("use that model") { try await $0.send(.prepare(sha256: sha, frameSkip: ModelStore.defaultFrameSkip, ref: row.ref)) }
  }

  public func unload() {
    perform("stop using the model") { try await $0.send(.unload) }
  }

  public func forget(_ row: ModelRow, artifacts: Bool, model: Bool) {
    guard let sha = row.sha256 else { return }
    perform("delete those files") { try await $0.send(.forget(sha256: sha, artifacts: artifacts, model: model)) }
  }

  public func importModel(at url: URL) {
    let path = url.path(percentEncoded: false)
    perform("import that model") { try await $0.send(.importModel(path: path)) }
  }

  public func clearError() {
    lastError = nil
  }

  private func perform(_ what: String, _ body: @escaping @MainActor (any ServerControlling) async throws -> ReplyEvent) {
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
  public static func preview(
    catalog: CatalogEvent?,
    inventory: InventoryEvent?,
    downloads: [String: DownloadEvent] = [:],
    engine: EngineEvent
  ) -> ModelStore {
    let server = PreviewServer(link: .waiting, engine: engine)
    let store = ModelStore(server: server, isLive: false)
    store.catalog = catalog
    store.inventory = inventory
    store.downloads = downloads
    store.engineChanged()
    return store
  }
}
