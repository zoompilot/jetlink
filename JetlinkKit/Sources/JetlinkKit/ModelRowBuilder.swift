import Foundation

public enum ModelStatus: Equatable, Sendable {
  /// A catalog row with no sha256. The model index gives every listed model
  /// one, so only a server from before the index sends such a row.
  case unresolved
  case notDownloaded
  case downloading(frac: Double, rateBps: Double)
  case downloaded
  case preparing(stage: String, frac: Double, msg: String)
  case prepared
  case loaded
  case failed(String)
}

public struct ModelRow: Identifiable, Equatable, Sendable {
  /// "ref:<ref>" for a catalog row, the sha256 for one only on disk. Two
  /// catalog models can name the same export (Flip Flop and Mountain Dew v3
  /// do), so a catalog row is never keyed by its sha256.
  public var id: String
  public var name: String
  public var ref: String?
  public var sha256: String?
  public var bytes: Int64?
  public var buildTime: String?
  public var status: ModelStatus
  public var preparedFor: [InventoryArtifact]
  public var isLoaded: Bool
  public var isDefault: Bool
  public var isRequestedByComma: Bool
  public var isLocal: Bool
  /// On disk, in no catalog and with no local record.
  public var isOrphan: Bool

  public init(
    id: String,
    name: String,
    ref: String? = nil,
    sha256: String? = nil,
    bytes: Int64? = nil,
    buildTime: String? = nil,
    status: ModelStatus = .notDownloaded,
    preparedFor: [InventoryArtifact] = [],
    isLoaded: Bool = false,
    isDefault: Bool = false,
    isRequestedByComma: Bool = false,
    isLocal: Bool = false,
    isOrphan: Bool = false
  ) {
    self.id = id
    self.name = name
    self.ref = ref
    self.sha256 = sha256
    self.bytes = bytes
    self.buildTime = buildTime
    self.status = status
    self.preparedFor = preparedFor
    self.isLoaded = isLoaded
    self.isDefault = isDefault
    self.isRequestedByComma = isRequestedByComma
    self.isLocal = isLocal
    self.isOrphan = isOrphan
  }
}

extension ModelRow {
  /// Something on disk to delete: the model, or an engine prepared from it.
  public func hasFiles(in inventory: InventoryEvent?) -> Bool {
    guard let sha256 else { return false }
    return !preparedFor.isEmpty || inventory?.models.contains { $0.sha256 == sha256 } == true
  }
}

extension DownloadEvent {
  /// Done, failed or cancelled: the row lingers a moment, then goes quiet.
  public var isTerminal: Bool { ["done", "failed", "cancelled"].contains(state) }
}

/// Turns the four server events plus the engine and link state into the list
/// the Models view renders. A pure function, so it is all the tests need.
public enum ModelRowBuilder {
  public static let activeDownloadStates: Set<String> = ["started", "progress"]

  public static func build(
    catalog: CatalogEvent?,
    inventory: InventoryEvent?,
    downloads: [String: DownloadEvent],
    engine: EngineEvent,
    link: LinkEvent,
    picks: [String: String] = [:]
  ) -> [ModelRow] {
    let models = inventory?.models ?? []
    let artifacts = inventory?.artifacts ?? []
    var modelsBySha: [String: InventoryModel] = [:]
    for model in models { modelsBySha[model.sha256] = model }
    var artifactsBySha: [String: [InventoryArtifact]] = [:]
    for artifact in artifacts { artifactsBySha[artifact.sha256, default: []].append(artifact) }

    var rows: [ModelRow] = []
    var seen: Set<String> = []
    var refs: Set<String> = []
    let owners = owners(of: catalog?.models ?? [], picks: picks)

    for entry in catalog?.models ?? [] {
      guard refs.insert(entry.ref).inserted else { continue }
      var row = ModelRow(
        id: "ref:\(entry.ref)",
        name: entry.name,
        ref: entry.ref,
        sha256: entry.sha256,
        bytes: entry.bytes,
        buildTime: entry.buildTime,
        status: .unresolved,
        isDefault: entry.ref == catalog?.defaultRef)
      if let sha = entry.sha256 {
        seen.insert(sha)
        // only the owner of a shared export shows its download, engine and load
        let owns = owners[sha] == entry.ref
        let engine = owns ? engine : .none
        row.preparedFor = artifactsBySha[sha] ?? []
        row.status = status(
          sha: sha,
          download: owns ? downloads[sha] : nil,
          engine: engine,
          artifacts: row.preparedFor,
          hasModelFile: modelsBySha[sha] != nil)
        row.isLoaded = engine.sha256 == sha && engine.state == .ready
        row.isRequestedByComma = engine.sha256 == sha && link.state == .connected
        if row.bytes == nil { row.bytes = modelsBySha[sha]?.bytes }
      }
      rows.append(row)
    }

    var extraShas: [String] = []
    for model in models where !seen.contains(model.sha256) {
      if !extraShas.contains(model.sha256) { extraShas.append(model.sha256) }
    }
    for artifact in artifacts where !seen.contains(artifact.sha256) {
      if !extraShas.contains(artifact.sha256) { extraShas.append(artifact.sha256) }
    }

    var localRows: [ModelRow] = []
    var orphanRows: [ModelRow] = []
    for sha in extraShas {
      let model = modelsBySha[sha]
      let name = model?.name
      let shaIsComplete = sha.count == 64
      var row = ModelRow(
        id: sha,
        name: name ?? "Unknown model \(sha.prefix(16))",
        ref: model?.ref,
        sha256: sha,
        bytes: model?.bytes,
        status: .notDownloaded,
        preparedFor: artifactsBySha[sha] ?? [],
        isLocal: name != nil,
        isOrphan: name == nil || !shaIsComplete)
      row.status = status(
        sha: sha,
        download: downloads[sha],
        engine: engine,
        artifacts: row.preparedFor,
        hasModelFile: model != nil)
      row.isLoaded = engine.sha256 == sha && engine.state == .ready
      row.isRequestedByComma = engine.sha256 == sha && link.state == .connected
      if row.isLocal && !row.isOrphan {
        localRows.append(row)
      } else {
        orphanRows.append(row)
      }
    }

    localRows.sort { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    orphanRows.sort { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    rows.append(contentsOf: localRows)
    rows.append(contentsOf: orphanRows)
    return rows
  }

  /// sha256 to the ref of the catalog model that shows that export's download,
  /// engine and load: several can name one export, and only one shows them,
  /// the one picked last (`picks`, sha256 to ref), else the first.
  public static func owners(of models: [CatalogModel], picks: [String: String]) -> [String: String] {
    var owners: [String: String] = [:]
    for model in models {
      guard let sha = model.sha256 else { continue }
      if owners[sha] == nil || model.ref == picks[sha] { owners[sha] = model.ref }
    }
    return owners
  }

  private static func status(
    sha: String,
    download: DownloadEvent?,
    engine: EngineEvent,
    artifacts: [InventoryArtifact],
    hasModelFile: Bool
  ) -> ModelStatus {
    if let download {
      if activeDownloadStates.contains(download.state) {
        return .downloading(frac: download.frac, rateBps: download.rateBps)
      }
      if download.state == "failed" {
        return .failed(download.detail)
      }
    }
    if engine.sha256 == sha {
      switch engine.state {
      case .building, .loading:
        return .preparing(stage: engine.stage ?? "", frac: engine.frac, msg: engine.msg)
      case .ready:
        return .loaded
      case .failed:
        return .failed(engine.detail)
      case .none:
        break
      }
    }
    if artifacts.contains(where: { $0.current }) { return .prepared }
    if hasModelFile { return .downloaded }
    return .notDownloaded
  }
}
