import Foundation

enum ModelStatus: Equatable, Sendable {
  case unresolved
  case notDownloaded
  case downloading(frac: Double, rateBps: Double)
  case downloaded
  case preparing(stage: String, frac: Double, msg: String)
  case prepared
  case loaded
  case failed(String)
}

struct ModelRow: Identifiable, Equatable, Sendable {
  /// The sha256 when it is known, else "ref:<ref>".
  var id: String
  var name: String
  var ref: String?
  var sha256: String?
  var bytes: Int64?
  var buildTime: String?
  var status: ModelStatus
  var preparedFor: [InventoryArtifact]
  var isLoaded: Bool
  var isDefault: Bool
  var isRequestedByComma: Bool
  var isLocal: Bool
  /// On disk, in no catalog and with no local record.
  var isOrphan: Bool

  init(
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

/// Turns the four server events plus the engine and link state into the list
/// the Models view renders. A pure function, so it is all the tests need.
enum ModelRowBuilder {
  static let activeDownloadStates: Set<String> = ["started", "progress"]

  static func build(
    catalog: CatalogEvent?,
    inventory: InventoryEvent?,
    downloads: [String: DownloadEvent],
    engine: EngineEvent,
    link: LinkEvent
  ) -> [ModelRow] {
    let models = inventory?.models ?? []
    let artifacts = inventory?.artifacts ?? []
    var modelsBySha: [String: InventoryModel] = [:]
    for model in models { modelsBySha[model.sha256] = model }
    var artifactsBySha: [String: [InventoryArtifact]] = [:]
    for artifact in artifacts { artifactsBySha[artifact.sha256, default: []].append(artifact) }

    var rows: [ModelRow] = []
    var seen: Set<String> = []

    for entry in catalog?.models ?? [] {
      var row = ModelRow(
        id: entry.sha256 ?? "ref:\(entry.ref)",
        name: entry.name,
        ref: entry.ref,
        sha256: entry.sha256,
        bytes: entry.bytes,
        buildTime: entry.buildTime,
        status: .unresolved,
        isDefault: entry.ref == catalog?.defaultRef)
      if let sha = entry.sha256 {
        seen.insert(sha)
        row.preparedFor = artifactsBySha[sha] ?? []
        row.status = status(
          sha: sha,
          download: downloads[sha],
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
