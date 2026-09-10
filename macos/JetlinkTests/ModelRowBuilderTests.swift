import Foundation
import Testing

@testable import Jetlink

struct ModelRowBuilderTests {
  // The five catalog refs whose pointers the fixture has resolved.
  private static let ctmv2 = "1c0f7e2b7bb1d0f4a2c58e6d3f9a4b7c8e5d2a1f0b3c6d9e8f7a4b1c2d3e4f50"
  private static let ctm = "2d1e8f3c9cc2e1f5b3d69f7e4a0b5c8d9f6e3b2a1c4d7e0f9a8b5c2d3e4f5061"
  private static let bmv4 = "a086d5249fc308bb73993d1e64630c669d4c7df5bde85f42ad61902543648525"
  private static let sad = "3e2f9a4d0dd3f2a6c4e70a8f5b1c6d9e0a7f4c3b2d5e8f1a0b9c6d3e4f506172"
  private static let bmv3 = "4f3a0b5e1ee4a3b7d5f81b9a6c2d7e0f1b8a5d4c3e6f9a2b1c0d7e4f50617283"
  private static let localSha = "5a4b1c6f2ff5b4c8e6a92cab7d3e8f102c9b6e5d4f7a0b3c2d1e8f5061728394"
  private static let orphanSha = "6b5c2d7a30061c5d9f7ba3bc8e4f90213dac7f6e5a8b1c4d3e2f90617283a4b5"

  private func catalogEvent() throws -> CatalogEvent {
    let data = try Fixture.data("catalog_event.json")
    guard case .catalog(let catalog) = try ControlEvent(jsonLine: data) else {
      throw CocoaError(.fileReadCorruptFile)
    }
    return catalog
  }

  private func artifact(sha: String, backend: String = "ort", device: String = "coreml-Apple_M1_Pro", current: Bool) -> InventoryArtifact {
    InventoryArtifact(
      sha256: sha,
      key: "\(sha.prefix(16)).\(backend)1.29.0.\(device)",
      path: "/cache/engines/\(sha.prefix(16)).\(backend)1.29.0.\(device).ortcache",
      bytes: 5_900_000_000,
      backend: backend,
      runtimeVersion: "1.29.0",
      device: device,
      builtAt: "2026-09-08T21:19:15Z",
      buildSeconds: 548.9,
      checkpoint: nil,
      current: current)
  }

  private func model(sha: String, name: String?, ref: String? = nil) -> InventoryModel {
    InventoryModel(sha256: sha, bytes: 765_953_504, path: "/cache/models/\(sha.prefix(16)).onnx", name: name, ref: ref)
  }

  private func inventory(models: [InventoryModel], artifacts: [InventoryArtifact], loaded: String? = nil) -> InventoryEvent {
    InventoryEvent(
      loaded: loaded,
      lastLoaded: loaded,
      models: models,
      artifacts: artifacts,
      disk: InventoryDisk(modelsBytes: 765_953_504, enginesBytes: 5_900_000_000, freeBytes: 120_000_000_000))
  }

  private func download(sha: String, state: String, frac: Double = 0.42, rate: Double = 41_000_000, detail: String = "") -> DownloadEvent {
    DownloadEvent(
      sha256: sha,
      ref: nil,
      state: state,
      frac: frac,
      bytes: 321_000_000,
      total: 765_953_504,
      rateBps: rate,
      detail: detail,
      source: nil)
  }

  @Test func catalogRowsKeepCatalogOrderAndFlagTheDefault() throws {
    let catalog = try catalogEvent()
    let rows = ModelRowBuilder.build(catalog: catalog, inventory: nil, downloads: [:], engine: .none, link: .waiting)
    #expect(rows.count == 13)
    #expect(rows.map(\.ref) == catalog.models.map { $0.ref })
    #expect(rows[0].name == "Cinque Terre Model V2 (September 08, 2026)")
    let defaults = rows.filter(\.isDefault)
    #expect(defaults.count == 1)
    #expect(defaults[0].sha256 == ModelRowBuilderTests.bmv4)
  }

  @Test func unresolvedRowsCarryARefIdentifier() throws {
    let rows = try ModelRowBuilder.build(catalog: catalogEvent(), inventory: nil, downloads: [:], engine: .none, link: .waiting)
    guard let row = rows.first(where: { $0.ref == "9d683c06518c0358fb402f38468a7030700c38ac" }) else {
      Issue.record("expected the BMV6 row")
      return
    }
    #expect(row.status == .unresolved)
    #expect(row.sha256 == nil)
    #expect(row.id == "ref:9d683c06518c0358fb402f38468a7030700c38ac")
  }

  @Test func everyStatusIsReachable() throws {
    let catalog = try catalogEvent()
    let engine = EngineEvent(
      state: .building,
      sha256: ModelRowBuilderTests.ctm,
      detail: "",
      stage: "build",
      frac: 0.6,
      msg: "compiling the graph",
      loadOnly: false)
    let event = inventory(
      models: [
        model(sha: ModelRowBuilderTests.bmv4, name: "BMRLNAP Model v4 (August 30, 2026)"),
        model(sha: ModelRowBuilderTests.sad, name: "Sad Model (August 28, 2026)"),
        model(sha: ModelRowBuilderTests.ctm, name: "Cinque Terre Model (September 04, 2026)"),
      ],
      artifacts: [artifact(sha: ModelRowBuilderTests.bmv4, current: true)])
    let downloads = [
      ModelRowBuilderTests.ctmv2: download(sha: ModelRowBuilderTests.ctmv2, state: "progress"),
      ModelRowBuilderTests.bmv3: download(sha: ModelRowBuilderTests.bmv3, state: "failed", detail: "the object is not on any server"),
    ]
    let rows = ModelRowBuilder.build(catalog: catalog, inventory: event, downloads: downloads, engine: engine, link: .waiting)
    var byShortRef: [String: ModelRow] = [:]
    for row in rows { if let sha = row.sha256 { byShortRef[sha] = row } }

    #expect(byShortRef[ModelRowBuilderTests.ctmv2]?.status == .downloading(frac: 0.42, rateBps: 41_000_000))
    #expect(byShortRef[ModelRowBuilderTests.bmv3]?.status == .failed("the object is not on any server"))
    #expect(byShortRef[ModelRowBuilderTests.ctm]?.status == .preparing(stage: "build", frac: 0.6, msg: "compiling the graph"))
    #expect(byShortRef[ModelRowBuilderTests.bmv4]?.status == .prepared)
    #expect(byShortRef[ModelRowBuilderTests.sad]?.status == .downloaded)
    let unresolved = rows.first { $0.ref == "9d683c06518c0358fb402f38468a7030700c38ac" }
    #expect(unresolved?.status == .unresolved)
    let noPointerYet = rows.first { $0.ref == "ba37c02c7c64225b8aec79411d1a8c2f5c1aa343" }
    #expect(noPointerYet?.status == .unresolved)
  }

  @Test func aReadyEngineIsLoadedAndRequestedWhenTheLinkIsUp() throws {
    let catalog = try catalogEvent()
    let engine = EngineEvent(
      state: .ready,
      sha256: ModelRowBuilderTests.bmv4,
      detail: "",
      stage: "load",
      frac: 1.0,
      msg: "",
      loadOnly: true)
    let event = inventory(
      models: [model(sha: ModelRowBuilderTests.bmv4, name: nil)],
      artifacts: [artifact(sha: ModelRowBuilderTests.bmv4, current: true)],
      loaded: ModelRowBuilderTests.bmv4)
    let connected = LinkEvent(state: .connected, detail: "", peer: "usb")
    let rows = ModelRowBuilder.build(catalog: catalog, inventory: event, downloads: [:], engine: engine, link: connected)
    guard let row = rows.first(where: { $0.sha256 == ModelRowBuilderTests.bmv4 }) else {
      Issue.record("expected the BMV4 row")
      return
    }
    #expect(row.status == .loaded)
    #expect(row.isLoaded)
    #expect(row.isRequestedByComma)
    #expect(row.preparedFor.count == 1)

    let waiting = ModelRowBuilder.build(catalog: catalog, inventory: event, downloads: [:], engine: engine, link: .waiting)
    #expect(waiting.first { $0.sha256 == ModelRowBuilderTests.bmv4 }?.isRequestedByComma == false)
  }

  @Test func preparedForListsEveryBackend() throws {
    let catalog = try catalogEvent()
    let event = inventory(
      models: [],
      artifacts: [
        artifact(sha: ModelRowBuilderTests.bmv4, current: true),
        artifact(sha: ModelRowBuilderTests.bmv4, backend: "tinygrad", device: "METAL", current: false),
      ])
    let rows = ModelRowBuilder.build(catalog: catalog, inventory: event, downloads: [:], engine: .none, link: .waiting)
    let row = rows.first { $0.sha256 == ModelRowBuilderTests.bmv4 }
    #expect(row?.preparedFor.count == 2)
    #expect(row?.status == .prepared)
  }

  @Test func aStaleArtifactAloneIsNotPrepared() throws {
    let catalog = try catalogEvent()
    let event = inventory(models: [], artifacts: [artifact(sha: ModelRowBuilderTests.bmv4, backend: "trt", device: "orin", current: false)])
    let rows = ModelRowBuilder.build(catalog: catalog, inventory: event, downloads: [:], engine: .none, link: .waiting)
    #expect(rows.first { $0.sha256 == ModelRowBuilderTests.bmv4 }?.status == .notDownloaded)
  }

  @Test func localAndOrphanRowsFollowTheCatalogInThatOrder() throws {
    let catalog = try catalogEvent()
    let event = inventory(
      models: [
        model(sha: ModelRowBuilderTests.localSha, name: "My exported model"),
        model(sha: ModelRowBuilderTests.orphanSha, name: nil),
      ],
      artifacts: [artifact(sha: "7c6d3e8b41172d6ea08bc4cd9f5a01324ebd8a7f6b9c2d5e4f3a01728394b5c6", current: true)])
    let rows = ModelRowBuilder.build(catalog: catalog, inventory: event, downloads: [:], engine: .none, link: .waiting)
    #expect(rows.count == 16)
    let extras = Array(rows.suffix(3))
    #expect(extras[0].name == "My exported model")
    #expect(extras[0].isLocal)
    #expect(!extras[0].isOrphan)
    #expect(extras[0].status == .downloaded)
    #expect(extras[1].isOrphan)
    #expect(extras[2].isOrphan)
    let orphanNames = Set(extras.dropFirst().map(\.name))
    #expect(orphanNames.contains("Unknown model \(ModelRowBuilderTests.orphanSha.prefix(16))"))
    #expect(orphanNames.contains("Unknown model 7c6d3e8b41172d6e"))
  }

  @Test func aShortShaFromTheInventoryIsAnOrphan() {
    let short = "a086d5249fc308bb"
    let event = inventory(models: [model(sha: short, name: "Something with no identity")], artifacts: [])
    let rows = ModelRowBuilder.build(catalog: nil, inventory: event, downloads: [:], engine: .none, link: .waiting)
    #expect(rows.count == 1)
    #expect(rows[0].isOrphan)
    #expect(rows[0].name == "Something with no identity")
  }

  @Test func aFinishedDownloadFallsThroughToTheDiskState() throws {
    let catalog = try catalogEvent()
    let event = inventory(models: [model(sha: ModelRowBuilderTests.bmv4, name: nil)], artifacts: [])
    let downloads = [ModelRowBuilderTests.bmv4: download(sha: ModelRowBuilderTests.bmv4, state: "done", frac: 1.0)]
    let rows = ModelRowBuilder.build(catalog: catalog, inventory: event, downloads: downloads, engine: .none, link: .waiting)
    #expect(rows.first { $0.sha256 == ModelRowBuilderTests.bmv4 }?.status == .downloaded)
  }

  @Test func noCatalogAndNoInventoryIsAnEmptyList() {
    #expect(ModelRowBuilder.build(catalog: nil, inventory: nil, downloads: [:], engine: .none, link: .waiting).isEmpty)
  }
}
