import Foundation
import JetlinkTestSupport
import Testing

@testable import JetlinkRegistry

/// tests/test_registry.py's importing, inventory and removal sections.
struct ImportTests {
  @Test func hashesCopiesAndRecords() async throws {
    let tmp = try TemporaryDirectory()
    let registry = Registry(layout: tmp.layout)
    let source = tmp.url.appending(path: "big_driving_supercombo.onnx")
    try RegistryFixture.blob.write(to: source)
    let seen = ProgressLog()

    let local = try await registry.importModel(at: source, progress: seen.callback)

    #expect(local.sha256 == RegistryFixture.blobSHA && local.bytes == Int64(RegistryFixture.blob.count))
    #expect(local.name == "big_driving_supercombo")
    #expect(try Data(contentsOf: registry.modelPath(sha256: RegistryFixture.blobSHA)) == RegistryFixture.blob)
    #expect(seen.all.last == 1.0)
    #expect(seen.all.contains(0.5), "the hash is the first half")
    #expect(tmp.names("models") == ["\(RegistryFixture.blobSHA.prefix(16)).onnx"], "no .part is left")

    _ = try await registry.importModel(at: source, name: "again")
    #expect(registry.localModels().map(\.name) == ["again"])
    #expect(registry.name(for: RegistryFixture.blobSHA) == ("again", nil))
  }

  @Test func refusesAFileThatIsNotAnONNX() async throws {
    let tmp = try TemporaryDirectory()
    let source = tmp.url.appending(path: "model.bin")
    try RegistryFixture.blob.write(to: source)
    let error = await #expect(throws: RegistryError.self) {
      try await Registry(layout: tmp.layout).importModel(at: source)
    }
    #expect(error?.message.contains("not an .onnx") == true)

    let missing = await #expect(throws: RegistryError.self) {
      try await Registry(layout: tmp.layout).importModel(at: tmp.url.appending(path: "gone.ONNX"))
    }
    #expect(missing?.message.hasSuffix("does not exist") == true)
  }

  @Test func aCancelledImportLeavesNothingBehind() async throws {
    let tmp = try TemporaryDirectory()
    let source = tmp.url.appending(path: "mine.onnx")
    try Data(count: 6 << 20).write(to: source)
    // six 1 MB hash chunks, then the first 4 MB copy chunk, then stop
    let stops = StopScript(Array(repeating: false, count: 7) + [true])
    let error = await #expect(throws: RegistryError.self) {
      try await Registry(layout: tmp.layout).importModel(at: source, shouldStop: stops.callback)
    }
    #expect(error?.message == "import cancelled")
    #expect(tmp.names("models").isEmpty)
    #expect(Registry(layout: tmp.layout).localModels().isEmpty)
  }

  @Test func concurrentImportsKeepEveryRecord() async throws {
    let tmp = try TemporaryDirectory()
    let registry = Registry(layout: tmp.layout)
    var sources: [URL] = []
    for i in 0..<12 {
      let source = tmp.url.appending(path: "m\(i).onnx")
      try Data("model \(i)".utf8).write(to: source)
      sources.append(source)
    }
    try await withThrowingTaskGroup(of: Void.self) { group in
      for source in sources {
        group.addTask { _ = try await registry.importModel(at: source) }
      }
      try await group.waitForAll()
    }
    #expect(Set(registry.localModels().map(\.name)) == Set((0..<12).map { "m\($0)" }))
  }
}

/// A cache with one fake artifact, one ort artifact (a directory), a sidecar
/// with no spec, a model and a .part.
private func buildCache(_ tmp: TemporaryDirectory) throws -> (fake: String, ort: String) {
  let fakeSHA = String(repeating: "b", count: 64)
  let ortSHA = String(repeating: "c", count: 64)
  let cache = EngineCache(layout: tmp.layout, tag: "fake0.1.test", suffix: ".fake", backend: "fake")
  let entry = try cache.entry(fakeSHA)
  try Data("plan".utf8).write(to: entry.path)
  try JSON.object([
    "backend": "fake", "device": "test", "built_at": "2026-09-08T21:19:15Z", "build_seconds": 548.9, "spec": makeSpec(fakeSHA),
  ]).data().write(to: entry.metaPath)

  let artifact = tmp.layout.engines.appending(path: "\(ortSHA.prefix(16)).ort1.29.0.coreml-Apple_M1_Pro.ortcache")
  try FileManager.default.createDirectory(at: artifact.appending(path: "inner"), withIntermediateDirectories: true)
  try Data(repeating: UInt8(ascii: "x"), count: 100).write(to: artifact.appending(path: "inner/model"))
  try Data(repeating: UInt8(ascii: "y"), count: 50).write(to: artifact.appending(path: "session"))
  try JSON.object([
    "backend": "ort", "onnxruntime": "1.29.0", "device": "coreml-Apple_M1_Pro",
    "built_at": "2026-09-08T21:19:15Z", "build_seconds": 548.9, "spec": makeSpec(ortSHA),
  ]).data().write(to: tmp.layout.engines.appending(path: "\(ortSHA.prefix(16)).ort1.29.0.coreml-Apple_M1_Pro.json"))

  try Data("x".utf8).write(to: tmp.layout.engines.appending(path: "nospec.fake"))
  try Data(#"{"backend": "fake"}"#.utf8).write(to: tmp.layout.engines.appending(path: "nospec.json"))
  try Data("onnx bytes".utf8).write(to: cache.layout.modelPath(sha256: fakeSHA))
  try Data("half".utf8).write(to: tmp.layout.models.appending(path: "\(ortSHA.prefix(16)).onnx.part"))
  return (fakeSHA, ortSHA)
}

struct InventoryTests {
  @Test func listsBothBackendsAndMarksTheCurrentOne() throws {
    let tmp = try TemporaryDirectory()
    let (fakeSHA, ortSHA) = try buildCache(tmp)

    let payload = Registry(layout: tmp.layout).inventory(artifactTag: "fake0.1.test", artifactSuffix: ".fake", loaded: nil)

    let bySHA = Dictionary(uniqueKeysWithValues: payload.artifacts.map { ($0.sha256, $0) })
    #expect(Set(bySHA.keys) == [fakeSHA, ortSHA])
    #expect(bySHA[fakeSHA]?.current == true)
    #expect(bySHA[ortSHA]?.current == false)
    #expect(bySHA[ortSHA]?.bytes == 150, "a directory artifact counts everything under it")
    #expect(bySHA[ortSHA]?.runtimeVersion == "1.29.0")
    #expect(bySHA[ortSHA]?.checkpoint == "b9facbcc")
    #expect(bySHA[ortSHA]?.key == "\(ortSHA.prefix(16)).ort1.29.0.coreml-Apple_M1_Pro")
    #expect(bySHA[ortSHA]?.backend == "ort")
    #expect(bySHA[ortSHA]?.device == "coreml-Apple_M1_Pro")
    #expect(bySHA[ortSHA]?.builtAt == "2026-09-08T21:19:15Z")
    #expect(bySHA[ortSHA]?.buildSeconds == 548.9)
    #expect(bySHA[fakeSHA]?.runtimeVersion == nil)
    #expect(bySHA[fakeSHA]?.path == tmp.layout.engines.appending(path: "\(fakeSHA.prefix(16)).fake0.1.test.fake").path(percentEncoded: false))

    #expect(payload.models.map(\.sha256) == [fakeSHA], "a .part is not a model")
    #expect(payload.models.first?.path == (try tmp.layout.modelPath(sha256: fakeSHA)).path(percentEncoded: false))
    #expect(payload.disk.enginesBytes == 154)
    #expect(payload.disk.modelsBytes == Int64("onnx bytes".utf8.count))
    #expect(payload.disk.freeBytes > 0)
    #expect(payload.loaded == nil && payload.lastLoaded == nil)
  }

  @Test func theOrtcacheOfThisDeviceIsCurrent() throws {
    let tmp = try TemporaryDirectory()
    let (fakeSHA, ortSHA) = try buildCache(tmp)
    let payload = Registry(layout: tmp.layout).inventory(artifactTag: "ort1.29.0.coreml-Apple_M1_Pro", artifactSuffix: ".ortcache", loaded: ortSHA)
    #expect(payload.artifacts.filter(\.current).map(\.sha256) == [ortSHA])
    #expect(payload.loaded == ortSHA)
    // the right key with another suffix is not what this server loads
    let wrongSuffix = Registry(layout: tmp.layout).inventory(artifactTag: "fake0.1.test", artifactSuffix: ".ortcache", loaded: nil)
    #expect(!wrongSuffix.artifacts.contains { $0.current })
    _ = fakeSHA
  }

  @Test func withoutABackendNothingIsCurrent() throws {
    let tmp = try TemporaryDirectory()
    _ = try buildCache(tmp)
    let payload = Registry(layout: tmp.layout).inventory(artifactTag: nil, artifactSuffix: ".ortcache", loaded: nil)
    #expect(payload.artifacts.count == 2)
    #expect(payload.artifacts.allSatisfy { !$0.current })
  }

  @Test func aLiteRtArtifactIsPairedAndReportsItsRuntime() throws {
    let tmp = try TemporaryDirectory()
    let sha = String(repeating: "e", count: 64)
    let key = "\(sha.prefix(16)).litert2.2.0.gpu-Tensor_G5"
    let artifact = tmp.layout.engines.appending(path: key + ".litertcache")
    try FileManager.default.createDirectory(at: artifact, withIntermediateDirectories: true)
    try Data(repeating: 1, count: 40).write(to: artifact.appending(path: "model.tflite"))
    try JSON.object(["backend": "litert", "litert": "2.2.0", "device": "gpu-Tensor_G5", "spec": makeSpec(sha)]).data()
      .write(to: tmp.layout.engines.appending(path: key + ".json"))
    let payload = Registry(layout: tmp.layout).inventory(artifactTag: "litert2.2.0.gpu-Tensor_G5", artifactSuffix: ".litertcache", loaded: nil)
    let entry = try #require(payload.artifacts.first)
    #expect(entry.path == artifact.path(percentEncoded: false) && entry.bytes == 40)
    #expect(entry.runtimeVersion == "2.2.0" && entry.current)
  }

  @Test func aModelOfUnknownIdentityIsListedByItsPrefix() throws {
    let tmp = try TemporaryDirectory()
    let registry = Registry(layout: tmp.layout)
    try Data("x".utf8).write(to: tmp.layout.models.appending(path: "\(String(repeating: "d", count: 16)).onnx"))
    try Data("x".utf8).write(to: tmp.layout.models.appending(path: "not-a-model.onnx"))
    let models = registry.inventory(artifactTag: nil, artifactSuffix: ".ortcache", loaded: nil).models
    #expect(models.map(\.sha256) == [String(repeating: "d", count: 16)])
    #expect(models.first?.name == nil && models.first?.ref == nil)
  }

  @Test func aDownloadedModelIsNamedFromTheIndex() async throws {
    let tmp = try TemporaryDirectory()
    let registry = Registry(layout: tmp.layout, session: MockNet(RegistryFixture.indexRoutes()).session)
    _ = await registry.catalog()
    try Data("x".utf8).write(to: registry.modelPath(sha256: RegistryFixture.oid))
    let model = registry.inventory(artifactTag: nil, artifactSuffix: ".ortcache", loaded: nil).models.first
    #expect(model?.sha256 == RegistryFixture.oid)
    #expect(model?.name == RegistryFixture.name)
    #expect(model?.ref == RegistryFixture.ref)
  }

  @Test func lastLoadedComesFromTheMarker() throws {
    let tmp = try TemporaryDirectory()
    let (fakeSHA, _) = try buildCache(tmp)
    EngineCache(layout: tmp.layout, tag: "fake0.1.test", suffix: ".fake", backend: "fake").rememberLoaded(sha256: fakeSHA, frameSkip: 4)
    #expect(Registry(layout: tmp.layout).inventory(artifactTag: nil, artifactSuffix: ".fake", loaded: nil).lastLoaded == fakeSHA)
  }

  @Test func theEventEncodesAsThePythonPayload() throws {
    let tmp = try TemporaryDirectory()
    _ = try buildCache(tmp)
    let payload = Registry(layout: tmp.layout).inventory(artifactTag: "fake0.1.test", artifactSuffix: ".fake", loaded: nil)
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let object = try JSONSerialization.jsonObject(with: encoder.encode(payload)) as? [String: Any]
    // the ort artifact has every field; Codable leaves a nil one out, which is the encoder's business
    let artifact = (object?["artifacts"] as? [[String: Any]])?.first { $0["backend"] as? String == "ort" }
    #expect(
      Set(artifact?.keys.map { $0 } ?? []) == [
        "sha256", "key", "path", "bytes", "backend", "runtime_version", "device", "built_at", "build_seconds", "checkpoint", "current",
      ])
    #expect(Set((object?["disk"] as? [String: Any])?.keys.map { $0 } ?? []) == ["models_bytes", "engines_bytes", "free_bytes"])
  }
}

struct RemoveTests {
  @Test func takesEveryBackendTheModelAndTheMarker() throws {
    let tmp = try TemporaryDirectory()
    let (fakeSHA, _) = try buildCache(tmp)
    let cache = EngineCache(layout: tmp.layout, tag: "fake0.1.test", suffix: ".fake", backend: "fake")
    let other = tmp.layout.engines.appending(path: "\(fakeSHA.prefix(16)).trt10.3.orin.plan")
    try Data("plan".utf8).write(to: other)
    try Data("{}".utf8).write(to: tmp.layout.engines.appending(path: "\(fakeSHA.prefix(16)).trt10.3.orin.json"))
    cache.rememberLoaded(sha256: fakeSHA, frameSkip: 4)
    let registry = Registry(layout: tmp.layout)

    try registry.remove(sha256: fakeSHA, artifacts: true, model: true)

    #expect(!tmp.names("engines").contains { $0.hasPrefix("\(fakeSHA.prefix(16)).") })
    #expect(!Files.exists(try tmp.layout.modelPath(sha256: fakeSHA)))
    #expect(!Files.exists(tmp.layout.lastLoadedURL))
    try registry.remove(sha256: fakeSHA, artifacts: true, model: true)  // nothing left to remove is not an error
  }

  @Test func ofTheModelOnlyKeepsTheEngines() throws {
    let tmp = try TemporaryDirectory()
    let (fakeSHA, _) = try buildCache(tmp)
    let registry = Registry(layout: tmp.layout)
    try registry.remove(sha256: fakeSHA, artifacts: false, model: true)
    #expect(!Files.exists(try registry.modelPath(sha256: fakeSHA)))
    #expect(registry.inventory(artifactTag: nil, artifactSuffix: ".fake", loaded: nil).artifacts.contains { $0.sha256 == fakeSHA })
  }

  @Test func removingAnImportedModelDropsItsLocalRecord() async throws {
    let tmp = try TemporaryDirectory()
    let registry = Registry(layout: tmp.layout)
    let source = tmp.url.appending(path: "mine.onnx")
    try RegistryFixture.blob.write(to: source)
    _ = try await registry.importModel(at: source, name: "mine")
    let keeper = tmp.url.appending(path: "other.onnx")
    try Data("other bytes".utf8).write(to: keeper)
    let other = try await registry.importModel(at: keeper, name: "other")

    try registry.remove(sha256: RegistryFixture.blobSHA, artifacts: false, model: true)

    #expect(registry.localModels().map(\.sha256) == [other.sha256])
    #expect(registry.name(for: RegistryFixture.blobSHA) == (nil, nil))
    #expect(registry.inventory(artifactTag: nil, artifactSuffix: ".ortcache", loaded: nil).models.map(\.sha256) == [other.sha256])
  }

  @Test func removingTheArtifactsOnlyKeepsTheLocalRecord() async throws {
    let tmp = try TemporaryDirectory()
    let registry = Registry(layout: tmp.layout)
    let source = tmp.url.appending(path: "mine.onnx")
    try RegistryFixture.blob.write(to: source)
    _ = try await registry.importModel(at: source, name: "mine")
    try registry.remove(sha256: RegistryFixture.blobSHA, artifacts: true, model: false)
    #expect(registry.localModels().map(\.name) == ["mine"])
  }

  @Test func ofADirectoryArtifact() throws {
    let tmp = try TemporaryDirectory()
    let (_, ortSHA) = try buildCache(tmp)
    try Registry(layout: tmp.layout).remove(sha256: ortSHA, artifacts: true, model: false)
    #expect(!tmp.names("engines").contains { $0.hasPrefix("\(ortSHA.prefix(16)).") })
  }

  @Test func refusesAnIdentityThatIsNotADigest() throws {
    let tmp = try TemporaryDirectory()
    let error = #expect(throws: RegistryError.self) {
      try Registry(layout: tmp.layout).remove(sha256: "../escape", artifacts: true, model: true)
    }
    #expect(error?.kind == .invalidIdentity)
  }
}
