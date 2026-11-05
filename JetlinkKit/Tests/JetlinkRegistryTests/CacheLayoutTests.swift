import Foundation
import JetlinkTestSupport
import Testing

@testable import JetlinkRegistry

/// tests/test_cache.py: keys, pruning, and what survives a restart.
struct CacheLayoutTests {
  private let sha = String(repeating: "b", count: 64)

  private func fake(_ tmp: TemporaryDirectory, version: String = "0.1", device: String = "test", suffix: String = ".fake") -> EngineCache {
    EngineCache(layout: tmp.layout, tag: "fake\(version).\(device)", suffix: suffix, backend: "fake")
  }

  /// An artifact and its sidecar, stamped at `mtime`.
  @discardableResult
  private func artifact(_ cache: EngineCache, _ name: String, mtime: TimeInterval, suffix: String = ".fake") throws -> URL {
    let url = cache.layout.engines.appending(path: name + suffix)
    try Data("plan".utf8).write(to: url)
    try Data("{}".utf8).write(to: cache.layout.engines.appending(path: name + ".json"))
    try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSince1970: mtime)], ofItemAtPath: url.path(percentEncoded: false))
    return url
  }

  @Test func theLayoutIsThePythonServers() {
    let layout = CacheLayout(root: URL(filePath: "/cache", directoryHint: .isDirectory))
    #expect(layout.models.path(percentEncoded: false) == "/cache/models/")
    #expect(layout.engines.path(percentEncoded: false) == "/cache/engines/")
    #expect(layout.indexURL.path(percentEncoded: false) == "/cache/registry/index.json")
    #expect(layout.localModelsURL.path(percentEncoded: false) == "/cache/registry/local-models.json")
    #expect(layout.lastLoadedURL.path(percentEncoded: false) == "/cache/last-loaded.json")
    #expect(CacheLayout.isSHA256(sha) && !CacheLayout.isSHA256(sha.uppercased()) && !CacheLayout.isSHA256(String(sha.dropLast())))
    #expect(CacheLayout.isRef(RegistryFixture.ref) && !CacheLayout.isRef(RegistryFixture.oid))
  }

  @Test func theKeyIsTheModelAndTheBackendTag() throws {
    let tmp = try TemporaryDirectory()
    let cache = fake(tmp, version: "2.0", device: "gpu9")
    let entry = try cache.entry(sha)
    #expect(entry.path == cache.layout.engines.appending(path: "\(sha.prefix(16)).fake2.0.gpu9.fake"))
    #expect(entry.metaPath == cache.layout.engines.appending(path: "\(sha.prefix(16)).fake2.0.gpu9.json"))
    #expect(try cache.layout.modelPath(sha256: sha) == cache.layout.models.appending(path: "\(sha.prefix(16)).onnx"))
  }

  @Test(arguments: ["../escape", "/tmp/escape", "", String(repeating: "a", count: 63), String(repeating: "z", count: 64)])
  func modelIdentityCannotEscapeTheCache(identity: String) throws {
    let tmp = try TemporaryDirectory()
    let cache = fake(tmp)
    let entryError = #expect(throws: RegistryError.self) { try cache.entry(identity) }
    #expect(entryError?.message.contains("SHA-256") == true)
    let pathError = #expect(throws: RegistryError.self) { try cache.layout.modelPath(sha256: identity) }
    #expect(pathError?.kind == .invalidIdentity)
  }

  @Test func twoBackendsKeepSeparateArtifactsForOneModel() throws {
    let tmp = try TemporaryDirectory()
    let a = fake(tmp, version: "1")
    let b = fake(tmp, version: "2")
    try Data("a".utf8).write(to: a.entry(sha).path)
    try JSON.object(["spec": makeSpec(sha)]).data().write(to: a.entry(sha).metaPath)
    #expect(a.inventory() == [sha])
    #expect(b.inventory() == [])
    #expect(!(try b.entry(sha).exists))
  }

  @Test func pruneKeepsTheNewestArtifactsOfThisBackendOnly() throws {
    let tmp = try TemporaryDirectory()
    let cache = fake(tmp)
    for i in 0..<4 {
      try artifact(cache, "m\(i)", mtime: 1_000_000 + Double(i))
    }
    let other = try artifact(cache, "theirs", mtime: 1, suffix: ".plan")
    cache.prune(keep: 2)
    #expect(tmp.names("engines").filter { $0.hasSuffix(".fake") } == ["m2.fake", "m3.fake"])
    #expect(!Files.exists(tmp.layout.engines.appending(path: "m0.json")))
    #expect(Files.exists(other), "another backend's artifact was pruned")
  }

  /// The Jetson boots at 1970 with no network, so a fresh plan can be the
  /// oldest file on disk. Pruning by mtime would delete the build that just
  /// finished and leave the caller reading a sidecar that no longer exists.
  @Test func pruneNeverDropsTheArtifactJustBuilt() throws {
    let tmp = try TemporaryDirectory()
    let cache = fake(tmp)
    try artifact(cache, "old_a", mtime: 2_000_000)
    try artifact(cache, "old_b", mtime: 2_000_001)
    let fresh = try artifact(cache, "fresh", mtime: 1)

    cache.prune(keep: 2, protect: fresh)

    #expect(Files.isFile(fresh))
    #expect(Files.isFile(tmp.layout.engines.appending(path: "fresh.json")))
    #expect(tmp.names("engines").filter { $0.hasSuffix(".fake") }.count == 2)
  }

  /// onnxruntime's artifact is a directory: the model plus a compiled cache.
  @Test func pruneRemovesADirectoryArtifactWhole() throws {
    let tmp = try TemporaryDirectory()
    let cache = fake(tmp, suffix: ".dir")
    for i in 0..<3 {
      let directory = tmp.layout.engines.appending(path: "m\(i).dir")
      try FileManager.default.createDirectory(at: directory.appending(path: "inner"), withIntermediateDirectories: true)
      try Data("x".utf8).write(to: directory.appending(path: "inner/model"))
      try Data("{}".utf8).write(to: tmp.layout.engines.appending(path: "m\(i).json"))
      let stamp = Date(timeIntervalSince1970: 1_000_000 + Double(i))
      try FileManager.default.setAttributes([.modificationDate: stamp], ofItemAtPath: directory.path(percentEncoded: false))
    }
    cache.prune(keep: 1)
    #expect(tmp.names("engines") == ["m2.dir", "m2.json"])
  }

  @Test func entryRemoveTakesBothHalves() throws {
    let tmp = try TemporaryDirectory()
    let cache = fake(tmp)
    let url = try artifact(cache, "gone", mtime: 1)
    let entry = CacheEntry(path: url, metaPath: tmp.layout.engines.appending(path: "gone.json"))
    #expect(entry.exists)
    entry.remove()
    #expect(!entry.exists && !Files.exists(url) && !Files.exists(entry.metaPath))
    entry.remove()  // nothing left to remove is not an error
  }

  @Test func sweepTempDropsOnlyStaleBuildDirectories() throws {
    let tmp = try TemporaryDirectory()
    let cache = fake(tmp)
    let stale = tmp.layout.engines.appending(path: "tmpstale")
    let fresh = tmp.layout.engines.appending(path: "tmpfresh")
    for directory in [stale, fresh] {
      try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
      try Data("x".utf8).write(to: directory.appending(path: "engine.plan"))
    }
    let old = Date().addingTimeInterval(-7 * 3600)
    try FileManager.default.setAttributes([.modificationDate: old], ofItemAtPath: stale.path(percentEncoded: false))

    cache.sweepTemp()

    #expect(!Files.exists(stale))
    #expect(Files.exists(fresh))
  }

  @Test func lastLoadedSurvivesAndNamesTheBackend() throws {
    let tmp = try TemporaryDirectory()
    let cache = fake(tmp)
    #expect(tmp.layout.lastLoaded() == nil)
    cache.rememberLoaded(sha256: sha, frameSkip: 4)
    #expect(tmp.layout.lastLoaded() == LastLoaded(sha256: sha, frameSkip: 4))
    #expect(Files.readJSON(tmp.layout.lastLoadedURL)?["backend"] == "fake")
  }

  /// The Jetson's marker has no backend field; it must still preload.
  @Test func aMarkerWrittenBeforeBackendsStillReads() throws {
    let tmp = try TemporaryDirectory()
    try JSON.object(["sha256": .string(sha), "frame_skip": 4]).data().write(to: tmp.layout.lastLoadedURL)
    #expect(tmp.layout.lastLoaded() == LastLoaded(sha256: sha, frameSkip: 4))
    try Data(#"{"sha256": "nothex", "frame_skip": 4}"#.utf8).write(to: tmp.layout.lastLoadedURL)
    #expect(tmp.layout.lastLoaded() == nil)
  }

  @Test func inventoryRequiresACompatibleArtifactAndSpec() throws {
    let tmp = try TemporaryDirectory()
    let cache = fake(tmp)
    let entry = try cache.entry(sha)
    try Data("plan".utf8).write(to: entry.path)
    try JSON.object(["spec": makeSpec(sha)]).data().write(to: entry.metaPath)
    #expect(cache.inventory() == [sha])
    try FileManager.default.removeItem(at: entry.path)
    #expect(cache.inventory() == [])
  }

  /// A Jetson's sidecars carry trt_version and no backend field.
  @Test func aSidecarFromBeforeBackendsCounts() throws {
    let tmp = try TemporaryDirectory()
    let cache = fake(tmp)
    let entry = try cache.entry(sha)
    try Data("plan".utf8).write(to: entry.path)
    try JSON.object(["trt_version": "10.3.0", "device": "Orin-sm87", "spec": makeSpec(sha)]).data().write(to: entry.metaPath)
    #expect(cache.inventory() == [sha])
  }

  @Test func pythonPathArithmetic() {
    #expect(PythonPath.suffix("a.onnx") == ".onnx")
    #expect(PythonPath.suffix(".onnx") == "")
    #expect(PythonPath.suffix("a.") == "")
    #expect(PythonPath.suffix("noext") == "")
    #expect(PythonPath.stem("x.ort1.29.0.coreml-M1.json") == "x.ort1.29.0.coreml-M1")
    #expect(PythonPath.withSuffix("m0.fake", ".json") == "m0.json")
  }
}
