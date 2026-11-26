import Foundation
import JetlinkKit
import JetlinkTestSupport
import Testing

@testable import JetlinkORT
@testable import JetlinkServer

/// onnxruntime's profiles: their sessions, layouts and options, which need a
/// Neural Engine or a Snapdragon to run, are only checked. The CPU profile
/// serves the golden frames in ServerTests.
@Suite("onnxruntime backend", .serialized)
struct OrtBackendTests {
  @Test("Each profile's sessions and layout")
  func profiles() {
    #if os(Android)
      let cpuLayout = "plain"
    #else
      let cpuLayout = "whole"
    #endif
    let table: [(OrtProfile, [String], [OrtUnit], String)] = [
      (.ane, ["vision", "policy"], [.coreML("CPUAndNeuralEngine"), .coreML("CPUAndGPU")], "split"),
      (.aneBoth, ["vision", "policy"], [.coreML("CPUAndNeuralEngine"), .coreML("CPUAndNeuralEngine")], "split"),
      (.coreml, ["model"], [.coreML("CPUAndGPU")], "whole"),
      (.aneWhole, ["model"], [.coreML("ALL")], "aneWhole"),
      (.htp, ["vision", "policy"], [.htp, .qnnGPU], "split"),
      (.htpWhole, ["model"], [.htp], "aneWhole"),
      (.gpu, ["model"], [.qnnGPU], "whole"),
      (.cpu, ["model"], [.cpu], cpuLayout),
    ]
    #expect(table.map(\.0) == OrtProfile.allCases)
    for (profile, names, units, layout) in table {
      #expect(profile.sessions.map(\.name) == names, "\(profile)")
      #expect(profile.sessions.map(\.unit) == units, "\(profile)")
      #expect("\(profile.layout)" == layout, "\(profile)")
    }
    #expect(OrtProfile(rawValue: "ane-whole") == .aneWhole && OrtProfile(rawValue: "htp-whole") == .htpWhole)
    #expect(OrtProfile(rawValue: "ane-both") == .aneBoth)
    // The policy's Neural Engine rewrites, and the rebuild they force, are for the profiles that run it there.
    #expect(OrtProfile.allCases.filter(\.policyOnNeuralEngine) == [.aneBoth, .aneWhole])
    #expect(OrtProfile.aneBoth.prepareVersion == 8 && OrtProfile.aneWhole.prepareVersion == 8)
    #expect(OrtProfile.ane.prepareVersion == 5 && OrtProfile.coreml.prepareVersion == 5)
    // The trunk is cut for the Neural Engine on iPhone and iPad only, and only where it runs there.
    #if os(iOS)
      #expect(OrtProfile.allCases.filter { $0.visionWeightLimit != nil } == [.ane, .aneBoth])
      #expect(OrtProfile.aneBoth.visionWeightLimit == 950_000_000)
    #else
      #expect(OrtProfile.allCases.allSatisfy { $0.visionWeightLimit == nil })
    #endif
    #if canImport(Metal)
      #expect(OrtProfile.available == [.ane, .aneBoth, .aneWhole, .coreml, .cpu])
    #elseif os(Android)
      #expect(OrtProfile.available == [.htp, .htpWhole, .gpu, .cpu])
    #else
      #expect(OrtProfile.available == [.cpu])
    #endif
  }

  @Test("Where it prepares for CoreML, the backend says what a model takes on disk")
  func storageNeed() {
    for profile in [OrtProfile.ane, .aneBoth, .aneWhole, .coreml] {
      #expect(OrtBackend(profile: profile, preparer: ONNXPreparer()).storageNeed(modelBytes: 1_000) == .coreML(modelBytes: 1_000), "\(profile)")
    }
    for profile in [OrtProfile.cpu, .htp, .gpu] {
      #expect(OrtBackend(profile: profile, preparer: ONNXPreparer()).storageNeed(modelBytes: 1_000) == nil, "\(profile)")
    }
  }

  @Test("An artifact of another runtime or an older preparation is stale; another profile's current one is not")
  func staleArtifacts() {
    let backend = OrtBackend(profile: .ane, preparer: ONNXPreparer())
    let chip = String(backend.deviceTag().dropFirst("ane-".count))
    func meta(_ profile: OrtProfile, prepare: Int, runtime: String = OrtRuntime.version, chip: String? = nil) -> [String: Any] {
      ["backend": "ort", "onnxruntime": runtime, "device": sanitize("\(profile.rawValue)-\(chip ?? "")"), "prepare": prepare]
    }
    #expect(!backend.isStale(artifactMeta: meta(.ane, prepare: OrtProfile.ane.prepareVersion, chip: chip)))
    #expect(!backend.isStale(artifactMeta: meta(.aneBoth, prepare: OrtProfile.aneBoth.prepareVersion, chip: chip)))
    #expect(backend.isStale(artifactMeta: meta(.aneBoth, prepare: OrtProfile.aneBoth.prepareVersion - 1, chip: chip)))
    #expect(backend.isStale(artifactMeta: meta(.ane, prepare: OrtProfile.ane.prepareVersion, runtime: "1.28.0", chip: chip)))
    // another chip's, another backend's, or one with no record of its preparation: not judged
    #expect(!backend.isStale(artifactMeta: meta(.ane, prepare: 1, chip: "Some_Other_Chip")))
    #expect(!backend.isStale(artifactMeta: ["backend": "trt", "device": "orin", "prepare": 1]))
    #expect(!backend.isStale(artifactMeta: ["backend": "ort", "onnxruntime": OrtRuntime.version, "device": sanitize("ane-\(chip)")]))
  }

  @Test("Pruning removes the stale artifacts and keeps the rest and the protected one")
  func pruneStale() throws {
    let tmp = try TemporaryDirectory()
    let backend = OrtBackend(profile: .aneBoth, preparer: ONNXPreparer())
    let cache = try ServerCache(root: tmp.url, backend: backend)
    let chip = String(backend.deviceTag().dropFirst("ane-both-".count))
    func artifact(_ name: String, _ profile: OrtProfile, prepare: Int) throws -> URL {
      let url = cache.layout.engines.appending(path: "\(name).ort\(OrtRuntime.version).\(profile.rawValue)-\(chip).ortcache")
      try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
      let meta: [String: Any] = ["backend": "ort", "onnxruntime": OrtRuntime.version, "device": sanitize("\(profile.rawValue)-\(chip)"), "prepare": prepare]
      try JSONSerialization.data(withJSONObject: meta).write(to: Artifact.sidecarURL(url))
      return url
    }
    let current = try artifact("aaaaaaaaaaaaaaaa", .aneBoth, prepare: OrtProfile.aneBoth.prepareVersion)
    let otherProfile = try artifact("bbbbbbbbbbbbbbbb", .ane, prepare: OrtProfile.ane.prepareVersion)
    let old = try artifact("cccccccccccccccc", .aneBoth, prepare: OrtProfile.aneBoth.prepareVersion - 1)
    let protected = try artifact("dddddddddddddddd", .aneBoth, prepare: OrtProfile.aneBoth.prepareVersion - 1)
    cache.pruneStale(protect: protected)
    let exists = { (url: URL) in FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) }
    #expect(exists(current) && exists(otherProfile) && exists(protected))
    #expect(!exists(old) && !exists(Artifact.sidecarURL(old)))
  }

  @Test("A trunk in pieces runs every piece on the vision session's unit")
  func piecesRunOnTheVisionUnit() throws {
    let pieces = ["vision", "vision2", "policy"]
    let both = try OrtBackend.sessions(for: pieces, OrtProfile.aneBoth.sessions)
    #expect(both.map(\.name) == pieces)
    #expect(both.map(\.unit) == Array(repeating: .coreML("CPUAndNeuralEngine"), count: 3))
    let ane = try OrtBackend.sessions(for: pieces, OrtProfile.ane.sessions)
    #expect(ane.map(\.unit) == [.coreML("CPUAndNeuralEngine"), .coreML("CPUAndNeuralEngine"), .coreML("CPUAndGPU")])
    #expect(try OrtBackend.sessions(for: ["vision", "policy"], OrtProfile.ane.sessions).map(\.name) == ["vision", "policy"])
    #expect(throws: (any Error).self) { try OrtBackend.sessions(for: ["vision", "policy", "vision2"], OrtProfile.ane.sessions) }
    #expect(throws: (any Error).self) { try OrtBackend.sessions(for: ["model"], OrtProfile.ane.sessions) }
  }

  /// Only a trunk prepared whole and over the limit rebuilds: Light Speed 2's
  /// 667 MB trunk, and pieces, load as they are.
  @Test("Only an oversized whole trunk is stale")
  func oversizedTrunk() throws {
    let dir = try TemporaryDirectory()
    func sparse(_ name: String, _ size: UInt64) throws {
      let url = dir.url.appending(path: name)
      #expect(FileManager.default.createFile(atPath: url.path, contents: nil))
      let handle = try FileHandle(forWritingTo: url)
      try handle.truncate(atOffset: size)
      try handle.close()
    }
    let limit: Int64 = 950_000_000
    let whole: [[String: Any]] = [["model": "vision.onnx"], ["model": "policy.onnx"]]
    try sparse("vision.onnx", 1_691_284_595)
    #expect(OrtBackend.oneOversizedTrunk(dir.url, whole, limit: limit))
    try sparse("vision.onnx", 854_634_215)
    try sparse("vision2.onnx", 836_651_214)
    #expect(!OrtBackend.oneOversizedTrunk(dir.url, [["model": "vision.onnx"], ["model": "vision2.onnx"], ["model": "policy.onnx"]], limit: limit))
    try sparse("vision.onnx", 699_000_000)
    #expect(!OrtBackend.oneOversizedTrunk(dir.url, whole, limit: limit))
    // a piece at the limit, its graph on top, is not mistaken for a whole trunk
    try sparse("vision.onnx", UInt64(limit) + 3_000_000)
    #expect(!OrtBackend.oneOversizedTrunk(dir.url, whole, limit: limit))
  }

  @Test("Each unit's session plan and provider options")
  func plans() {
    let backend = OrtBackend(profile: .htp, preparer: ONNXPreparer())
    #expect(backend.providerOptions(.htp)["backend_type"] == "htp")
    #expect(backend.providerOptions(.htp)["htp_performance_mode"] == "burst")
    #expect(backend.providerOptions(.qnnGPU)["backend_type"] == "gpu")
    let sustained = OrtBackend(profile: .htp, preparer: ONNXPreparer(), keepAlive: false)
    #expect(sustained.providerOptions(.htp)["htp_performance_mode"] == "sustained_high_performance")

    let model = URL(fileURLWithPath: "/tmp/model.onnx")
    let npu = backend.plan(.htp, model: model, cache: nil)
    #expect(npu.provider == "QNN" && npu.label == "QNN(htp)" && npu.usesNeuralEngine && !npu.usesGPU)
    let gpu = backend.plan(.qnnGPU, model: model, cache: nil)
    #expect(gpu.provider == "QNN" && gpu.label == "QNN(gpu)" && gpu.usesGPU)
    let cpu = backend.plan(.cpu, model: model, cache: nil)
    #expect(cpu.provider == nil && cpu.label == "CPU" && cpu.threads == HostChip.cpuThreads)
    let coreML = backend.plan(.coreML("ALL"), model: model, cache: URL(fileURLWithPath: "/tmp/coreml-model"))
    #expect(coreML.provider == "CoreML" && coreML.label == "CoreML(ALL)" && coreML.usesGPU && coreML.usesNeuralEngine)
    #expect(coreML.options["ModelCacheDirectory"] == "/tmp/coreml-model")
    #expect(coreML.options["SpecializationStrategy"] == "FastPrediction")
    #expect(backend.plan(.coreML("CPUAndGPU"), model: model, cache: nil).options["SpecializationStrategy"] == nil)
  }

  /// An entry names its unit as the backend that wrote it did: QNN's and the
  /// CPU's by unit, CoreML's by compute units, the CPU as null units in older
  /// Apple artifacts.
  @Test("Manifest entries, old and new, read back to their units")
  func manifestEntries() {
    for unit in [OrtUnit.coreML("CPUAndNeuralEngine"), .coreML("ALL"), .htp, .qnnGPU, .cpu] {
      #expect(OrtUnit(entry: unit.entry(model: "model.onnx", session: "model")) == unit, "\(unit)")
    }
    #expect(OrtUnit.coreML("CPUAndGPU").entry(model: "policy.onnx", session: "policy")["cache"] as? String == "coreml-policy")
    #expect(OrtUnit(entry: ["model": "model.onnx", "units": NSNull(), "cache": NSNull()]) == .cpu)
    #expect(OrtUnit(entry: ["model": "model.onnx"]) == nil)
    #expect(OrtUnit(entry: ["model": "model.onnx", "unit": "npu"]) == nil)
  }

  @Test("The chip names the artifacts")
  func chip() {
    let backend = OrtBackend(profile: .htp, preparer: ONNXPreparer(), chip: "SM8650")
    #expect(backend.deviceTag() == "htp-SM8650")
    #expect(backend.tag() == "ort\(sanitize(OrtRuntime.version)).htp-SM8650")
    #expect(OrtBackend(profile: .htp, preparer: ONNXPreparer(), chip: "").deviceTag() == "htp-unknown")
    #expect(OrtBackend(profile: .cpu, preparer: ONNXPreparer()).deviceTag() == sanitize("cpu-\(HostChip.name())"))
  }
}
