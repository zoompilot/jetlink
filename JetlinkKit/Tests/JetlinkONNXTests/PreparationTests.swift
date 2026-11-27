import Foundation
import JetlinkTestSupport
import Testing

@testable import JetlinkONNX

/// The Swift preparation held to what Python's makes of the same graphs.
/// Scripts/make_onnx_fixtures.py wrote the graphs, Python's prepared files and
/// python.json (the counts, the weight bytes, or the error Python raised),
/// with cache keys from the prefix "fixture". The files must match byte for
/// byte; Scripts/check_onnx_prep.py does the field-by-field comparison and
/// runs both with onnxruntime.
@Suite struct PreparationTests {
  struct PythonResult: Decodable, Sendable {
    let error: String?
    let stripped: Int?
    let retypedImages: Bool?
    let gathers: Int?
    let gemms: Int?
    let tiles: Int?
    let norms: Int?
    let heads: Int?
    let parts: [String: Int64]?
  }

  static let python = results("python.json")

  /// A results file Python wrote, by case.
  static func results(_ name: String) -> [String: PythonResult] {
    (try? JSONDecoder().decode([String: PythonResult].self, from: Data(contentsOf: Fixtures.url(name)))) ?? [:]
  }

  /// "<fixture>.<layout>" for every case Python ran.
  static let cases: [String] = python.keys.sorted()

  static let layouts: [String: CoreMLPreparation.Layout] = ["split": .split, "whole": .whole, "ane-whole": .aneWhole]

  @Test func fixturesArePresent() {
    #expect(Self.cases.count == 54)
    #expect(Self.cases.contains("stateful.split"))
    #expect(Self.cases.contains("variants.ane-whole"))
  }

  @Test(arguments: cases)
  func matchesPython(_ name: String) throws {
    let expected = try #require(Self.python[name])
    let fixture = String(name.split(separator: ".")[0])
    let layoutName = String(name.split(separator: ".")[1])
    let layout = try #require(Self.layouts[layoutName])
    let out = try TemporaryDirectory()

    let prepare = {
      try CoreMLPreparation.prepare(
        source: Fixtures.url("\(fixture).onnx"), into: out.url, layout: layout,
        cacheKey: { CoreMLPreparation.cacheKey(stem: "fixture", part: $0) })
    }

    if fixture == "unrecorded", layout != .split {
      // The one place the two differ by design: Python asks onnx's shape
      // inferrer for the Gather's input and rewrites it; Swift has no
      // inferrer and leaves a Gather whose input shape is not recorded alone.
      #expect(expected.gathers == 1)
      let report = try prepare()
      #expect(report.gathers == 0)
      #expect(report.retypedImages)
      return
    }
    if fixture == "noshape", layout == .split {
      // Python asks onnx's shape inferrer for the trunk's shape; Swift has none.
      #expect(throws: OnnxError("the export records no shape for trunk; this model cannot be prepared on iPhone or iPad")) {
        try prepare()
      }
      #expect(try FileManager.default.contentsOfDirectory(atPath: out.url.path).isEmpty)
      return
    }
    if fixture == "notype" || fixture == "noentry", layout == .aneWhole {
      // Python asks onnx's shape inferrer for the type of a policy norm's
      // input (notype) or a head's entry (noentry) and carries on; Swift
      // refuses rather than prepare something else.
      #expect(expected.norms == 3 && expected.heads == 6)
      let tensor = fixture == "notype" ? "p" : "vm"
      #expect(throws: OnnxError("the export records no type for \(tensor); this model cannot be prepared on iPhone or iPad")) {
        try prepare()
      }
      #expect(try FileManager.default.contentsOfDirectory(atPath: out.url.path).isEmpty)
      return
    }
    if let message = expected.error {
      #expect(throws: OnnxError(message)) { try prepare() }
      #expect(try FileManager.default.contentsOfDirectory(atPath: out.url.path).isEmpty)
      return
    }

    let report = try prepare()
    #expect(report.stripped == expected.stripped)
    #expect(report.retypedImages == expected.retypedImages)
    #expect(report.gathers == expected.gathers)
    #expect(report.gemms == expected.gemms)
    #expect(report.tiles == expected.tiles)
    #expect(report.norms == (expected.norms ?? 0))
    #expect(report.heads == (expected.heads ?? 0))
    #expect(report.parts.map(\.name) == (layout == .split ? ["vision", "policy"] : ["model"]))
    for part in report.parts {
      #expect(part.weightBytes == expected.parts?[part.name], "\(part.name)")
      #expect(part.url == out.url.appendingPathComponent("\(part.name).onnx"))
      let mine = try Data(contentsOf: part.url)
      let theirs = try Data(contentsOf: Fixtures.url("\(fixture).\(layoutName).\(part.name).expected.onnx"))
      #expect(mine == theirs, "\(name) \(part.name): \(mine.count) bytes against Python's \(theirs.count)")
    }
  }

  /// The names and order tests/test_ane_whole.py pins, read back from the
  /// file the Swift preparation wrote (the bytes are checked above).
  @Test func aneWholeVariantsLayout() throws {
    let out = try TemporaryDirectory()
    let report = try CoreMLPreparation.prepare(
      source: Fixtures.url("variants.onnx"), into: out.url, layout: .aneWhole, cacheKey: { $0 })
    #expect(report.norms == 3 && report.heads == 6)
    let data = try Data(contentsOf: report.parts[0].url)
    let model = try data.withUnsafeBytes { try Decode.model(Source(bytes: $0)) }
    let g = try #require(model.graph)
    let names = g.nodes.map(\.displayName)
    let i = try #require(names.firstIndex(of: "vm"))
    #expect(
      Array(names[(i + 1)..<(i + 10)]) == [
        "vm__cast_fp32", "hln", "h1mm__gemm", "hg", "h2", "hres", "hres__cast_fp16", "hsc", "hsc__cast_fp16",
      ])
    #expect(names.firstIndex(of: "p__prescale") == names.firstIndex(of: "ln1")! - 1)
    let node = { (name: String) in g.nodes.first { $0.displayName == name }! }
    #expect(node("ln1").inputs[0] == "p__scaled" && node("ln3").inputs[0] == "p__scaled" && node("ln4").inputs[0] == "p32")
    #expect(node("hln").inputs == ["vm__fp32", "hs__fp32", "hb__fp32"])
    #expect(node("h1mm__gemm").inputs == ["hln", "h1mm__wt__fp32", "hb1__fp32"])
    #expect(node("hsc").inputs == ["hres__fp32", "s__fp32"] && node("hsc").outputs == ["hsc__fp32"])
    #expect(node("vm__cast_fp32").attribute("to")?.i == 1 && node("hres__cast_fp16").attribute("to")?.i == 10)
    #expect(node("pre").inputs.suffix(2) == ["hres", "hsc"])
    let inits = g.initializers.map(\.key)
    #expect(inits.filter { $0.hasSuffix("__fp32") } == ["h1mm__wt__fp32", "hW2__fp32", "hb__fp32", "hb1__fp32", "hs__fp32", "s__fp32"])
    #expect(inits.contains("s") && !inits.contains("hW2") && !inits.contains("h1mm__wt"))
    #expect(inits.last == "s__fp32")
    let const = try #require(g.initializers.first { $0.key == "__layernorm_prescale_8" })
    #expect(const.dims.isEmpty && const.elementType == DataType.float16)
    let vi = Set(g.valueInfo.map(\.key))
    #expect(vi.isDisjoint(with: ["hln", "hg", "h2"]) && vi.isSuperset(of: ["vm", "hres", "hsc"]))
  }

  @Test func progressRisesToOne() throws {
    let out = try TemporaryDirectory()
    var seen: [Double] = []
    _ = try CoreMLPreparation.prepare(
      source: Fixtures.url("stateful.onnx"), into: out.url, layout: .split,
      cacheKey: { "k\($0)" }, progress: { seen.append($0) })
    #expect(!seen.isEmpty)
    #expect(seen == seen.sorted())
    #expect(seen.last == 1.0)
    #expect(seen.allSatisfy { (0...1).contains($0) })
  }

  /// The caller's key is written as it is given, replacing CACHE_KEY.
  @Test func cacheKeyReplacesOldKeys() throws {
    let out = try TemporaryDirectory()
    let report = try CoreMLPreparation.prepare(
      source: Fixtures.url("queued.onnx"), into: out.url, layout: .whole, cacheKey: { "abc\($0)" })
    let meta = try OnnxMeta.read(contentsOf: report.parts[0].url)
    #expect(meta.props.map(\.key) == ["output_slices", "model_checkpoint", "COREML_CACHE_KEY"])
    #expect(meta.prop("COREML_CACHE_KEY") == "abcmodel")
  }

  /// The split parts carry only the key: Extractor copies no metadata_props.
  @Test func splitPartsCarryOnlyTheKey() throws {
    let out = try TemporaryDirectory()
    let report = try CoreMLPreparation.prepare(
      source: Fixtures.url("queued.onnx"), into: out.url, layout: .split, cacheKey: { $0 })
    let vision = try OnnxMeta.read(contentsOf: report.parts[0].url)
    #expect(vision.props == [.init(key: "COREML_CACHE_KEY", value: "vision")])
    #expect(vision.inputs.map(\.name) == ["img", "big_img"])
    #expect(vision.inputs.allSatisfy { $0.elemType == 10 })
    #expect(vision.outputs.map(\.name) == ["trunk"])
    let policy = try OnnxMeta.read(contentsOf: report.parts[1].url)
    #expect(policy.inputs.map(\.name) == ["trunk", "desire_pulse", "traffic_convention", "features_buffer"])
    #expect(policy.outputs.map(\.name) == ["outputs"])
  }

  /// Two branches that never meet: the cut is every tensor the policy reads.
  @Test func noSingleCut() throws {
    let out = try TemporaryDirectory()
    let report = try CoreMLPreparation.prepare(
      source: Fixtures.url("nocut.onnx"), into: out.url, layout: .split, cacheKey: { $0 })
    let vision = try OnnxMeta.read(contentsOf: report.parts[0].url)
    #expect(vision.outputs.map(\.name) == ["a", "b", "vision_out"])
    let policy = try OnnxMeta.read(contentsOf: report.parts[1].url)
    #expect(policy.inputs.map(\.name) == ["a", "b", "desire"])
  }

  @Test func cacheKeyMatchesPython() {
    // re.sub('[^A-Za-z0-9]', '', stem + part)[:63]
    #expect(
      CoreMLPreparation.cacheKey(stem: "404a18cfd86d2963.ort1.29.0.ane-Apple M1 Pro", part: "vision")
        == "404a18cfd86d2963ort1290aneAppleM1Provision")
    #expect(
      CoreMLPreparation.cacheKey(stem: String(repeating: "é-x", count: 70), part: "policy")
        == String(repeating: "x", count: 63))
  }

  @Test func missingSourceIsAnError() throws {
    let out = try TemporaryDirectory()
    #expect(throws: (any Error).self) {
      try CoreMLPreparation.prepare(
        source: out.url.appendingPathComponent("none.onnx"), into: out.url, layout: .whole, cacheKey: { $0 })
    }
  }

  /// A real model prepared on request, to check against the original with
  /// onnxruntime or to measure: JETLINK_PREPARE=<layout>:<model.onnx>:<directory>,
  /// the layout whole, ane-whole, split, split-pieces (split with the
  /// Neural Engine's weight limit), split-pieces-ane-both (that with the
  /// policy's Neural Engine rewrites, as ane-both on a phone) or plain.
  @Test(.enabled(if: ProcessInfo.processInfo.environment["JETLINK_PREPARE"] != nil))
  func preparesARealModel() throws {
    let spec = ProcessInfo.processInfo.environment["JETLINK_PREPARE"]!.split(separator: ":", maxSplits: 2).map(String.init)
    let layouts: [String: CoreMLPreparation.Layout] = [
      "whole": .whole, "ane-whole": .aneWhole, "split": .split, "split-pieces": .split, "split-pieces-ane-both": .split,
      "plain": .plain,
    ]
    let layout = try #require(layouts[spec[0]])
    let limit = spec[0].hasPrefix("split-pieces") ? CoreMLPreparation.neuralEngineWeightLimit : nil
    let start = Date()
    let report = try CoreMLPreparation.prepare(
      source: URL(fileURLWithPath: spec[1]), into: URL(fileURLWithPath: spec[2]), layout: layout,
      policyOnNeuralEngine: spec[0] == "split-pieces-ane-both", visionWeightLimit: limit, cacheKey: { "jetlink\($0)" })
    print("prepared in \(Date().timeIntervalSince(start)) s: \(report)")
  }
}
