import Foundation
import JetlinkTestSupport
import Testing

@testable import JetlinkONNX

/// The vision trunk in pieces the Neural Engine takes on iPhone and iPad
/// (Split.chunks, CoreMLPreparation's `visionWeightLimit`).
struct ChunkTests {
  /// img through four MatMuls of 64 weight bytes each to trunk. With `skip`,
  /// the last adds img back, so img is read after every possible cut.
  static func chain(skip: Bool = false) -> Model {
    var b = OnnxGraphBuilder()
    b.input("img", DataType.float, [1, 4])
    for i in 1...4 {
      b.fp32("w\(i)", [4, 4], Array(repeating: Float(i), count: 16))
    }
    b.node("MatMul", ["img", "w1"], ["a1"])
    b.node("MatMul", ["a1", "w2"], ["a2"])
    b.node("MatMul", ["a2", "w3"], ["a3"])
    if skip {
      b.node("MatMul", ["a3", "w4"], ["a4"])
      b.node("Add", ["a4", "img"], ["trunk"])
      b.valueInfo("a4", DataType.float, [1, 4])
    } else {
      b.node("MatMul", ["a3", "w4"], ["trunk"])
    }
    for name in ["a1", "a2", "a3"] {
      b.valueInfo(name, DataType.float, [1, 4])
    }
    b.output("trunk", DataType.float, [1, 4])
    return Model(irVersion: 8, graph: b.graph)
  }

  static func io(_ m: Model) -> (inputs: [String], outputs: [String], weights: Int64) {
    let g = m.graph!
    return (g.inputs.map(\.key), g.outputs.map(\.key), g.initializers.reduce(0) { $0 + Int64($1.raw?.count ?? 0) })
  }

  @Test func aTrunkWithinTheLimitStaysWhole() throws {
    let pieces = try Split.chunks(Self.chain(), limit: 256)
    #expect(pieces.count == 1)
    #expect(Self.io(pieces[0]).outputs == ["trunk"])
  }

  @Test func twoEvenPiecesWhenTheyFit() throws {
    let pieces = try Split.chunks(Self.chain(), limit: 150).map(Self.io)
    #expect(pieces.map(\.inputs) == [["img"], ["a2"]])
    #expect(pieces.map(\.outputs) == [["a2"], ["trunk"]])
    #expect(pieces.map(\.weights) == [128, 128])
  }

  @Test func asManyPiecesAsTheLimitNeeds() throws {
    let pieces = try Split.chunks(Self.chain(), limit: 100).map(Self.io)
    #expect(pieces.map(\.weights).allSatisfy { $0 <= 100 })
    #expect(pieces.first?.inputs == ["img"] && pieces.last?.outputs == ["trunk"])
    #expect(pieces.count == 3 || pieces.count == 4)
    for (a, b) in zip(pieces, pieces.dropFirst()) {
      #expect(a.outputs == b.inputs)
    }
  }

  /// No point has one tensor crossing, so nothing is cut.
  @Test func anImageReadLateLeavesItWhole() throws {
    let pieces = try Split.chunks(Self.chain(skip: true), limit: 150)
    #expect(pieces.count == 1)
  }

  /// Under the limit, the preparation writes what it writes without one,
  /// byte for byte.
  @Test func aTrunkUnderTheLimitPreparesAsBefore() throws {
    let a = try TemporaryDirectory()
    let b = try TemporaryDirectory()
    let split = try CoreMLPreparation.prepare(source: Fixtures.url("queued.onnx"), into: a.url, layout: .split, cacheKey: { $0 })
    let limited = try CoreMLPreparation.prepare(
      source: Fixtures.url("queued.onnx"), into: b.url, layout: .split,
      visionWeightLimit: CoreMLPreparation.neuralEngineWeightLimit, cacheKey: { $0 })
    #expect(limited.parts.map(\.name) == ["vision", "policy"])
    for (x, y) in zip(split.parts, limited.parts) {
      #expect(try Data(contentsOf: x.url) == Data(contentsOf: y.url))
    }
  }

  /// Over the limit, the trunk is written as vision.onnx and vision2.onnx,
  /// each piece's output the next one's input, and the policy reads the last.
  @Test func aTrunkOverTheLimitPreparesInPieces() throws {
    var b = OnnxGraphBuilder()
    b.input("img", DataType.float, [1, 4])
    b.input("desire", DataType.float, [1, 4])
    for i in 1...4 {
      b.fp32("w\(i)", [4, 4], Array(repeating: Float(i), count: 16))
    }
    b.node("MatMul", ["img", "w1"], ["a1"])
    b.node("MatMul", ["a1", "w2"], ["a2"])
    b.node("MatMul", ["a2", "w3"], ["a3"])
    b.node("MatMul", ["a3", "w4"], ["trunk"])
    b.node("Add", ["trunk", "desire"], ["outputs"])
    for name in ["a1", "a2", "a3", "trunk"] {
      b.valueInfo(name, DataType.float, [1, 4])
    }
    b.output("outputs", DataType.float, [1, 4])
    let dir = try TemporaryDirectory()
    let source = dir.url.appendingPathComponent("chain.onnx")
    try Data(b.bytes).write(to: source)
    let report = try CoreMLPreparation.prepare(
      source: source, into: dir.url.appendingPathComponent("out"), layout: .split, visionWeightLimit: 150, cacheKey: { "k\($0)" })
    #expect(report.parts.map(\.name) == ["vision", "vision2", "policy"])
    #expect(report.parts.map(\.weightBytes) == [128, 128, 0])
    let metas = try report.parts.map { try OnnxMeta.read(contentsOf: $0.url) }
    #expect(metas.map { $0.inputs.map(\.name) } == [["img"], ["a2"], ["trunk", "desire"]])
    #expect(metas.map { $0.outputs.map(\.name) } == [["a2"], ["trunk"], ["outputs"]])
    #expect(metas.map { $0.prop("COREML_CACHE_KEY") } == ["kvision", "kvision2", "kpolicy"])
  }
}
