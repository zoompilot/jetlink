import Foundation
import Testing

@testable import JetlinkONNX

/// The rewrites that keep the policy on the Neural Engine, which are Swift's
/// alone: there is no Python output to hold them to, so each is held to the
/// graph it should leave.
@Suite struct NeuralEngineRewriteTests {
  /// Nothing is read from a source: every initializer here is owned.
  static var none: Source { Source(bytes: UnsafeRawBufferPointer(start: nil, count: 0)) }
  static let f16 = DataType.float16

  /// `Softmax(Where(mask, fill, scores))` over [1, 2, 3, 3], the mask through a Not or direct.
  static func attention(mask: [Bool], throughNot: Bool, fill: Float = -.infinity, reader: String = "Softmax") -> Graph {
    var g = OnnxGraphBuilder()
    g.input("scores", f16, [1, 2, 3, 3])
    g.bool("mask", [1, 1, 3, 3], mask)
    g.fp16("fill", [], [fill])
    if throughNot {
      g.node("Not", ["mask"], ["masked"])
      g.valueInfo("masked", DataType.bool, [1, 1, 3, 3])
    }
    g.node("Where", [throughNot ? "masked" : "mask", "fill", "scores"], ["filled"])
    g.node(reader, ["filled"], ["p"], reader == "Softmax" ? [.int("axis", -1)] : [])
    g.output("p", f16, [1, 2, 3, 3])
    return g.graph
  }

  static func values(_ t: Tensor) throws -> [Float] {
    try Elements.floats(Elements.littleEndian(t, none), as: t.elementType, for: t.key)
  }

  @Test func aMaskThroughANotBecomesAnAdd() throws {
    // The export's shape: the constant says what to keep, its Not what to fill.
    let keep = (0..<9).map { $0 / 3 >= $0 % 3 }
    var g = Self.attention(mask: keep, throughNot: true)
    #expect(try Patches.maskedFillAsAdd(&g, Self.none) == 1)
    #expect(g.nodes.map(\.op) == ["Add", "Softmax"])
    #expect(g.nodes[0].inputs == ["scores", "filled__mask"])
    #expect(g.nodes[0].outputs == ["filled"])
    #expect(g.initializers.map(\.key) == ["filled__mask"])
    #expect(!g.valueInfo.contains { $0.key == "masked" })
    let mask = g.initializers[0]
    #expect(mask.dims == [1, 1, 3, 3])
    #expect(mask.elementType == Self.f16)
    #expect(try Self.values(mask) == keep.map { $0 ? 0 : -10_000 })
  }

  @Test func aConstantConditionFillsWhereItIsTrue() throws {
    let fillHere = [false, true, true, false, false, true, false, false, false]
    var g = Self.attention(mask: fillHere, throughNot: false)
    #expect(try Patches.maskedFillAsAdd(&g, Self.none) == 1)
    #expect(g.nodes.map(\.op) == ["Add", "Softmax"])
    #expect(try Self.values(g.initializers[0]) == fillHere.map { $0 ? -10_000 : 0 })
  }

  @Test func whatAnAddWouldChangeStaysAWhere() throws {
    let keep = (0..<9).map { $0 / 3 >= $0 % 3 }
    // A row filled whole: its weights depend on the fill.
    var wholeRow = Self.attention(mask: [false, false, false] + Array(keep.dropFirst(3)), throughNot: true)
    #expect(try Patches.maskedFillAsAdd(&wholeRow, Self.none) == 0)
    #expect(wholeRow.nodes.map(\.op) == ["Not", "Where", "Softmax"])
    // A fill a score could reach.
    var small = Self.attention(mask: keep, throughNot: true, fill: -100)
    #expect(try Patches.maskedFillAsAdd(&small, Self.none) == 0)
    // A reader that is not a softmax sees the fill itself.
    var relu = Self.attention(mask: keep, throughNot: true, reader: "Relu")
    #expect(try Patches.maskedFillAsAdd(&relu, Self.none) == 0)
    #expect(relu.initializers.map(\.key) == ["mask", "fill"])
  }

  @Test func aFourAxisMatMulFoldsItsMiddleAxes() throws {
    var g = OnnxGraphBuilder()
    g.input("x", Self.f16, [1, 9, 32, 4])
    g.input("y", Self.f16, [1, 9, 4])
    g.fp16("w", [4, 6], [Float](repeating: 0.5, count: 24))
    g.node("MatMul", ["x", "w"], ["wide"])
    g.node("MatMul", ["y", "w"], ["narrow"])
    g.output("wide", Self.f16, [1, 9, 32, 6])
    g.output("narrow", Self.f16, [1, 9, 6])
    var graph = g.graph
    #expect(Patches.foldMatMulAxes(&graph) == 1)
    #expect(graph.nodes.map(\.op) == ["Reshape", "MatMul", "Reshape", "MatMul"])
    #expect(graph.nodes[0].inputs == ["x", "wide__folded_shape"])
    #expect(graph.nodes[1].inputs == ["wide__folded", "w"])
    #expect(graph.nodes[2].inputs == ["wide__product", "wide__out_shape"])
    #expect(graph.nodes[2].outputs == ["wide"])
    #expect(graph.nodes[3].inputs == ["y", "w"])
    let shapes = Dictionary(
      uniqueKeysWithValues: try graph.initializers.filter { $0.elementType == DataType.int64 }.map {
        ($0.key, try Elements.integers($0, Self.none))
      })
    #expect(shapes == ["wide__folded_shape": [1, 288, 4], "wide__out_shape": [1, 9, 32, 6]])
  }

  @Test func theRewritesAreForThePolicyOnTheNeuralEngineOnly() throws {
    let keep = (0..<9).map { $0 / 3 >= $0 % 3 }
    var g = OnnxGraphBuilder()
    g.graph = Self.attention(mask: keep, throughNot: true)
    let bytes = g.bytes
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let source = directory.appendingPathComponent("source.onnx")
    try Data(bytes).write(to: source)
    for (on, masks) in [(false, 0), (true, 1)] {
      let report = try CoreMLPreparation.prepare(
        source: source, into: directory.appendingPathComponent("\(on)"), layout: .whole, policyOnNeuralEngine: on, cacheKey: { $0 })
      #expect(report.masks == masks)
      #expect(report.folds == 0)
    }
  }
}
