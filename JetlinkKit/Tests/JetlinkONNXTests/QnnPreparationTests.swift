import Testing

@testable import JetlinkONNX

@Suite struct QnnPreparationTests {
  @Test("Prescaling retains a native normalization and its original learned parameters")
  func nativeNormalization() throws {
    var builder = OnnxGraphBuilder()
    builder.input("x", DataType.float16, [1, 2])
    builder.fp16("scale", [2], [1, 2])
    builder.fp16("bias", [2], [0.25, -0.5])
    builder.node("LayerNormalization", ["x", "scale", "bias"], ["y"], [.float("epsilon", 1e-4)])
    builder.output("y", DataType.float16, [1, 2])
    let weights = builder.graph.initializers
    var graph = builder.graph
    #expect(try QnnPreparation.fusedLayerNorms(&graph) == 1)
    #expect(graph.nodes.map(\.op) == ["Mul", "LayerNormalization"])
    #expect(graph.nodes[1].inputs.dropFirst() == ["scale", "bias"])
    #expect(graph.nodes[1].outputs == ["y"])
    #expect(graph.nodes[1].attribute("epsilon")?.f == Float(1e-4) / 16)
    #expect(graph.initializers.prefix(weights.count).map(\.key) == weights.map(\.key))
    let raw: (Tensor) -> [UInt8]? = { tensor in
      guard case .owned(let bytes)? = tensor.raw else { return nil }
      return bytes
    }
    #expect(graph.initializers.prefix(weights.count).map(raw) == weights.map(raw))
    #expect(graph.initializers.last?.key == "__qnn_fused_norm_scale4")
    // A second transformation must not silently apply another factor of four.
    #expect(throws: OnnxError.self) { _ = try QnnPreparation.fusedLayerNorms(&graph) }
  }

  @Test("Optional normalization statistics are rejected rather than changing their meaning")
  func optionalStatistics() {
    var builder = OnnxGraphBuilder()
    builder.input("x", DataType.float16, [1, 2])
    builder.node("LayerNormalization", ["x", "scale"], ["y", "mean"])
    var graph = builder.graph
    #expect(throws: OnnxError.self) { _ = try QnnPreparation.fusedLayerNorms(&graph) }
    #expect(graph.nodes.map(\.op) == ["LayerNormalization"])
  }
}
