import Foundation

/// Range-safe fp16 preparation for the validated experimental QNN profile.
public enum QnnPreparation {
  /// Keep the native LayerNormalization kernel instead of expanding it into
  /// seventeen elementwise/reduction operations. A power-of-two input scale
  /// and epsilon/k^2 preserve the real-valued expression; device parity is
  /// still required because the HTP's fp16 arithmetic differs from the CPU.
  static func fusedLayerNorms(_ graph: inout Graph, k: Float = 4) throws -> Int {
    let types = Patches.staticTypes(graph)
    let dimensions = Patches.staticDims(graph)
    let constant = "__qnn_fused_norm_scale4"
    guard !graph.initializers.contains(where: { $0.key == constant }) else {
      throw OnnxError("reserved QNN normalization constant already present")
    }
    var rewritten: [Node] = []
    var count = 0
    for (index, original) in graph.nodes.enumerated() {
      var node = original
      if node.op == "LayerNormalization", let input = node.inputs.first, types[input] == DataType.float16 {
        guard node.outputs.dropFirst().allSatisfy({ $0.isEmpty }) else {
          throw OnnxError("experimental fused normalization does not rewrite optional statistics outputs")
        }
        let scaled = "__qnn_fused_norm_\(index)"
        rewritten.append(Node(inputs: [input, constant], outputs: [scaled], name: scaled, opType: "Mul"))
        node.inputs[0] = scaled
        let epsilon = (node.attribute("epsilon")?.f ?? 1e-5) / (k * k)
        node.attributes.removeAll { $0.name == "epsilon" }
        var encoded = Encoded()
        encoded.stringField(1, "epsilon")
        encoded.tag(2, .fixed32)
        let bits = epsilon.bitPattern
        for shift in stride(from: 0, to: 32, by: 8) { encoded.byte(UInt8(truncatingIfNeeded: bits >> shift)) }
        encoded.intField(20, 1)
        node.attributes.append(Attribute(bytes: .owned(encoded.tail), name: "epsilon", type: 1, i: 0, f: epsilon, ints: [], s: "", floats: [], t: nil))
        if let shape = dimensions[input] { graph.valueInfo.append(.tensor(scaled, DataType.float16, shape)) }
        count += 1
      }
      rewritten.append(node)
    }
    if count > 0 {
      graph.initializers.append(
        Tensor(
          name: constant, dims: [], dataType: DataType.float16,
          raw: .owned(Elements.encode([1 / k], as: DataType.float16))))
    }
    graph.nodes = rewritten
    return count
  }

}
