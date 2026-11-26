import Foundation
import JetlinkONNX

/// What a model preparation produced: the parts to run as a chain, in order.
public struct PreparedModel: Sendable {
  public struct Part: Sendable {
    public let name: String
    public let file: String
    public let weightBytes: Int64

    public init(name: String, file: String, weightBytes: Int64) {
      self.name = name
      self.file = file
      self.weightBytes = weightBytes
    }
  }

  public let parts: [Part]
  public let summary: String

  public init(parts: [Part], summary: String) {
    self.parts = parts
    self.summary = summary
  }
}

/// Reads and rewrites ONNX: what JetlinkONNX does, behind a seam so the
/// server builds and tests without it.
public protocol ModelPreparer: Sendable {
  func readSpec(model: URL, sha256: String, nbytes: Int64, frameSkip: Int) throws -> ModelSpec
  /// Writes the parts into `directory` in the device's layout, with the
  /// rewrites the Neural Engine needs to take the policy when it runs there.
  func prepare(
    model: URL, into directory: URL, layout: CoreMLPreparation.Layout, policyOnNeuralEngine: Bool,
    cacheKey: @escaping (String) -> String
  ) throws -> PreparedModel
}

/// The server's reading and preparing of an ONNX, through JetlinkONNX: the
/// spec from the graph's inputs, outputs and output_slices, and the CoreML
/// preparation the Python backend does, streamed from a mapped file.
public struct ONNXPreparer: ModelPreparer {
  public init() {}

  public func readSpec(model: URL, sha256: String, nbytes: Int64, frameSkip: Int) throws -> ModelSpec {
    let meta = try OnnxMeta.read(contentsOf: model)
    return ModelSpec(
      sha256: sha256,
      nbytes: nbytes,
      frameSkip: frameSkip,
      inputShapes: meta.inputs.map { NamedShape($0.name, $0.dims.map { Int($0) }) },
      outputShapes: meta.outputs.map { NamedShape($0.name, $0.dims.map { Int($0) }) },
      outputSlices: try meta.outputSlices().map { NamedSlice($0.name, start: $0.start, stop: $0.stop) },
      checkpoint: meta.modelCheckpoint)
  }

  public func prepare(
    model: URL, into directory: URL, layout: CoreMLPreparation.Layout, policyOnNeuralEngine: Bool,
    cacheKey: @escaping (String) -> String
  ) throws -> PreparedModel {
    let report = try CoreMLPreparation.prepare(
      source: model, into: directory, layout: layout, policyOnNeuralEngine: policyOnNeuralEngine, cacheKey: cacheKey)
    var summary =
      "stripped \(report.stripped) tinygrad op(s), \(report.retypedImages ? "images retyped to fp16" : "inputs left as declared"), "
      + "\(report.gathers) negative Gather index(es) normalized, \(report.gemms) MatMul+Add rewritten as Gemm(transB=1), \(report.tiles) Expand(s) as Tile"
    if layout == .aneWhole {
      summary += "; for the whole Neural Engine: \(report.norms) policy LayerNormalization(s) prescaled, \(report.heads) vision head node(s) in fp32"
    }
    if policyOnNeuralEngine {
      summary += "; for the policy on the Neural Engine: \(report.masks) attention mask(s) as Add, \(report.folds) MatMul(s) folded to three axes"
    }
    return PreparedModel(
      parts: report.parts.map { PreparedModel.Part(name: $0.name, file: $0.url.lastPathComponent, weightBytes: $0.weightBytes) },
      summary: summary)
  }
}
