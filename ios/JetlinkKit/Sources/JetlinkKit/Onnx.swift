// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.
//
// An ONNX model as the rewrite in OnnxPrepare.swift needs it: the graph's
// nodes, initializers and value infos as editable values, everything else
// carried through as the bytes it was. Field numbers are onnx.proto's.

import Foundation

enum F {
  // ModelProto
  static let modelGraph = 7, modelOpset = 8, modelMetadata = 14, modelFunctions = 25
  // GraphProto
  static let graphNode = 1, graphInitializer = 5, graphInput = 11, graphOutput = 12, graphValueInfo = 13
  // NodeProto
  static let nodeInput = 1, nodeOutput = 2, nodeName = 3, nodeOpType = 4, nodeAttribute = 5, nodeDomain = 7
  // AttributeProto
  static let attrName = 1, attrI = 3, attrType = 20
  static let attrTypeInt: Int64 = 2
  // TensorProto
  static let tensorDims = 1, tensorDataType = 2, tensorInt32Data = 5, tensorInt64Data = 7, tensorName = 8
  static let tensorRawData = 9, tensorExternalData = 13, tensorDataLocation = 14
  // ValueInfoProto, TypeProto, TypeProto.Tensor, TensorShapeProto, Dimension
  static let viName = 1, viType = 2, typeTensor = 1, tensorElemType = 1, tensorShape = 2, shapeDim = 1
  static let dimValue = 1
  // OperatorSetIdProto, StringStringEntryProto
  static let opsetDomain = 1, entryKey = 1, entryValue = 2
}

/// onnx.TensorProto.DataType
public enum OnnxType {
  public static let float: Int32 = 1, uint8: Int32 = 2, int32: Int32 = 6, int64: Int32 = 7, float16: Int32 = 10

  private static let names = [
    "", "float32", "uint8", "int8", "uint16", "int16", "int32", "int64", "string", "bool", "float16", "float64",
  ]

  static func name(_ t: Int32) -> String {
    t > 0 && Int(t) < names.count ? names[Int(t)] : "unknown(\(t))"
  }
}

public struct OnnxError: Error, CustomStringConvertible {
  public let description: String
  init(_ d: String) { description = d }
}

// MARK: - graph pieces

struct Attribute {
  var name: String
  var i: Int64?
  var field: ProtoField  // as it came, for writing back

  static func int(_ name: String, _ value: Int64) -> Attribute {
    let msg = ProtoMessage(fields: [.string(F.attrName, name), .int(F.attrI, value), .int(F.attrType, F.attrTypeInt)])
    return Attribute(name: name, i: value, field: ProtoField(F.nodeAttribute, .message(msg)))
  }
}

struct Node {
  // Any change marks the node edited, so it is re-encoded rather than copied.
  var inputs: [String] { didSet { edited = true } }
  var outputs: [String] { didSet { edited = true } }
  var name: String? { didSet { edited = true } }
  var opType: String { didSet { edited = true } }
  var domain: String? { didSet { edited = true } }
  var attributes: [Attribute] { didSet { edited = true } }
  var rest: [ProtoField] { didSet { edited = true } }  // every other field
  var source: Range<Int>?  // the bytes it was read from
  var edited = false

  init(opType: String, inputs: [String], outputs: [String], name: String, attributes: [Attribute] = []) {
    self.opType = opType
    self.inputs = inputs
    self.outputs = outputs
    self.name = name
    self.domain = nil
    self.attributes = attributes
    self.rest = []
    self.source = nil
    self.edited = true
  }

  init(_ file: MappedFile, _ range: Range<Int>) throws {
    let m = try ProtoMessage.decode(file, range)
    var inputs: [String] = [], outputs: [String] = [], attrs: [Attribute] = [], rest: [ProtoField] = []
    var name: String?, opType: String?, domain: String?
    for f in m.fields {
      switch f.number {
      case F.nodeInput: inputs.append(f.string(file) ?? "")
      case F.nodeOutput: outputs.append(f.string(file) ?? "")
      case F.nodeName: name = f.string(file)
      case F.nodeOpType: opType = f.string(file)
      case F.nodeDomain: domain = f.string(file)
      case F.nodeAttribute:
        guard let a = try f.message(file) else { throw OnnxError("a node attribute is not a message") }
        attrs.append(Attribute(name: a.first(F.attrName)?.string(file) ?? "", i: a.first(F.attrI)?.int, field: f))
      default: rest.append(f)
      }
    }
    self.inputs = inputs
    self.outputs = outputs
    self.name = name
    self.opType = opType ?? ""
    self.domain = domain
    self.attributes = attrs
    self.rest = rest
    self.source = range
  }

  /// The node's name as Python reads it: absent is "".
  var nameOrEmpty: String { name ?? "" }
  var domainOrEmpty: String { domain ?? "" }

  func attr(_ name: String) -> Attribute? { attributes.first { $0.name == name } }

  var field: ProtoField {
    if !edited, let source { return ProtoField(F.graphNode, .raw(.source(source))) }
    var fields: [ProtoField] = []
    fields += inputs.map { .string(F.nodeInput, $0) }
    fields += outputs.map { .string(F.nodeOutput, $0) }
    if let name { fields.append(.string(F.nodeName, name)) }
    fields.append(.string(F.nodeOpType, opType))
    fields += attributes.map(\.field)
    if let domain { fields.append(.string(F.nodeDomain, domain)) }
    fields += rest
    return ProtoField(F.graphNode, .message(ProtoMessage(fields: fields)))
  }
}

struct Initializer {
  var name: String
  var dims: [Int64]
  var dataType: Int32
  var raw: Range<Int>?  // raw_data, when that is how the values are stored
  var typed: [ProtoField]  // int32_data / int64_data, when they are
  var external: Bool
  var field: ProtoField

  init(_ file: MappedFile, _ range: Range<Int>) throws {
    let m = try ProtoMessage.decode(file, range)
    name = m.first(F.tensorName)?.string(file) ?? ""
    dims = try ProtoField.int64s(m.all(F.tensorDims), file)
    dataType = Int32(truncatingIfNeeded: m.first(F.tensorDataType)?.int ?? 0)
    if case .raw(.source(let r))? = m.first(F.tensorRawData)?.payload { raw = r } else { raw = nil }
    typed = m.fields.filter { $0.number == F.tensorInt32Data || $0.number == F.tensorInt64Data }
    external = (m.first(F.tensorDataLocation)?.int ?? 0) == 1 || m.has(F.tensorExternalData)
    field = ProtoField(F.graphInitializer, .raw(.source(range)))
  }

  /// numpy_helper.from_array: dims, data_type, name, raw_data.
  init(name: String, dims: [Int64], dataType: Int32, data: Chunk) {
    self.name = name
    self.dims = dims
    self.dataType = dataType
    self.raw = nil
    self.typed = []
    self.external = false
    var fields: [ProtoField] = dims.map { .int(F.tensorDims, $0) }
    fields.append(.int(F.tensorDataType, Int64(dataType)))
    fields.append(.string(F.tensorName, name))
    fields.append(ProtoField(F.tensorRawData, .raw(data)))
    field = ProtoField(F.graphInitializer, .message(ProtoMessage(fields: fields)))
  }

  static func int64(_ name: String, _ values: [Int64], dims: [Int64]) -> Initializer {
    var bytes = [UInt8]()
    bytes.reserveCapacity(values.count * 8)
    for v in values { withUnsafeBytes(of: v.littleEndian) { bytes.append(contentsOf: $0) } }
    return Initializer(name: name, dims: dims, dataType: OnnxType.int64, data: .bytes(bytes))
  }

  var count: Int { dims.reduce(1) { $0 * Int($1) } }

  /// The values as integers, for index and shape constants.
  func integers(_ file: MappedFile) throws -> [Int64] {
    if external { throw OnnxError("\(name) keeps its data in an external file") }
    switch dataType {
    case OnnxType.int64:
      if let raw {
        guard raw.count == count * 8 else { throw OnnxError("\(name): raw_data is \(raw.count) bytes for \(count) int64") }
        return (0..<count).map { loadLE(file, raw.lowerBound + 8 * $0, Int64.self) }
      }
      return try ProtoField.int64s(typed.filter { $0.number == F.tensorInt64Data }, file)
    case OnnxType.int32:
      if let raw {
        guard raw.count == count * 4 else { throw OnnxError("\(name): raw_data is \(raw.count) bytes for \(count) int32") }
        return (0..<count).map { Int64(Int32(bitPattern: loadLE(file, raw.lowerBound + 4 * $0, UInt32.self))) }
      }
      return try ProtoField.int64s(typed.filter { $0.number == F.tensorInt32Data }, file).map {
        Int64(Int32(truncatingIfNeeded: $0))
      }
    default:
      throw OnnxError("\(name) is \(OnnxType.name(dataType)), not an integer tensor")
    }
  }

  /// float16 values as their bit patterns, from raw_data or int32_data.
  func float16Bits(_ file: MappedFile) throws -> [UInt16] {
    guard dataType == OnnxType.float16 else { throw OnnxError("\(name) is not float16") }
    if external { throw OnnxError("\(name) keeps its data in an external file") }
    if let raw {
      guard raw.count == count * 2 else { throw OnnxError("\(name): raw_data is \(raw.count) bytes for \(count) float16") }
      return (0..<count).map { loadLE(file, raw.lowerBound + 2 * $0, UInt16.self) }
    }
    let v = try ProtoField.int64s(typed.filter { $0.number == F.tensorInt32Data }, file)
    guard v.count == count else { throw OnnxError("\(name) holds \(v.count) values for \(count)") }
    return v.map { UInt16(truncatingIfNeeded: $0) }
  }
}

struct ValueInfo {
  var name: String
  var elemType: Int32?  // nil when the type is absent or not a tensor
  var dims: [Int64]?  // nil when there is no shape; -1 for a dimension without a value
  var message: ProtoMessage?  // decoded when edited
  var source: Range<Int>

  init(_ file: MappedFile, _ range: Range<Int>) throws {
    source = range
    let m = try ProtoMessage.decode(file, range)
    name = m.first(F.viName)?.string(file) ?? ""
    elemType = nil
    dims = nil
    if let type = try m.first(F.viType)?.message(file), let tensor = try type.first(F.typeTensor)?.message(file) {
      if let e = tensor.first(F.tensorElemType)?.int, e != 0 { elemType = Int32(truncatingIfNeeded: e) }
      if let shape = try tensor.first(F.tensorShape)?.message(file) {
        var d: [Int64] = []
        for dimField in shape.all(F.shapeDim) {
          let dim = try dimField.message(file) ?? ProtoMessage()
          d.append(dim.first(F.dimValue)?.int ?? -1)
        }
        dims = d
      }
    }
  }

  /// The same value info with its tensor element type changed.
  mutating func retype(_ t: Int32, _ file: MappedFile) throws {
    var m = try message ?? ProtoMessage.decode(file, source)
    guard var type = try m.first(F.viType)?.message(file), var tensor = try type.first(F.typeTensor)?.message(file)
    else { throw OnnxError("\(name) has no tensor type to change") }
    tensor.set(.int(F.tensorElemType, Int64(t)))
    type.set(ProtoField(F.typeTensor, .message(tensor)))
    m.set(ProtoField(F.viType, .message(type)))
    message = m
    elemType = t
  }

  func field(_ number: Int) -> ProtoField {
    if let message { return ProtoField(number, .message(message)) }
    return ProtoField(number, .raw(.source(source)))
  }
}

// MARK: - the model

public final class OnnxModel {
  let file: MappedFile
  var top: ProtoMessage  // ModelProto, with the graph field replaced on write
  var graphRest: [ProtoField]  // graph fields other than the five below
  var nodes: [Node]
  var initializers: [Initializer]
  var inputs: [ValueInfo]
  var outputs: [ValueInfo]
  var valueInfo: [ValueInfo]

  public init(path: String) throws {
    file = try MappedFile(path: path)
    top = try ProtoMessage.decode(file, 0..<file.count)
    let graphFields = top.all(F.modelGraph)
    guard graphFields.count == 1, case .raw(.source(let g)) = graphFields[0].payload else {
      throw OnnxError("the model has \(graphFields.count) graphs, expected one")
    }
    let graph = try ProtoMessage.decode(file, g)
    var nodes: [Node] = [], inits: [Initializer] = [], ins: [ValueInfo] = [], outs: [ValueInfo] = []
    var vis: [ValueInfo] = [], rest: [ProtoField] = []
    for f in graph.fields {
      guard case .raw(.source(let r)) = f.payload else {
        rest.append(f)
        continue
      }
      switch f.number {
      case F.graphNode: nodes.append(try Node(file, r))
      case F.graphInitializer: inits.append(try Initializer(file, r))
      case F.graphInput: ins.append(try ValueInfo(file, r))
      case F.graphOutput: outs.append(try ValueInfo(file, r))
      case F.graphValueInfo: vis.append(try ValueInfo(file, r))
      default: rest.append(f)
      }
    }
    self.nodes = nodes
    self.initializers = inits
    self.inputs = ins
    self.outputs = outs
    self.valueInfo = vis
    self.graphRest = rest
  }

  // MARK: model-level pieces

  var opsetDomains: [String] {
    top.all(F.modelOpset).map { (try? $0.message(file))?.first(F.opsetDomain)?.string(file) ?? "" }
  }

  /// Removes the first opset import for `domain`, as the Python does with `del`.
  func removeOpset(_ domain: String) {
    for (i, f) in top.fields.enumerated() where f.number == F.modelOpset {
      if ((try? f.message(file))?.first(F.opsetDomain)?.string(file) ?? "") == domain {
        top.fields.remove(at: i)
        return
      }
    }
  }

  public var metadata: [(String, String)] {
    top.all(F.modelMetadata).compactMap { f in
      guard let m = try? f.message(file) else { return nil }
      return (m.first(F.entryKey)?.string(file) ?? "", m.first(F.entryValue)?.string(file) ?? "")
    }
  }

  /// Drops metadata entries with any of `keys` and appends key=value.
  func setMetadata(_ key: String, _ value: String, replacing keys: Set<String>) {
    top.fields.removeAll { f in
      guard f.number == F.modelMetadata, let m = try? f.message(file) else { return false }
      return keys.contains(m.first(F.entryKey)?.string(file) ?? "")
    }
    top.fields.append(
      ProtoField(F.modelMetadata, .message(ProtoMessage(fields: [.string(F.entryKey, key), .string(F.entryValue, value)]))))
  }

  // MARK: writing

  var graphMessage: ProtoMessage {
    var fields: [ProtoField] = nodes.map(\.field)
    fields += initializers.map(\.field)
    fields += inputs.map { $0.field(F.graphInput) }
    fields += outputs.map { $0.field(F.graphOutput) }
    fields += valueInfo.map { $0.field(F.graphValueInfo) }
    fields += graphRest
    return ProtoMessage(fields: fields)
  }

  /// Writes the model to `path`, replacing whatever is there.
  public func write(to path: String) throws {
    var model = top
    model.set(ProtoField(F.modelGraph, .message(graphMessage)))
    let fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
    if fd < 0 { throw OnnxError("cannot create \(path): \(String(cString: strerror(errno)))") }
    defer { close(fd) }
    var w = ProtoWriter(fd: fd, file: file)
    try model.write(to: &w)
    try w.flush()
    if fsync(fd) != 0 { throw OnnxError("cannot sync \(path): \(String(cString: strerror(errno)))") }
  }

  // MARK: static shape information, as onnx_patch._static_info

  /// Static shapes and element types from the graph's inputs, value_info and
  /// outputs, later ones overriding earlier ones. There is no shape inference
  /// here; callers say what they cannot do without.
  func staticInfo() -> (dims: [String: [Int64]], types: [String: Int32]) {
    var dims: [String: [Int64]] = [:], types: [String: Int32] = [:]
    for vi in inputs + valueInfo + outputs {
      if let t = vi.elemType { types[vi.name] = t }
      if let d = vi.dims { dims[vi.name] = d }
    }
    return (dims, types)
  }
}

// MARK: - metadata, as jetlink/onnx_meta.py

public struct OnnxMeta: Sendable {
  public struct IO: Sendable {
    public var name: String
    public var shape: [Int]
    public var type: String
  }

  public var inputs: [IO] = []
  public var outputs: [IO] = []
  public var props: [String: String] = [:]

  public init(path: String) throws {
    let model = try OnnxModel(path: path)
    func io(_ v: ValueInfo) -> IO {
      // A dimension without a value reads as 0, as both Python parsers give.
      IO(name: v.name, shape: (v.dims ?? []).map { $0 < 0 ? 0 : Int($0) }, type: OnnxType.name(v.elemType ?? 0))
    }
    inputs = model.inputs.map(io)
    outputs = model.outputs.map(io)
    for (k, v) in model.metadata { props[k] = v }
  }

  init(inputs: [IO], outputs: [IO], props: [String: String]) {
    self.inputs = inputs
    self.outputs = outputs
    self.props = props
  }

  /// openpilot's base64 pickle of the output slice map.
  public func outputSlices() throws -> [String: SliceRange] {
    guard let raw = props["output_slices"] else { throw OnnxError("output_slices not in model metadata_props") }
    guard let data = Data(base64Encoded: raw, options: .ignoreUnknownCharacters) else {
      throw OnnxError("output_slices is not base64")
    }
    let total = outputs.first { $0.name == "outputs" }.map { product($0.shape) }
    var out: [String: SliceRange] = [:]
    for (name, (start, stop)) in try unpickleSliceMap([UInt8](data)) {
      let s = start ?? 0
      guard let e = stop ?? total else { throw OnnxError("output_slices[\(name)] has no end") }
      guard s >= 0, e >= s else { throw OnnxError("output_slices[\(name)] is \(s)..<\(e)") }
      out[name] = SliceRange(start: s, stop: e)
    }
    return out
  }
}
