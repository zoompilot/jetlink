// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.
//
// The graph surgery jetlink/onnx_patch.py does before onnxruntime sees a
// model, ported pass for pass so a phone can prepare what a comma uploads.
//
//   strip_tinygrad_ops         onnxruntime rejects the org.tinygrad domain
//   patch_uint8_inputs         images arrive as fp16 from the queues
//   normalize_gather_indices   the Neural Engine gathers garbage at index -1
//   gemm_with_transposed_weight  CoreML stores the weights in its weight
//                              file instead of 4 GB of MIL text
//   expand_to_tile             two Expands CoreML will not take, as Tiles,
//                              so the graph stays one CoreML program
//
// and for the phone's Neural Engine build, which keeps the whole model on the
// Neural Engine (the Mac's `ane` runs what follows the vision trunk on the
// GPU, which on a phone is far slower), the passes in ios/scripts/ane_passes.py:
//
//   prescale_layernorm         LayerNorm(x / 8) in fp16, on the Neural Engine
//   heads_in_fp32              the small heads at the end of the vision trunk
//                              in fp32, off the Neural Engine
//
// Each pass does what its Python original does, in the same order, with the
// same names for everything it adds, so ios/scripts/check_prepare.py can hold
// the two outputs side by side. The one difference is where the Python falls
// back to onnx's shape inference for a tensor the file carries no shape for.
// There is none here: for the Neural Engine's two correctness passes that is
// an error rather than a model that would compute garbage there; for the Gemm
// rewrite, which changes no arithmetic, the MatMul is left and a warning says
// its weight will load slowly. The driving models record every shape.

import Foundation

let imgInputs: Set<String> = ["img", "big_img"]
let tinygradDomain = "org.tinygrad"
let passthroughOps: Set<String> = ["Contiguous"]
/// What the policy's LayerNorm inputs are divided by on the Neural Engine;
/// see ios/scripts/ane_passes.py.
let layernormPrescale = 8
/// The ops headsInFP32 moves into fp32, and how many nodes it will move at
/// most; see ios/scripts/ane_passes.py.
let headOps: Set<String> = ["Gemm", "MatMul", "LayerNormalization", "Gelu", "Add", "Sub", "Mul", "Div", "Relu", "Sigmoid", "Tanh"]
let headMaxNodes = 64
/// A weight smaller than this stays a MatMul; see onnx_patch.BLOB_MIN_ELEMENTS.
let blobMinElements: Int64 = 1024
/// The metadata key onnxruntime's CoreML provider keys its compile cache on.
public let coremlCacheKeyProp = "COREML_CACHE_KEY"

public struct PrepareReport: Sendable, CustomStringConvertible {
  public var stripped = 0
  public var retyped = false
  public var gathers = 0
  public var norms = 0
  public var gemms = 0
  public var tiles = 0
  public var heads = 0
  /// What the convert stage writes towards: the initializers' bytes.
  public var weightsBytes = 0

  public var description: String {
    "stripped \(stripped) tinygrad op(s), \(retyped ? "images retyped to fp16" : "inputs left as declared"), "
      + "\(gathers) negative Gather index(es) normalized, "
      + "\(gemms) MatMul+Add rewritten as Gemm(transB=1), \(tiles) Expand(s) as Tile, "
      + "\(norms) LayerNormalization(s) pre-scaled by 1/\(layernormPrescale), \(heads) head node(s) in fp32"
  }
}

public enum OnnxPrepare {
  /// _prepared_model plus _with_cache_key: the ONNX as onnxruntime will see
  /// it, written to `dst`.
  @discardableResult
  public static func prepare(src: String, dst: String, forANE: Bool, forCoreML: Bool, cacheKey: String?) throws
    -> PrepareReport
  {
    let m = try OnnxModel(path: src)
    if let bad = m.initializers.first(where: \.external) {
      throw OnnxError("\(bad.name) keeps its weights in an external file; upload a single-file model")
    }
    var r = PrepareReport()
    r.stripped = try stripTinygradOps(m)
    if needsPatch(m) {
      try patchUInt8Inputs(m)
      r.retyped = true
    }
    r.gathers = try normalizeGatherIndices(m, required: forANE)
    r.gemms = forCoreML ? try gemmWithTransposedWeight(m) : 0
    r.tiles = forCoreML ? try expandToTile(m) : 0
    if forANE {
      let vision = try visionNodes(m)
      let policy = Set(m.nodes.map(\.nameOrEmpty)).subtracting(vision)
      r.norms = try prescaleLayerNorm(m, only: policy)
      r.heads = try headsInFP32(m, vision: vision)
    }
    r.weightsBytes = m.initializers.reduce(0) { $0 + $1.rawBytes }
    if let cacheKey {
      m.setMetadata(coremlCacheKeyProp, cacheKey, replacing: [coremlCacheKeyProp, "CACHE_KEY"])
    }
    try m.write(to: dst)
    return r
  }

  // MARK: strip_tinygrad_ops

  /// Bypass tinygrad's layout-hint nodes. Returns how many went.
  static func stripTinygradOps(_ m: OnnxModel) throws -> Int {
    let graphOutputs = Set(m.outputs.map(\.name))
    var dead = Set<Int>()
    // A snapshot, as the Python takes one, processed in graph order.
    for t in m.nodes.indices where m.nodes[t].domainOrEmpty == tinygradDomain {
      let node = m.nodes[t]
      if !passthroughOps.contains(node.opType) {
        throw OnnxError("unknown \(tinygradDomain) op \(node.opType); it may not be a no-op, so dropping it is not safe")
      }
      if node.inputs.count != 1 || node.outputs.count != 1 || !node.attributes.isEmpty {
        throw OnnxError("\(node.opType) is not a plain one-in one-out passthrough")
      }
      let source = node.inputs[0], produced = node.outputs[0]
      if graphOutputs.contains(produced) {
        // The output keeps its name: output_slices address it. So the
        // producer takes the name over.
        for i in m.nodes.indices where !dead.contains(i) {
          for j in m.nodes[i].outputs.indices where m.nodes[i].outputs[j] == source { m.nodes[i].outputs[j] = produced }
        }
        for i in m.nodes.indices where !dead.contains(i) {
          for j in m.nodes[i].inputs.indices where m.nodes[i].inputs[j] == source { m.nodes[i].inputs[j] = produced }
        }
      } else {
        for i in m.nodes.indices where !dead.contains(i) {
          for j in m.nodes[i].inputs.indices where m.nodes[i].inputs[j] == produced { m.nodes[i].inputs[j] = source }
        }
      }
      dead.insert(t)
    }
    if dead.isEmpty { return 0 }
    m.nodes = m.nodes.enumerated().filter { !dead.contains($0.offset) }.map(\.element)
    m.removeOpset(tinygradDomain)
    var live = Set<String>()
    for n in m.nodes {
      live.formUnion(n.inputs)
      live.formUnion(n.outputs)
    }
    m.valueInfo.removeAll { !live.contains($0.name) }
    return dead.count
  }

  // MARK: patch_uint8_inputs

  static func needsPatch(_ m: OnnxModel) -> Bool {
    m.inputs.contains { imgInputs.contains($0.name) && $0.elemType == OnnxType.uint8 }
  }

  /// The Cast nodes turning the uint8 image inputs into fp16, in either of
  /// the two graph shapes in the wild. Indices into m.nodes.
  static func headCasts(_ m: OnnxModel) -> [Int] {
    let perInput = m.nodes.indices.filter {
      m.nodes[$0].opType == "Cast" && m.nodes[$0].inputs.count == 1 && imgInputs.contains(m.nodes[$0].inputs[0])
    }
    if !perInput.isEmpty { return perInput }
    guard let concat = m.nodes.first(where: { $0.opType == "Concat" && $0.inputs.allSatisfy(imgInputs.contains) }),
      let cat = concat.outputs.first
    else { return [] }
    if let cast = m.nodes.indices.first(where: { m.nodes[$0].opType == "Cast" && m.nodes[$0].inputs == [cat] }) {
      return [cast]
    }
    return []
  }

  static func patchUInt8Inputs(_ m: OnnxModel) throws {
    let img = m.inputs.indices.filter { imgInputs.contains(m.inputs[$0].name) }
    if img.isEmpty { throw OnnxError("model has none of img, big_img as graph inputs") }
    let casts = headCasts(m)
    if casts.isEmpty {
      throw OnnxError("could not find the head Cast on the image inputs; neither a Cast per input nor one after their Concat")
    }
    for c in casts {
      guard let to = m.nodes[c].attr("to")?.i else { throw OnnxError("a head Cast has no 'to' attribute") }
      if to != Int64(OnnxType.float16) {
        throw OnnxError("head Cast targets \(to), expected FLOAT16 (\(OnnxType.float16))")
      }
    }
    for i in img {
      if m.inputs[i].elemType != OnnxType.uint8 { throw OnnxError("\(m.inputs[i].name) is not uint8; model already patched?") }
      try m.inputs[i].retype(OnnxType.float16, m.file)
    }
    // Remove casts one at a time, as the Python does, rewiring what reads each.
    var retyped = Set<String>()
    let castNodes = casts.map { m.nodes[$0] }
    var removed = Set<Int>()
    for (k, c) in casts.enumerated() {
      let source = castNodes[k].inputs[0], produced = castNodes[k].outputs[0]
      for i in m.nodes.indices where !removed.contains(i) {
        for j in m.nodes[i].inputs.indices where m.nodes[i].inputs[j] == produced { m.nodes[i].inputs[j] = source }
      }
      removed.insert(c)
      retyped.insert(source)
    }
    m.nodes = m.nodes.enumerated().filter { !removed.contains($0.offset) }.map(\.element)
    // onnxruntime rejects a model whose value_info still says uint8.
    for i in m.valueInfo.indices where retyped.contains(m.valueInfo[i].name) {
      try m.valueInfo[i].retype(OnnxType.float16, m.file)
    }
  }

  // MARK: normalize_gather_indices

  /// Rewrites Gather nodes whose constant index is negative to the positive
  /// equivalent, each with its own index initializer. `required`: a node that
  /// needs rewriting but whose data has no static shape is an error rather
  /// than left alone, because the Neural Engine would gather garbage.
  static func normalizeGatherIndices(_ m: OnnxModel, required: Bool) throws -> Int {
    let initByName = m.initializerMap
    let (dims, _) = m.staticInfo()
    var rewritten = 0
    for i in m.nodes.indices {
      let node = m.nodes[i]
      guard node.opType == "Gather", node.inputs.count > 1, let idx = initByName[node.inputs[1]] else { continue }
      guard idx.dataType == OnnxType.int64 || idx.dataType == OnnxType.int32 else { continue }
      let index = try idx.integers(m.file)
      guard let lowest = index.min(), lowest < 0 else { continue }
      let axis = node.attr("axis")?.i ?? 0
      guard let shape = dims[node.inputs[0]] else {
        if required {
          throw OnnxError(
            "\(node.label): Gather with a negative index on \(node.inputs[0]), whose shape the model does not record")
        }
        Prep.log.warning("\(node.label): no static shape for \(node.inputs[0]); negative Gather index left as it is")
        continue
      }
      // Python indexes the shape tuple with the axis as given, negative included.
      if axis >= Int64(shape.count) { continue }
      let a = axis < 0 ? Int64(shape.count) + axis : axis
      if a < 0 { throw OnnxError("\(node.label): axis \(axis) out of range for rank \(shape.count)") }
      let size = shape[Int(a)]
      if size <= 0 { continue }
      let fixed = index.map { $0 < 0 ? $0 + size : $0 }
      if fixed.contains(where: { $0 < 0 || $0 >= size }) {
        throw OnnxError("\(node.label): Gather index \(index) out of range for axis \(axis) of size \(size)")
      }
      let name = "\(node.outputs[0])__index"
      m.initializers.append(.integers(name, fixed, dims: idx.dims, dataType: idx.dataType))
      m.nodes[i].inputs[1] = name
      rewritten += 1
    }
    return rewritten
  }

  // MARK: vision_nodes and layernorm_in_fp32

  /// Names of the nodes that depend on the image inputs alone.
  static func visionNodes(_ m: OnnxModel) throws -> Set<String> {
    let inits = Set(m.initializers.map(\.name))
    let images = Set(m.inputs.map(\.name).filter(imgInputs.contains))
    if images.isEmpty { throw OnnxError("model has none of img, big_img as graph inputs") }
    var visionTensors = images
    var names = Set<String>()
    for node in m.nodes {
      let data = node.inputs.filter { !$0.isEmpty && !inits.contains($0) }
      if !data.isEmpty && data.allSatisfy(visionTensors.contains) {
        names.insert(node.nameOrEmpty)
        visionTensors.formUnion(node.outputs)
      }
    }
    return names
  }

  // MARK: the Neural Engine passes (ios/scripts/ane_passes.py)

  /// Rewrites an Expand with a constant shape of the input's rank, which only
  /// repeats size-1 axes, as the equivalent Tile. Returns how many.
  static func expandToTile(_ m: OnnxModel) throws -> Int {
    let initByName = m.initializerMap
    let (dims, _) = m.staticInfo()
    var done = 0
    for i in m.nodes.indices {
      let n = m.nodes[i]
      guard n.opType == "Expand", n.inputs.count == 2, let shapeInit = initByName[n.inputs[1]],
        let shape = dims[n.inputs[0]]
      else { continue }
      let target = try shapeInit.integers(m.file)
      guard shape.count == target.count, shape.allSatisfy({ $0 > 0 }) else { continue }
      var repeats: [Int64] = []
      for (have, want) in zip(shape, target) {
        if want == 1 || want == have {
          repeats.append(1)
        } else if have == 1 {
          repeats.append(want)
        } else {
          repeats = []
          break
        }
      }
      guard repeats.count == shape.count else { continue }
      let name = "\(n.outputs[0])__repeats"
      m.initializers.append(.integers(name, repeats, dims: [Int64(repeats.count)], dataType: OnnxType.int64))
      m.nodes[i].opType = "Tile"
      m.nodes[i].inputs[1] = name
      done += 1
    }
    return done
  }

  /// Feeds the fp16 LayerNorms named in `only` their input times 1/k, one Mul
  /// per distinct input: the same normalization, with the fp16 squares in range.
  static func prescaleLayerNorm(_ m: OnnxModel, only: Set<String>, k: Int = layernormPrescale) throws -> Int {
    let const = "__layernorm_prescale_\(k)"
    let (_, types) = m.staticInfo()
    var new: [Node] = []
    var scaled: [String: String] = [:]
    var done = 0
    for var n in m.nodes {
      if n.opType == "LayerNormalization" && only.contains(n.nameOrEmpty) {
        guard let t = types[n.inputs[0]] else {
          // A policy LayerNorm left in fp16 unscaled overflows on the Neural Engine.
          throw OnnxError("\(n.label): the model does not record the type of \(n.inputs[0])")
        }
        if t != OnnxType.float16 {
          new.append(n)
          continue
        }
        let x = n.inputs[0]
        if scaled[x] == nil {
          scaled[x] = "\(x)__scaled"
          new.append(Node(opType: "Mul", inputs: [x, const], outputs: ["\(x)__scaled"], name: "\(n.nameOrEmpty)__prescale"))
        }
        n.inputs[0] = scaled[x]!
        done += 1
      }
      new.append(n)
    }
    if done > 0 {
      let bits = Float16(1.0 / Double(k)).bitPattern.littleEndian
      m.initializers.append(Initializer(name: const, dims: [], dataType: OnnxType.float16, data: .bytes([UInt8(bits & 0xFF), UInt8(bits >> 8)])))
      m.nodes = new
    }
    return done
  }

  /// The indices, in graph order, of the heads that end the vision trunk:
  /// the largest set of vision nodes with ops in headOps whose outputs are
  /// all read, and read only, by each other or by a Concat that makes a graph
  /// output. Empty when there is no such Concat or the set is too big.
  static func visionHeads(_ m: OnnxModel, vision: Set<String>) -> [Int] {
    let outputs = Set(m.outputs.map(\.name))
    let ends = Set(m.nodes.indices.filter { m.nodes[$0].opType == "Concat" && m.nodes[$0].outputs.contains(where: outputs.contains) })
    if ends.isEmpty { return [] }
    var readers: [String: Set<Int>] = [:]
    for (i, n) in m.nodes.enumerated() {
      for x in n.inputs { readers[x, default: []].insert(i) }
    }
    var region = Set<Int>()
    var grew = true
    while grew {
      grew = false
      for i in m.nodes.indices.reversed() {
        let n = m.nodes[i]
        if region.contains(i) || !vision.contains(n.nameOrEmpty) || !headOps.contains(n.opType)
          || n.outputs.contains(where: outputs.contains)
        {
          continue
        }
        let inside = region.union(ends)
        if n.outputs.allSatisfy({ o in readers[o].map { !$0.isEmpty && $0.isSubset(of: inside) } ?? false }) {
          region.insert(i)
          grew = true
        }
      }
    }
    return region.count <= headMaxNodes ? region.sorted() : []
  }

  /// Runs visionHeads in fp32: what they read from the trunk cast up, their
  /// fp16 weights as fp32 copies, and what they hand the output Concat cast
  /// back down. The Neural Engine cannot run fp32, so CoreML places them on
  /// the GPU or CPU; in fp16 there, road_transform fails the parity gate.
  /// Returns how many nodes moved.
  static func headsInFP32(_ m: OnnxModel, vision: Set<String>) throws -> Int {
    let index = visionHeads(m, vision: vision)
    if index.isEmpty { return 0 }
    let heads = Set(index)
    let initByName = m.initializerMap
    let produced = Set(index.flatMap { m.nodes[$0].outputs })
    let entries = Set(index.flatMap { m.nodes[$0].inputs }.filter { !$0.isEmpty && initByName[$0] == nil && !produced.contains($0) })
    let (_, types) = m.staticInfo()
    for x in entries.sorted() {
      guard let t = types[x] else { throw OnnxError("the model does not record the type of \(x), which its heads read") }
      if t != OnnxType.float16 { return 0 }
    }
    let weights = Set(index.flatMap { m.nodes[$0].inputs }.filter { initByName[$0] != nil })
    if weights.contains(where: { ![OnnxType.float16, OnnxType.float].contains(initByName[$0]!.dataType) }) { return 0 }
    var readOutside = Set<String>()
    for (j, n) in m.nodes.enumerated() where !heads.contains(j) { readOutside.formUnion(n.inputs) }
    let exits = produced.intersection(readOutside)
    var wide: [String: String] = [:]
    for w in weights.sorted() where initByName[w]!.dataType == OnnxType.float16 {
      wide[w] = "\(w)__fp32"
      // layernorm_in_fp32 names its copies the same way; one is enough
      if initByName["\(w)__fp32"] == nil {
        m.initializers.append(try initByName[w]!.widened(name: "\(w)__fp32", m.file))
      }
    }
    var new: [Node] = []
    var cast = Set<String>()
    for (j, var n) in m.nodes.enumerated() {
      guard heads.contains(j) else {
        new.append(n)
        continue
      }
      for (i, x) in n.inputs.enumerated() {
        if entries.contains(x) {
          if cast.insert(x).inserted {
            new.append(
              Node(opType: "Cast", inputs: [x], outputs: ["\(x)__fp32"], name: "\(x)__cast_fp32",
                attributes: [.int("to", Int64(OnnxType.float))]))
          }
          n.inputs[i] = "\(x)__fp32"
        } else if exits.contains(x) {
          n.inputs[i] = "\(x)__fp32"
        } else if let w = wide[x] {
          n.inputs[i] = w
        }
      }
      let back = n.outputs.filter(exits.contains)
      n.outputs = n.outputs.map { exits.contains($0) ? "\($0)__fp32" : $0 }
      new.append(n)
      for o in back {
        new.append(
          Node(opType: "Cast", inputs: ["\(o)__fp32"], outputs: [o], name: "\(o)__cast_fp16",
            attributes: [.int("to", Int64(OnnxType.float16))]))
      }
    }
    m.nodes = new
    // The fp16 originals, unless something outside the heads reads them too.
    m.initializers.removeAll { wide[$0.name] != nil && !readOutside.contains($0.name) }
    // What the heads compute inside is fp32 now; the value infos said fp16.
    let inside = produced.subtracting(exits)
    m.valueInfo.removeAll { inside.contains($0.name) }
    return index.count
  }

  // MARK: gemm_with_transposed_weight

  /// Rewrites MatMul(x, W) + Add(b) as Gemm(x, W.T, b, transB=1), with the
  /// Reshape pair onnxruntime's own fusion adds around a rank-3+ input.
  static func gemmWithTransposedWeight(_ m: OnnxModel) throws -> Int {
    let initByName = m.initializerMap
    var consumers: [String: [Int]] = [:]
    for (i, n) in m.nodes.enumerated() {
      for name in n.inputs { consumers[name, default: []].append(i) }
    }
    let graphOutputs = Set(m.outputs.map(\.name))
    let (dims, _) = m.staticInfo()

    var replacements: [Int: [Node]] = [:]
    var drop = Set<Int>()
    var rewritten = 0
    for (i, node) in m.nodes.enumerated() {
      guard node.opType == "MatMul", node.inputs.count == 2, let weight = initByName[node.inputs[1]],
        let stem = node.outputs.first, !graphOutputs.contains(stem)
      else { continue }
      guard weight.dims.count == 2, weight.dims[0] * weight.dims[1] >= blobMinElements else { continue }
      let after = consumers[stem] ?? []
      guard after.count == 1, m.nodes[after[0]].opType == "Add" else { continue }
      let add = m.nodes[after[0]]
      // The bias has to broadcast over the output's last axis.
      guard let bias = add.inputs.first(where: { initByName[$0] != nil }), initByName[bias]!.dims == [weight.dims[1]]
      else { continue }
      guard let shape = dims[node.inputs[0]] else {
        // The Python asks onnx's shape inference here. Leaving the MatMul
        // changes no arithmetic, only how CoreML stores this weight: as text,
        // which for a large one makes loads slow and big.
        Prep.log.warning(
          "\(node.label): the model does not record the shape of \(node.inputs[0]), so its \(weight.dims) weight "
            + "stays a MatMul; CoreML stores it as text, which is slow to load if it is large")
        continue
      }
      guard shape.count >= 2, shape.last == weight.dims[0], shape.allSatisfy({ $0 > 0 }) else { continue }

      let k = weight.dims[0], n = weight.dims[1]
      let transposed = "\(stem)__wt"
      m.initializers.append(try weight.transposed(name: transposed, m.file))

      var new: [Node] = []
      var a = node.inputs[0]
      if shape.count > 2 {
        let flat = "\(stem)__flat_shape"
        m.initializers.append(.integers(flat, [-1, k], dims: [2], dataType: OnnxType.int64))
        a = "\(stem)__flat"
        new.append(Node(opType: "Reshape", inputs: [node.inputs[0], flat], outputs: [a], name: "\(stem)__reshape_in"))
      }
      let gemmOut = shape.count == 2 ? add.outputs[0] : "\(stem)__gemm"
      new.append(
        Node(
          opType: "Gemm", inputs: [a, transposed, bias], outputs: [gemmOut], name: "\(stem)__gemm",
          attributes: [.int("transB", 1)]))
      if shape.count > 2 {
        let back = "\(stem)__out_shape"
        let outShape = Array(shape.dropLast()) + [n]
        m.initializers.append(.integers(back, outShape, dims: [Int64(outShape.count)], dataType: OnnxType.int64))
        new.append(Node(opType: "Reshape", inputs: [gemmOut, back], outputs: [add.outputs[0]], name: "\(stem)__reshape_out"))
      }
      replacements[i] = new
      drop.insert(after[0])
      rewritten += 1
    }
    if rewritten == 0 { return 0 }

    var rebuilt: [Node] = []
    for (i, node) in m.nodes.enumerated() where !drop.contains(i) {
      rebuilt.append(contentsOf: replacements[i] ?? [node])
    }
    m.nodes = rebuilt
    // The originals the transposed copies replace, and anything else unread.
    let used = Set(m.nodes.flatMap(\.inputs))
    m.initializers.removeAll { !used.contains($0.name) }
    return rewritten
  }
}

extension Node {
  /// How a message names a node: its name, or its op and first output.
  var label: String { nameOrEmpty.isEmpty ? "\(opType) \(outputs.first ?? "")" : nameOrEmpty }
}

enum Prep {
  static let log = JLogger("jetlink.prepare")
}

extension OnnxModel {
  /// Name to initializer, the last of a duplicated name winning as in a dict
  /// comprehension.
  var initializerMap: [String: Initializer] {
    var d: [String: Initializer] = [:]
    for t in initializers { d[t.name] = t }
    return d
  }
}

extension Initializer {
  /// An integer constant of the given type, as numpy_helper.from_array writes one.
  static func integers(_ name: String, _ values: [Int64], dims: [Int64], dataType: Int32) -> Initializer {
    var bytes = [UInt8]()
    if dataType == OnnxType.int32 {
      bytes.reserveCapacity(values.count * 4)
      for v in values { withUnsafeBytes(of: Int32(truncatingIfNeeded: v).littleEndian) { bytes.append(contentsOf: $0) } }
    } else {
      bytes.reserveCapacity(values.count * 8)
      for v in values { withUnsafeBytes(of: v.littleEndian) { bytes.append(contentsOf: $0) } }
    }
    return Initializer(name: name, dims: dims, dataType: dataType, data: .bytes(bytes))
  }

  /// Bytes of tensor data this initializer carries, for progress totals.
  var rawBytes: Int {
    if let raw { return raw.count }
    if case .message(let msg) = field.payload, case .raw(let c)? = msg.first(F.tensorRawData)?.payload { return c.count }
    return 0
  }

  static func elementSize(_ t: Int32) -> Int? {
    switch t {
    case OnnxType.float, OnnxType.int32: 4
    case OnnxType.int64, 11: 8  // 11 = double
    case OnnxType.float16, 4, 5, 16: 2  // uint16, int16, bfloat16
    case OnnxType.uint8, 3, 9: 1  // int8, bool
    default: nil
    }
  }

  /// This float16 tensor as a float32 one named `name`, as numpy's astype
  /// makes it. Converted while the model is written when the values are in
  /// the source file.
  func widened(name: String, _ file: MappedFile) throws -> Initializer {
    guard dataType == OnnxType.float16 else { throw OnnxError("\(self.name) is not float16") }
    if external { throw OnnxError("\(self.name) keeps its data in an external file") }
    let n = count
    if let raw {
      guard raw.count == n * 2 else { throw OnnxError("\(self.name): raw_data is \(raw.count) bytes for \(n) float16") }
      let src = raw.lowerBound
      return Initializer(
        name: name, dims: dims, dataType: OnnxType.float,
        data: .lazy(n * 4) { w in
          // raw_data sits wherever protobuf put it; the conversion wants it aligned.
          let half = UnsafeMutablePointer<Float16>.allocate(capacity: max(n, 1))
          let out = UnsafeMutablePointer<Float>.allocate(capacity: max(n, 1))
          defer {
            half.deallocate()
            out.deallocate()
          }
          UnsafeMutableRawPointer(half).copyMemory(from: file.base + src, byteCount: n * 2)
          Convert.f16ToF32(half, out, count: n)
          try w.write(UnsafeRawBufferPointer(start: out, count: n * 4))
        })
    }
    // Written by an earlier pass, or kept as int32_data.
    var bits: [UInt16]
    if case .message(let msg) = field.payload, case .raw(let c)? = msg.first(F.tensorRawData)?.payload {
      var bytes: [UInt8]
      switch c {
      case .source(let r): bytes = Array(UnsafeRawBufferPointer(start: file.base + r.lowerBound, count: r.count))
      case .bytes(let b): bytes = b
      case .lazy(_, let produce):
        var w = ProtoWriter(memory: file)
        try produce(&w)
        bytes = w.memory
      }
      guard bytes.count == n * 2 else { throw OnnxError("\(self.name): raw_data is \(bytes.count) bytes for \(n) float16") }
      bits = bytes.withUnsafeBytes { b in (0..<n).map { UInt16(littleEndian: b.loadUnaligned(fromByteOffset: 2 * $0, as: UInt16.self)) } }
    } else {
      bits = try float16Bits(file)
    }
    var out = [UInt8]()
    out.reserveCapacity(n * 4)
    for b in bits { withUnsafeBytes(of: Float(Float16(bitPattern: b)).bitPattern.littleEndian) { out.append(contentsOf: $0) } }
    return Initializer(name: name, dims: dims, dataType: OnnxType.float, data: .bytes(out))
  }

  /// This 2-D weight transposed, produced a tensor at a time while the model
  /// is written rather than held for the whole rewrite.
  func transposed(name: String, _ file: MappedFile) throws -> Initializer {
    guard let raw, !external else { throw OnnxError("\(self.name): only raw_data weights can be transposed") }
    guard let e = Initializer.elementSize(dataType) else { throw OnnxError("\(self.name): unsupported type \(dataType)") }
    let rows = Int(dims[0]), cols = Int(dims[1])
    guard raw.count == rows * cols * e else {
      throw OnnxError("\(self.name): raw_data is \(raw.count) bytes for \(rows)x\(cols) of \(e)-byte values")
    }
    let src = raw.lowerBound
    let total = raw.count
    let data = Chunk.lazy(total) { w in
      let out = UnsafeMutableRawPointer.allocate(byteCount: max(total, 1), alignment: 16)
      defer { out.deallocate() }
      transpose(UnsafeRawPointer(file.base + src), out, rows: rows, cols: cols, elementSize: e)
      try w.write(UnsafeRawBufferPointer(start: out, count: total))
    }
    return Initializer(name: name, dims: [dims[1], dims[0]], dataType: dataType, data: data)
  }
}

/// out[c][r] = in[r][c] for a rows x cols matrix, in cache-sized tiles.
func transpose(_ src: UnsafeRawPointer, _ dst: UnsafeMutableRawPointer, rows: Int, cols: Int, elementSize e: Int) {
  let tile = 64
  func run<T>(_: T.Type) {
    let s = src.assumingMemoryBound(to: T.self)
    let d = dst.assumingMemoryBound(to: T.self)
    var r0 = 0
    while r0 < rows {
      let r1 = min(r0 + tile, rows)
      var c0 = 0
      while c0 < cols {
        let c1 = min(c0 + tile, cols)
        for r in r0..<r1 {
          for c in c0..<c1 { d[c * rows + r] = s[r * cols + c] }
        }
        c0 = c1
      }
      r0 = r1
    }
  }
  switch e {
  case 1: run(UInt8.self)
  case 2: run(UInt16.self)
  case 4: run(UInt32.self)
  default: run(UInt64.self)
  }
}
