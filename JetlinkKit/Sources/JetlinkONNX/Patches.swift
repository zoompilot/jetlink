import Foundation

#if canImport(os)
  import os
#else
  import JetlinkLog
#endif

/// The graph rewrites of jetlink/onnx_patch.py on the `simplify` branch, on
/// the decoded model. Each one follows its Python function step by step,
/// including the order it appends initializers in and the errors it raises,
/// so the file it leads to is the one Python writes. The reasons for each
/// rewrite are in the Python docstrings; the comments here are about the port.
///
/// There is no shape inferrer in Swift. Where the Python asks onnx's shape
/// inferrer for a shape the file does not record, the rewrite that needed it
/// is skipped here, as the Python skips it when the inferrer finds none;
/// where it asks for a type (the ane-whole passes), the preparation refuses,
/// since the pass would otherwise go a different way. The driving models'
/// exports record every tensor's shape and type, so on them nothing is
/// skipped or refused and the output is the Python's byte for byte.
enum Patches {
  /// Where the vision trunk starts (onnx_patch.IMG_INPUTS).
  static let imageInputs = ["img", "big_img", "new_img", "state_img_q"]
  static let imageInputsRepr = "('img', 'big_img', 'new_img', 'state_img_q')"
  /// Ops that only move image bytes around (onnx_patch.LAYOUT_OPS).
  static let layoutOps: Set<String> = [
    "Concat", "Slice", "Gather", "Reshape", "Unsqueeze", "Squeeze", "Transpose", "Flatten", "Identity", "Expand",
  ]
  static let tinygradDomain = "org.tinygrad"
  static let passthroughOps: Set<String> = ["Contiguous"]
  /// A weight smaller than this stays as it is (onnx_patch.BLOB_MIN_ELEMENTS).
  static let blobMinElements: Int64 = 1024
  /// LayerNorm(x / k) is LayerNorm(x) with epsilon scaled by k^2
  /// (onnx_patch.LAYERNORM_PRESCALE).
  static let layerNormPrescale = 8
  /// The ops heads_in_fp32 moves to fp32 (onnx_patch.HEAD_OPS).
  static let headOps: Set<String> = [
    "Gemm", "MatMul", "LayerNormalization", "Gelu", "Add", "Sub", "Mul", "Div", "Relu", "Sigmoid", "Tanh",
  ]
  /// Larger than that is the trunk itself (onnx_patch.HEAD_MAX_NODES).
  static let headMaxNodes = 64

  /// Where the Python's `log.warning` lines go: the cases a pass declines
  /// and leaves the graph as it was.
  private static let log = Logger(subsystem: "io.zoompilot.jetlink", category: "onnx")

  // MARK: strip_tinygrad_ops

  /// Bypasses tinygrad's layout-hint nodes and returns how many went.
  static func stripTinygradOps(_ g: inout Graph, _ opsets: inout [OpsetImport]) throws -> Int {
    let graphOutputs = Set(g.outputs.map(\.key))
    var removed = 0
    var k = 0
    while k < g.nodes.count {
      let node = g.nodes[k]
      guard node.domainName == tinygradDomain else {
        k += 1
        continue
      }
      guard passthroughOps.contains(node.op) else {
        throw OnnxError("unknown \(tinygradDomain) op \(pyRepr(node.op)); it may not be a no-op, so dropping it is not safe")
      }
      guard node.inputs.count == 1, node.outputs.count == 1, node.attributes.isEmpty else {
        throw OnnxError("\(node.op) is not a plain one-in one-out passthrough")
      }
      let source = node.inputs[0]
      let produced = node.outputs[0]
      if graphOutputs.contains(produced) {
        // The output keeps its name, so its producer takes the name over.
        rename(&g, outputs: source, to: produced)
        rename(&g, inputs: source, to: produced)
      } else {
        rename(&g, inputs: produced, to: source)
      }
      // Nothing before it moved, so it is still at k; the next node is now there.
      g.nodes.remove(at: k)
      removed += 1
    }

    if removed > 0 {
      if let i = opsets.firstIndex(where: { $0.domain == tinygradDomain }) {
        opsets.remove(at: i)
      }
      var live = Set<String>()
      for node in g.nodes {
        live.formUnion(node.inputs)
        live.formUnion(node.outputs)
      }
      g.valueInfo.removeAll { !live.contains($0.key) }
    }
    return removed
  }

  // MARK: patch_uint8_inputs

  static func needsPatch(_ g: Graph) -> Bool {
    g.inputs.contains { $0.elemType == DataType.uint8 }
  }

  /// Retypes the uint8 inputs, the images, to fp16 and drops the head Cast.
  static func patchUint8Inputs(_ g: inout Graph) throws {
    let sources = g.inputs.filter { $0.elemType == DataType.uint8 }.map(\.key)
    guard !sources.isEmpty else {
      throw OnnxError("model has no uint8 inputs; already patched?")
    }

    // _image_dataflow: nodes are in topological order, so one pass sees every
    // producer before its consumers.
    var retyped = Set(sources)
    var casts: [Int] = []
    for (k, node) in g.nodes.enumerated() where node.inputs.contains(where: retyped.contains) {
      if node.op == "Cast" {
        casts.append(k)
      } else if layoutOps.contains(node.op) {
        retyped.formUnion(node.outputs)
      } else {
        throw OnnxError(
          "\(node.op) \(pyRepr(node.displayName)) reads the uint8 images; only layout ops "
            + "and the head Cast are expected there")
      }
    }
    guard !casts.isEmpty else {
      throw OnnxError("could not find the head Cast: no Cast ends the uint8 image chain")
    }
    let graphOutputs = Set(g.outputs.map(\.key))
    for k in casts {
      let cast = g.nodes[k]
      guard let to = cast.attribute("to") else {
        throw OnnxError("head Cast \(cast.displayName) has no 'to' attribute")
      }
      guard to.i == Int64(DataType.float16) else {
        throw OnnxError("head Cast targets \(to.i), expected FLOAT16 (\(DataType.float16))")
      }
      if let out = cast.outputs.first, graphOutputs.contains(out) {
        throw OnnxError("head Cast \(cast.displayName) feeds a graph output; dropping it would rename one")
      }
    }

    // Everything downstream reads each cast's source directly now. Python
    // removes each cast right after its renames; removing them all after all
    // the renames ends the same, since a rename that reaches a cast only
    // touches a node that is dropped anyway.
    for k in casts {
      let cast = g.nodes[k]
      if let source = cast.inputs.first, let produced = cast.outputs.first {
        rename(&g, inputs: produced, to: source)
      }
    }
    for k in casts.reversed() {
      g.nodes.remove(at: k)
    }

    retype(&g.inputs, retyped)
    retype(&g.valueInfo, retyped)
    retype(&g.outputs, retyped)
  }

  private static func retype(_ values: inout [ValueInfo], _ names: Set<String>) {
    for i in values.indices where names.contains(values[i].key) {
      var type = values[i].type ?? TypeInfo()
      if type.tensor == nil {
        // Setting tensor_type in Python selects it in TypeProto's oneof,
        // which clears whichever other member was set.
        type.extras.removeAll { [4, 5, 8, 9].contains($0.number) }
        type.tensor = TensorType()
      }
      type.tensor!.elemType = DataType.float16
      values[i].type = type
    }
  }

  // MARK: normalize_gather_indices

  /// Rewrites Gathers whose constant index is negative to the positive
  /// equivalent, each with an index initializer of its own.
  static func normalizeGatherIndices(_ g: inout Graph, _ src: Source) throws -> Int {
    let initializers = lastIndexByName(g.initializers)
    let gathers = g.nodes.indices.filter { k in
      let n = g.nodes[k]
      return n.op == "Gather" && n.inputs.count > 1 && initializers[n.inputs[1]] != nil
    }
    let dims = staticDims(g)
    var rewritten = 0
    for k in gathers {
      let node = g.nodes[k]
      let index = g.initializers[initializers[node.inputs[1]]!]
      guard DataType.isInteger(index.elementType) else { continue }
      let values = try Elements.integers(index, src)
      guard let lowest = values.min(), lowest < 0 else { continue }
      let axis = node.attribute("axis")?.i ?? 0
      guard let shape = dims[node.inputs[0]], axis < Int64(shape.count) else { continue }
      // Python indexes the shape tuple with the axis, so a negative one counts from the end.
      let position = axis < 0 ? Int(axis) + shape.count : Int(axis)
      guard position >= 0 else {
        throw OnnxError("\(node.displayName): Gather axis \(axis) is out of range for a rank-\(shape.count) input")
      }
      let size = shape[position]
      guard size > 0 else { continue }
      let fixed = values.map { $0 < 0 ? $0 + size : $0 }
      if fixed.contains(where: { $0 < 0 || $0 >= size }) {
        throw OnnxError(
          "\(node.displayName): Gather index \(pyList(values, index.dims)) out of range "
            + "for axis \(axis) of size \(size)")
      }
      let name = "\(node.outputs.first ?? "")__index"
      g.initializers.append(
        Tensor(
          name: name, dims: index.dims, dataType: index.dataType,
          raw: .owned(try Elements.encode(fixed, as: index.elementType, for: name))))
      g.nodes[k].inputs[1] = name
      rewritten += 1
    }
    return rewritten
  }

  // MARK: expand_to_tile

  /// Rewrites an Expand with a constant shape of the input's rank, which only
  /// repeats size-1 axes, as the equivalent Tile.
  static func expandToTile(_ g: inout Graph, _ src: Source) throws -> Int {
    let initializers = lastIndexByName(g.initializers)
    let dims = staticDims(g)
    var done = 0
    for k in g.nodes.indices {
      let n = g.nodes[k]
      guard n.op == "Expand", n.inputs.count == 2, let shapeIndex = initializers[n.inputs[1]] else { continue }
      let shape = dims[n.inputs[0]]
      let target = try Elements.integers(g.initializers[shapeIndex], src)
      guard let shape, shape.count == target.count, !shape.contains(where: { $0 <= 0 }) else { continue }
      var repeats: [Int64] = []
      var repeatsOnly = true
      for (have, want) in zip(shape, target) {
        if want == 1 || want == have {
          repeats.append(1)
        } else if have == 1 {
          repeats.append(want)
        } else {
          repeatsOnly = false
          break
        }
      }
      guard repeatsOnly else { continue }
      let name = "\(n.outputs.first ?? "")__repeats"
      g.initializers.append(int64Tensor(repeats, name))
      g.nodes[k].opType = "Tile"
      g.nodes[k].inputs[1] = name
      done += 1
    }
    return done
  }

  // MARK: gemm_with_transposed_weight

  /// Rewrites `MatMul(x, W)` followed by `Add(b)` as `Gemm(x, W.T, b,
  /// transB=1)`, with a Reshape either side when x has more than two axes.
  ///
  /// The transposed weights are not made here. Each new initializer is a
  /// recipe over the original's bytes in the mapped file, carried out block
  /// by block when the part is written, so no two are ever in memory at once.
  static func gemmWithTransposedWeight(_ g: inout Graph, _ src: Source) throws -> Int {
    let initializers = lastIndexByName(g.initializers)
    var consumers: [String: [Int]] = [:]
    for (k, node) in g.nodes.enumerated() {
      for name in node.inputs {
        consumers[name, default: []].append(k)
      }
    }
    let outputs = Set(g.outputs.map(\.key))
    let candidates = g.nodes.indices.filter { k in
      let n = g.nodes[k]
      return n.op == "MatMul" && n.inputs.count == 2 && initializers[n.inputs[1]] != nil
        && !(n.outputs.first.map(outputs.contains) ?? true)
    }
    let dims = staticDims(g)

    var replacements: [Int: [Node]] = [:]
    var drop = Set<Int>()
    var rewritten = 0
    for k in candidates {
      let node = g.nodes[k]
      let weight = g.initializers[initializers[node.inputs[1]]!]
      guard weight.dims.count == 2, weight.dims[0] * weight.dims[1] >= blobMinElements else { continue }
      let after = consumers[node.outputs[0]] ?? []
      guard after.count == 1, g.nodes[after[0]].op == "Add" else { continue }
      let add = g.nodes[after[0]]
      // The bias has to broadcast over the output's last axis; anything else
      // is a real elementwise Add and not a Gemm's C.
      guard let bias = add.inputs.first(where: { initializers[$0] != nil }),
        g.initializers[initializers[bias]!].dims == [weight.dims[1]]
      else { continue }
      guard let shape = dims[node.inputs[0]], shape.count >= 2, shape.last == weight.dims[0],
        !shape.contains(where: { $0 <= 0 })
      else { continue }

      let stem = node.outputs[0]
      let transposed = "\(stem)__wt"
      g.initializers.append(try transposedTensor(weight, transposed, src))

      var new: [Node] = []
      var a = node.inputs[0]
      if shape.count > 2 {
        let flat = "\(stem)__flat_shape"
        g.initializers.append(int64Tensor([-1, weight.dims[0]], flat))
        a = "\(stem)__flat"
        new.append(Node(inputs: [node.inputs[0], flat], outputs: [a], name: "\(stem)__reshape_in", opType: "Reshape"))
      }
      let gemmOut = shape.count == 2 ? add.outputs[0] : "\(stem)__gemm"
      new.append(
        Node(
          inputs: [a, transposed, bias], outputs: [gemmOut], name: "\(stem)__gemm", opType: "Gemm",
          attributes: [.int("transB", 1)]))
      if shape.count > 2 {
        let back = "\(stem)__out_shape"
        g.initializers.append(int64Tensor(Array(shape.dropLast()) + [weight.dims[1]], back))
        new.append(
          Node(
            inputs: [gemmOut, back], outputs: [add.outputs[0]], name: "\(stem)__reshape_out",
            opType: "Reshape"))
      }
      replacements[k] = new
      drop.insert(after[0])
      rewritten += 1
    }

    guard rewritten > 0 else { return 0 }

    var rebuilt: [Node] = []
    rebuilt.reserveCapacity(g.nodes.count + 2 * rewritten)
    for (k, node) in g.nodes.enumerated() where !drop.contains(k) {
      if let new = replacements[k] {
        rebuilt.append(contentsOf: new)
      } else {
        rebuilt.append(node)
      }
    }
    g.nodes = rebuilt
    dropUnusedInitializers(&g)
    return rewritten
  }

  /// The weights the rewrite left behind (_drop_unused_initializers).
  static func dropUnusedInitializers(_ g: inout Graph) {
    var used = Set<String>()
    for node in g.nodes {
      used.formUnion(node.inputs)
    }
    g.initializers.removeAll { !used.contains($0.key) }
  }

  /// `numpy_helper.from_array(np.ascontiguousarray(W.T), name)`: the same
  /// element type, the dims swapped, the data as raw_data.
  private static func transposedTensor(_ weight: Tensor, _ name: String, _ src: Source) throws -> Tensor {
    guard let size = DataType.size(weight.elementType) else {
      throw OnnxError("\(weight.key): cannot transpose a weight of element type \(weight.elementType)")
    }
    let transpose = Transpose(
      elements: try transposeElements(of: weight, src), rows: Int(weight.dims[0]), cols: Int(weight.dims[1]), elementSize: size)
    return Tensor(
      name: name, dims: [weight.dims[1], weight.dims[0]], dataType: weight.dataType,
      raw: .transposed(transpose))
  }

  /// A weight's elements as a transpose reads them while it is written:
  /// raw_data or a packed float or double field where it lies, a weight in
  /// the other typed fields decoded then.
  static func transposeElements(of weight: Tensor, _ src: Source) throws -> Transpose.Elements {
    let type = weight.elementType
    guard let size = DataType.size(type) else {
      throw OnnxError("\(weight.key): cannot transpose a weight of element type \(type)")
    }
    let expected = weight.elementCount * size
    switch weight.raw {
    case .source(let r):
      guard r.count == expected else { throw sizeMismatch(weight, r.count, expected) }
      return .source(r)
    case .owned(let bytes):
      guard bytes.count == expected else { throw sizeMismatch(weight, bytes.count, expected) }
      return .owned(bytes)
    case .transposed:
      throw OnnxError("\(weight.key): a transposed weight cannot be transposed again")
    case .widened:
      throw OnnxError("\(weight.key): a widened weight cannot be transposed")
    case nil:
      // float_data and double_data, packed in one run, are already the raw
      // layout. Anything else is decoded when it is written.
      let field = Elements.typedField(type)
      if let field, field == 4 || field == 10, type == DataType.float || type == DataType.double,
        let ranges = weight.typed[field], ranges.count == 1, ranges[0].count == expected
      {
        return .source(ranges[0])
      }
      let count = try Elements.typedCount(weight, src)
      guard count == weight.elementCount else { throw sizeMismatch(weight, count * size, expected) }
      return .typed(weight)
    }
  }

  private static func sizeMismatch(_ t: Tensor, _ have: Int, _ want: Int) -> OnnxError {
    OnnxError("initializer \(t.key) has \(have) bytes of data where its dims \(t.dims) need \(want)")
  }

  // MARK: prescale_layernorm

  /// Feeds every fp16 LayerNormalization on the policy side its input times
  /// 1/k, one Mul per distinct input, and returns how many norms.
  ///
  /// Python looks a norm's input type up in the file and asks the shape
  /// inferrer for one the file does not record; Swift has no inferrer, so a
  /// policy norm whose input type is not recorded is an error rather than a
  /// silent difference. The driving models' exports record every tensor's.
  static func prescaleLayerNorm(_ g: inout Graph, k: Int = layerNormPrescale) throws -> Int {
    let const = "__layernorm_prescale_\(k)"
    let mask = try Split.visionMask(g)
    var norms = Set<Int>()
    for (i, node) in g.nodes.enumerated() where node.op == "LayerNormalization" && !mask[i] {
      norms.insert(i)
    }
    var already = Set<String>()
    for node in g.nodes where node.op == "Mul" && node.inputs.count == 2 && node.inputs[1] == const {
      if let out = node.outputs.first { already.insert(out) }
    }
    let types = staticTypes(g)
    var new: [Node] = []
    new.reserveCapacity(g.nodes.count + norms.count)
    var scaled: [String: String] = [:]
    var done = 0
    for (i, node) in g.nodes.enumerated() {
      var node = node
      if norms.contains(i), let x = node.inputs.first, !already.contains(x) {
        guard let type = types[x] else {
          throw OnnxError("the export records no type for \(x); this model cannot be prepared on iPhone or iPad")
        }
        if type == DataType.float16 {
          if scaled[x] == nil {
            scaled[x] = "\(x)__scaled"
            new.append(Node(inputs: [x, const], outputs: ["\(x)__scaled"], name: "\(x)__prescale", opType: "Mul"))
          }
          node.inputs[0] = scaled[x]!
          done += 1
        }
      }
      new.append(node)
    }
    if done > 0 {
      // np.float16(1 / k), a scalar: no dims, the two bytes as raw_data.
      let bits = Float16(1 / Float(k)).bitPattern.littleEndian
      g.initializers.append(
        Tensor(
          name: const, dims: [], dataType: DataType.float16,
          raw: .owned([UInt8(truncatingIfNeeded: bits), UInt8(truncatingIfNeeded: bits >> 8)])))
      g.nodes = new
    }
    return done
  }

  // MARK: vision_heads

  /// The indices, in graph order, of the heads that end the vision trunk:
  /// grown backwards to a fixed point from the Concats that make a graph
  /// output. Empty when there is no such Concat or the set would exceed
  /// headMaxNodes.
  static func visionHeads(_ g: Graph) throws -> [Int] {
    let mask = try Split.visionMask(g)
    let outputs = Set(g.outputs.map(\.key))
    var ends = Set<Int>()
    for (i, n) in g.nodes.enumerated() where n.op == "Concat" && n.outputs.contains(where: outputs.contains) {
      ends.insert(i)
    }
    guard !ends.isEmpty else { return [] }
    var readers: [String: Set<Int>] = [:]
    for (i, n) in g.nodes.enumerated() {
      for x in n.inputs {
        readers[x, default: []].insert(i)
      }
    }
    var region = Set<Int>()
    var grew = true
    while grew {
      grew = false
      for i in g.nodes.indices.reversed() {
        let n = g.nodes[i]
        if region.contains(i) || !mask[i] || !headOps.contains(n.op) || n.outputs.contains(where: outputs.contains) {
          continue
        }
        let allowed = region.union(ends)
        let joins = n.outputs.allSatisfy { o in
          guard let r = readers[o], !r.isEmpty else { return false }
          return r.isSubset(of: allowed)
        }
        if joins {
          region.insert(i)
          grew = true
        }
      }
    }
    return region.count <= headMaxNodes ? region.sorted() : []
  }

  // MARK: heads_in_fp32

  /// Runs `visionHeads` in fp32: a Cast up before the first head reads each
  /// entry, fp32 copies of the fp16 head weights, a Cast down after each
  /// exit. Returns how many nodes moved, 0 with a warning when it found no
  /// heads it could move.
  ///
  /// The fp32 copies are not made here: each is a recipe over the fp16
  /// bytes, carried out band by band when the part is written.
  static func headsInFP32(_ g: inout Graph, _ src: Source) throws -> Int {
    let index = try visionHeads(g)
    guard !index.isEmpty else {
      log.warning(
        "heads_in_fp32: no heads found after the vision trunk (no output Concat they feed, or more than \(headMaxNodes) nodes); the whole graph stays fp16"
      )
      return 0
    }
    let heads = Set(index)
    let initializers = lastIndexByName(g.initializers)
    var produced = Set<String>()
    for i in index {
      produced.formUnion(g.nodes[i].outputs)
    }
    var entries = Set<String>()
    var weights = Set<String>()
    for i in index {
      for x in g.nodes[i].inputs where !x.isEmpty {
        if initializers[x] != nil {
          weights.insert(x)
        } else if !produced.contains(x) {
          entries.insert(x)
        }
      }
    }
    let types = staticTypes(g)
    // Python asks the shape inferrer for an entry the file does not type;
    // Swift has none (see prescaleLayerNorm).
    if let untyped = entries.filter({ types[$0] == nil }).sorted(by: pyLess).first {
      throw OnnxError("the export records no type for \(untyped); this model cannot be prepared on iPhone or iPad")
    }
    let notFP16 = entries.filter { types[$0] != DataType.float16 }.sorted(by: pyLess)
    guard notFP16.isEmpty else {
      log.warning("heads_in_fp32: a head reads \(pyStrList(notFP16)), which is not fp16; the whole graph stays fp16")
      return 0
    }
    let odd = weights.filter {
      let type = g.initializers[initializers[$0]!].elementType
      return type != DataType.float16 && type != DataType.float
    }.sorted(by: pyLess)
    guard odd.isEmpty else {
      log.warning("heads_in_fp32: head weights \(pyStrList(odd)) are neither fp16 nor fp32; the whole graph stays fp16")
      return 0
    }
    var readOutside = Set<String>()
    for (j, n) in g.nodes.enumerated() where !heads.contains(j) {
      readOutside.formUnion(n.inputs)
    }
    let exits = produced.intersection(readOutside)
    var wide: [String: String] = [:]
    for w in weights.sorted(by: pyLess) {
      let weight = g.initializers[initializers[w]!]
      guard weight.elementType == DataType.float16 else { continue }
      wide[w] = "\(w)__fp32"
      if initializers[wide[w]!] == nil {
        g.initializers.append(try widenedTensor(weight, wide[w]!, src))
      }
    }
    var new: [Node] = []
    new.reserveCapacity(g.nodes.count + entries.count + exits.count)
    var cast = Set<String>()
    for (j, node) in g.nodes.enumerated() {
      guard heads.contains(j) else {
        new.append(node)
        continue
      }
      var n = node
      for i in n.inputs.indices {
        let x = n.inputs[i]
        if entries.contains(x) {
          if !cast.contains(x) {
            cast.insert(x)
            new.append(
              Node(
                inputs: [x], outputs: ["\(x)__fp32"], name: "\(x)__cast_fp32", opType: "Cast",
                attributes: [.int("to", Int64(DataType.float))]))
          }
          n.inputs[i] = "\(x)__fp32"
        } else if exits.contains(x) {
          n.inputs[i] = "\(x)__fp32"
        } else if let w = wide[x] {
          n.inputs[i] = w
        }
      }
      let back = n.outputs.filter(exits.contains)
      for i in n.outputs.indices where exits.contains(n.outputs[i]) {
        n.outputs[i] = "\(n.outputs[i])__fp32"
      }
      new.append(n)
      for o in back {
        new.append(
          Node(
            inputs: ["\(o)__fp32"], outputs: [o], name: "\(o)__cast_fp16", opType: "Cast",
            attributes: [.int("to", Int64(DataType.float16))]))
      }
    }
    g.nodes = new
    // The fp16 originals, unless something outside the heads reads them too.
    g.initializers.removeAll { wide[$0.key] != nil && !readOutside.contains($0.key) }
    // What the heads compute inside is fp32 now; the value infos said fp16.
    let inside = produced.subtracting(exits)
    g.valueInfo.removeAll { inside.contains($0.key) }
    return index.count
  }

  /// `numpy_helper.from_array(numpy_helper.to_array(w).astype(np.float32),
  /// name)`: the same dims, FLOAT, the data as raw_data.
  private static func widenedTensor(_ weight: Tensor, _ name: String, _ src: Source) throws -> Tensor {
    let count = weight.elementCount
    let expected = count * 2
    let elements: Widen.Elements
    switch weight.raw {
    case .source(let r):
      guard r.count == expected else { throw sizeMismatch(weight, r.count, expected) }
      elements = .source(r)
    case .owned(let bytes):
      guard bytes.count == expected else { throw sizeMismatch(weight, bytes.count, expected) }
      elements = .owned(bytes)
    case .transposed(let t):
      guard t.byteCount == expected, t.elementSize == 2 else { throw sizeMismatch(weight, t.byteCount, expected) }
      elements = .transposed(t)
    case .widened:
      throw OnnxError("\(weight.key): a widened weight cannot be widened again")
    case nil:
      let have = try Elements.typedCount(weight, src)
      guard have == count else { throw sizeMismatch(weight, have * 2, expected) }
      elements = .typed(weight)
    }
    return Tensor(name: name, dims: weight.dims, dataType: DataType.float, raw: .widened(Widen(elements: elements, count: count)))
  }

  // MARK: Neural Engine policy rewrites

  /// What a masked position adds to its attention score: far enough below
  /// any score that its softmax weight is 0, and small enough that the sum
  /// stays finite in fp16.
  static let additiveMask: Float = -10_000

  /// Rewrites `Softmax(Where(mask, fill, x))` with a constant mask and a
  /// constant fill of `additiveMask` or lower as `Softmax(x + m)`, m being 0
  /// where x is kept and `additiveMask` where it was filled.
  ///
  /// The Neural Engine's compiler on macOS 27 takes no `select`, which is
  /// what CoreML makes of the Where, and the attention around each one falls
  /// to the CPU with it. A masked score's weight is 0 either way. A mask that
  /// fills a whole row is left alone: that row's weights depend on the fill.
  ///
  /// Not in the Python: this pass is Swift's alone.
  static func maskedFillAsAdd(_ g: inout Graph, _ src: Source) throws -> Int {
    let initializers = lastIndexByName(g.initializers)
    var producers: [String: Int] = [:]
    var consumers: [String: [Int]] = [:]
    for (k, node) in g.nodes.enumerated() {
      for name in node.outputs { producers[name] = k }
      for name in node.inputs { consumers[name, default: []].append(k) }
    }
    let outputs = Set(g.outputs.map(\.key))
    let dims = staticDims(g)

    var done = 0
    for k in g.nodes.indices {
      let n = g.nodes[k]
      guard n.op == "Where", n.inputs.count == 3, n.outputs.count == 1, !outputs.contains(n.outputs[0]) else { continue }
      // The mask: a constant, or the Not of one.
      var filledWhere = true
      var maskName = n.inputs[0]
      if initializers[maskName] == nil, let p = producers[maskName], g.nodes[p].op == "Not", g.nodes[p].inputs.count == 1 {
        maskName = g.nodes[p].inputs[0]
        filledWhere = false
      }
      guard let maskIndex = initializers[maskName], let fillIndex = initializers[n.inputs[1]] else { continue }
      let mask = g.initializers[maskIndex]
      let fill = g.initializers[fillIndex]
      guard mask.elementType == DataType.bool, !mask.dims.isEmpty, !mask.dims.contains(where: { $0 <= 0 }),
        fill.elementType == DataType.float16 || fill.elementType == DataType.float, fill.elementCount == 1
      else { continue }
      let fillValue = try Elements.floats(Elements.littleEndian(fill, src), as: fill.elementType, for: fill.key)[0]
      guard fillValue <= additiveMask else { continue }
      // Only a softmax over the mask's last axis reads it.
      let readers = consumers[n.outputs[0]] ?? []
      guard readers.count == 1, g.nodes[readers[0]].op == "Softmax", let shape = dims[n.inputs[2]],
        shape.count >= mask.dims.count, !shape.contains(where: { $0 <= 0 })
      else { continue }
      let axis = g.nodes[readers[0]].attribute("axis")?.i ?? -1
      guard axis == -1 || axis == Int64(shape.count - 1) else { continue }

      let filled = try Elements.littleEndian(mask, src).map { ($0 != 0) == filledWhere }
      let row = Int(mask.dims.last!)
      guard row == Int(shape.last!) else { continue }
      let wholeRow = stride(from: 0, to: filled.count, by: row).contains { start in
        filled[start..<(start + row)].allSatisfy { $0 }
      }
      guard !wholeRow else { continue }

      let name = "\(n.outputs[0])__mask"
      g.initializers.append(
        Tensor(
          name: name, dims: mask.dims, dataType: fill.elementType,
          raw: .owned(Elements.encode(filled.map { $0 ? additiveMask : 0 }, as: fill.elementType))))
      g.nodes[k] = Node(inputs: [n.inputs[2], name], outputs: n.outputs, name: n.name, opType: "Add")
      done += 1
    }
    guard done > 0 else { return 0 }

    // The Nots nothing reads now, then the masks and fills only they and the Wheres read.
    var read = Set(g.outputs.map(\.key))
    for node in g.nodes { read.formUnion(node.inputs) }
    var gone = Set<String>()
    g.nodes.removeAll { node in
      guard node.op == "Not", node.outputs.allSatisfy({ !read.contains($0) }) else { return false }
      gone.formUnion(node.outputs)
      return true
    }
    g.valueInfo.removeAll { gone.contains($0.key) }
    dropUnusedInitializers(&g)
    return done
  }

  /// Rewrites a `MatMul(x, W)` of a four-axis x by a constant matrix, which
  /// `gemmWithTransposedWeight` left because no bias follows it, as the same
  /// MatMul on x with its two middle axes folded into one, a Reshape either
  /// side.
  ///
  /// The Neural Engine's compiler on macOS 27 takes the three-axis product
  /// and not the four-axis one.
  ///
  /// Not in the Python: this pass is Swift's alone.
  static func foldMatMulAxes(_ g: inout Graph) -> Int {
    let initializers = lastIndexByName(g.initializers)
    let dims = staticDims(g)
    var rebuilt: [Node] = []
    rebuilt.reserveCapacity(g.nodes.count)
    var done = 0
    for n in g.nodes {
      guard n.op == "MatMul", n.inputs.count == 2, n.outputs.count == 1, let w = initializers[n.inputs[1]],
        g.initializers[w].dims.count == 2, let shape = dims[n.inputs[0]], shape.count == 4,
        !shape.contains(where: { $0 <= 0 }), shape[3] == g.initializers[w].dims[0]
      else {
        rebuilt.append(n)
        continue
      }
      let stem = n.outputs[0]
      let folded = "\(stem)__folded_shape"
      let back = "\(stem)__out_shape"
      g.initializers.append(int64Tensor([shape[0], shape[1] * shape[2], shape[3]], folded))
      g.initializers.append(int64Tensor(Array(shape.dropLast()) + [g.initializers[w].dims[1]], back))
      rebuilt.append(Node(inputs: [n.inputs[0], folded], outputs: ["\(stem)__folded"], name: "\(stem)__fold", opType: "Reshape"))
      rebuilt.append(
        Node(inputs: ["\(stem)__folded", n.inputs[1]], outputs: ["\(stem)__product"], name: n.name, opType: "MatMul", attributes: n.attributes))
      rebuilt.append(Node(inputs: ["\(stem)__product", back], outputs: [stem], name: "\(stem)__unfold", opType: "Reshape"))
      done += 1
    }
    if done > 0 { g.nodes = rebuilt }
    return done
  }

  // MARK: helpers

  /// Static shapes from what the file carries (_static_dims), inputs first,
  /// then value_info, then outputs, a later entry replacing an earlier one.
  /// A dimension without a value is -1.
  static func staticDims(_ g: Graph) -> [String: [Int64]] {
    var dims: [String: [Int64]] = [:]
    for vi in g.inputs + g.valueInfo + g.outputs {
      if let shape = vi.shape {
        dims[vi.key] = shape.dims.map { $0.value ?? -1 }
      }
    }
    return dims
  }

  /// Element types from what the file carries (_static_types), in the same
  /// order as `staticDims`. A value info without an element type is absent.
  static func staticTypes(_ g: Graph) -> [String: Int32] {
    var types: [String: Int32] = [:]
    for vi in g.inputs + g.valueInfo + g.outputs where vi.elemType != 0 {
      types[vi.key] = vi.elemType
    }
    return types
  }

  /// `{t.name: t for t in g.initializer}` as positions: a repeated name maps
  /// to its last tensor, as a dict built that way does.
  static func lastIndexByName(_ tensors: [Tensor]) -> [String: Int] {
    var map: [String: Int] = [:]
    for (i, t) in tensors.enumerated() {
      map[t.key] = i
    }
    return map
  }

  static func int64Tensor(_ values: [Int64], _ name: String) -> Tensor {
    var bytes: [UInt8] = []
    bytes.reserveCapacity(values.count * 8)
    for v in values {
      withUnsafeBytes(of: v.littleEndian) { bytes.append(contentsOf: $0) }
    }
    return Tensor(name: name, dims: [Int64(values.count)], dataType: DataType.int64, raw: .owned(bytes))
  }

  private static func rename(_ g: inout Graph, inputs from: String, to: String) {
    for k in g.nodes.indices {
      for i in g.nodes[k].inputs.indices where g.nodes[k].inputs[i] == from {
        g.nodes[k].inputs[i] = to
      }
    }
  }

  private static func rename(_ g: inout Graph, outputs from: String, to: String) {
    for k in g.nodes.indices {
      for i in g.nodes[k].outputs.indices where g.nodes[k].outputs[i] == from {
        g.nodes[k].outputs[i] = to
      }
    }
  }
}

// MARK: Python's formatting, for error messages that match

/// Python's repr of a str.
func pyRepr(_ s: String) -> String {
  let quote: Character = (s.contains("'") && !s.contains("\"")) ? "\"" : "'"
  var out = String(quote)
  for scalar in s.unicodeScalars {
    switch scalar {
    case "\\": out += "\\\\"
    case "\n": out += "\\n"
    case "\r": out += "\\r"
    case "\t": out += "\\t"
    default:
      if Character(scalar) == quote {
        out += "\\\(quote)"
      } else if scalar.value < 0x20 || scalar.value == 0x7f {
        out += String(format: "\\x%02x", scalar.value)
      } else {
        out.unicodeScalars.append(scalar)
      }
    }
  }
  out.append(quote)
  return out
}

/// numpy's `tolist()` of integers, as Python prints it.
func pyList(_ values: [Int64], _ dims: [Int64]) -> String {
  guard let first = dims.first else { return values.first.map(String.init) ?? "[]" }
  let rest = Array(dims.dropFirst())
  let stride = max(1, rest.reduce(1) { $0 * Int($1) })
  let items = (0..<Int(first)).map { i -> String in
    if rest.isEmpty { return String(values[i]) }
    return pyList(Array(values[(i * stride)..<((i + 1) * stride)]), rest)
  }
  return "[" + items.joined(separator: ", ") + "]"
}

/// Python's print of a list of str.
func pyStrList(_ values: [String]) -> String {
  "[" + values.map(pyRepr).joined(separator: ", ") + "]"
}

/// Python's sort order for str: by code point, with no normalisation.
func pyLess(_ a: String, _ b: String) -> Bool {
  a.unicodeScalars.lexicographicallyPrecedes(b.unicodeScalars) { $0.value < $1.value }
}
