// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.
//
// Every size on the wire, derived from the model's ONNX metadata. The same
// derivations as jetlink/spec.py; the two ends must agree on every one.

import Foundation

public let modelRunFreq = 20
public let modelContextFreq = 5
public let defaultFrameSkip = modelRunFreq / modelContextFreq  // 4
public let uploadChunk = 4 << 20

public struct SliceRange: Equatable, Sendable {
  public var start: Int
  public var stop: Int
  public var count: Int { stop - start }
}

public struct ModelSpec: Equatable, Sendable {
  public var sha256: String
  public var nbytes: Int
  public var frameSkip: Int
  public var inputShapes: [String: [Int]]
  public var outputShapes: [String: [Int]]
  public var outputSlices: [String: SliceRange]
  public var checkpoint: String?

  public init(
    sha256: String, nbytes: Int, frameSkip: Int, inputShapes: [String: [Int]], outputShapes: [String: [Int]],
    outputSlices: [String: SliceRange], checkpoint: String?
  ) {
    self.sha256 = sha256
    self.nbytes = nbytes
    self.frameSkip = frameSkip
    self.inputShapes = inputShapes
    self.outputShapes = outputShapes
    self.outputSlices = outputSlices
    self.checkpoint = checkpoint
  }

  func shape(_ name: String) throws -> [Int] {
    guard let s = inputShapes[name] else { throw SpecError("the model has no \(name) input") }
    return s
  }

  // MARK: vision

  public var imgShape: [Int] { inputShapes["img"] ?? [] }  // (1, 12, H, W)
  public var nFrames: Int { imgShape[1] / 6 }
  public var modelHW: (Int, Int) { (imgShape[2], imgShape[3]) }
  public var imgBufShape: [Int] { [frameSkip * (nFrames - 1) + 1, 6, imgShape[2], imgShape[3]] }
  /// What `warp` on the comma produces: narrow and wide stacked.
  public var warpedShape: [Int] { [2, 6, imgShape[2], imgShape[3]] }
  public var warpedNbytes: Int { product(warpedShape) }  // uint8

  // MARK: recurrent and scalar inputs

  /// Flattened per-frame feature size. (1,32,32,512) -> 16384.
  public var featDim: Int { product(Array((inputShapes["features_buffer"] ?? []).dropFirst(2))) }

  /// The packed float32 inputs in wire order: desire, traffic_convention, action_t, prev_feat.
  public var packedShapes: [(String, [Int])] {
    let dp = inputShapes["desire_pulse"] ?? [0, 0, 0]
    let fb = inputShapes["features_buffer"] ?? [0]
    return [
      ("desire", [dp[2]]),
      ("traffic_convention", inputShapes["traffic_convention"] ?? []),
      ("action_t", inputShapes["action_t"] ?? []),
      ("prev_feat", [fb[0], featDim]),
    ]
  }

  public var packedSizes: [Int] { packedShapes.map { product($0.1) } }
  public var packedNelem: Int { packedSizes.reduce(0, +) }
  public var packedNbytes: Int { packedNelem * 4 }

  public var featQShape: [Int] {
    let fb = inputShapes["features_buffer"] ?? [0, 0]
    return [frameSkip * fb[1], fb[0], featDim]
  }

  public var desireQShape: [Int] {
    let dp = inputShapes["desire_pulse"] ?? [0, 0, 0]
    return [frameSkip * dp[1], dp[0], dp[2]]
  }

  // MARK: output

  public var outputNelem: Int { product(outputShapes["outputs"] ?? []) }
  public var outputNbytes: Int { outputNelem * 4 }  // float32, as openpilot's JIT returns

  // MARK: wire sizes

  public var inferReqNbytes: Int { Wire.inferReqSize + warpedNbytes + packedNbytes }
  public var inferRespNbytes: Int { Wire.inferRespSize + outputNbytes }

  /// Checks the inputs the queues and the wire depend on are there and sane,
  /// so a model of another shape is refused at load rather than misread.
  public func validate() throws {
    for name in ["img", "big_img", "desire_pulse", "traffic_convention", "action_t", "features_buffer"] {
      let s = try shape(name)
      if s.contains(where: { $0 <= 0 }) { throw SpecError("\(name) has a dynamic shape \(s)") }
    }
    guard imgShape.count == 4, imgShape[1] % 6 == 0, imgShape[1] >= 6 else {
      throw SpecError("img is \(imgShape), expected (1, 6k, H, W)")
    }
    guard inputShapes["big_img"] == imgShape else { throw SpecError("big_img differs from img") }
    guard (inputShapes["desire_pulse"] ?? []).count == 3 else { throw SpecError("desire_pulse is not 3-D") }
    guard (inputShapes["features_buffer"] ?? []).count >= 3 else { throw SpecError("features_buffer is under 3-D") }
    guard frameSkip > 0 else { throw SpecError("frame_skip must be positive") }
    guard outputNelem > 0 else { throw SpecError("the model has no outputs tensor") }
  }

  // MARK: the wire form, as ModelSpec.to_dict / from_dict

  public var json: JSON {
    .object([
      "sha256": .string(sha256),
      "nbytes": .int(nbytes),
      "frame_skip": .int(frameSkip),
      "checkpoint": .str(checkpoint),
      "input_shapes": .object(inputShapes.mapValues { .array($0.map(JSON.int)) }),
      "output_shapes": .object(outputShapes.mapValues { .array($0.map(JSON.int)) }),
      "output_slices": .object(outputSlices.mapValues { .array([.int($0.start), .int($0.stop)]) }),
    ])
  }

  public init(json d: JSON) throws {
    guard let sha = d["sha256"]?.string, let nbytes = d["nbytes"]?.int else {
      throw SpecError("a spec needs sha256 and nbytes")
    }
    func shapes(_ key: String) throws -> [String: [Int]] {
      guard let o = d[key]?.object else { throw SpecError("a spec needs \(key)") }
      return try o.mapValues { v in
        guard let a = v.array?.compactMap(\.int), a.count == v.array?.count else {
          throw SpecError("\(key) holds a non-integer shape")
        }
        return a
      }
    }
    guard let slices = d["output_slices"]?.object else { throw SpecError("a spec needs output_slices") }
    self.sha256 = sha
    self.nbytes = nbytes
    self.frameSkip = d["frame_skip"]?.int ?? defaultFrameSkip
    self.inputShapes = try shapes("input_shapes")
    self.outputShapes = try shapes("output_shapes")
    self.outputSlices = try slices.mapValues { v in
      guard let a = v.array, a.count == 2, let s = a[0].int, let e = a[1].int else {
        throw SpecError("output_slices holds something other than [start, stop]")
      }
      return SliceRange(start: s, stop: e)
    }
    self.checkpoint = d["checkpoint"]?.string
  }

  /// The same spec, stamped with another frame_skip, as {**d, 'frame_skip': x}.
  public func with(frameSkip: Int) -> ModelSpec {
    var s = self
    s.frameSkip = frameSkip
    return s
  }
}

public struct SpecError: Error, CustomStringConvertible {
  public let description: String
  public init(_ d: String) { description = d }
}

@inline(__always)
func product(_ a: [Int]) -> Int { a.reduce(1, *) }

extension ModelSpec {
  /// spec_from_meta: the spec for a model file, from its graph metadata.
  public init(meta: OnnxMeta, sha256: String, nbytes: Int, frameSkip: Int = defaultFrameSkip) throws {
    self.init(
      sha256: sha256, nbytes: nbytes, frameSkip: frameSkip,
      inputShapes: Dictionary(meta.inputs.map { ($0.name, $0.shape) }, uniquingKeysWith: { a, _ in a }),
      outputShapes: Dictionary(meta.outputs.map { ($0.name, $0.shape) }, uniquingKeysWith: { a, _ in a }),
      outputSlices: try meta.outputSlices(), checkpoint: meta.props["model_checkpoint"])
  }
}
