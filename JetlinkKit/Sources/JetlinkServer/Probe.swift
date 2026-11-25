import Foundation
import JetlinkKit

/// The loaded engine's outputs on a fixed run of frames, for holding one
/// processor's outputs to another's on the device itself
/// (`OutputParity.compare`).
public struct ProbeCapture: Sendable {
  /// Each frame's whole driving output.
  public let outputs: [[Float]]
  public let slices: [(name: String, range: Range<Int>)]
}

/// Stand-in frames, the same on every run: scripts/verify_parity.py's
/// `make_inputs`. Gradients and moving blobs rather than white noise, which
/// drives a vision network into activations no road ever produces. The noise
/// is this file's own generator, not numpy's, so the frames match another run
/// here and not the Python script's.
enum ProbeFrames {
  static func warped(_ spec: ModelSpec, frame i: Int, noise: inout SplitMix) -> [UInt8] {
    let (h, w) = spec.modelHW
    let shape = spec.warpedShape
    var out = [UInt8](repeating: 0, count: spec.warpedBytes)
    var at = 0
    for cam in 0..<shape[0] {
      for ch in 0..<shape[1] {
        let phase = 0.7 * Double(i) + 1.3 * Double(ch) + 2.1 * Double(cam)
        for y in 0..<h {
          for x in 0..<w {
            let base = 96 + 64 * sin(Double(x) / 37.0 + phase) + 48 * cos(Double(y) / 23.0 - phase) + 24 * sin(Double(x + y) / 61.0)
            out[at] = UInt8(min(max(base + 6 * noise.normal(), 0), 255))
            at += 1
          }
        }
      }
    }
    return out
  }

  static func packed(_ spec: ModelSpec, frame i: Int) -> [Float] {
    var out = [Float](repeating: 0, count: spec.packedCount)
    for (name, range, _) in spec.packedLayout {
      switch name {
      case "traffic_convention":
        for (k, value) in [Float(1), 0].prefix(range.count).enumerated() { out[range.lowerBound + k] = value }
      case "action_t":
        // Action horizons in seconds; modeld sends 0.2 to 0.4 s on a real car.
        let values = [Float(0.25 + 0.1 * sin(0.5 * Double(i))), Float(0.35 + 0.1 * cos(0.5 * Double(i)))]
        for (k, value) in values.prefix(range.count).enumerated() { out[range.lowerBound + k] = value }
      case "desire":
        // a pulse every eighth frame through the real desires; index 0 is "none"
        if i % 8 == 0, range.count > 1 {
          out[range.lowerBound + 1 + (i / 8) % (range.count - 1)] = 1
        }
      default: break
      }
    }
    return out
  }
}

/// A small seeded generator: the probe's frames must be the same every run.
struct SplitMix {
  private var state: UInt64

  init(seed: UInt64) { state = seed }

  mutating func next() -> UInt64 {
    state &+= 0x9E37_79B9_7F4A_7C15
    var z = state
    z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
    z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
    return z ^ (z >> 31)
  }

  /// Uniform in (0, 1].
  mutating func unit() -> Double { (Double(next() >> 11) + 1) / 9_007_199_254_740_992 }

  mutating func normal() -> Double {
    (-2 * log(unit())).squareRoot() * cos(2 * .pi * unit())
  }
}

extension EngineHost {
  /// As many frames as scripts/verify_parity.py captures.
  public static let probeFrames = 32

  /// Runs the loaded engine over `frames` fixed frames from a reset, the
  /// hidden state fed back as a drive does, and returns each frame's driving
  /// output. Holds the engine as a benchmark does (`withEngineHeld`).
  /// Blocks; call it off the main thread.
  public func probe(frames: Int = EngineHost.probeFrames) throws -> ProbeCapture {
    try withEngineHeld(to: "compare") { l in
      let spec = l.spec
      var noise = SplitMix(seed: 0)
      var outputs: [[Float]] = []
      lock.withLock { l.staging.reset() }
      for i in 0..<frames {
        var output = [Float](repeating: 0, count: spec.outputCount)
        let frame = try runFrame(
          l, warped: ProbeFrames.warped(spec, frame: i, noise: &noise), packed: ProbeFrames.packed(spec, frame: i), into: &output)
        guard frame.finite else { throw HostError.failed("frame \(i) of the comparison was not finite") }
        outputs.append(output)
      }
      return ProbeCapture(outputs: outputs, slices: spec.outputSlices.map { ($0.name, $0.range(in: spec.outputCount)) })
    }
  }
}
